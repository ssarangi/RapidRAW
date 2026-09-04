#!/usr/bin/env python3
"""Package the official BioCLIP 2 model for RapidRAW's local ONNX runtime.

The script deliberately uses the pinned TreeOfLife-200M revision used by the
upstream BioCLIP 2 release.  It exports the ViT-L/14 image encoder and writes
the complete 867k-species taxonomy as normalized, sharded f32 embeddings.

Usage:
  python scripts/export_bioclip2_onnx.py --output-dir /tmp/bioclip-v2

Dependencies: torch, open_clip_torch, onnx, numpy, huggingface_hub
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
from huggingface_hub import hf_hub_download


MODEL_ID = "hf-hub:imageomics/bioclip-2"
MODEL_REVISION = "2957b322090f9cb17ae72c71981c7218a28d81e0"
TAXONOMY_REPOSITORY = "imageomics/TreeOfLife-200M"
TAXONOMY_REVISION = "a8f38b4388579862c56ae57d6f094c2ac0e92e12"
TAXONOMY_EMBEDDINGS = "embeddings/txt_emb_species.npy"
TAXONOMY_LABELS = "embeddings/txt_emb_species.json"
# Below GitHub's per-release-asset limit, with enough headroom for tooling.
MAX_EMBEDDING_PART_BYTES = 1_000_000_000

RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


def export_vision_encoder(model: torch.nn.Module, output_dir: Path) -> None:
    output_path = output_dir / "vision_encoder.onnx"
    print(f"Exporting BioCLIP 2 ViT-L/14 image encoder to {output_path}...", flush=True)
    visual = model.visual.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    # The legacy exporter emits an external .data file automatically once the
    # tensor payload exceeds ONNX's 2 GiB protobuf limit.
    torch.onnx.export(
        visual,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print("Vision encoder export complete.", flush=True)


def normalize_taxon_label(entry: list[object]) -> dict[str, str]:
    ranks, common_name = entry
    if not isinstance(ranks, list) or len(ranks) != len(RANKS):
        raise ValueError(f"Unexpected Tree-of-Life taxonomy label: {entry!r}")
    values = [str(value).strip() for value in ranks]
    genus, species = values[-2:]
    scientific_name = " ".join(value for value in (genus, species) if value)
    if not scientific_name:
        for rank, value in reversed(list(zip(RANKS, values))):
            if value:
                scientific_name = value
                taxon_rank = rank
                break
        else:
            raise ValueError(f"Tree-of-Life taxonomy label has no name: {entry!r}")
    else:
        taxon_rank = "species" if species else "genus"
    result = {"scientificName": scientific_name, "taxonRank": taxon_rank}
    if isinstance(common_name, str) and common_name.strip():
        result["commonName"] = common_name.strip()
    return result


def package_taxonomy(output_dir: Path) -> None:
    print("Downloading pinned BioCLIP 2 Tree-of-Life taxonomy...", flush=True)
    labels_path = hf_hub_download(
        TAXONOMY_REPOSITORY,
        TAXONOMY_LABELS,
        repo_type="dataset",
        revision=TAXONOMY_REVISION,
    )
    embeddings_path = hf_hub_download(
        TAXONOMY_REPOSITORY,
        TAXONOMY_EMBEDDINGS,
        repo_type="dataset",
        revision=TAXONOMY_REVISION,
    )
    with open(labels_path, encoding="utf-8") as source:
        labels = [normalize_taxon_label(entry) for entry in json.load(source)]

    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.dtype != np.float32:
        raise ValueError(f"Expected a two-dimensional float32 taxonomy matrix, got {embeddings.shape} {embeddings.dtype}")
    # Upstream stores features as [dimension, taxonomy_count]. RapidRAW scans
    # each taxonomy row, so transpose the memory-mapped view without copying it.
    if len(labels) != embeddings.shape[1]:
        raise ValueError(f"Taxonomy labels ({len(labels)}) and embeddings ({embeddings.shape[1]}) differ")
    embeddings = embeddings.T

    # The published matrix has a zero-padded tail. Those rows can never score
    # as a taxonomy match; retaining them would violate the runtime invariant
    # that every candidate has a usable normalized embedding.
    valid_count = len(labels)
    for start in range(0, len(embeddings), 8192):
        chunk = np.asarray(embeddings[start : start + 8192], dtype=np.float32)
        norms = np.linalg.norm(chunk, axis=1)
        invalid = np.flatnonzero(~np.isfinite(norms) | (norms <= 1e-8))
        if invalid.size:
            valid_count = start + int(invalid[0])
            break
    if valid_count != len(labels):
        remaining = np.asarray(embeddings[valid_count:], dtype=np.float32)
        if np.any(np.isfinite(remaining).all(axis=1) & (np.linalg.norm(remaining, axis=1) > 1e-8)):
            raise ValueError("Tree-of-Life taxonomy has invalid embeddings between valid rows")
        print(f"Excluding {len(labels) - valid_count:,} zero-padded taxonomy rows.", flush=True)
        labels = labels[:valid_count]
        embeddings = embeddings[:valid_count]

    labels_output = output_dir / "species_labels.json"
    labels_output.write_text(json.dumps(labels, separators=(",", ":")), encoding="utf-8")

    dimension = int(embeddings.shape[1])
    rows_per_part = max(1, MAX_EMBEDDING_PART_BYTES // (dimension * np.dtype("<f4").itemsize))
    parts = []
    for part_index, start in enumerate(range(0, embeddings.shape[0], rows_per_part)):
        end = min(start + rows_per_part, embeddings.shape[0])
        name = f"species_embeddings.{part_index:03d}.bin"
        part_path = output_dir / name
        row_bytes = dimension * np.dtype("<f4").itemsize
        expected_bytes = (end - start) * row_bytes
        existing_bytes = part_path.stat().st_size if part_path.exists() else 0
        if existing_bytes > expected_bytes or existing_bytes % row_bytes:
            raise ValueError(f"Existing taxonomy part has an invalid size: {part_path}")
        resume_row = start + existing_bytes // row_bytes
        if resume_row == end:
            print(f"Taxonomy part {part_index + 1} is already complete.", flush=True)
            parts.append(name)
            continue
        mode = "ab" if existing_bytes else "wb"
        print(f"Writing normalized taxonomy part {part_index + 1}: rows {resume_row:,}-{end:,}...", flush=True)
        with part_path.open(mode) as destination:
            for chunk_start in range(resume_row, end, 8192):
                chunk_end = min(chunk_start + 8192, end)
                chunk = np.asarray(embeddings[chunk_start:chunk_end], dtype=np.float32)
                norms = np.linalg.norm(chunk, axis=1, keepdims=True)
                if not np.all(np.isfinite(norms)) or np.any(norms <= 1e-8):
                    raise ValueError("Taxonomy contains a non-finite or zero-length embedding")
                (chunk / norms).astype("<f4", copy=False).tofile(destination)
        parts.append(name)

    manifest = {
        "embeddingDimension": dimension,
        "embeddingParts": parts,
        "taxonomyCount": len(labels),
        "source": {
            "model": MODEL_ID,
            "modelRevision": MODEL_REVISION,
            "taxonomyRepository": TAXONOMY_REPOSITORY,
            "taxonomyRevision": TAXONOMY_REVISION,
        },
    }
    (output_dir / "taxonomy_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Packaged {len(labels):,} taxa with {dimension}-dimensional embeddings.", flush=True)


def finalize_existing_taxonomy(output_dir: Path) -> None:
    """Validate completed shards and write a manifest after an interrupted run."""
    labels = json.loads((output_dir / "species_labels.json").read_text(encoding="utf-8"))
    parts = sorted(path.name for path in output_dir.glob("species_embeddings.*.bin"))
    if not labels or not parts:
        raise ValueError("No completed BioCLIP 2 taxonomy files were found")
    total_bytes = sum((output_dir / part).stat().st_size for part in parts)
    if total_bytes % (len(labels) * np.dtype("<f4").itemsize):
        raise ValueError("Embedding files do not contain an integral vector dimension")
    dimension = total_bytes // (len(labels) * np.dtype("<f4").itemsize)
    manifest = {
        "embeddingDimension": dimension,
        "embeddingParts": parts,
        "taxonomyCount": len(labels),
        "source": {
            "model": MODEL_ID,
            "modelRevision": MODEL_REVISION,
            "taxonomyRepository": TAXONOMY_REPOSITORY,
            "taxonomyRevision": TAXONOMY_REVISION,
        },
    }
    (output_dir / "taxonomy_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Validated {len(labels):,} taxa with {dimension}-dimensional embeddings.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--skip-vision-export",
        action="store_true",
        help="Reuse an already exported vision_encoder.onnx in --output-dir",
    )
    parser.add_argument(
        "--finalize-existing-taxonomy",
        action="store_true",
        help="Validate existing taxonomy shards and only write taxonomy_manifest.json",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID}@{MODEL_REVISION}...", flush=True)
    if args.skip_vision_export:
        if not (args.output_dir / "vision_encoder.onnx").is_file():
            raise ValueError("--skip-vision-export requires an existing vision_encoder.onnx")
        print("Reusing existing vision_encoder.onnx.", flush=True)
    else:
        model, _ = open_clip.create_model_from_pretrained(
            MODEL_ID, device="cpu", return_transform=True
        )
        export_vision_encoder(model, args.output_dir)
    if args.finalize_existing_taxonomy:
        finalize_existing_taxonomy(args.output_dir)
    else:
        package_taxonomy(args.output_dir)
    print(f"BioCLIP 2 package is ready in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
