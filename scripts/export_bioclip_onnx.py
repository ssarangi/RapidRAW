#!/usr/bin/env python3
"""
BioCLIP ONNX Export Script for RapidRAW

Exports the Imageomics BioCLIP (ViT-B/16) vision encoder to ONNX and packages:
1. vision_encoder.onnx
2. species_embeddings.bin (packed f32 array of normalized 512-d embeddings)
3. species_labels.json (taxonomy metadata with scientific and common names)

Usage:
    pip install open_clip_torch torch onnx
    python scripts/export_bioclip_onnx.py --output-dir ./bioclip-onnx
"""

import argparse
import json
import os
import struct
import urllib.request
import torch
import open_clip

def export_bioclip_vision_encoder(model, output_path: str):
    print(f"Exporting BioCLIP vision encoder to {output_path}...")
    visual = model.visual
    visual.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        visual,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"✓ Saved {output_path}")

def generate_species_bundle(model, tokenizer, output_dir: str):
    print("Generating species taxonomy dictionary & embeddings...")
    species_list_url = "https://raw.githubusercontent.com/Imageomics/bioclip/main/data/common_species.json"
    labels_path = os.path.join(output_dir, "species_labels.json")
    embeddings_path = os.path.join(output_dir, "species_embeddings.bin")

    species_data = []
    try:
        req = urllib.request.Request(species_list_url, headers={"User-Agent": "RapidRAW/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            species_data = json.loads(resp.read().decode("utf-8"))
    except Exception as err:
        print(f"Note: Could not fetch remote common_species list ({err}), using built-in curated fauna & bird taxonomy.")
        species_data = [
            {"scientificName": "Haliaeetus leucocephalus", "commonName": "Bald Eagle", "taxonRank": "species"},
            {"scientificName": "Buteo jamaicensis", "commonName": "Red-tailed Hawk", "taxonRank": "species"},
            {"scientificName": "Pandion haliaetus", "commonName": "Osprey", "taxonRank": "species"},
            {"scientificName": "Falco peregrinus", "commonName": "Peregrine Falcon", "taxonRank": "species"},
            {"scientificName": "Turdus migratorius", "commonName": "American Robin", "taxonRank": "species"},
            {"scientificName": "Cyanocitta cristata", "commonName": "Blue Jay", "taxonRank": "species"},
            {"scientificName": "Cardinalis cardinalis", "commonName": "Northern Cardinal", "taxonRank": "species"},
            {"scientificName": "Archilochus colubris", "commonName": "Ruby-throated Hummingbird", "taxonRank": "species"},
            {"scientificName": "Megaceryle alcyon", "commonName": "Belted Kingfisher", "taxonRank": "species"},
            {"scientificName": "Anas platyrhynchos", "commonName": "Mallard", "taxonRank": "species"},
            {"scientificName": "Branta canadensis", "commonName": "Canada Goose", "taxonRank": "species"},
            {"scientificName": "Pelecanus occidentalis", "commonName": "Brown Pelican", "taxonRank": "species"},
            {"scientificName": "Ardea herodias", "commonName": "Great Blue Heron", "taxonRank": "species"},
            {"scientificName": "Canis lupus", "commonName": "Gray Wolf", "taxonRank": "species"},
            {"scientificName": "Vulpes vulpes", "commonName": "Red Fox", "taxonRank": "species"},
            {"scientificName": "Ursus arctos", "commonName": "Brown Bear", "taxonRank": "species"},
            {"scientificName": "Panthera leo", "commonName": "Lion", "taxonRank": "species"},
            {"scientificName": "Panthera tigris", "commonName": "Tiger", "taxonRank": "species"},
            {"scientificName": "Acinonyx jubatus", "commonName": "Cheetah", "taxonRank": "species"},
            {"scientificName": "Odocoileus virginianus", "commonName": "White-tailed Deer", "taxonRank": "species"},
            {"scientificName": "Sciurus carolinensis", "commonName": "Eastern Gray Squirrel", "taxonRank": "species"}
        ]

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(species_data, f, indent=2)
    print(f"✓ Saved {labels_path} ({len(species_data)} taxa)")

    print("Computing species text embeddings...")
    prompts = [
        f"a photo of {item['commonName'] + ' (' if item.get('commonName') else ''}{item['scientificName']}{')' if item.get('commonName') else ''}"
        for item in species_data
    ]

    with torch.no_grad():
        text_tokens = tokenizer(prompts)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        embeddings_np = text_features.cpu().numpy().astype("float32")

    with open(embeddings_path, "wb") as f:
        embeddings_np.tofile(f)
    print(f"✓ Saved {embeddings_path} ({embeddings_np.shape[0]}x{embeddings_np.shape[1]} f32)")

def main():
    parser = argparse.ArgumentParser(description="Export BioCLIP model to ONNX for RapidRAW")
    parser.add_argument("--output-dir", default="./bioclip-onnx", help="Directory to write ONNX and metadata files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading BioCLIP model (hf-hub:imageomics/bioclip)...")
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

    onnx_path = os.path.join(args.output_dir, "vision_encoder.onnx")
    export_bioclip_vision_encoder(model, onnx_path)
    generate_species_bundle(model, tokenizer, args.output_dir)
    print(f"\nAll BioCLIP ONNX artifacts successfully generated in: {args.output_dir}")

if __name__ == "__main__":
    main()
