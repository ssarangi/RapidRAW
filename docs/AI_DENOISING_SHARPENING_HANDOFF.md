# RapidRAW AI Denoising and Sharpening Handoff

Status: Research-backed implementation proposal  
Audience: Antigravity / RapidRAW maintainers  
Research date: 2026-08-26

## Decision

Build this as two deliberately separate, non-destructive capabilities:

1. **Raw-first AI denoise** for difficult RAW files, producing a derived linear asset before normal editing.
2. **RGB restoration** for developed images, with distinct controls for denoise, deblur, and upscale. Do not market upscaling or generative detail synthesis as sharpening.

The recommended first production path is:

| Priority | Capability | Recommended implementation | Why |
| --- | --- | --- | --- |
| 1 | RAW AI denoise | RawNIND / UtNet2 model family, using Darktable's published packaging and preprocessing as the interoperability reference | It operates before demosaic on Bayer data and is built around paired RAW noise data. |
| 2 | RGB AI denoise | NAFNet SIDD width-32 ONNX | A small, documented ONNX package with fixed tile behavior and permissive model license. |
| 3 | Conservative detail enlargement | RealPLKSR 2x, then 4x | Its published Darktable package intentionally uses pre-GAN weights to reduce invented texture. |
| 4 | Optional experimental deblur | Restormer or MPRNet | Both have public code and weights, but should be opt-in because deblur changes image evidence more aggressively than denoise. |

Do **not** use Intel Open Image Denoise as RapidRAW's photographic RAW denoiser. Intel OIDN is a high-quality local library for rendered beauty/HDR buffers; it is not trained or documented as a camera-sensor RAW denoise/demosaic model. It may be useful later for a render-oriented experimental filter, but it is not a substitute for a camera-photo model. [OIDN documentation](https://www.openimagedenoise.org/documentation.html)

## Why Raw-First Matters

Sensor noise should be addressed before demosaic when the source is a supported RAW mosaic. Demosaic spreads noise across channels and makes subsequent denoising less physically direct. Darktable's current neural-restore design makes the same separation: Bayer images use a combined raw denoise/demosaic inference path, while non-Bayer sensors fall back to a linear RGB path; both are emitted as a new derivative and leave the original unchanged. [Darktable neural restore](https://docs.darktable.org/usermanual/development/en/module-reference/utility-modules/shared/neural-restore/)

The linked wildlife-focused article is a useful warning, not benchmark evidence: it stresses that AI restoration predicts plausible detail and therefore needs a fidelity-first workflow for feathers, fur, and other forensic-looking texture. Its proof of concept sends a Darktable-produced denoised derivative into RawTherapee rather than replacing the RAW original. [MarcRPhoto workflow article](https://marcrphoto.wordpress.com/2026/06/29/can-darktables-ai-raw-denoise-produce-a-better-starting-point-for-rawtherapee/)

RapidRAW should adopt that discipline: **source RAW remains immutable; analysis output is a versioned derivative with a visible provenance record.**

## Model Assessment

### RAW denoise: RawNIND / UtNet2

**Recommendation: first serious integration and benchmark target.**

Darktable's `rawdenoise-nind` package documents the two relevant variants of the RawNIND work:

- Bayer path: black/white normalize each channel, pack the CFA as `R, G1, G2, B`, crop to dimensions divisible by 16, infer, gain-match, then convert camera RGB to linear Rec.2020.
- Non-Bayer path: use normal RAW development to linear Rec.2020, crop to multiples of 16, then infer in linear RGB.

The upstream work introduces paired RAW images across multiple camera sensors and explicitly evaluates a raw Bayer path and a more broadly compatible linear-RGB path. [RawNIND paper](https://arxiv.org/abs/2501.08924)  The Darktable model card is unusually valuable because it specifies model inputs, checkpoint variants, preprocessing, output scaling, licenses, data provenance, and ONNX conversion assumptions. [RawNIND package card](https://github.com/darktable-org/darktable-ai/blob/master/models/rawdenoise-nind/README.md)

Constraints:

- Bayer model and preprocessing are model-specific. Do not feed arbitrary demosaiced RGB into it.
- Output gain matching is required for the documented Bayer package.
- Tile halo, CFA phase, black level, white level, white balance, and crop origin are correctness requirements, not optimization details.
- X-Trans, Foveon, monochrome, and unsupported CFA patterns need explicit fallback behavior. Do not silently run Bayer inference on them.
- The upstream model card says GPL-3.0. RapidRAW is AGPL-3.0, which is generally compatible with GPL-3.0 when the combined work is conveyed under AGPL-3.0, but model-weight and training-data terms must still be reviewed and recorded before distribution.

### RGB denoise: NAFNet SIDD width-32

**Recommendation: first downloadable RGB denoiser.**

NAFNet is a compact restoration architecture introduced as a deliberately simple baseline. Its public code covers denoising, deblurring, and restoration tasks. [NAFNet paper](https://arxiv.org/abs/2204.04676)  Darktable has already produced a model card for a practical NAFNet SIDD package: FP32 ONNX, input/output `[1, 3, 768, 768]`, values in `[0,1]`, and explicit tile requirements. [Darktable NAFNet model card](https://github.com/darktable-org/darktable-ai/blob/master/models/denoise-nafnet/README.md)

Why it is a good Phase 1 RGB model:

- ONNX Runtime matches RapidRAW's existing local inference direction.
- Static tile dimensions make memory behavior predictable.
- The model and code are MIT licensed according to the published model card.
- It is intended for real-image denoising through SIDD, not only synthetic Gaussian noise.

Limitations:

- SIDD is smartphone imagery, not a guarantee of optimal results for every interchangeable-lens camera sensor.
- It should run after demosaic in linear RGB, before display-referred contrast/grading, unless the user explicitly chooses an export-oriented late-stage denoise.
- It needs overlap tiling and blend/stitch logic to avoid seams.

### RGB denoise alternatives

| Model | Role | Decision |
| --- | --- | --- |
| NIND denoise | Real-image RGB alternative with strong Darktable packaging support | Add as an evaluation pack after NAFNet. |
| Restormer | Transformer restoration model for denoise, deblur, derain, and defocus blur | Keep as an experimental multi-task pack; verify export, tile behavior, speed, and weight terms before shipping. [Official repository](https://github.com/swz30/Restormer) |
| MIRNet-v2 | Strong published restoration family | Benchmark candidate, not the first implementation. [Paper](https://arxiv.org/abs/2205.01649) |
| DRUNet / DPIR | Classical deep-denoiser baseline and plug-and-play restoration prior | Useful internal quality baseline; not a first UX-facing model without a maintained package/manifest. |

### Sharpening, deblur, and upscaling are different operations

RapidRAW must not expose a single ambiguous `AI Sharpen` button.

- **Capture sharpening** restores modest local contrast after demosaic. Keep the existing deterministic sharpen/detail tools as the default.
- **Deblur** tries to invert optical/camera-motion blur. It can alter fine structures. Restormer and MPRNet have published deblur capabilities, but should always be labeled `Experimental deblur` and produce a derivative. [MPRNet](https://github.com/swz30/MPRNet)
- **Super-resolution** creates additional pixels. It is not evidence recovery and must be treated as an enlargement/export tool.

For conservative super-resolution, prefer **RealPLKSR** over GAN-finetuned Real-ESRGAN variants. Darktable's RealPLKSR package uses an approximately 7M-parameter pure CNN and explicitly selects MSSIM-pretrain weights, noting lower texture-hallucination risk than GAN-finetuned output. It has documented static ONNX tile sizes for 2x and 4x. [RealPLKSR model card](https://github.com/darktable-org/darktable-ai/blob/master/models/upscale-realplksr/README.md)

Real-ESRGAN is a valid open-source general restoration model under BSD-3-Clause, but it was trained with synthetic degradations and is more appropriate as an explicitly creative/enhancement option than the default for wildlife or evidence-sensitive photo work. [Real-ESRGAN repository](https://github.com/xinntao/Real-ESRGAN)

## Product Architecture

### Preserve the original and make derivatives first-class

Add a durable catalog record such as `image_derivatives` rather than overwriting a source path:

```text
image_derivatives
  id
  source_image_id
  operation_kind          -- raw_denoise | rgb_denoise | deblur | upscale
  model_id
  model_revision
  recipe_json             -- strength, tile size, overlap, preprocessing, execution provider
  input_hash
  output_path
  output_hash
  output_format           -- dng | tiff | exr
  width, height
  state                   -- queued | running | paused | completed | failed | cancelled
  created_at, completed_at
```

Rules:

1. A source image and a derived image are linked but independently browsable.
2. Every output records model ID, immutable model checksum/revision, settings, source file hash/mtime, compute backend, and output checksum.
3. A source edit never modifies the generated derivative. Re-run creates a new derivative revision or replaces only an unreferenced failed/incomplete file.
4. Sidecar recipes remain portable where feasible; the catalog stores operational history and availability state.
5. The UI always provides `Original`, `Derivative`, and split/zoom comparison before export or promote-to-edit actions.

### Model-pack contract

Extend the visual-model registry instead of making ad-hoc downloads. Borrow Darktable's useful discipline: model packages include a manifest, conversion provenance, static tensor contract, normalization, tile size, overlap, format, license, training-data provenance, and a validation sample. Darktable's packaging repository is a strong reference for this model-card and ONNX-validation workflow. [darktable-ai](https://github.com/darktable-org/darktable-ai)

Required manifest fields beyond RapidRAW's current pack metadata:

```json
{
  "id": "rawnind-utnet2-bayer",
  "revision": "immutable-upstream-or-conversion-revision",
  "task": "raw_denoise",
  "input": { "layout": "NCHW", "channels": 4, "multipleOf": 16, "range": "0..1" },
  "output": { "colorSpace": "linear-rec2020", "format": "float32" },
  "tiling": { "tileSize": 0, "overlap": 0, "padMode": "reflect" },
  "preprocess": "rawnind_bayer_v1",
  "license": { "model": "GPL-3.0", "code": "GPL-3.0", "weights": "verify" },
  "artifacts": [{ "name": "model.onnx", "sha256": "..." }],
  "validation": { "inputHash": "...", "outputHash": "..." }
}
```

Do not mark a model usable merely because expected filenames exist. Validate ONNX load, tensor names/types/shapes, output range sanity, checksum, and one deterministic fixture before enabling it.

### Execution and jobs

Use the existing catalog background-job system:

- Job kinds: `raw_denoise`, `rgb_denoise`, `deblur`, `upscale`.
- Jobs are cancellable. Long tiled operations are pausable between tiles/images, not in the middle of an ONNX inference call.
- Set concurrency to one image/model session by default; allow bounded parallelism only after GPU memory measurement.
- Persist image/derivative state after every completed output, not only at batch end.
- Write to a temporary sibling file, fsync, validate, and atomically rename before marking completed.
- Network/NAS source reads must remain outside the UI thread and follow existing stall-tolerant file-access policy.

## UX Proposal

Put daily use in the Library/image workflow, not Settings. Settings only manages model installation, cache location, compute provider, and defaults.

### Per-image actions

`Enhance` menu:

- `RAW denoise...` when the source supports the selected raw model.
- `RGB denoise...`
- `Experimental deblur...`
- `Upscale 2x...` / `Upscale 4x...`
- `Compare derivative`
- `Show provenance`

Each operation opens a compact tool surface with model selector, strength, estimated output type/size, and an explicit `Create derivative` action. A preview must show original/processed crops at 100% or 200%, never only a fitted full image.

### Batch actions

From Library selection or a catalog smart collection:

- `Run RAW denoise on selected`
- `Run RGB denoise on selected`
- `Queue failed/pending only`
- `Pause`, `resume`, `cancel`

The existing status bar should show current item, tile/image progress, estimated output count, failures, and a click-through modal. A completed job opens a review set; it must not silently replace the source selection.

### Defaults and warnings

- Default strength: conservative, with source/processed blend available for RAW denoise.
- Warn before 4x upscale with projected dimensions and storage.
- Show a `Detail may be reconstructed` label for deblur/upscale and any GAN/diffusion option.
- Do not show AI denoise as a cure for soft focus or motion blur.

## CLI Contract

The existing `rapidraw-cli` should gain one-shot commands that use the same recipe and job engine where possible:

```bash
# Inspect packs and their validation status
rapidraw-cli models list
rapidraw-cli models verify --id rawnind-utnet2-bayer

# Create non-destructive derivatives
rapidraw-cli enhance raw-denoise --model rawnind-utnet2-bayer --strength 0.75 --output-dir /path/out input.ARW
rapidraw-cli enhance denoise --model denoise-nafnet-sidd --strength 0.60 --output-dir /path/out input.tif
rapidraw-cli enhance upscale --model realplksr-x2 --output-dir /path/out input.tif

# Catalog batch; returns a persistent job ID
rapidraw-cli catalog enhance --database catalog.db --query saved:"High ISO wildlife" --operation raw-denoise
rapidraw-cli jobs list --database catalog.db --state running
```

Exit codes: `0` completed, `1` processing/model failure, `2` invalid request or incompatible source, `130` user cancellation. JSON output should include source, derivative, model revision, recipe, duration, and warnings.

## Phased Implementation Plan

### Phase A: infrastructure and test harness

1. Add `image_derivatives`, a model-manifest schema, derivative state, and atomic output helper.
2. Add a visual regression corpus: low/high ISO Bayer RAW, X-Trans RAW, wildlife feathers, foliage, skin, night city, flat-color areas, and intentional motion blur.
3. Build crop comparison, output provenance, job telemetry, and per-model fixture validation before shipping any model.
4. Keep deterministic sharpening as the default and run all neural operations as derivatives.

### Phase B: RawNIND evaluation integration

1. Implement Bayer packing, black/white normalization, CFA phase preservation, mod-16 crop/pad, gain matching, and color conversion against the documented RawNIND reference.
2. Implement a clearly labeled linear-RGB fallback for non-Bayer sources.
3. Start with TIFF/EXR output if valid linear DNG writing is not proven; add DNG only with correct metadata/color handling.
4. Compare output with Darktable's released rawdenoise package on fixtures before exposing a default action.

### Phase C: NAFNet RGB denoise

1. Package a checksum-pinned FP32 ONNX model with exact `768x768` tile contract.
2. Implement reflect padding, overlap, weighted stitching, output clipping, and input/output color-space verification.
3. Add conservative strength mixing in linear RGB and a processed-vs-original crop review.

### Phase D: enhancement/deblur experiments

1. Add RealPLKSR 2x as the first enlargement option, then 4x after memory and seam tests.
2. Add Restormer or MPRNet behind an Experimental deblur label.
3. Keep GAN or diffusion models disabled by default and separate from fidelity-preserving tools.

## Evaluation Gates

Do not select on PSNR alone. Require all of the following:

1. **Correctness:** no CFA phase/color cast, tile seams, NaNs, clipped highlights, orientation loss, or corrupted metadata.
2. **Fidelity:** blinded 100%/200% review of feathers, fur, foliage, skin, stars, text, and flat gradients. Report destructive artifacts separately from noise reduction.
3. **Repeatability:** same input + model checksum + recipe produces the same output hash on the same execution provider, or a documented bounded numeric tolerance where providers differ.
4. **Performance:** capture wall time, peak RAM/VRAM, tile size, provider, source resolution, and effective megapixels per second.
5. **Workflow:** the user can find original, derivative, model revision, and recipe from the Library without opening Settings.
6. **Legal/provenance:** every distributed model records weight license, source URL, checksum, conversion script revision, training-data provenance, and any unresolved limitation.

Suggested acceptance corpus: at least 100 images across cameras and ISO ranges, with a dedicated wildlife subset. For raw denoise, compare both against a conventional profiled denoiser and Darktable's RawNIND output. For deblur/upscale, require explicit reviewer sign-off because no-reference metrics cannot prove generated texture is real.

## Risks and Non-goals

- Do not promise that AI restores lost detail. Denoising can suppress noise while also suppressing or inventing texture.
- Do not use a model trained on a different sensor domain without labeling it as an experimental fallback.
- Do not run deep restoration automatically during import, thumbnail generation, or catalog scans.
- Do not overwrite RAWs, `.rrdata`, XMP, or current edits.
- Do not make cloud inference part of the default path.
- Do not collapse denoise, sharpen, deblur, and super-resolution into one score or one slider.

## Source Ledger

Primary and first-party sources consulted:

- [Darktable neural restore documentation](https://docs.darktable.org/usermanual/development/en/module-reference/utility-modules/shared/neural-restore/), accessed 2026-08-26. RAW-vs-RGB positioning, derived-output workflow, model activation, strengths, and limits.
- [Darktable AI overview](https://docs.darktable.org/usermanual/development/en/special-topics/ai/overview/), accessed 2026-08-26. Local narrow-AI task model and supported task categories.
- [darktable-ai model packaging repository](https://github.com/darktable-org/darktable-ai), accessed 2026-08-26. ONNX packaging, reproducibility, validation, licensing/provenance criteria, and current denoise/upscale packs.
- [RawNIND / UtNet2 model card](https://github.com/darktable-org/darktable-ai/blob/master/models/rawdenoise-nind/README.md), accessed 2026-08-26. Bayer and linear preprocessing, tensor constraints, output handling, license, and limitations.
- [RawNIND paper](https://arxiv.org/abs/2501.08924), 2025. Raw paired-data motivation and raw/linear denoise approaches.
- [NAFNet model card](https://github.com/darktable-org/darktable-ai/blob/master/models/denoise-nafnet/README.md), accessed 2026-08-26. ONNX tensor and tiling contract, SIDD training scope, and licensing.
- [NAFNet paper](https://arxiv.org/abs/2204.04676), 2022. Architecture and restoration scope.
- [RealPLKSR model card](https://github.com/darktable-org/darktable-ai/blob/master/models/upscale-realplksr/README.md), accessed 2026-08-26. Conservative non-GAN model choice, tile contracts, and known limits.
- [Restormer official repository](https://github.com/swz30/Restormer), accessed 2026-08-26. Public multi-task restoration implementation and weights.
- [MPRNet official repository](https://github.com/swz30/MPRNet), accessed 2026-08-26. Public multi-stage denoise/deblur restoration implementation.
- [Real-ESRGAN official repository](https://github.com/xinntao/Real-ESRGAN), accessed 2026-08-26. General restoration scope and BSD-3-Clause code license.
- [Intel Open Image Denoise documentation](https://www.openimagedenoise.org/documentation.html), accessed 2026-08-26. Library purpose and beauty/HDR image API.
- [MarcRPhoto Darktable-to-RawTherapee article](https://marcrphoto.wordpress.com/2026/06/29/can-darktables-ai-raw-denoise-produce-a-better-starting-point-for-rawtherapee/), 2026-06-29. Practitioner workflow observation and fidelity caution; not treated as controlled benchmark evidence.
