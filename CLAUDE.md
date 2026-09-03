# Project notes for Claude

## Lesson: TS error counts are not a diff

During the "Continue Session" crash investigation, a `ReferenceError: Can't find
variable: useSettingsStore` in `src/hooks/useAppNavigation.ts` caused Continue
Session to fail. The actual bug was trivial: `useSettingsStore` was used via
`.getState()` in three places but never imported.

It took three wrong theories (stale `activeLibraryDbPath`, missing
`active_catalog_root_id` field, a settings-persistence race) and three rounds of
"still broken" from the user before the real cause was found — and only because
the user was asked directly for the console error text, which named the exact
file, line, and undefined variable.

The reason it took so long: throughout the session, `npx tsc --noEmit -p . | wc -l`
was used as a "no new errors introduced" check, comparing the total count before
and after edits (e.g. "152 → 149, looks fine"). That count was never actually
opened and read. The missing-import error was `TS2304: Cannot find name
'useSettingsStore'` and was sitting in that baseline the entire time, plainly
visible, just never inspected.

**Rule going forward:**
- Never treat a tsc/build error *count* as a stand-in for a diff. If the count
  matters, capture the actual list before and after and diff the lists, not
  just the totals — an unchanged count can still hide a severe, unrelated,
  crash-causing error.
- When a user reports a runtime crash (not a logic/UX bug), ask for the actual
  error message/stack trace from devtools/console *before* proposing a fix.
  `ReferenceError: Can't find variable: X` almost always means a missing
  import or a variable used before declaration — check that first, not last,
  and don't reach for state-race or persistence-timing theories until a
  missing-reference explanation has been ruled out by looking at the code.
- If a second attempted fix doesn't resolve a reported crash, stop theorizing
  and get the real error text instead of proposing a third theory.

## Lesson: stringly-typed enum values silently rot across merges

A dark-thumbnail bug traced back to `_DSF4819.RAF.rrdata` containing
`"toneMapper": "AGX"` (uppercase). Every current write site only ever writes
lowercase `'agx'`, but `image_processing.rs` compared it with `tone_mapper ==
"agx"` (case-sensitive) — silently resolving the file's actual tonemapper to
Basic, permanently, with no error. The same exact bug pattern existed in two
*other* places in the same file (`resolve_tonemapper_override`) and in
`file_management.rs`, because the "compare a JSON/settings string against a
literal" pattern had been copy-pasted rather than centralized. It also meant a
prior manual test the user ran ("toggle to AGX and see if it helps") produced
no observable effect and looked like a dead end — when the real problem was
that the toggle's *write* silently didn't matter due to a *read*-side bug
elsewhere, not that AGX itself was ineffective.

Upstream merges are a likely source of this class of drift: this codebase is
periodically merged from `upstream/main` (a different fork), and upstream may
still represent a field as a plain string (or with different casing/values)
even where local code has already modeled it as a Rust `enum` / TS `enum`.
A merge can silently reintroduce a raw string literal or a new call site that
does its own `== "some_value"` comparison instead of going through the typed
representation.

**Rule going forward:**
- Any field that has a small, fixed set of valid string values (tonemapper,
  status/state fields, mode flags, etc.) should be represented as a real Rust
  `enum` (with a case-insensitive/lenient `parse()`, not raw `==` string
  comparisons) and a real TS `enum`, not `string` / `'a' | 'b'` unions compared
  ad-hoc at each call site. Centralize the parse-from-JSON logic in one place
  (e.g. `ToneMapper::from_adjustments_json`) so there is exactly one spot that
  can be wrong, and unit-test it (including a case-insensitivity /
  unknown-value-falls-back-to-default test) rather than each call site
  re-deriving the mapping.
- After merging from `upstream/main`, grep the diff for fields that this
  codebase has already converted to an enum (tonemapper, status/state fields,
  etc.) and check whether upstream reintroduced a raw string comparison, a new
  call site with its own `== "literal"` check, or a differently-cased literal
  for that field. Convert any such reintroduced string handling to use the
  existing enum/parse function instead of leaving it as a parallel, driftable
  string check.
- If a user says "I toggled/changed X and it didn't help," don't assume X (the
  feature) is ineffective — check whether the *write* of X actually happened
  and whether the *read* of X is looking at the same representation
  (case, key name, storage location) before concluding X doesn't matter.

## Open TODO: main-thread stall during RAW image load

While investigating why a fast embedded-JPEG placeholder (shown while a RAW
file's full preview decodes) never actually became visible on screen, direct
instrumentation (a `setInterval` heartbeat ticking every 200ms, since
WebKitGTK does not reliably support the Long Tasks `PerformanceObserver` API)
showed **repeated ~300-600ms main-thread stalls, roughly one per tick**,
starting *before* any RAW-load code even runs and continuing throughout the
load. This is not one single blocking call — it's a chronic, repeated stutter.

Consequence: `originalSize` (and therefore `imageRenderSize`, and therefore
the WebGL canvas's ability to draw anything at a nonzero size) gets set
correctly and quickly by a fast, non-demosaic dimension probe
(`get_fast_image_dimensions` in `file_management.rs`), but the React re-render
that would actually reflect that state doesn't get a chance to run for
several seconds because the main thread is chronically busy — by which point
the real decoded preview has usually already replaced the placeholder anyway.

**Not caused by session's WebGL/rendering work** — the stalls were present
before any of the new code paths ran, so this is a pre-existing, systemic
issue, not a regression from the WebGL canvas or the fast-dimensions fetch.

**Needs:** profiling with WebKit's Web Inspector (or equivalent) attached to
the running app to identify exactly what JS work is running repeatedly during
image load. No further guessing without that data — this exact investigation
already went through several wrong turns (CSS/compositor theories, WGPU
surface bugs, texture-loading CORS issues) before isolating this as the real
bottleneck; don't re-litigate those from scratch if this comes up again.

## Status: RAW Develop pipeline (demosaic + preprocess + denoise + sharpen) — shipped, live by default for standard Bayer RAWs

Implemented in `src-tauri/src/custom_raw_pipeline.rs`, `demosaic_algorithms.rs`,
`raw_preprocess.rs`, `raw_denoise.rs`, `raw_sharpen.rs`. **Hard constraint
honored: all code stays on our side.** rawler is used only to decode a RAW
file into pre-demosaic sensor data + calibration metadata (`RawImage`'s
public fields); demosaic, raw-domain preprocessing, white balance,
camera→sRGB color-matrix conversion, highlight compression, denoise,
sharpening, cropping (active area + default crop), and orientation are all
reimplemented in this repo. This was necessary, not just preferred: rawler's
own calibration step (`map_3ch_to_rgb`) is `pub(crate)`-only inside rawler
and cannot be called or hooked into from outside the crate, so there was no
way to reuse rawler's pipeline for anything past our own demosaic step even
if we'd wanted to.

**Pipeline stages, in order:** raw-domain preprocess (PDAF pixel correction,
hot/dead pixel + CFA row-banding correction, green-channel equalization, on
the Bayer mosaic) → demosaic (AMaZE/IGV/LMMSE/Bilinear) → white balance →
camera→sRGB color matrix → highlight compression → wavelet luminance/
chrominance denoise → luminance sharpening (unsharp mask or Richardson-Lucy
deconvolution) → crop (active area, then default crop) → orientation.

- **Demosaic**: three algorithms validated against real RawTherapee source
  (`Beep6581/RawTherapee`'s `amaze_demosaic_RT.cc`, `demosaic_algos.cc`
  (`igv_interpolate`), `lmmse_demosaic.cc`) — not guessed from names. AMaZE
  and LMMSE follow the reference formulas closely; IGV's real ±6-neighborhood
  Gaussian-vector-variance refinement pass is approximated with simple
  neighbor averaging (documented in `demosaic_algorithms.rs`'s module doc).
  ISO auto-selection (`select_by_iso`): AMaZE below 800, IGV 800–1599, LMMSE
  ≥1600 — starting thresholds, not tuned against a real test set.
- **Preprocess** (`raw_preprocess.rs`):
  - Hot/dead pixel correction (outlier vs. same-CFA-color neighbor median,
    using the fact that a standard Bayer CFA repeats with period 2 in each
    axis) and CFA row-banding denoise (per-row-type local-average offset
    correction).
  - **PDAF pixel correction** (`correct_pdaf_pixels`): cameras with
    on-sensor phase-detect pixels replace some green photosites on specific
    rows, which read back brighter than their neighbors. Which rows are
    affected is camera-model-specific and can't be derived generically -
    ported the `pdaf_pattern`/`pdaf_offset` table for ~13 camera model
    groups from ART's `rtengine/camconst.json` into `raw_pdaf_data.rs`
    (RapidRAW is AGPLv3, ART/RawTherapee is GPLv3 - compatible, GPLv3
    content may be combined into an AGPLv3 work), matched against
    `rawler`'s `clean_make`/`clean_model`. The actual per-pixel anomaly
    detection mirrors ART's `PDAFLinesFilter::markLine`
    (`rtengine/pdaflinesfilter.cc`) exactly (brighter than all 4 diagonal
    green neighbors + a local-consistency check), then corrects via the
    same same-color-median approach as hot/dead pixel correction. A no-op
    for any camera not in the table.
  - **Green channel equalization** (`equalize_green_channels`): unlike PDAF
    correction, this needs no per-camera data - computes the average of
    each of a Bayer CFA's two green sub-populations (green-on-red-row vs.
    green-on-blue-row) directly from the image and rescales both toward
    their shared mean. Mirrors ART's `green_equilibrate_global`
    (`rtengine/green_equil_RT.cc`).
  - Deliberately still narrower than ART's full `preprocess()`:
    dark-frame/flat-field subtraction is out of scope (needs a
    user-managed calibration-frame library - a separate feature, not part
    of this pipeline).
- **Denoise** (`raw_denoise.rs`): runs *after* demosaic, not before — checked
  ART's actual source (`rtengine/simpleprocess.cc`) and confirmed its
  `denoise()` call happens after `demosaic()`, contradicting an earlier
  assumption in this project that raw denoise belonged pre-demosaic. Uses a
  simplified "à trous" (undecimated) wavelet transform, luminance and
  chrominance denoised independently in YCbCr space with per-level
  attenuation (finer levels attenuated more) — a real, standard multiscale
  technique, but NOT a port of ART's actual directional complex-wavelet
  transform (`cplx_wavelet_dec.cc`) or its NLMeans/guided-smoothing passes.
  `suggest_strength_for_iso` gives the ISO-based default. Still one combined
  strength knob rather than ART's separate luma/chroma-red-green/
  chroma-blue-yellow sliders — backlogged, not started.
- **Sharpening** (`raw_sharpen.rs`, `SharpenMethod`): two selectable
  methods, matching ART's own choices (`rtengine/ipsharpen.cc`), both
  luminance-only with the same edge-aware blend mask (suppresses sharpening
  in flat/noisy regions):
  - `UnsharpMask` (default): classic blur-and-subtract, matches ART's
    `unsharp_mask` + `buildBlendMask`.
  - `RlDeconvolution`: real iterative Richardson-Lucy deconvolution against
    a symmetric Gaussian PSF (`estimate *= blur(observed / blur(estimate))`,
    20 iterations), simplified from ART's `deconvsharpening` ("rld" method)
    - skips ART's per-pixel early-stopping divergence check and dedicated
    impulse-noise exclusion mask (the latter is covered by
    `raw_preprocess`'s hot/dead pixel correction already running first).
  - ART's third method (`psf`, deconvolution against a real measured
    per-lens PSF) is not implemented - backlogged, needs lens PSF data we
    don't have.
  `suggest_amount_for_iso` gives the ISO-based default (lower at high ISO,
  so sharpening doesn't re-amplify noise denoise didn't fully remove).
- **Tone curve: not needed, already solved.** Originally tracked here as
  "not done," based on evaluating this pipeline in isolation - checked
  `src/shaders/shader.wgsl` and confirmed the app's GPU tonemap stage
  (downstream of every RAW *and* non-RAW image alike) already has real
  filmic tonemapping: `legacy_tonemap` (ACES) and a full `agx_tonemap`
  (sigmoid toe/shoulder curve + gamut compression), with AgX as the default
  (`default_raw_tonemapper` in `app_settings.rs`). This pipeline staying
  linear (no curve, no gamma) is *correct* - it's the same contract
  `raw_processing::develop_raw_image` already has, and the downstream
  shader is what applies the curve. Building a curve inside this pipeline
  would double up with AgX/ACES and over-contrast every image.
- Eligibility gate unchanged: only a standard 2×2 RGB Bayer CFA
  (`RawSensorData::is_standard_bayer`). X-Trans, 4-channel (CMY/RGBE),
  monochrome, and DNG `LinearRaw` sources are ineligible and always fall
  back to rawler's own PPG pipeline, with no user-visible error. X-Trans
  (Fuji) demosaic is backlogged - a 6×6-pattern algorithm, substantially
  different from Bayer, only worth it if Fuji support is actually needed.

**Live-app wiring — two paths, deliberately different scope:**
- `raw_processing::develop_raw_image` (used by thumbnails, export, culling,
  HDR, focus stacking, panorama, restoration, etc. — every caller except the
  editor's own image-open path) is **unchanged by default**, still gated by
  the `RAPIDRAW_CUSTOM_DEMOSAIC=1` env var for ad-hoc testing.
- `raw_processing::develop_raw_image_for_editor` (new) is what
  `image_loader::load_base_image_from_bytes` actually calls whenever a
  `raw_develop_adjustments: Option<&serde_json::Value>` is passed in - it
  reads `rawDemosaicAlgorithm`/`rawDenoiseAmount`/`rawSharpenAmount`/
  `rawSharpenMethod`/`rawPreprocessEnabled` from that JSON blob itself now
  (moved here from `image_loader.rs` so every caller shares one extraction
  path instead of duplicating it), falling back to `develop_raw_image`
  (rawler PPG) on any error or ineligible sensor. `None` (no
  adjustments in scope) behaves exactly like all-default/auto overrides.
  **Now wired into every real `load_base_image_from_bytes` caller that can
  cheaply reach an adjustments/sidecar value**, not just the editor: the
  editor's own load path, `generate_preview_for_path`, `load_and_composite`
  callers (export), HDR merge, panorama stitching, negative-film conversion,
  focus stacking, and AI restoration's RAW fallback path all now load that
  image's `.rrdata` sidecar (via `exif_processing::load_sidecar`, since
  these functions didn't already have adjustments in scope) and pass it
  through. Callers that pass `fast_demosaic: true` (culling/duplicate
  analysis, community-preset thumbnails, one `export_processing.rs`
  preview call, `generate_all_community_previews`) are deliberately left
  as `None` - `develop_raw_image_for_editor` only engages the custom
  pipeline when `!fast_demosaic`, so wiring adjustments through would have
  had zero effect there anyway.
  **Known limitation**: `load_image`'s `decoded_image_cache`/`original_image`
  cache is not specifically invalidated when only raw-develop adjustments
  change on an already-loaded image — same pre-existing limitation as
  `raw_highlight_compression`/`linear_raw_mode`, not something this work
  introduced or fixed. Re-selecting/reopening the image picks up the change.
- CLI (`rapidraw-cli raw inspect --input <file>` /
  `rapidraw-cli raw develop --input <file> --output <file.png>
  [--demosaic auto|amaze|igv|lmmse|bilinear] [--denoise auto|<0-1>]
  [--sharpen auto|<0-1>] [--sharpen-method unsharp|rld] [--no-preprocess]
  [--linear]`) exercises the same `custom_raw_pipeline` functions directly
  (not through `develop_raw_image_for_editor`) — this is the intended way
  to debug the raw pipeline outside the full Tauri app, per explicit
  request. Required making `custom_raw_pipeline`, `demosaic_algorithms`,
  `raw_denoise`, `raw_preprocess`, `raw_pdaf_data`, `raw_sharpen`,
  `raw_processing` all `pub mod` in `lib.rs`.

**UI**: right-rail Adjustments panel has a new "Raw Develop" section
(`src/components/adjustments/RawDevelop.tsx`, `ControlsPanel.tsx`),
positioned before Basic — demosaic dropdown, denoise/sharpen sliders each
with an "Auto" toggle (auto = -1 sentinel, resolved ISO-side), a sharpen
method dropdown (Unsharp Mask / Richardson-Lucy Deconvolution), and a
preprocess on/off switch. Hidden/no-op for non-RAW files. The former
"RAW Denoise & Sharpening" section (`Restore.tsx`, the on-demand
RawNIND/NAFNet ONNX AI restoration feature — a fundamentally different
thing: a background job producing a saved derivative, not a live pipeline
stage) was renamed "AI Denoise" and moved into the AI tab
(`AIPanel.tsx`, next to `GeminiCritiquePanel`) — it was never part of this
pipeline and user feedback flagged the naming/placement as confusing.

**Backlogged (explicitly deferred, not started):**
- ART's third sharpening method (`psf`, real per-lens measured PSF
  deconvolution) - needs lens PSF data we don't have.
- Separate luma / chroma-red-green / chroma-blue-yellow denoise strength
  sliders (ART exposes three; we collapse to one "amount" that internally
  weights luma/chroma differently - see `raw_denoise.rs`'s
  `LUMA_LEVEL_WEIGHTS`/`CHROMA_LEVEL_WEIGHTS`).
- X-Trans (Fuji) demosaic - a 6×6-pattern algorithm, substantially
  different from Bayer.
- Dark frame / flat field subtraction - a separate feature (calibration
  frame library: capture, store, match by camera/ISO/exposure), not part
  of this pipeline.
