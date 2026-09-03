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

## Fixed: color cast + missing parallelism + two-phase progressive load

After the pipeline above went live by default, real usage surfaced two
serious bugs and a UX problem that weren't caught by CLI/visual spot-checks
during development. Root causes and fixes, in case anything like this
resurfaces:

- **Strong purple/magenta color cast on every image.** Root cause:
  `custom_raw_pipeline.rs`'s `decode_raw_sensor_data` read
  `raw_image.xyz_to_cam` for the camera-to-XYZ matrix - a field rawler's
  own source (`rawimage.rs`) explicitly marks `// TODO: deprecated, use
  color_matrix`. It doesn't reliably hold the file's real calibrated
  matrix. Fixed to read `raw_image.color_matrix` instead (a
  `HashMap<Illuminant, FlatColorMatrix>`), preferring the `Illuminant::D65`
  entry and falling back to whatever's first available - exactly mirroring
  rawler's own real `Calibrate` step in `imgop::develop::RawDevelop`
  (`pub(crate)`, so this was re-derived, not called directly). **Lesson**:
  when reading a field from an external crate for something
  correctness-critical (a color matrix, not a cosmetic value), check the
  crate's own doc comments/TODOs on that field before trusting it, even if
  it type-checks and looks plausible.
  **This fix alone was insufficient** - real usage still showed a strong
  cast after it shipped. Two more bugs were compounding it, found only by
  reading rawler's actual (private, `pub(crate)`) `map_3ch_to_rgb` source
  directly from the vendored git checkout on disk rather than
  reverse-engineering behavior from output alone:
  - `apply_white_balance` normalized `wb_coeffs` by dividing by whichever
    channel had the numerically largest raw coefficient
    (`gain = wb[c] / wb_max`) before applying it. `wb_coeffs` are already
    the direct per-channel multipliers a neutral subject needs (green is
    the ~1.0 reference channel by convention) - dividing by the max
    channel *inverts* the correction, suppressing exactly the channels
    that needed boosting. Fixed to apply `wb_coeffs` as-is, matching
    rawler's real `Calibrate` step (`let wb = rawimage.wb_coeffs;`, used
    directly with no re-normalization).
  - `apply_color_matrix` built the camera->sRGB matrix by inverting
    `xyz_to_cam` first and *then* row-normalizing the result so it would
    be neutral-preserving. This is mathematically the wrong matrix to
    normalize: rawler's real algorithm builds the *forward* sRGB->camera
    matrix (`xyz_to_cam * srgb_to_xyz`), row-normalizes *that* first, and
    only then inverts it to get camera->sRGB. Normalizing after inverting
    is not the same operation as normalizing before inverting (they only
    coincide for special-case matrices), even though both trivially pass
    a "does pure gray map to pure gray" sanity check in isolation - the
    bug only shows up on real, non-neutral colors. Fixed to replicate
    rawler's actual order exactly (see `apply_color_matrix`'s doc comment
    for the full derivation).
  Verified via a debug pixel dump comparing `develop_raw_image_custom_with_algorithm`
  against `raw_processing::develop_raw_image`'s rawler-PPG reference output
  for the *same real photo* at the *same pixel* - both fixes were confirmed
  by checking that `wb_coeffs`/`color_matrix`/`black_level`/`white_level`
  inputs matched exactly between the two pipelines, then that the actual
  matrix arithmetic (re-derived independently in Python/numpy from the
  same inputs) reproduced the Rust output bit-for-bit.
- **A third, separate, much larger color-cast bug specific to LMMSE**:
  even after the two fixes above, LMMSE (the ISO>=1600 auto-selected
  algorithm - so any high-ISO low-light shot, exactly the case that
  triggered the original bug report) still showed a strong magenta/purple
  cast, while AMaZE, IGV, and plain bilinear all demosaiced the *same*
  file correctly. This was the actual dominant cause of the originally
  reported cast, not the two matrix bugs above (which were real but
  comparatively minor). Root cause, in `demosaic_algorithms.rs`'s
  `lmmse_green_plane`: stage 1 computes a (green - same_channel)
  difference estimate only at native red/blue sites, leaving every
  interleaved green site's slot at a placeholder `0.0`. Stage 2's 9-tap
  Gaussian smoothing pass then walked that plane with *unit* pixel
  offsets (`offset = i`) - but a Bayer CFA's red (or blue) sites repeat
  with period 2 along any row/column, so half of the "neighbor" taps a
  unit-offset kernel reads are actually those bogus green-slot zeros, not
  real same-color difference samples. This systematically pulled every
  smoothed diff toward 0, which pulled LMMSE's reconstructed green
  (`sensor.data[idx] + diff`) toward the raw same-channel value instead of
  the true local green level - and since a camera's raw blue channel
  reads low under warm/tungsten light (exactly why `wb_coeffs` needs a
  large blue gain), this specifically under-estimated green in the blue
  channel's own reconstructed positions, which is what a magenta/purple
  cast *is*. Fixed by changing the tap offset to `i * 2` (stride 2), so
  every tap lands on the same-color lattice the diff plane actually has
  data on. **Diagnosis method**: bisected by algorithm first (AMaZE/IGV/
  bilinear all correct, only LMMSE wrong - ruled out the shared WB/
  color-matrix/preprocess code, which is identical across all four),
  *then* by pixel-level dbgpx comparison of the LMMSE green plane against
  the others, not by staring at the matrix math in isolation - the actual
  numeric divergence (LMMSE's green channel at a sampled dark pixel came
  out roughly half of AMaZE/IGV/bilinear's value for the same pixel) is
  what pointed at the green-plane reconstruction rather than anything
  downstream. Verified fixed via the same reference-pipeline pixel-patch
  comparison: a 40x40 dark-background patch went from `[49, 35, 72]`
  (LMMSE, broken) to `[44.7, 47.9, 56.6]` (matching the rawler-PPG
  reference's `[43.9, 48.1, 57.0]` almost exactly).
- **No parallelism anywhere in the new pipeline.** None of
  `demosaic_algorithms.rs`, `custom_raw_pipeline.rs`, `raw_denoise.rs`,
  `raw_sharpen.rs`, `raw_preprocess.rs` used `rayon`, while the rest of
  this codebase parallelizes exactly this kind of per-row/per-pixel work
  throughout (`image_processing.rs`, `image_restoration.rs`, etc.). Added
  `rayon::prelude::*` and converted the hot loops (demosaic per-pixel
  passes, à trous wavelet smoothing, Gaussian blur, preprocess corrections)
  to `par_chunks_mut`/`par_iter_mut`/`rayon::join`. Also found and fixed a
  compounding allocation bug in `demosaic_algorithms.rs`'s LMMSE final
  combine stage: `Vec::with_capacity(5)` called twice per red/blue pixel -
  ~21 million heap allocations for a 42MP image - replaced with fixed-size
  `[f32; 5]` arrays.
- **Even after both fixes, the full pipeline (demosaic + denoise + sharpen)
  still takes several seconds** (measured cleanly, machine otherwise idle:
  ~7.7s total for a 42MP ISO-3200 file - demosaic ~2.1s, denoise ~3.7s,
  sharpen ~0.8s, preprocess ~0.7s). This is real algorithmic cost (AMaZE/
  IGV/LMMSE + wavelet denoise + sharpening are inherently far more
  expensive than rawler's PPG demosaic, the same tradeoff RawTherapee/ART
  make with their own equivalent pipeline), not a residual bug - not
  something to keep chasing with more micro-optimization.

**Fix: two-phase progressive load**, instead of blocking first paint on
the slow pipeline:
- `raw_processing::develop_raw_image_for_editor` gained an
  `allow_custom_pipeline: bool` parameter (separate from `fast_demosaic`,
  which is rawler's own quarter-resolution thumbnail mode, not a fast
  full-res option) - `false` forces the fast rawler-PPG path even for a
  full-quality decode. Threaded through `image_loader::load_base_image_from_bytes`
  the same mechanical way as `raw_develop_adjustments` was - `true` at every
  existing call site (preserves current behavior everywhere), `false` only
  at `load_image`'s own two decode call sites.
- `image_loader::load_image` (the real editor-open command) now decodes
  **phase 1** with `allow_custom_pipeline: false` (fast, full-resolution,
  matches the pre-this-session ~1s baseline) and returns immediately. If
  the image is RAW, it then spawns a background OS thread (`std::thread::spawn`,
  matching the existing pattern in `generate_uncropped_preview` - not
  `tokio::spawn`, since `tauri::State`'s borrow can't outlive the command)
  that re-decodes with `allow_custom_pipeline: true` (**phase 2**, the real
  AMaZE/IGV/LMMSE+denoise+sharpen pipeline). On success, guarded by the
  same `load_image_generation` atomic-counter cancellation mechanism
  already used everywhere else in this function (so a stale phase-2 result
  from an image the user has since navigated away from is silently
  dropped), it swaps the upgraded image into `state.original_image`,
  clears the same downstream caches `load_image` itself clears on a fresh
  load (`cached_preview`, `gpu_image_cache`, `full_warped_cache`,
  `full_transformed_cache`, `mask_cache`, `patch_cache`, `geometry_cache`),
  updates `decoded_image_cache`, and emits a `"raw-develop-upgraded"`
  Tauri event with `{ path }`.
- Frontend: `useImageProcessing.ts` listens for `raw-develop-upgraded` and,
  if it's for the currently-selected+ready image, calls the same
  `applyAdjustments` the normal adjustment-change path uses - so the
  on-screen image quietly upgrades in place a few seconds after first
  paint, with no user action needed and no new UI chrome added.
- **Not independently verified live** (no GUI access in this environment):
  compiles cleanly, TS diffed clean against baseline, and the logic was
  traced carefully against the existing cancellation/cache-clearing
  patterns already established in this exact file - but the actual
  runtime behavior (does the event fire, does the swap look smooth, is
  there a visible "pop" when phase 2 lands) needs a real test in the
  running app.

## Fixed: over-aggressive luma denoise, RAW Develop panel not actually live, Switch duplicate-id bug

Three separate issues reported after real usage of `_DSC1713.ARW` (Sony
ILCE-7RM3, ISO 3200) compared directly against its in-camera JPEG:

- **Soft image, halo instead of texture (leather jacket detail lost).**
  `raw_denoise.rs`'s `LUMA_LEVEL_WEIGHTS` was `[0.9, 0.7, 0.4, 0.15]` - at
  this file's auto strength (~0.49), the finest luminance band (exactly
  the frequency range fine texture like leather grain, skin pores, and
  fabric weave lives in) was attenuated by ~44%. Denoise ran before
  sharpening, so by the time unsharp mask ran there was no fine texture
  left to sharpen - it could only amplify the remaining large-scale edges
  (stitching, garment silhouette), which is what a "soft interior, halo at
  edges" look actually is. Chrominance denoise was untouched - chroma
  noise isn't texture-bearing, so `CHROMA_LEVEL_WEIGHTS` staying aggressive
  is correct. Fixed `LUMA_LEVEL_WEIGHTS` to `[0.25, 0.45, 0.35, 0.15]`
  (much gentler at the finest level). Verified by cropping the exact same
  region from the pipeline output and the camera JPEG side-by-side -
  leather grain/wrinkle texture is now visibly present and close to the
  JPEG, not smoothed away.
- **The "Raw Develop" panel's demosaic/denoise/sharpen/preprocess controls
  had no live effect at all** (a bigger problem than the reported "no
  progress indication" - investigating that surfaced this). These fields
  are read only by `develop_raw_image_for_editor`/
  `develop_raw_image_custom_resolved`, which only run when the RAW file is
  *decoded* - never by `process_preview_job`/`apply_adjustments` (the
  generic preview-recompute pipeline every other adjustment panel uses,
  which works on the already-decoded `state.original_image` and never
  reads these fields). Changing the dropdown *did* trigger the generic
  debounced preview recompute (explaining "the process starts") but that
  recompute is a no-op for these fields - the only way the change ever
  took effect was fully re-selecting/reopening the image (a limitation
  already flagged, but not yet fixed, in the two-phase-load section
  above). Fixed by actually wiring it up:
  - Extracted `load_image`'s phase-2 background-upgrade thread (decode
    with `allow_custom_pipeline: true`, swap into `state.original_image`,
    clear caches, emit `"raw-develop-upgraded"`) into a shared
    `image_loader::spawn_raw_develop_upgrade` function.
  - New command `reprocess_raw_develop(js_adjustments)`: reads the
    currently-loaded image's path (no-op if not RAW), bumps
    `load_image_generation` (so a still-in-flight previous upgrade is
    superseded), and calls `spawn_raw_develop_upgrade` with the *live*
    in-editor `js_adjustments` (not the last-saved sidecar - the user is
    actively editing, unsaved).
  - Frontend (`useImageProcessing.ts`): a `useEffect` watches
    `rawDemosaicAlgorithm`/`rawDenoiseAmount`/`rawSharpenAmount`/
    `rawSharpenMethod`/`rawPreprocessEnabled` (joined into one string key),
    explicitly distinguishing "these fields changed" from "the selected
    image changed" (a path change must never itself count as a field
    edit, even though a different image's own raw-develop values will
    differ) via a ref tracking `{path, fieldsKey}` from the previous run.
    A genuine field edit sets `isRawReprocessing: true` and calls a
    lodash-debounced (500ms) `invoke(ReprocessRawDevelop)` - the debounce
    means rapid dropdown/slider changes settle on one request instead of
    each one starting (and wasting several CPU-seconds on) a full
    multi-second re-decode that the next change immediately supersedes.
    The existing `raw-develop-upgraded` listener (shared with the initial
    phase-2 load) clears `isRawReprocessing` and re-renders; a new
    `raw-develop-reprocess-error` event (emitted by
    `spawn_raw_develop_upgrade` on a read/decode failure) also clears it,
    so the busy indicator can't get stuck on if the reprocess fails.
  - UI: `RawDevelop.tsx` shows a small spinner + "Reprocessing..." label
    (via a new `isRawReprocessing` field on `useEditorStore`) next to the
    panel description whenever a reprocess is in flight - the actual ask
    behind the "no indication" report.
  - **Known remaining limitation**: this does not *cancel* an
    already-*running* re-decode mid-flight (the debounce only prevents
    *starting* redundant ones) - true cancellation would need a
    cancellation check threaded through `custom_raw_pipeline`'s stages,
    which wasn't done here. In practice the debounce should make this rare
    (a decode only starts after 500ms of no further changes), but a decode
    already in progress when a *new* change lands will still run to
    completion before the newer one starts.
- **`Switch.tsx`'s `id` prop was declared on `SwitchProps` but never read**
  - every switch derived its DOM id purely from its `label` text
    (`switch-${label...}`), so two switches sharing a label (the "Auto"
    toggles next to RAW Denoise and RAW Sharpening) rendered the exact
    same `id`. The click/label association itself still worked (the
    `<input>` is a DOM descendant of its own `<label>`, so implicit
    association doesn't depend on `id` matching), but this was a real bug
    for anything else that resolves by id and a likely contributor to the
    reported "can't turn off Sharpen's Auto, seems connected to Denoise's"
    confusion. Fixed to prefer an explicit `id` when the caller passes
    one; `RawDevelop.tsx`'s two "Auto" switches (and the preprocess
    switch) now pass distinct explicit ids. Also added `gap-3`/`shrink-0`
    to the label+switch row layout in `RawDevelop.tsx` as a defensive fix
    for the reported "text and toggle with no gap" look, but this was
    **not visually confirmed** (no GUI access in this environment) - if it
    persists, the actual screenshot is needed to diagnose further.
    **Follow-up**: the "Reprocessing..." indicator itself turned out to be
    the actual cause of a *different* layout complaint - putting it inline
    with the panel description text made that text wrap and the whole
    panel shift height every time reprocessing started/stopped. Moved it
    out of `RawDevelop.tsx` entirely and onto the canvas instead
    (`ImageCanvas.tsx`, a small pill top-right, mirroring the existing
    "Embedded Preview" pill top-left) - it now overlays the rendered image
    rather than pushing panel content around, and is anchored to the thing
    that's actually changing.

## Fixed: severely underexposed RAW file misread as a decode bug (it wasn't)

A different RAW file (`_DSC9252.ARW`, same Sony ILCE-7RM3) rendered
extremely dark in the app, while ART and darktable showed it reasonably
exposed - reported as "what gives?", i.e. suspected as a pipeline bug like
the earlier color-cast one. **It is not a bug.** Added a temporary raw-ADU
percentile dump to `custom_raw_pipeline.rs`'s existing `debug_color_matrix`
test (gated behind `RAPIDRAW_TEST_RAW_PATH`, kept as a permanent
diagnostic) and confirmed: `black_level=512`, `white_level=15360`, and the
actual raw sensor data's `p50=524` (12 units above black - the median
pixel), `p99.9=2881`, `max=4561` (~30% of the sensor's full range even at
the single brightest pixel in the frame). This is a real, severely
underexposed capture (roughly 3-4 stops under a full-range exposure), not
a metadata or color-matrix misread - confirmed further by rendering the
same file through `raw_processing::develop_raw_image` (rawler's own PPG
pipeline, completely independent code path from `custom_raw_pipeline.rs`)
and getting the *same* near-black average brightness (`14.3, 9.6, 8.4` vs
the custom pipeline's `14.25, 9.57, 8.41`) - both pipelines agree, because
both are rendering the real (dark) sensor data faithfully.

**Why ART/darktable look different**: this app's pipeline is deliberately
linear/scene-referred up through a single tonemap step (AgX/ACES in the
GPU shader, see the "Tone curve: not needed, already solved" note above) -
it doesn't apply any automatic exposure compensation. ART and darktable's
*default* views typically DO apply an automatic base curve / filmic
exposure boost that actively brightens an underexposed capture as part of
their out-of-the-box rendering intent. Both are legitimate rendering
philosophies for the same underlying (genuinely dark) raw data - this is
a product/UX difference in default rendering intent, not a correctness
bug, and the shot is recoverable via a manual `+3`-ish EV push in the
Basic panel's Exposure slider, the same way it would need pushing in
ART/darktable if their auto-exposure heuristic didn't already do it for
you.

**If auto-exposure-on-load is ever wanted**: that would be a real,
deliberate feature (estimate a default exposure compensation from the
raw histogram, e.g. targeting a scene-average or highlight-relative
midpoint) - it isn't something to bolt onto the RAW Develop pipeline
itself, since the underlying color/demosaic pipeline is already correct
here. Not started.

## Added: PPG as an explicit selectable Demosaic Algorithm option

`DemosaicAlgorithm` (`adjustments.ts`) gained a `Ppg = 'ppg'` value,
listed right after `Auto` and labeled `"PPG (Default)"` - rawler's own
demosaic, i.e. what a RAW file gets when this custom pipeline doesn't run
at all. Wired in `raw_processing::develop_raw_image_for_editor`: computes
`demosaic_override` once (previously only computed inside the
custom-pipeline branch) and short-circuits past
`develop_raw_image_custom_resolved` entirely when it's `Some("ppg")`,
falling straight through to the existing `develop_raw_image` (rawler PPG)
call at the bottom of the function - this is the same fallback path
already used when the custom pipeline errors out (ineligible sensor,
etc.), just reached deliberately instead of via an `Err`.

## Fixed: embedded-preview thumbnail rendered as a 0x0 (invisible) WebGL quad

Reported as "the Embedded Preview badge shows but no image is visible."
Root-caused (diagnosis credited to an external analysis the user ran
independently, verified line-by-line against the actual current code
before acting on it - it was accurate): `Editor.tsx`'s `croppedDimensions`
- which feeds `useImageRenderSize`, which sizes the actual WebGL canvas
quad in `WebglTexturedCanvas.tsx` - reads only `selectedImage.width`/
`selectedImage.height`. Those stay `0` until the *full* image decode
resolves (`useImageLoader.ts`'s `loadFullImageData`), which for a RAW
file can take seconds. Meanwhile `originalSize` gets a real value almost
immediately, via a separate fast/non-demosaic dimension probe
(`loadFastDimensions`, `GetFastImageDimensions`) - but `croppedDimensions`
never looked at `originalSize`, and `loadFastDimensions` never wrote to
`selectedImage.width`/`height` either. Net effect: the embedded-preview
thumbnail (already loaded, ready to show) had a real image to display but
a `{width:0, height:0}` box computed for the canvas to draw it into, so
the WebGL vertex shader collapsed the quad to a single point - zero
pixels rasterized, fully transparent, while the badge (driven by a
different, correct condition) still showed. Fixed both ends: `Editor.tsx`
now falls back to `originalSize` when `selectedImage.width/height` are
`0`, and `loadFastDimensions` now also mirrors its result onto
`selectedImage.width`/`height` (not just `originalSize`), so the
placeholder gets a correct-aspect-ratio box to render into within
whatever the fast dimensions probe takes (near-instant), not whatever the
full RAW decode takes.
