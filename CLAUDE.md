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

## Status: custom demosaic pipeline (AMaZE / IGV / LMMSE) — implemented, opt-in

Implemented in `src-tauri/src/custom_raw_pipeline.rs` +
`src-tauri/src/demosaic_algorithms.rs`. **Hard constraint honored: all code
stays on our side.** rawler is used only to decode a RAW file into
pre-demosaic sensor data + calibration metadata (`RawImage`'s public fields);
demosaic, white balance, camera→sRGB color-matrix conversion, highlight
compression, cropping (active area + default crop), and orientation are all
reimplemented in this repo. This was necessary, not just preferred: rawler's
own calibration step (`map_3ch_to_rgb`) is `pub(crate)`-only inside rawler
and cannot be called or hooked into from outside the crate, so there was no
way to reuse rawler's pipeline for anything past our own demosaic step even
if we'd wanted to — the whole post-demosaic chain had to be reimplemented
too, mirroring `raw_processing::develop_internal`'s exact rescale/highlight-
compression formulas so output stays numerically compatible with the app's
GPU tonemap stage.

- Three algorithms implemented and validated against real RawTherapee source
  (`Beep6581/RawTherapee`'s `amaze_demosaic_RT.cc`, `demosaic_algos.cc`
  (`igv_interpolate`), `lmmse_demosaic.cc`) — not guessed from names. AMaZE
  and LMMSE follow the reference formulas closely; IGV's real ±6-neighborhood
  Gaussian-vector-variance refinement pass is approximated with simple
  neighbor averaging (documented in `demosaic_algorithms.rs`'s module doc).
- ISO auto-selection: `demosaic_algorithms::select_by_iso` — AMaZE below 800,
  IGV 800–1599, LMMSE ≥1600. Starting thresholds, not tuned against a real
  test set.
- Eligibility: only a standard 2×2 RGB Bayer CFA (`RawSensorData::is_standard_bayer`).
  X-Trans, 4-channel (CMY/RGBE), monochrome, and DNG `LinearRaw` sources are
  ineligible and always fall back to rawler's own PPG pipeline.
- **Live-app wiring is opt-in, default OFF**, gated by the
  `RAPIDRAW_CUSTOM_DEMOSAIC=1` env var, checked inside
  `raw_processing::develop_raw_image` (the single choke point every real
  call site already goes through). Falls back to the normal rawler-PPG path
  on any error or ineligible sensor, so leaving the env var unset — the
  default — can never change existing behavior. This was a deliberate
  choice over silently replacing the default pipeline: the surrounding
  production path (`develop_internal`) also handles linear-RAW formats,
  monochrome, multi-exposure WB neutralization, and highlight compression
  that would all need independent re-verification before trusting this for
  every user by default.
- **CLI support** (for debugging this pipeline outside the full Tauri app,
  per explicit request): `rapidraw-cli raw inspect --input <file.raw>`
  (reports ISO/CFA-eligibility/orientation/crop without developing) and
  `rapidraw-cli raw develop --input <file.raw> --output <file.png>
  [--demosaic auto|amaze|igv|lmmse|bilinear] [--highlight-compression <f32>]
  [--linear]` (`--linear` reproduces the exact linear pre-tonemap
  intermediate the live app pipeline consumes, re-encoded to sRGB only for
  PNG viewability; default is a display-ready gamma-encoded preview). This
  required making `custom_raw_pipeline`, `demosaic_algorithms`, and
  `raw_processing` `pub mod` in `lib.rs` (they were private `mod` before).
  **Next work (raw denoise, raw sharpen, tone curve) should hang off this
  same `raw develop` command as additional flags**, not a separate CLI
  entry point.

**Not yet done:**
- Tone curve: ART/RawTherapee use a mild S-curve (contrast/brightness), not
  the straight linear→gamma passthrough this pipeline currently does
  (`srgb_gamma` in `custom_raw_pipeline.rs`). Matters for the upcoming
  tone-mapping phase.
- Raw denoise, raw sharpening — not started.
- `develop_raw_custom_with_algorithm` (the non-`--linear` CLI/test path)
  only crops to `active_area`, not the default `crop_area` — a Phase-1 gap
  that `develop_raw_image_custom(_with_algorithm)` (the production-parity
  path) does not have. Low priority since it only affects the CLI preview
  PNG, not the live app, but worth fixing for consistency if it causes
  confusion when comparing CLI output to the app.
