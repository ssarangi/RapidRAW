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

## Open TODO: pluggable demosaic algorithms (AMaZE / IGV / LMMSE) via rawler

Goal: implement AMaZE, IGV, and LMMSE demosaic algorithms, with automatic
selection based on ISO (LMMSE/IGV for high-ISO images, AMaZE otherwise —
exact ISO thresholds still to be decided).

**Hard constraint (explicitly stated by the user): all code must stay on our
side.** A local, path-based `[patch]` override of the vendored `rawler`
git dependency was tried earlier this session (see git history around the
WebGL rendering investigation) and explicitly rejected: it lived in an
untracked `vendor/` directory outside version control, which is unreproducible
and not acceptable long-term. Before implementing new demosaic algorithms,
figure out an approach that doesn't require modifying/forking the upstream
`rawler` crate at all — e.g. decoding to a pre-demosaic intermediate ourselves
and running our own demosaic step entirely within this repo's own Rust code,
rather than swapping rawler's internal `Demosaic` trait implementation.
