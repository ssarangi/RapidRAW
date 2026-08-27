# RapidRAW Implementation Handoff for Antigravity

Status: Active handoff  
Date: 2026-08-26  
Branch: `main`

## Objective

Continue RapidRAW as a filesystem-first RAW editor with an optional local SQLite catalog and local-first AI DAM capabilities. Do not regress folder-based browsing for users who never create a catalog.

The immediate objective is to finish and harden the existing catalog/DAM/AI work. Only after that completion pass should Antigravity build non-destructive image restoration, beginning with RAW denoise.

Read [AI Denoising and Sharpening Handoff](AI_DENOISING_SHARPENING_HANDOFF.md) before selecting models or writing inference code. It is the research and model-selection decision record.

## Non-negotiable Product Rules

1. Source files, `.rrdata`, and XMP are never overwritten by AI operations.
2. SQLite catalog mode is optional. Folder mode must continue to open and browse images without a database.
3. All catalog work is local by default and runs off the UI thread.
4. Every long-running AI operation is a durable background job with progress, pause/resume where technically safe, cancellation, and error history.
5. Model outputs are versioned by model ID, immutable revision/checksum, source modification state, and recipe.
6. AI-generated restoration is a derivative, not a replacement for an original.
7. Search, People, Insights, and culling are main-library workflows; Settings is for configuration and model installation only.
8. Do not make network filesystem access part of synchronous startup or UI rendering. CIFS/SMB calls can enter uninterruptible kernel sleep.

## Current State

### Implemented and committed

- Optional SQLite library creation/opening, collection roots, metadata extraction, catalog search, metrics, smart collections, and scan jobs.
- Catalog search facets: camera, lens, year, person, tags, AI tags, rating, filename/caption/free text.
- Background-job database, status-bar indicator/modal, pause/resume/cancel controls, and persisted job history.
- Local face workflow: YuNet + SFace detection/recognition, unreviewed faces, clustering, naming, rename, merge, safe person removal, and people-to-library navigation.
- Culling plans, persistent sessions, decision overrides, culling history, and CLI inspection.
- RAM++ broad image tagging as a catalog background job, including model download/status and coverage metrics.
- Visual model registry with direct RAM++ download, validated manual BioCLIP bundle installation, removal, and model listing via `rapidraw-cli models list`.

### Important limitations

- Only YuNet + SFace is a working face inference path. Other face packs are registry/evaluation entries, not runtime adapters.
- BioCLIP has a bundle contract but no actual inference runner, taxonomy parser, result storage, or bird-specific review flow.
- There is no implemented denoise, deblur, sharpen, or upscale inference pipeline yet.
- CLI currently inspects catalog/model state; it does not yet execute AI operations on files.
- Google Photos and Amazon Photos integrations are intentionally deferred.

## Relevant Code Map

| Area | Primary files |
| --- | --- |
| Tauri command registration and app state | `src-tauri/src/lib.rs` |
| SQLite schema, catalog APIs, jobs, faces, culling, metrics | `src-tauri/src/library_db.rs` |
| RAM++ inference and tagging job | `src-tauri/src/tagging.rs` |
| Face detection/recognition runtime | `src-tauri/src/face_detection.rs` |
| Face model registry/downloads | `src-tauri/src/face_model_registry.rs` |
| Visual model registry, RAM++, BioCLIP bundle installation | `src-tauri/src/visual_model_registry.rs` |
| Standalone CLI | `src-tauri/src/bin/rapidraw-cli.rs` |
| Catalog header/search/AI actions | `src/components/panel/library/LibraryHeader.tsx` |
| Library shell | `src/components/panel/MainLibrary.tsx` |
| Status bar and job modal | `src/components/panel/BottomBar.tsx` |
| Catalog insights | `src/components/views/InsightsView.tsx` |
| People workflow | `src/components/views/PeopleView.tsx` |
| Visual model Settings UI | `src/components/settings/VisualModelsSettings.tsx` |
| Frontend invoke types/contracts | `src/components/ui/AppProperties.tsx` |

## Mandatory Phase 0: Finish Existing Work

Do not start denoise, sharpening, deblur, or upscale work until this phase is complete. The current repository has useful scaffolding, but several user-visible features are not yet end-to-end implementations.

### 0.1 Catalog and library workflow hardening

Complete the following before adding new AI output types:

1. Verify catalog creation/opening, root addition/removal, scan, sync, delete-library, and missing-root recovery on local disks and reachable NAS roots.
2. Ensure opening a library never blocks splash/startup on an unavailable SMB/CIFS mount. Root probing, tree expansion, folder image discovery, metadata reads, and thumbnail generation must have deadlines/worker isolation and publish non-blocking state to the UI.
3. Preserve physical folder hierarchy by default. Selecting a folder shows child folders and direct images; recursive/flattened display is an explicit persisted checkbox, never an accidental catalog-search side effect.
4. Ensure RAW/JPEG pairing follows the existing `Prefer RAW` view option in both filesystem and catalog results.
5. Make empty states truthful: no active filter must not say an image filter excluded all images.
6. Keep Settings as library/model configuration. Library search, People, Insights, culling, and job monitoring stay accessible from the main Library shell.
7. Exercise dark-theme contrast in all native selects/dropdowns/buttons, especially catalog search, Settings actions, and View Options.

### 0.2 Background job completion

The current job table and status UI exist. Finish the product contract:

1. All scan, metadata extraction, thumbnail, face, RAM++, model-download, BioCLIP, and future derivative work uses a durable job record.
2. Job state must be visible without opening a modal; current operation, filename, progress, failure reason, and queue position must be available in the status surface.
3. Pause/resume/cancel must be offered only when the worker can honor it. Cancellation must not leave jobs in `running` or `cancelling` after a terminal error.
4. Add `retry failed` and `retry all eligible` paths per job kind.
5. On app restart, reconcile interrupted jobs into a clear terminal/retryable state. Do not show stale active spinners.
6. For batch analyzers, persist per-image progress/results frequently enough to resume safely and avoid SQLite lock contention.
7. Validate WAL/busy-timeout/short transaction boundaries under a scan plus tagging/face job. The observed `database is locked` failure is a release blocker.

### 0.3 Face detection and recognition completion

The People view and YuNet + SFace runtime exist, but model management must not imply unsupported inference:

1. Clearly label the only currently runnable face stack: YuNet detector plus SFace recognizer.
2. For every additional visible/downloadable face pack, either implement an adapter with preprocessing, tensors, embedding dimensions, thresholds, and tests, or label it `Conversion/runtime support required`; never present it as ready to recognize faces.
3. Keep embeddings from different recognizers isolated. Rebuild clusters and suggestions when a recognition model revision changes.
4. Complete People review operations: name, rename, merge, safe removal/relabeling, reject/restore face, cluster assign/ignore, and navigation to that person's images.
5. Refresh People results when jobs finish, but do not continuously reorder a user actively reviewing faces.
6. Add model-specific precision/recall fixture checks and an explicit review-first policy. Face recognition must never auto-confirm identities merely from confidence.

### 0.4 Broad tagging and species recognition completion

1. RAM++ is the broad-tag production path. Validate its installed ONNX pack, model checksums, preprocessing, threshold format, and per-image state transitions.
2. Make the tag-review queue efficient for large catalogs: accept/reject single and batch suggestions, expose source model/confidence, and ensure rejected tags do not appear in ordinary facet search.
3. Keep the current BioCLIP bundle installer, but do not market BioCLIP as implemented until it has an executable ONNX vision encoder, taxonomy-label parser, matching embedding matrix reader, persistent result schema, review UI, and batch job.
4. Gate BioCLIP by RAM++ broad categories such as bird/wildlife, with a user override and an audit trail. A broad label is a compute heuristic, not a claim that a species is present.
5. Store species output separately from user tags and accepted AI tags until reviewed. Preserve taxonomy rank, confidence, model revision, and source image.

### 0.5 Culling completion

1. Verify culling plans against real burst/RAW+JPEG groups and ensure user overrides survive apply/reopen.
2. Finish session history operations: inspect decision provenance, filter sessions, reopen/review a session, and clearly distinguish proposed from applied decisions.
3. Apply operations must never silently delete/move source files. Keep explicit user confirmation and log target paths/results.
4. Add tests for keep/reject override, no-op apply, failed filesystem operation, persistence, and source-root mapping.

### 0.6 Search, Insights, and metadata completion

1. Confirm metadata extraction handles EXIF/XMP values and exposes consistent camera, lens, year, caption, rating, person, tag, and AI-tag facets.
2. Dropdown facets must contain unique library values with counts; only filename/caption/free-text fields are typed inputs.
3. Catalog search must combine criteria deterministically, preserve search state in saved/smart collections, and avoid flattening ordinary folder browsing.
4. Insights must distinguish review suggestions, accepted tags, analyzer coverage, failures, missing files, culling sessions, and overrides. Metrics must be queryable in the CLI too.
5. Add an image-level provenance/detail view for metadata, catalog state, AI tags, face records, and background-job result/error summaries.

### 0.7 CLI completion

The CLI currently inspects state. Complete the command surface before adding restoration commands:

1. Document stable JSON output and exit codes for every existing command.
2. Add model status/verification and explicit catalog-job inspection detail.
3. Add controlled local commands for supported operations only: catalog scan/sync, RAM++ tag batch, YuNet/SFace face detection/recognition batch, suggested-tag review export, and job cancel/retry.
4. Do not require a Tauri window or app handle for CLI execution. Extract reusable services from Tauri commands.
5. Treat each CLI operation as a one-shot command or a durable catalog job with a returned job ID.

### 0.8 Tests, migration, and release verification

1. Add database migration coverage for all catalog tables, including jobs, faces, model state, tags, collections, culling sessions, and people mutations.
2. Add focused backend tests for job terminal states, cancellation, retry, database busy handling, model manifest validation, and catalog query correctness.
3. Add frontend tests or component-level checks for the critical runtime imports and library startup paths. Past `useShallow` and `TextVariants` omissions crashed the entire view; prevent this class of regression.
4. Run `cargo +1.98 check --manifest-path src-tauri/Cargo.toml --offline`, `npm run build`, focused Rust tests, and the CLI smoke tests before every coherent feature commit.
5. Before calling Phase 0 complete, run a manual smoke matrix: new library, existing library, local root, available NAS root, unavailable NAS root, scan/pause/resume/cancel, folder hierarchy, RAW preference, catalog search, RAM++, People, culling, model install/remove, and application restart.

### Phase 0 Completion Gate

Phase 0 is complete only when the following statement is true:

> A user can create/open a local catalog, index local or available NAS folders without freezing the UI, browse the intended folder hierarchy, search/inspect metadata, run supported local face and RAM++ analysis as monitored jobs, review results, and use culling without stale UI, database locks, or untruthful model capability claims.

Record unresolved model/runtime limitations in the UI and documentation. Do not hide them behind download buttons.

## Phase 1: Non-destructive Derivative Infrastructure

Start only after Phase 0 completion. This is the shared foundation for RAW denoise, RGB denoise, deblur, upscale, and later generated image variants.

### Database

Add `image_derivatives` in `library_db.rs` with at least:

```sql
id INTEGER PRIMARY KEY,
source_image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
operation_kind TEXT NOT NULL,
model_id TEXT NOT NULL,
model_revision TEXT NOT NULL,
recipe_json TEXT NOT NULL,
input_hash TEXT,
output_path TEXT NOT NULL,
output_hash TEXT,
output_format TEXT NOT NULL,
width INTEGER,
height INTEGER,
state TEXT NOT NULL,
error_message TEXT,
created_at INTEGER NOT NULL,
completed_at INTEGER,
updated_at INTEGER NOT NULL
```

Requirements:

- Add a schema migration and migration test.
- Keep output paths outside source directories by default, with an explicit configurable derivative root.
- Store recipe/model provenance before executing, then atomically transition state.
- Never represent a derivative as the same `images` row as its source.

### Shared backend API

Create a focused module, for example `src-tauri/src/image_restoration.rs`, rather than adding model-specific code to `file_management.rs`.

It should own:

- `RestorationRecipe`, `DerivativeOperation`, and output-path policy.
- Manifest verification: model checksum, tensor contract, preprocessing ID, tile size/overlap.
- Temp-file write, validation, fsync, atomic rename, and failure cleanup.
- A generic catalog job launcher that writes `image_derivatives` and updates `background_jobs`.
- Input preparation and tile scheduler hooks, without hard-coding a particular model.

Register only explicit Tauri commands in `lib.rs`; keep reusable logic outside command functions so CLI can use it later.

### Initial UI

Add a compact `Enhance` menu in the Library/image action area. For this milestone, it can show unavailable model state and a `Show provenance` stub, but it must not advertise a working denoise operation until one is executable.

The job modal in `BottomBar.tsx` already supports generic jobs. Add readable labels for the new kinds:

- `raw_denoise`
- `rgb_denoise`
- `deblur`
- `upscale`

### Acceptance Criteria

1. Creating a mock derivative job produces a durable `image_derivatives` row and a background job.
2. Cancellation leaves the source untouched and records `cancelled`.
3. A synthetic output written through the helper is atomically published only after validation.
4. A completed derivative can be listed by source image and its model recipe is returned.
5. Existing `cargo +1.98 check --manifest-path src-tauri/Cargo.toml --offline` and `npm run build` pass.

## Phase 2: Image Restoration Milestones

### 1. RawNIND / UtNet2 evaluation runner

- Use the raw-first contract in the denoise research handoff.
- Start behind an experimental model flag.
- Implement Bayer handling only after CFA phase, black/white normalization, mod-16 handling, gain matching, and output color conversion have fixture tests.
- Non-Bayer sources use a clearly labeled linear-RGB fallback.
- Compare results against Darktable's documented package on the test corpus before enabling by default.

### 2. NAFNet RGB runner

- Package a checksum-pinned ONNX model with the documented `768x768` tile input.
- Implement reflect padding, overlap, weighted stitching, and linear-RGB strength blending.
- Add per-image preview/crop comparison and catalog batch job.

### 3. RealPLKSR upscale

- Implement 2x first, then 4x after memory/seam testing.
- Label it enlargement, not sharpening.
- Use derivative output and explicit provenance.

### 4. BioCLIP bird workflow

- Parse a validated taxonomy-label artifact and matching embedding matrix.
- Run only after RAM++ has a qualifying broad tag such as bird/wildlife, with user override.
- Persist model-specific species suggestions with confidence, taxonomy rank, review state, and image links.
- Add searchable accepted species tags; suggestions remain reviewable.

### 5. Additional face runtime adapters

- One adapter at a time.
- Do not mix embeddings from different models.
- Store detector/recognizer preprocessing, dimensions, thresholds, and model revision.
- Add a migration/rebuild path for each model pack.

## Testing Policy

Add focused tests next to the storage/runtime code. Use fixtures with no personal images.

- SQLite migration and state-transition tests.
- Manifest/tensor-contract validation tests.
- Tile overlap and seam tests.
- CFA phase / black level / white level tests for RAW preprocessing.
- Atomic output and cancellation tests.
- Provenance serialization tests.
- Visual regression corpus checks for wildlife feathers, foliage, skin, flat gradients, stars, and high-ISO RAW.

Do not rely solely on PSNR/SSIM. Review at 100% and 200% and explicitly flag invented texture, color shifts, highlight clipping, and seams.

## Commands

Use these before each commit:

```bash
cargo +1.98 check --manifest-path src-tauri/Cargo.toml --offline
npm run build
git diff --check
```

Known environment caveat: full `cargo test` / full binary links can intermittently leave or wait on a Cargo target-directory lock in this environment. Do not run competing Cargo commands. If a link stalls, wait for the existing process/lock to clear; do not delete target artifacts or use destructive Git operations.

Current CLI examples:

```bash
npm run cli:debug -- models list
npm run cli:debug -- library metrics --database /path/to/rapidraw-library.db
npm run cli:debug -- jobs list --database /path/to/rapidraw-library.db
npm run cli:debug -- cull sessions --database /path/to/rapidraw-library.db
```

## Git and Workspace Rules

- Work on `main` unless the user explicitly requests another branch.
- Commit each coherent feature after checks pass.
- Do not revert unrelated work in a dirty tree.
- Use `apply_patch` for manual edits.
- Do not use `git reset --hard`, checkout-based reverts, or destructive cleanup without explicit user instruction.

## Definition of Done for AI Features

An AI capability is not done when its model downloads or its button appears. It is done only when:

1. The model is checksum-pinned and validated.
2. The operation is non-destructive and produces a durable derivative/provenance record.
3. It runs as a controllable background job.
4. Its result can be compared with the original, found later, and deleted independently.
5. Its failures are visible and retryable.
6. It has focused automated tests and representative visual fixtures.
