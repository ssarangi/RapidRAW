# RapidRAW AI and DAM Architecture

Status: Proposed design

This document defines how RapidRAW can grow from a filesystem-first RAW editor into an optional, local-first digital asset manager. It covers face detection and recognition, model downloads, migration from cloud photo services, catalog search and statistics, and an explainable AI-assisted culling workflow.

The existing filesystem workflow remains supported. Persistent face recognition, cross-folder search, durable AI analysis, and collection-wide statistics require an open SQLite library.

## Executive Decisions

1. AI work is local by default, asynchronous, resumable, and non-destructive.
2. Library, People, Cull, and Insights are first-class workspaces. Settings configures them but does not contain their daily workflows.
3. Face detection and face recognition are separate, replaceable model stages.
4. All supported face models are downloadable through a model registry. Models are not silently downloaded during startup or catalog browsing.
5. Model outputs are versioned. Embeddings from different recognition models are never mixed.
6. Confirmed people, rejected matches, ratings, and culling decisions are durable user data. Embeddings, quality scores, and similarity groups are rebuildable cache data.
7. Google Photos and Amazon Photos do not expose their internal People recognition through supported public APIs. Google Takeout offers a useful image-level people-label migration path. Amazon Rekognition can be an optional separate cloud recognizer, but it does not import Amazon Photos identities.
8. Culling defaults to assisted review. Automated culling may assign flags, ratings, or color labels, but it does not delete or move originals without a separate explicit action.
9. RAW/JPEG pairs and virtual copies are treated as one logical capture for face indexing and culling unless the user explicitly requests otherwise.
10. Risky network filesystem reads must be isolated from the UI and database writer. A background thread alone cannot protect the process from a CIFS call stuck in uninterruptible kernel sleep.

## Product Principles

### Local First and Private

- Source images, face crops, embeddings, and identity labels remain local unless the user explicitly enables a cloud provider.
- No image, face crop, embedding, or culling decision is sent as telemetry.
- Cloud features state exactly what leaves the computer, where it is stored, and how to delete it.
- Face data is treated as sensitive biometric data even when a jurisdiction does not legally classify it that way.

### Optional Catalog

- Filesystem mode continues to open folders without SQLite.
- Catalog mode indexes multiple roots and enables People, collection-wide search, Insights, saved searches, and durable AI jobs.
- Culling can operate on an ad hoc filesystem folder, but results are substantially more useful when stored in a catalog.

### Explainability

- A culling result exposes component scores and reasons such as `subject soft`, `left eye closed`, `best expression in burst`, or `highlight clipping`.
- A face suggestion exposes the candidate, similarity, model, and representative confirmed faces.
- Thresholds are model-specific and never presented as if scores from different models were comparable.

### User Authority

- Recognition produces suggestions until a user confirms them.
- High-confidence batch confirmation is available but remains an explicit action.
- Rejected person matches are retained so the same bad suggestion is not repeatedly shown.
- Re-running analysis does not erase confirmations, manual face regions, or culling decisions.

## Information Architecture

The main application should have four peer workspaces while a catalog is open:

| Workspace | Purpose | Persistent entry point |
| --- | --- | --- |
| Library | Browse folders/albums, search, filter, rate, tag, edit | Existing Library view |
| People | Detect, recognize, cluster, name, and review faces | People icon in the Sources navigation |
| Cull | Analyze a shoot and compare duplicate/burst candidates | Cull icon and folder context action |
| Insights | Catalog statistics, coverage, health, and AI metrics | Insights icon in the Sources navigation |

Settings remains reachable from the Sources pane, immediately left of the image-information control as previously requested. It contains library configuration, model management, compute limits, privacy, cloud credentials, and default scan behavior.

Search, statistics, face scanning, recognition review, and culling results must not be buried in Settings.

### Application Shell

```text
+----------------------------------------------------------------------------------+
| Library / People / Cull / Insights | Unified search                 Jobs  View   |
+----------+-----------------------------------------------------------------------+
| Sources  | Breadcrumb / active scope                         Sort  Filter  More   |
|          |-----------------------------------------------------------------------|
| Folders  |                                                                       |
| Albums   |                         Main workspace                                |
| People   |                                                                       |
| Searches |                                                                       |
|          |                                                                       |
|----------|-----------------------------------------------------------------------|
| Settings | Background status: Indexing 1,204/8,412  [pause] [details]            |
+----------+-----------------------------------------------------------------------+
```

The existing draggable editor panels remain editor tools. Library workspace navigation should not be implemented as another draggable adjustment panel because it defines what occupies the main canvas.

## Library Workspace

### Header

The Library header owns:

- Breadcrumb and active source.
- Unified search field.
- Active query chips.
- Sort, view, recursive-folder, RAW/JPEG grouping, and filter controls.
- Save Search action.
- Background job indicator.

Only filename, caption, title, and free text are text inputs. Camera, lens, year, person, tag, rating, file type, edit state, and root are facet controls populated from unique catalog values and counts.

Example query presentation:

```text
[maui sunset____________] [2025 x] [Sony FE 24-70mm x] [Alice x] [4+ stars x]
```

The query engine uses one structured representation for the search bar, smart albums, People photo filters, Cull scopes, and Insights drill-downs. Clicking a chart segment in Insights should open the equivalent Library query rather than a bespoke result page.

### Sources Pane

The Sources pane groups navigational items:

```text
Folders
  Pictures
  NAS Photos
Albums
  Portfolio
  Maui 2025
People
  All People
  Unknown
  Suggestions (142)
Smart Collections
  Five stars
  Unedited RAW
  Missing files
```

Folders retain hierarchy. Selecting a folder displays only direct images unless `Show images inside subfolders` is enabled. Catalog folders and filesystem folders follow the same visual rule.

## People Workspace

People is a durable DAM workflow, not a dialog.

### Top-Level Views

- **All People:** named people ordered by recent activity, name, or image count.
- **Unknown:** unlabeled face clusters and individual unclustered faces.
- **Suggestions:** proposed assignments waiting for confirmation.
- **Needs Attention:** low-quality detections, conflicting labels, missing source images, and stale embeddings.
- **Ignored:** intentionally ignored faces that should not be suggested again.

### People Grid

Each person row or compact tile shows a representative face, name, confirmed image count, pending suggestion count, and latest year. Person tiles are repeated entities and therefore appropriate card-like items; the surrounding page should remain an unframed work surface.

Selecting a person opens a main-canvas detail view:

```text
Alice                                              Search within Alice   More
Confirmed 326   Suggestions 18   Rejected 4   2009-2026
--------------------------------------------------------------------------------
Representative faces: [face] [face] [face] [face]
--------------------------------------------------------------------------------
Photos | Faces | Suggestions
[photo grid filtered to Alice]
```

### Face Review

The review surface should optimize high-volume keyboard work:

- Large face crop plus enough surrounding image context.
- Candidate people ranked with similarity and representative examples.
- Confirm, reject, ignore, rename, merge, or create person.
- Batch-select a cluster and assign once.
- Open the original photo without losing review position.
- Correct the face rectangle when detection is wrong.
- Compare uncertain faces side by side.

Suggested keyboard actions should be discoverable through tooltips and the existing shortcut system, not permanent instructional text in the workspace.

### Face Scan Wizard

`Scan Faces` opens a focused wizard:

1. **Scope:** all catalog images, new/unscanned images, selected folders, date range, album, current results, or selected images.
2. **Operation:** detect new faces, recognize unknown faces, cluster unknown faces, rebuild embeddings, or full rebuild.
3. **Models:** detection model, recognition model, and whether missing models should be downloaded.
4. **Quality:** minimum face size, detection sensitivity, normal or high-resolution scan, and whether to include rejected/hidden images.
5. **Performance:** worker count, CPU/GPU provider, pause on battery, and network-source policy.
6. **Review:** estimated images, downloaded data, disk cache growth, and operations that will be invalidated.

Starting the scan closes the wizard and creates a persistent background job. Clicking its status-bar item opens job details with current image thumbnail, path, faces detected, model, rate, elapsed time, failures, and pause/resume/cancel controls.

## Cull Workspace

Cull is a shoot-oriented review environment and should not remain only a modal.

### Entry Points

- `Cull` workspace followed by scope selection.
- Folder, album, or selection context action: `Start Culling Session`.
- Optional ingest step: `Copy and Cull` for memory cards in a later phase.

### Session Wizard

1. Select scope and RAW/JPEG policy.
2. Choose `Assisted` or `Automated`.
3. Choose genre profile: general, wedding/event, portrait, family/group, sports/action, wildlife, or custom.
4. Choose desired strictness or approximate keep percentage.
5. Customize duplicate grouping, blur, eye state, exposure, subject priority, and scene coverage.
6. Choose result mapping: flags, stars, colors, or catalog-only labels.
7. Review and start.

### Review Views

- **Grid:** selected images first with duplicate stacks.
- **Loupe:** one image with key-face crops, analysis reasons, and duplicate filmstrip.
- **Compare:** two to six burst candidates synchronized for zoom and pan.
- **Timeline:** scenes and bursts along capture time.

The left filter list contains Selected, Highlights, Maybe, Technical Warnings, Closed Eyes, Soft Focus, Duplicates, and Unrated. The right context pane contains key faces, component scores, and alternatives for the active image.

Automated culling writes ratings/flags only after preview confirmation. Moving rejects or sending them to trash is a distinct later command and never part of analysis.

## Insights Workspace

Insights is the catalog dashboard and maintenance overview. It is not a marketing dashboard and should favor compact charts, tables, and drill-down controls over decorative cards.

### Catalog Overview

- Total logical photos, physical files, RAW/JPEG pairs, videos, and virtual copies.
- Images by year/month and root.
- Camera, lens, focal length, aperture, ISO, and rating distributions.
- Tagged, edited, rated, and unreviewed coverage.
- Missing/offline roots and stale metadata.
- Storage estimate by file type and root.

### People Metrics

- Face-scan coverage.
- Detected, unknown, suggested, confirmed, ignored, and low-quality face counts.
- People by image count and year.
- Stale embedding count by model version.
- Recognition confirmation and rejection rates.

### Culling Metrics

- Sessions, images analyzed, duplicate groups, and technical warnings.
- Suggested keep rate versus final keep rate.
- User override rate by reason and genre.
- Pairwise winner accuracy within duplicate groups.
- Processing throughput and failure counts.

Every count or chart segment is actionable. Selecting it opens the corresponding Library, People, or Cull result set.

## Settings Responsibilities

Settings contains configuration and maintenance only.

### Library

- Active library, database path, roots, availability, backup, relink, optimize, and delete.
- Metadata and background indexing defaults.
- Cache locations and size limits.
- `Return to Folder Browsing` without deleting the library.

### AI Models

- Installed, available, downloading, failed, and update-available model packs.
- Download, retry, cancel, remove, verify, benchmark, and `Download All` actions.
- Default detector, recognizer, culling models, and execution provider.
- Disk footprint, source, checksum, license notice, runtime, and last benchmark.
- Import a custom model manifest.

### AI Performance

- CPU thread budget and inference worker count.
- CPU, CUDA, TensorRT, DirectML, OpenVINO, CoreML, or automatic execution provider when supported by the build.
- Pause heavy jobs on battery or while editing.
- NAS concurrency and unavailable-root retry policy.

### Privacy and Cloud

- Local-only default.
- Google Takeout import history.
- Optional AWS Rekognition configuration and collection deletion.
- Delete face crops, embeddings, recognition data, culling analyses, or all AI-derived data.
- Credentials stored through an OS credential facility, never in the catalog SQLite file.

## Face Model System

RapidRAW already uses `ort` and `ndarray`. Face models should use the same Rust/ONNX Runtime foundation rather than introducing Python into the desktop runtime.

### Model Tasks

Face analysis is composed from independent tasks:

```text
oriented preview
    -> face detector
    -> bounding box + landmarks
    -> alignment adapter
    -> recognition embedder
    -> normalized embedding
    -> local clustering / person search
```

Some detectors produce five landmarks, some six, and some only rectangles. A detector without sufficient landmarks requires a landmark model before recognition. Each recognition adapter owns its alignment template, input dimensions, color order, normalization, output extraction, and embedding normalization.

### Initially Supported Detectors

| ID | Family | Runtime artifact | Role |
| --- | --- | --- | --- |
| `yunet` | OpenCV YuNet | ONNX | Fast default and digiKam-compatible baseline |
| `scrfd-500m` | InsightFace SCRFD | ONNX | Compact detector |
| `scrfd-2.5g` | InsightFace SCRFD | ONNX | Balanced detector |
| `scrfd-10g` | InsightFace SCRFD | ONNX | Higher-accuracy small-face detector |
| `retinaface` | RetinaFace | ONNX conversion/validated artifact | Quality-oriented detector |
| `blazeface-short` | MediaPipe BlazeFace | Validated ONNX conversion | Very fast close-range detector |
| `blazeface-full` | MediaPipe BlazeFace | Validated ONNX conversion | Fast full-range detector |

### Initially Supported Recognizers

| ID | Family | Runtime artifact | Role |
| --- | --- | --- | --- |
| `sface` | OpenCV SFace/MobileFaceNet | ONNX | Fast default and digiKam-compatible baseline |
| `arcface-r18` | ArcFace | ONNX | Compact ArcFace experiment |
| `arcface-r50` | ArcFace | ONNX | Balanced ArcFace experiment |
| `arcface-r100` | ArcFace/Glint360K | ONNX | Accuracy-oriented experiment |
| `adaface-ir18` | AdaFace | Validated ONNX conversion | Compact quality-adaptive experiment |
| `adaface-ir50` | AdaFace | Validated ONNX conversion | Quality-adaptive experiment |
| `facenet-128` | FaceNet | Validated ONNX artifact | Legacy comparison baseline |
| `facenet-512` | FaceNet | Validated ONNX artifact | Larger FaceNet baseline |
| `openface-nn4` | OpenFace nn4.small2 | Validated ONNX conversion | Legacy comparison baseline |

`All models` means all validated entries in the registry are downloadable. It does not mean arbitrary combinations share preprocessing or comparable scores.

Source-code licenses, model-file licenses, and training-dataset terms are separate facts. The registry records all three when known and displays uncertainty when provenance is incomplete. The experimental model setting may allow a user to proceed after acknowledgement, but the application must not silently describe an artifact as unrestricted merely because its inference code is open source.

### Recommended Model Packs

- **Fast:** YuNet + SFace quantized.
- **Balanced:** SCRFD 2.5G + ArcFace R50.
- **Quality:** SCRFD 10G or RetinaFace + ArcFace R100.
- **Low-quality faces:** RetinaFace + AdaFace IR50.
- **Mobile experiment:** BlazeFace + FaceNet.
- **digiKam comparison:** YuNet + SFace.
- **Legacy comparison:** YuNet + OpenFace.

The advanced model picker may choose detector and recognizer separately, but it must reject incompatible landmark/alignment contracts.

### Model Manifest

Hard-coded URL constants should evolve into a versioned registry. A simplified manifest is:

```json
{
  "id": "sface",
  "version": "2021dec",
  "task": "face_embedding",
  "displayName": "OpenCV SFace",
  "runtime": "onnx",
  "artifacts": [
    {
      "url": "https://upstream.example/model.onnx",
      "fileName": "model.onnx",
      "sha256": "...",
      "size": 38696353
    }
  ],
  "license": {
    "name": "Apache-2.0",
    "url": "https://upstream.example/LICENSE",
    "acceptanceRequired": false,
    "redistribution": "external-download"
  },
  "input": {
    "width": 112,
    "height": 112,
    "colorOrder": "rgb",
    "normalization": "sface"
  },
  "output": {
    "decoder": "flat_embedding",
    "dimensions": 128,
    "l2Normalize": true
  },
  "alignment": "opencv-five-point",
  "defaultThresholds": {
    "suggest": 0.0,
    "highConfidence": 0.0
  }
}
```

Thresholds in the example are deliberately unset until calibrated against a RapidRAW validation corpus. Published verification thresholds are useful starting points but are not automatically safe for open-set identification across thousands of people comparisons.

### Download Manager

- Model downloads start only from Model Settings or a scan wizard confirmation.
- Show exact total size before `Download All`.
- Limit concurrent downloads and inference initialization.
- Download to `.part`, support HTTP range resume when possible, verify SHA-256, then atomically rename.
- Persist progress and allow cancel/retry.
- Validate ONNX inputs and outputs before marking a model installed.
- Keep exact upstream source and license text in the installed manifest.
- Restricted model families require explicit license acknowledgement before download.
- Removal is disabled while a model is used by an active job.
- A model update installs beside the old version until dependent embeddings are rebuilt or discarded.

Models live globally under the app data model directory. Catalogs record model IDs and versions but do not duplicate the weights.

### Runtime Interface

Conceptually, the Rust backend needs these interfaces:

```text
FaceDetector.detect(image) -> [FaceDetection]
FaceAligner.align(image, detection, contract) -> FaceCrop
FaceEmbedder.embed(crops) -> [Embedding]
FaceRecognitionProvider.enroll/search/delete -> provider-specific identity operations
```

ONNX adapters implement the first three. AWS Rekognition implements the provider interface but does not expose embeddings. Google Takeout implements an import interface rather than a recognizer.

## Recognition and Clustering

### Detection

- Correct EXIF orientation before inference.
- Store normalized coordinates relative to the oriented logical image.
- Use embedded RAW previews for the normal pass.
- Offer a tiled high-resolution pass for group photos and small background faces.
- Store detection confidence, landmarks, detector ID/version, preview dimensions, and quality metrics.
- Deduplicate faces for a RAW/JPEG pair at the logical-capture level.

### Embeddings

- Align each face using the recognizer's required template.
- Batch compatible face crops for inference.
- Store float embeddings as BLOBs with dimensions, model ID, and model version.
- Retain embeddings for multiple recognition models during experiments.
- Never compare vectors across model IDs or versions.

### Person Matching

Recognition should use multiple confirmed exemplars per person rather than one centroid only. Age, pose, glasses, facial hair, and lighting can produce separate appearance modes.

A practical first matcher is:

1. Find likely people using one or more person centroids.
2. Compare the query with confirmed exemplars for the candidate people.
3. Aggregate the best few quality-weighted similarities.
4. Apply model-specific thresholds and a margin over the second candidate.
5. Store the suggestion and explanation.
6. Require confirmation unless an explicit high-confidence batch rule is active.

Rejected person/face pairs are negative evidence and should suppress repeated suggestions. They should not directly fine-tune the embedding network.

### Unknown Clustering

Cluster unknown embeddings in the background using cosine distance. Start with conservative density-based clustering and expose clusters as review aids, not truths. Temporal proximity, co-occurrence, and imported weak labels may refine ranking, but must not override strong facial disagreement.

For early catalogs, brute-force vector comparison is acceptable. A catalog with 100,000 128-dimensional float embeddings is still tractable in memory. Add a vector index only after profiling demonstrates a need.

## Database Ownership

Use two SQLite files beside the library:

```text
rapidraw-library.db    durable catalog and user decisions
rapidraw-ai-cache.db   rebuildable embeddings, features, and analysis cache
```

The core database must remain sufficient to preserve user work if the AI cache is deleted.

### Migration From Catalog Schema Version 1

The current catalog work stores `person:<name>` sidecar values as tags with `kind = 'person'`. During the People schema migration:

1. Create one stable `people` row for each unique normalized person tag.
2. Preserve the original tag spelling and image association.
3. Import the association as image-level evidence with its original source, not as a confirmed face assignment.
4. Run local face detection when requested.
5. If an image has one detected face and one person tag, offer a high-confidence migration suggestion.
6. If an image has multiple people or faces, use the same constrained weak-label assignment used by Google Takeout.
7. Keep the legacy tag searchable until the user confirms migration results.

The schema migration is transactional and does not rewrite `.rrdata` or XMP sidecars.

### Durable Core Tables

```sql
CREATE TABLE people (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  sort_name TEXT,
  representative_face_id INTEGER,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  UNIQUE(name)
);

CREATE TABLE faces (
  id INTEGER PRIMARY KEY,
  logical_image_id INTEGER NOT NULL,
  left_norm REAL NOT NULL,
  top_norm REAL NOT NULL,
  width_norm REAL NOT NULL,
  height_norm REAL NOT NULL,
  landmarks_json TEXT,
  detector_model_id TEXT NOT NULL,
  detector_model_version TEXT NOT NULL,
  detection_confidence REAL,
  region_source TEXT NOT NULL,
  state TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE face_assignments (
  face_id INTEGER PRIMARY KEY REFERENCES faces(id) ON DELETE CASCADE,
  person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,
  state TEXT NOT NULL,
  source TEXT NOT NULL,
  confidence REAL,
  recognition_model_id TEXT,
  recognition_model_version TEXT,
  confirmed_at INTEGER,
  updated_at INTEGER NOT NULL
);

CREATE TABLE face_match_rejections (
  face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
  person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
  model_id TEXT NOT NULL,
  model_version TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY(face_id, person_id, model_id, model_version)
);
```

`logical_image_id` represents a grouped capture so RAW/JPEG files do not create duplicate People results. The implementation may introduce a dedicated logical-capture table or use the existing group identity with a stable catalog ID.

### Rebuildable AI Cache Tables

```sql
CREATE TABLE face_embeddings (
  face_id INTEGER NOT NULL,
  model_id TEXT NOT NULL,
  model_version TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  vector BLOB NOT NULL,
  quality REAL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY(face_id, model_id, model_version)
);

CREATE TABLE image_ai_features (
  logical_image_id INTEGER NOT NULL,
  feature_set_id TEXT NOT NULL,
  feature_set_version TEXT NOT NULL,
  features_json TEXT NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY(logical_image_id, feature_set_id, feature_set_version)
);

CREATE TABLE similarity_groups (
  id INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL,
  representative_logical_image_id INTEGER,
  group_type TEXT NOT NULL,
  confidence REAL
);
```

Face crop thumbnails are files in the cache directory, addressed by catalog ID and model-independent crop revision. They do not belong in the core database BLOB pages.

### Jobs and Sessions

Persist jobs so they can resume after restart:

- `ai_jobs`: type, scope, state, progress, model contract, settings, timestamps, and error summary.
- `ai_job_items`: image ID, stage, attempts, result, and last error.
- `cull_sessions`: scope, genre, settings, feature-set version, and state.
- `cull_decisions`: suggested and final status, scores, reasons, and user override.
- `preference_events`: pairwise choices and context used by personalization.

Use one catalog writer actor or queue. Inference workers send bounded result batches to it; they do not each open long write transactions.

## Google Photos and Amazon Options

### Feasibility Matrix

| Integration | Access existing face groups | Can recognize a supplied face | Recommended use |
| --- | --- | --- | --- |
| Google Photos Library API | No | No | Do not build recognition around it |
| Google Photos Picker API | No labels; user-selected media only | No | Optional manual seed-photo import |
| Google Cloud Vision | No | Explicitly does not identify individuals | Not useful for recognition |
| Google Takeout | Sometimes image-level people names, no face boxes | Offline migration only | Build an import wizard |
| Amazon Photos | No supported public People/face API found | No | Do not build around it |
| Amazon Rekognition | Separate collection, not Amazon Photos identities | Yes | Optional cloud provider |

Google restricted the Photos Library API after March 31, 2025 to app-created media and directs full-library selection to the Picker API. Neither API exposes private People clusters. Cloud Vision explicitly does not support recognition of specific individuals.

### Google Takeout Import

Some Takeout JSON sidecars contain an image-level `people` array such as `Alice` and `Bob`, but do not identify which detected face belongs to which name. Availability varies, so the importer must inspect the archive and report coverage before changing the catalog.

Import process:

1. Select a Takeout `Google Photos` directory or archive.
2. Resolve each JSON sidecar to its media file despite Takeout naming variants.
3. Match exported media to catalog images by content hash when possible, then filename plus capture time and dimensions as a fallback.
4. Import names as weak image-level labels with source `google_takeout`.
5. Run local face detection on matched images.
6. Use unambiguous single-face/single-name images as high-quality identity seeds.
7. Build initial local prototypes from those seeds.
8. For images with multiple faces and names, solve a constrained face-to-name assignment using prototype similarity and one-to-one bipartite matching.
9. Iterate assignments and prototypes while retaining confidence and provenance.
10. Present inferred assignments in Suggestions for confirmation.

This transfers useful recognition knowledge without pretending Takeout provides face rectangles. Ambiguous image-level labels remain searchable as `contains person` metadata but do not become confirmed face assignments.

The optional Picker workflow can ask the user to enter a person's name and then select representative Google Photos images for that person. It is manual enrollment, not access to Google's recognition model.

### Amazon Rekognition Provider

AWS Rekognition collections can store face vectors, associate multiple faces with a user, and search by an input image. A RapidRAW provider could:

1. Create one collection per library and AWS region.
2. Upload confirmed face crops to `IndexFaces`.
3. Associate several faces with a provider user ID.
4. Send a locally detected face crop to `SearchUsersByImage`.
5. Store only AWS IDs and match scores locally.
6. Delete faces, users, and the collection from RapidRAW.

This does not reuse anything from Amazon Photos. It is a paid cloud alternative to local embeddings. AWS states that submitted images may be stored and used to improve the service unless the account opts out, so setup must disclose that policy and link to the opt-out controls.

Local detection remains useful because RapidRAW can send one aligned crop rather than an entire family photograph. The provider must still treat AWS's returned score as provider-specific.

## Background Execution and NAS Safety

### Job Pipeline

```text
scope query
  -> enumerate catalog IDs without filesystem probes
  -> bounded decode queue
  -> preview/crop cache
  -> bounded inference workers
  -> result queue
  -> single batched SQLite writer
  -> UI events and persistent job progress
```

- Enumeration reads catalog rows first and does not synchronously `stat` every path.
- Decode and inference have independent concurrency limits.
- Jobs checkpoint after small batches and can pause, resume, or cancel.
- UI events are throttled so thousands of image updates do not cause React churn.
- A failed image records an error and does not abort the whole job.
- The editor and thumbnail loader retain reserved capacity while background AI runs.

### Unresponsive Network Filesystems

`spawn_blocking` prevents an async executor thread from blocking, but it cannot interrupt a Linux thread stuck in CIFS `D` state. For paths on network roots, filesystem reads should be delegated to a small helper process with a bounded request protocol.

- The main Tauri process sends one explicit file request to the helper.
- A blocked helper does not hold the UI, catalog lock, or inference pool.
- A timeout marks the root temporarily unavailable and stops scheduling more work there.
- Only a small bounded number of helpers may exist, preventing a pile-up of stuck processes.
- Cancellation abandons the request and job immediately even if the kernel does not release the helper yet.
- Local roots may use the normal in-process decoder for lower overhead.

Mount configuration still affects recovery, but the application must remain usable when a mount is stale.

## Culling Analysis

Aftershoot publicly describes a pipeline that scores images from 1 to 100, groups similar images, selects within each group, identifies key faces, and separates selected, highlight, duplicate, blurry, and closed-eye results. It also changes priorities by genre and learns from corrected culls. Its exact networks and formula are proprietary.

RapidRAW should implement the observable product behavior as a transparent feature ensemble rather than claim to reproduce Aftershoot's hidden models.

### Current RapidRAW Baseline

The catalog branch's current auto-cull implementation already has:

- Double-gradient perceptual hashing for similarity groups.
- Global Laplacian variance.
- Center-crop sharpness.
- Optional U2Net subject-mask sharpness.
- Histogram-based highlight and shadow clipping.
- A weighted quality score.
- RAW/JPEG logical candidate handling.
- Preview-before-apply behavior.

Its principal limitations are:

- A 720-pixel analysis image can conceal focus differences visible at delivery resolution.
- Perceptual hash distance alone may merge different moments or split a burst when framing changes.
- Connected-component grouping can bridge dissimilar endpoints through intermediate images.
- Absolute Laplacian thresholds vary with camera resolution, noise, demosaicing, sharpening, and subject texture.
- It has no face-local sharpness, eye-state, pose, expression, scene coverage, or learned preference model.
- It maps a small set of metrics directly to destructive workflow choices too early.

### Proposed Stages

#### 1. Logical Capture Resolution

- Collapse RAW/JPEG pairs and configured derivative files.
- Choose the fastest representative preview that preserves enough detail.
- Retain backing paths so final ratings and moves apply consistently.

#### 2. Scene and Burst Segmentation

Combine:

- Capture-time gaps and camera sequence numbers.
- Perceptual hashes.
- Existing CLIP image embeddings or a compact visual embedding.
- Composition geometry and detected subject overlap.
- Camera/lens metadata.

First divide a shoot into scenes, then build duplicate/burst groups only within plausible temporal neighborhoods. Use complete-linkage or representative-constrained grouping to avoid transitive chains that join distinct endpoints.

#### 3. Objective Technical Features

- Global sharpness at multiple scales.
- Subject-region sharpness.
- Face-local and eye-local sharpness.
- Motion blur direction and magnitude.
- Noise estimate separated from genuine detail.
- Highlight and shadow clipping by channel.
- Exposure balance and dynamic range.
- White-balance outlier relative to the scene.
- Face occlusion and extreme pose.
- Eye-open probability per key face.
- Source preview quality and confidence in every metric.

Technical features should be relative to comparable images in the same scene whenever possible. A soft portrait should not be judged against a high-texture landscape using one global threshold.

#### 4. Subject and Semantic Features

- Primary subject coverage and prominence.
- Key-face identity consistency across a burst.
- Subject separation from background.
- Saliency and background clutter.
- Gaze and expression changes.
- Composition features such as horizon, headroom, edge intersections, and subject crop.
- Aesthetic predictor output as a weak feature, never a hard reject.

Emotion classification is culturally and contextually fragile. It may help rank near-identical expressions but should not label a person's emotional state or remove a unique moment.

#### 5. Relative Ranking

Within a group, compute an explainable ranking:

```text
rank = technical_quality
     + subject_quality
     + key_face_quality
     + expression_preference
     + composition
     + personalized_preference
     - severe_fault_penalties
```

Weights depend on genre. Sports emphasizes subject/body focus and motion; portraits emphasize eye and face sharpness; weddings protect scene coverage and key faces; boudoir and artistic profiles reduce closed-eye penalties.

The system selects at least one candidate from every distinct scene or duplicate group. Unique images are never discarded solely because an aesthetic model gives them a low score.

#### 6. Result Buckets

- **Selected:** best candidates satisfying scene coverage.
- **Highlights:** strongest subset of Selected.
- **Maybe:** best available image in a weak group or an uncertain decision.
- **Technical Warnings:** severe blur, clipping, corruption, or uncertain decode.
- **Closed Eyes:** review category, not an automatic reject in every genre.
- **Duplicates:** alternatives grouped under a selected representative.
- **Unrated:** not analyzed or deliberately reset.

### Candidate Models and Methods

The culling model registry should eventually support:

- Existing CLIP embedding for semantic similarity and weak aesthetic heads.
- U2Net or a newer compact segmentation model for subject localization.
- Face detector and landmarks shared with People.
- A compact eye-state classifier or landmark-derived eye feature with confidence fallback.
- A no-reference image-quality model such as NIMA, MUSIQ, or a validated compact alternative.
- A compact learned blur/noise classifier in addition to deterministic metrics.
- Optional aesthetic linear heads over CLIP embeddings.

No quality model should be adopted only from benchmark scores. It must be evaluated on RAW-derived previews, varied genres, diverse people, and the actual failure cases photographers reject.

### Personalization

Do not fine-tune the face recognizer or a large vision model on every catalog. Learn a lightweight per-user ranking head over cached features.

Training signals include:

- Replacing the selected image in a duplicate group.
- Keeping or rejecting an AI suggestion.
- Pairwise compare choices.
- Final stars, flags, and colors after a cull.
- Genre and camera context.

Pairwise ranking is preferable to predicting an absolute universal score. The question is usually `which of these near-duplicates does this photographer prefer?`

Use a staged model roadmap:

1. Collect explicit overrides and optional one-click reasons from the first cull onward, but do not personalize recommendations yet.
2. After at least 200 meaningful cross-group preference pairs, activate a regularized local linear ranker as the conservative, explainable fallback.
3. After several thousand diverse pairs and a held-out improvement over the linear ranker, train a small per-user DNN ranking head over frozen visual embeddings plus technical, face, subject, composition, and metadata features. Keep its output as a bounded adjustment to the transparent baseline rather than a deletion authority.
4. Use a contextual-bandit policy only to select the most informative uncertain comparisons to ask the user. It must never autonomously decide which source images to remove.

Every proposed keep or reject stores measured values, comparative evidence, model versions, the personalized adjustment, and the final user decision. The review UI must distinguish direct measurements from model inferences and personalized preferences.

Keep separate genre profiles and a global fallback. Show sample count and confidence. Provide reset, export, and disable controls. A correction updates the lightweight preference model without re-running image feature extraction.

## Search and Smart Collections

AI-derived predicates join the structured catalog query model:

- Person confirmed/suggested/unknown.
- Contains faces and face count.
- Face scan state and model version.
- Duplicate group and representative state.
- Culling session, bucket, score range, and warning type.
- Similar to selected image.
- Semantic text similarity.
- AI tag and confidence.

Saved searches store structured JSON, not generated SQL. Schema migrations can then translate old predicates safely.

Examples:

- `Alice + 2025 + 4 stars and up`
- `Unknown faces from Maui folder`
- `Sony 85mm portraits with closed-eye warning`
- `Unedited RAW highlights from last culling session`
- `Images not face-scanned with available source files`

## Privacy, Security, and Data Lifecycle

- Store cloud credentials in the OS credential store.
- Keep model and provider IDs in SQLite, not secrets.
- Make cloud operations opt-in per provider and library.
- Show upload count and estimated cost before cloud enrollment or search.
- Provide delete-provider-data and verify-deletion actions.
- Encrypt transport and respect provider region selection.
- Consider optional SQLCipher or an encrypted AI cache later; normal SQLite is not encrypted at rest.
- Delete cached face crops when face data is cleared.
- Preserve user-confirmed person tags when only embeddings are cleared.
- Do not write person names or face regions into image metadata unless the user explicitly enables XMP face-region export.
- Record provenance for imported labels and every automatic assignment.

## Evaluation

### Face Detection

- Recall for large, medium, and small faces.
- False positives per image.
- Landmark/alignment failures.
- Throughput and peak memory by model and execution provider.
- Group photos, profiles, glasses, masks, children, aging, low light, and varied skin tones.

### Face Recognition

- True accept and false accept rates on a catalog-specific labeled validation set.
- Open-set rejection accuracy for people with no confirmed identity.
- Confirmation/rejection rate of top-one suggestions.
- Accuracy across age gaps and appearance changes.
- Threshold and second-candidate margin calibrated separately for each model.

### Culling

- Pairwise winner agreement within duplicate groups.
- Recall of the user's final delivered images.
- Catastrophic miss rate: unique or essential moments not selected.
- Duplicate grouping precision and recall.
- Blur and closed-eye precision/recall.
- Override rate by reason and genre.
- Time saved in review, not just model runtime.

The app should offer `Benchmark on this computer` and an advanced `Compare models on selected labeled faces` experiment. Results remain local and are clearly separated from production assignments.

## Implementation Plan

## Headless CLI

RapidRAW should ship a companion binary, `rapidraw-cli`, for batch work, automation, server-side catalog maintenance, and reproducible model evaluation. It is intentionally ImageMagick-like: every operation can run without a window, returns a non-zero exit code on failure, and can emit structured JSON for scripts.

The CLI must call the same Rust services as Tauri commands. It must not duplicate scanner, model, matching, or SQLite logic in a second implementation.

```text
rapidraw-cli library create --name "Archive" --database /data/archive.db
rapidraw-cli library add-root --database /data/archive.db /photos/2026
rapidraw-cli catalog scan --database /data/archive.db --root /photos/2026 --recursive --json-progress
rapidraw-cli models list --json
rapidraw-cli models install opencv-yunet-sface
rapidraw-cli faces detect --database /data/archive.db --model opencv-yunet-sface --root /photos/2026
rapidraw-cli faces cluster --database /data/archive.db --model opencv-yunet-sface
rapidraw-cli faces recognize --database /data/archive.db --model opencv-yunet-sface --json-report report.json
rapidraw-cli people list --database /data/archive.db --json
rapidraw-cli cull analyze /photos/session --preset portraits --json-report cull-report.json
```

### CLI Contract

- `--database` is required for catalog, People, and durable AI commands. Filesystem-only operations accept paths directly.
- `--json` produces a single machine-readable result on stdout. `--json-progress` emits newline-delimited JSON progress events on stdout; diagnostics remain on stderr.
- `--quiet` suppresses human progress output. `--dry-run` validates scope, model availability, and write access without changing a catalog.
- `--wait`, `--pause`, `--resume`, `--cancel`, and `jobs` operate on the same durable job records shown in the desktop status bar.
- Commands that may download models require `--accept-license <pack-id>` where acknowledgement is required. They never download models implicitly.
- Recognition commands produce suggestions only by default. `--confirm` requires an explicit person identifier and must be auditable in the catalog.
- Model, scan, benchmark, and culling reports include model revision, source paths, counts, elapsed time, skipped files, and recoverable errors.
- The command exit status is stable: `0` complete, `1` operational failure, `2` invalid arguments, `3` cancelled, and `4` partial completion.

### CLI Test Strategy

- Unit-test argument parsing and exit-code mapping.
- Run integration tests against a temporary SQLite catalog and fixture images.
- Snapshot JSON output schemas so shell automation does not break silently.
- Exercise cancellation, an unavailable root, a missing model, and a model-license rejection in CI.

## Batch Analysis Execution

Every AI analysis feature, including tagging, faces, wildlife classification, embeddings, and culling features, runs as a durable batch job. A UI action only enqueues work and returns immediately.

- Enumerate candidate images in a short-lived database read transaction, then process bounded batches rather than holding a SQLite reader or writer for the full collection.
- Use a bounded worker pool sized by configurable CPU/GPU limits. Image decode, inference, and database writes are separate stages connected by bounded queues, so a fast scanner cannot exhaust memory while inference is slower.
- Serialize catalog writes through the existing writer boundary in small transactions. Each batch commits independently; a restart resumes or retries only unfinished items.
- Persist per-image analysis state keyed by image revision and model revision. Never recompute an unchanged successful item unless the user explicitly requests reanalysis.
- Throttle status events by time and batch boundaries, not by image. The UI receives job state, completed/total, current item, rate, ETA when reliable, and aggregated failures without rendering on every file.
- UI updates must be non-blocking: no synchronous filesystem or database calls on the render path, no modal progress loops, and no image decoding solely for status display. Preview thumbnails use the existing thumbnail queue with a bounded request rate.
- Pause stops dequeuing new work; cancel drains or abandons pending work at safe checkpoints; already committed results remain durable. Jobs expose retry-failed and retry-all modes.
- Remote/NAS access runs through the isolated, timeout-aware discovery path. A stuck filesystem call must not occupy the database writer, UI thread, or all analysis workers.

### Phase 0: Catalog Integration

- Land the SQLite catalog work on the active development branch.
- Introduce stable logical-capture IDs for RAW/JPEG groups.
- Move catalog writes behind a single writer queue.
- Add durable job/session tables and schema migrations.
- Extract catalog operations into services callable by both Tauri and `rapidraw-cli`.

### Phase 1: AI Registry and Job Manager

- Replace hard-coded model downloads with manifest-backed model management.
- Implement list, download, download-all, cancel, verify, remove, and benchmark commands.
- Build persistent background jobs with pause/resume/cancel and throttled status events.
- Keep existing AI models represented in the same registry over time.
- Add the CLI shell with `models`, `library`, `catalog`, `jobs`, and JSON progress conventions before face processing.

### Phase 2: Vertical Face Slice

- Implement YuNet detection and SFace embeddings first.
- Add face/person schema, face crop cache, matching, and model versioning.
- Add Scan Faces wizard and background status details.
- Build Unknown, Suggestions, All People, and Person Detail views.
- Add equivalent `faces detect`, `faces cluster`, `faces recognize`, and `people` CLI commands.

This first pair proves alignment, persistence, review, model migration, and UI before multiplying adapters.

### Phase 3: All Face Model Adapters

- Add SCRFD, RetinaFace, and BlazeFace detector adapters.
- Add ArcFace, AdaFace, FaceNet, and OpenFace recognizer adapters.
- Publish validated manifest entries, checksums, conversion provenance, and compatibility rules.
- Add pack presets, advanced independent selection, side-by-side benchmarks, and Download All.

### Phase 4: Google Migration and Optional AWS

- Build Takeout archive inspection and report.
- Import image-level people labels and run weak-label assignment.
- Add manual Google Picker seed workflow only if it materially improves enrollment.
- Implement AWS Rekognition behind the provider interface after local behavior is stable.

### Phase 5: Culling Foundation

- Replace global-only scoring with feature extraction cached by version.
- Improve scene/burst grouping and preserve scene coverage.
- Add face-local focus, key faces, eye state, and technical warning explanations.
- Persist cull sessions and build Grid, Loupe, and Compare review views.
- Keep results non-destructive.

### Visual Tagging And Wildlife Identification

- Add RAM++ as the first local broad-tagging adapter for concepts such as animal, bird, tree, lake, mountain, car, sky, and activity phrases. Store its results as AI tags with model/version/confidence provenance; never overwrite manual tags.
- Use Tag2Text only as an optional captioning companion, not as the canonical keyword source.
- Add BioCLIP as the biology-specific classifier. It should receive either the whole image or an automatically detected animal/bird crop and return ranked taxonomic candidates.
- For bird photography, use a cascade: RAM++ first supplies a broad image-level `bird` confidence gate; a bird-capable local object detector then produces one or more crops; BioCLIP ranks species for every crop. RAM++ is not the crop detector because its tags are image-level rather than bounding boxes. Present top-k common and scientific names with confidence, taxonomy, and an explicit `Uncertain` outcome; do not silently apply species tags below a calibrated threshold.
- Support regional candidate lists derived from capture GPS/date, but preserve a global mode and clearly mark regional filtering as a ranking aid rather than proof of identification.
- Evaluate a packaged ONNX BioCLIP export or a sidecar adapter first. The Rust core should consume a stable embedding/classifier interface so a specialist bird model can later replace it for a chosen geography.
- Add taxonomy fields to the catalog (`kingdom` through `species`, common name, model version, confidence, review state) and expose them as searchable facets and smart collections.

### Phase 6: Culling Personalization

- Record pairwise decisions and overrides.
- Train lightweight local ranking profiles per genre.
- Add profile confidence, reset/export, and A/B evaluation.
- Add highlights and configurable automated selection only after assisted-mode quality is measured.

### Phase 7: Insights and Advanced Search

- Build catalog, People, and Cull metrics.
- Add AI predicates to structured search and smart collections.
- Make every metric drill into a concrete result set.

## Acceptance Criteria

- Opening the app or catalog never downloads or initializes face models.
- Every listed model can be downloaded, verified, cancelled, removed, and benchmarked from the UI.
- Every catalog, model, face, and culling operation has an equivalent documented `rapidraw-cli` command with stable JSON output.
- A recognition model can be changed without repeating face detection when stored landmarks are compatible.
- Model/version changes never silently compare incompatible embeddings.
- Face indexing and culling run in the background with accurate status, current item, pause, resume, cancel, and error details.
- An unavailable NAS cannot block the splash screen, catalog UI, SQLite writer, or editor.
- RAW/JPEG pairs do not duplicate People results or culling candidates.
- Search by person uses confirmed assignments by default and can explicitly include suggestions.
- Google Takeout import reports matched, unmatched, labeled, ambiguous, and inferred counts before confirmation.
- Automated culling cannot delete or move originals as part of analysis.
- User corrections survive cache deletion and model upgrades.

## Research Sources

- [digiKam People workflow](https://docs.digikam.org/en/left_sidebar/people_view.html)
- [digiKam face maintenance](https://docs.digikam.org/en/maintenance_tools/maintenance_faces.html)
- [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [OpenCV SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
- [InsightFace model restrictions](https://github.com/deepinsight/insightface/tree/master/model_zoo)
- [AdaFace](https://github.com/mk-minchul/AdaFace)
- [MediaPipe face detection](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector)
- [FaceNet](https://github.com/davidsandberg/facenet)
- [OpenFace](https://github.com/cmusatyalab/openface)
- [Rust ONNX Runtime binding](https://github.com/pykeio/ort)
- [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/)
- [Google Photos API changes](https://developers.google.com/photos/support/updates)
- [Google Photos Picker](https://developers.google.com/photos/picker/guides/get-started-picker)
- [Google Cloud Vision face detection limitations](https://cloud.google.com/vision/docs/detecting-faces)
- [Google Photos face groups](https://support.google.com/photos/answer/6128838)
- [AWS Rekognition face collections](https://docs.aws.amazon.com/rekognition/latest/dg/collections.html)
- [AWS Rekognition data handling](https://docs.aws.amazon.com/rekognition/latest/dg/security-data-encryption.html)
- [Aftershoot culling workflow](https://support.aftershoot.com/en/articles/5223473-get-started-with-aftershoot-culling)
- [Aftershoot culling preferences](https://support.aftershoot.com/en/articles/6508163-setting-your-ai-automated-culling-preferences-in-aftershoot)
- [Aftershoot technical overview](https://support.aftershoot.com/en/articles/10601968-technical-answers-about-aftershoot)
- [Aftershoot genre behavior](https://support.aftershoot.com/en/articles/10570203-aftershoot-culling-genres)

## Immediate Recommendation

Build the shared model registry and persistent AI job manager before implementing a large set of face adapters. Then deliver one complete YuNet/SFace People workflow, including review and migrations, before adding the remaining models. This ordering does not reduce the intended model scope; it ensures every subsequent detector and recognizer plugs into a tested download, schema, job, and UI contract instead of creating parallel one-off implementations.
