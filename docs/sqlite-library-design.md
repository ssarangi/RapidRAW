# SQLite Library Design

RapidRAW should keep the current filesystem-first workflow as the default and add an optional SQLite-backed catalog for users who want a curated, searchable library. The main compatibility rule is that existing image loading, editing, thumbnailing, export, culling, and external edit code should still receive normal RapidRAW paths. The database becomes an index and organization layer, not a replacement for the source files.

## Goals

- Keep "open a folder and edit files" working without creating a database.
- Let users create one or more named libraries stored as local SQLite files.
- Make large libraries fast to browse, filter, search, tag, rate, and organize.
- Preserve RapidRAW sidecars as the portable edit format.
- Allow later expansion into thumbnails, perceptual similarity, face/person data, and advanced saved searches.

## Inspiration From digiKam

digiKam's important architectural ideas are worth borrowing, but not copying table-for-table:

- A core database stores collection roots, albums, images, tags, metadata, and searches.
- Thumbnail, similarity, and face recognition data can be separated from core catalog data.
- Album roots are explicit, so relative image paths survive when a root moves.
- Tags are normalized globally and assigned through join tables.
- WAL mode is recommended for large SQLite catalogs.

RapidRAW should start with a smaller single-file catalog and keep cache-like data optional:

- `rapidraw-library.db`: durable catalog state.
- Existing thumbnail cache directory: continue storing generated thumbnails outside the core DB initially.
- Future optional `rapidraw-cache.db`: thumbnails, hashes, similarity vectors, face embeddings.

## Library Modes

### Filesystem Mode

This is the current behavior. A folder path is selected, `list_images_in_dir` or `list_images_recursive` scans the filesystem, and `ImageFile.path` remains the identity. Ratings, tags, virtual copies, and edits continue to resolve from `.rrdata` sidecars and optional XMP sync.

No SQLite connection is opened unless the user explicitly creates or opens a library.

### Catalog Mode

A library has a SQLite file and a set of imported roots. The UI can browse either:

- Physical folders from indexed roots.
- Albums and album groups.
- Smart albums/saved searches.
- Tag/date/rating/color/camera/lens views.

Every result row maps back to an `ImageFile` shape so existing frontend and editor code keep working.

## Storage Location

Default local path:

```text
<app_data_dir>/libraries/<library-id>/rapidraw-library.db
```

Users should also be able to create a portable library beside photo roots:

```text
<chosen-folder>/rapidraw-library.db
```

For performance and reliability, SQLite files should live on local SSD/NVMe storage. Photo files can live elsewhere, but the DB should not be placed directly on flaky network shares.

## SQLite Configuration

On open:

```sql
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA temp_store = MEMORY;
PRAGMA busy_timeout = 5000;
```

Use one write connection guarded by a Rust mutex or async task queue, plus short-lived read connections. Catalog writes should be explicit transactions.

## Core Schema

```sql
CREATE TABLE schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at INTEGER NOT NULL
);

CREATE TABLE libraries (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  app_version TEXT
);

CREATE TABLE collection_roots (
  id INTEGER PRIMARY KEY,
  library_id TEXT NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
  label TEXT,
  absolute_path TEXT NOT NULL,
  canonical_path TEXT,
  volume_id TEXT,
  is_available INTEGER NOT NULL DEFAULT 1,
  last_scan_at INTEGER,
  UNIQUE(library_id, absolute_path)
);

CREATE TABLE folders (
  id INTEGER PRIMARY KEY,
  root_id INTEGER NOT NULL REFERENCES collection_roots(id) ON DELETE CASCADE,
  parent_id INTEGER REFERENCES folders(id) ON DELETE CASCADE,
  relative_path TEXT NOT NULL,
  name TEXT NOT NULL,
  modified_at INTEGER,
  indexed_at INTEGER,
  image_count INTEGER NOT NULL DEFAULT 0,
  UNIQUE(root_id, relative_path)
);

CREATE TABLE images (
  id INTEGER PRIMARY KEY,
  root_id INTEGER NOT NULL REFERENCES collection_roots(id) ON DELETE CASCADE,
  folder_id INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
  file_name TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  extension TEXT,
  file_size INTEGER,
  modified_at INTEGER NOT NULL,
  content_hash TEXT,
  status TEXT NOT NULL DEFAULT 'present',
  is_raw INTEGER NOT NULL DEFAULT 0,
  is_cloud_placeholder INTEGER NOT NULL DEFAULT 0,
  imported_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  UNIQUE(root_id, relative_path)
);

CREATE INDEX idx_images_folder ON images(folder_id, file_name);
CREATE INDEX idx_images_modified ON images(modified_at);
CREATE INDEX idx_images_hash ON images(content_hash);
```

`relative_path` is the root-relative path including filename. Absolute paths are resolved at query time as `collection_roots.absolute_path + images.relative_path`.

## RapidRAW Edit State

Sidecars remain the source of truth for non-destructive edits. The database caches enough state to make browsing and filtering fast.

```sql
CREATE TABLE image_versions (
  id INTEGER PRIMARY KEY,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  copy_id TEXT,
  display_name TEXT,
  sidecar_path TEXT,
  rating INTEGER NOT NULL DEFAULT 0,
  color_label TEXT,
  is_edited INTEGER NOT NULL DEFAULT 0,
  adjustments_json TEXT,
  sidecar_modified_at INTEGER,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  UNIQUE(image_id, copy_id)
);

CREATE INDEX idx_versions_rating ON image_versions(rating);
CREATE INDEX idx_versions_color ON image_versions(color_label);
CREATE INDEX idx_versions_edited ON image_versions(is_edited);
```

For the original image, `copy_id` is null. Virtual copies use the existing `?vc=<id>` path convention at the API boundary.

Write policy:

- Editing writes `.rrdata` first.
- The catalog is updated after the sidecar write succeeds.
- If the DB update fails, the sidecar still preserves the edit and the catalog can repair itself on next scan.

## Metadata

Metadata should be separated into searchable typed fields and flexible key/value properties.

```sql
CREATE TABLE image_metadata (
  image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
  date_taken INTEGER,
  width INTEGER,
  height INTEGER,
  orientation INTEGER,
  camera_make TEXT,
  camera_model TEXT,
  lens_model TEXT,
  focal_length REAL,
  aperture REAL,
  shutter TEXT,
  iso INTEGER,
  exposure_compensation REAL,
  gps_lat REAL,
  gps_lon REAL,
  title TEXT,
  caption TEXT,
  author TEXT,
  copyright TEXT,
  metadata_hash TEXT,
  updated_at INTEGER NOT NULL
);

CREATE INDEX idx_metadata_date_taken ON image_metadata(date_taken);
CREATE INDEX idx_metadata_camera ON image_metadata(camera_make, camera_model);
CREATE INDEX idx_metadata_lens ON image_metadata(lens_model);

CREATE TABLE image_properties (
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  namespace TEXT NOT NULL,
  name TEXT NOT NULL,
  value TEXT,
  PRIMARY KEY(image_id, namespace, name)
);
```

The typed metadata table powers common library filters. `image_properties` keeps less common EXIF/XMP fields without schema churn.

## Tags And Labels

```sql
CREATE TABLE tags (
  id INTEGER PRIMARY KEY,
  parent_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'user',
  icon TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  UNIQUE(parent_id, name, kind)
);

CREATE TABLE image_tags (
  image_version_id INTEGER NOT NULL REFERENCES image_versions(id) ON DELETE CASCADE,
  tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
  source TEXT NOT NULL DEFAULT 'user',
  confidence REAL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY(image_version_id, tag_id, source)
);

CREATE INDEX idx_image_tags_tag ON image_tags(tag_id, image_version_id);
```

Existing tag strings map cleanly:

- `user:<name>` -> `tags.kind = 'user'`
- AI tags -> `tags.kind = 'ai'`
- `color:<name>` should be stored in `image_versions.color_label`, not as a general tag, but can be exposed through the old `tags` array for compatibility.

## Albums

The current JSON album tree should migrate into normalized tables.

```sql
CREATE TABLE album_nodes (
  id TEXT PRIMARY KEY,
  parent_id TEXT REFERENCES album_nodes(id) ON DELETE CASCADE,
  library_id TEXT NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
  type TEXT NOT NULL CHECK(type IN ('group', 'album')),
  name TEXT NOT NULL,
  icon TEXT,
  sort_order INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE album_images (
  album_id TEXT NOT NULL REFERENCES album_nodes(id) ON DELETE CASCADE,
  image_version_id INTEGER NOT NULL REFERENCES image_versions(id) ON DELETE CASCADE,
  sort_order INTEGER NOT NULL DEFAULT 0,
  added_at INTEGER NOT NULL,
  PRIMARY KEY(album_id, image_version_id)
);
```

Frontend `AlbumItem` can remain the transport shape. Backend commands convert rows into the existing tree.

## Saved Searches And Smart Albums

```sql
CREATE TABLE saved_searches (
  id TEXT PRIMARY KEY,
  library_id TEXT NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
  parent_id TEXT REFERENCES album_nodes(id) ON DELETE SET NULL,
  name TEXT NOT NULL,
  icon TEXT,
  query_json TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
```

`query_json` should model the existing `filterCriteria`, `searchCriteria`, and `sortCriteria`, plus future predicates like date ranges, camera, lens, focal length, dimensions, path prefix, edited status, and duplicate/similarity state.

## Full Text Search

Add FTS once typed metadata and tags are stable:

```sql
CREATE VIRTUAL TABLE image_search_fts USING fts5(
  path,
  file_name,
  title,
  caption,
  tags,
  camera,
  lens,
  content='',
  tokenize='unicode61'
);
```

Maintain it from catalog writes and background indexing.

## Future Cache Schema

These are intentionally not needed for phase one:

```sql
CREATE TABLE image_thumbnails (
  image_version_id INTEGER PRIMARY KEY REFERENCES image_versions(id) ON DELETE CASCADE,
  cache_key TEXT NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  format TEXT NOT NULL,
  bytes BLOB,
  file_path TEXT,
  generated_at INTEGER NOT NULL
);

CREATE TABLE image_fingerprints (
  image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
  phash TEXT,
  color_hash TEXT,
  embedding BLOB,
  model TEXT,
  updated_at INTEGER NOT NULL
);

CREATE TABLE people (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE face_regions (
  id INTEGER PRIMARY KEY,
  image_version_id INTEGER NOT NULL REFERENCES image_versions(id) ON DELETE CASCADE,
  person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,
  x REAL NOT NULL,
  y REAL NOT NULL,
  width REAL NOT NULL,
  height REAL NOT NULL,
  confidence REAL,
  embedding BLOB,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
```

## Rust Module Shape

Add a new backend module instead of expanding `file_management.rs`:

```text
src-tauri/src/library_db/
  mod.rs
  connection.rs
  migrations.rs
  scan.rs
  queries.rs
  albums.rs
  tags.rs
  metadata.rs
  sync.rs
```

Suggested dependency:

```toml
rusqlite = { version = "0.32", features = ["bundled", "chrono", "serde_json"] }
```

`rusqlite` is a pragmatic fit because RapidRAW's Tauri backend already uses mostly synchronous Rust commands with `spawn_blocking` for heavier work. `sqlx` is also reasonable, but it adds compile-time DB setup and async complexity that is not necessary for a local catalog.

## Tauri Commands

Keep current commands. Add catalog-specific commands:

```rust
create_library(name, path?) -> LibraryInfo
open_library(path) -> LibraryInfo
close_library()
get_active_library() -> Option<LibraryInfo>
add_library_root(path) -> CollectionRoot
remove_library_root(root_id)
scan_library_root(root_id, recursive, refresh_metadata)
list_catalog_folder(folder_id, mode, query) -> Vec<ImageFile>
list_catalog_root(root_id, relative_path, recursive, query) -> Vec<ImageFile>
search_catalog(query) -> Vec<ImageFile>
get_catalog_albums() -> Vec<AlbumItem>
save_catalog_album_tree(tree)
add_catalog_images_to_album(album_id, paths)
get_catalog_album_images(album_id) -> Vec<ImageFile>
set_catalog_rating_for_paths(paths, rating)
set_catalog_color_label_for_paths(paths, color)
add_catalog_tag_for_paths(paths, tag)
remove_catalog_tag_for_paths(paths, tag)
```

Phase one can internally keep old command names for albums and route by mode, but explicit catalog commands are cleaner long-term.

## Frontend State

Add a small library source concept:

```ts
type LibrarySource =
  | { type: 'filesystem' }
  | { type: 'catalog'; libraryId: string; dbPath: string };
```

`useLibraryStore` can add:

```ts
librarySource: LibrarySource;
catalogRoots: CatalogRoot[];
activeCatalogFolderId: number | null;
activeSavedSearchId: string | null;
```

Existing `imageList`, `imageRatings`, `albumTree`, and selection state remain unchanged.

## Library Setup UI

Catalog mode needs a real setup experience, not just a settings toggle. The UI should make the difference between "browse folders directly" and "manage a curated library" explicit.

### Entry Points

- First-run empty state: show "Open Folder" and "Create Library" as peer actions.
- Folder tree header: add a library selector/switcher for filesystem mode versus active catalog.
- Settings > Library: manage libraries, roots, database location, scan behavior, backups, and maintenance.
- Existing folder context menu: add "Add Folder to Library" when a catalog is active.

### Create Library Wizard

Use a short multi-step wizard:

1. Name the library.
2. Choose database location:
   - App data directory, recommended.
   - Custom local folder.
   - Portable database beside a photo collection.
3. Add initial photo folders.
4. Choose scan options:
   - Current folder only or recursive.
   - Import existing `.rrdata` sidecars.
   - Sync XMP metadata if enabled.
   - Generate metadata index now or in background.
5. Review estimated work and start indexing.

After creation, the app should open the catalog immediately and show indexing progress in the library header/status area.

### Open Existing Library Wizard

Opening a library should validate:

- The SQLite file exists and has a supported schema.
- Required roots are available.
- Roots that moved can be relinked before browsing.
- The DB is local or, if on a network filesystem, the user gets a clear warning.

If the schema is old, show a migration confirmation before modifying the DB. Always recommend backing up the DB first.

### Import Existing RapidRAW Workflow

Many users will already have folders with `.rrdata` sidecars and JSON albums. Provide a migration wizard:

1. Pick existing root folders or use currently pinned/root folders.
2. Scan for supported images.
3. Import sidecar ratings, edits, color labels, tags, and virtual copies.
4. Import `albums.json` if present.
5. Show unresolved album paths and missing files.
6. Finish with a catalog plus unchanged original files and sidecars.

This should be additive. The app must not delete sidecars or `albums.json` during migration.

### Library Management Screen

Settings > Library should include:

- Active library name, DB path, schema version, size, item count.
- Roots table with path, availability, last scan, image count, and relink/remove/rescan actions.
- Scan options and background indexing status.
- Maintenance actions:
  - Check missing files.
  - Relink moved roots.
  - Optimize database.
  - Backup database.
  - Export catalog manifest.
  - Import albums from JSON.
- A clear "Return to Folder Browsing" action that switches back to filesystem mode without closing or deleting the catalog.

### Relink Wizard

If a root is missing, users should be guided to:

1. Select the new folder location.
2. Confirm by matching known child paths and sample files.
3. Update `collection_roots.absolute_path`.
4. Rescan changed files.

Album membership should survive because albums reference `image_version_id`, not raw paths.

### UI Copy Principles

- Avoid presenting SQLite as a technical prerequisite. Use "Library" in UI copy and put database details in advanced/diagnostic areas.
- Keep "Open Folder" prominent for users who do not want a catalog.
- Explain irreversible or high-impact operations before running them, especially schema migration, cleanup, and root removal.
- Show background scan progress without blocking browsing once enough rows are available.

## Background Indexing

Scanning should be incremental:

1. Walk root folders and upsert folders/images by `(root_id, relative_path)`.
2. Mark missing files as `missing`, not immediately deleted.
3. Read sidecars for ratings, color labels, virtual copies, and edit state.
4. Read EXIF/XMP only when file size or modified time changed.
5. Emit progress events to the frontend.
6. Batch DB writes in transactions of 500 to 2000 images.

Use a single indexing task handle similar to current AI tagging cancellation.

## File Operations Sync

All existing file operations should update catalog rows when a catalog is active:

- Rename file: update `images.relative_path`, `file_name`, sidecar paths, album references remain stable through IDs.
- Move file: update `root_id`, `folder_id`, and `relative_path`.
- Delete/trash file: mark `status = 'trashed'` or `missing`, remove only after explicit cleanup.
- Rename folder: update affected folder paths and image relative paths in one transaction.
- Import/tether capture: insert rows immediately after filesystem write succeeds.

This is why album membership should reference `image_version_id` instead of raw paths.

## Migration From Current JSON Albums

When opening or creating the first catalog:

1. Read existing `albums.json`.
2. Resolve each path to a catalog image/version where possible.
3. Insert groups/albums into `album_nodes`.
4. Insert resolved images into `album_images`.
5. Keep unresolved paths in an import report so the user can relink roots.
6. Do not delete `albums.json` automatically.

## Compatibility Boundary

Catalog queries return the existing frontend `ImageFile` contract:

```ts
interface ImageFile {
  path: string;
  modified: number;
  is_edited: boolean;
  rating: number;
  tags?: string[] | null;
  exif?: Record<string, string> | null;
  is_virtual_copy: boolean;
  is_cloud_placeholder: boolean;
  is_raw: boolean;
  group_id?: string | null;
}
```

For virtual copies, return `absolute/path.ext?vc=<copy_id>` so editor code and sidecar lookup keep working.

## Implementation Phases

### Phase 1: Catalog Foundation

- Add `library_db` module with migrations and connection management.
- Add create/open/close library commands.
- Add root scanning and `list_catalog_folder`.
- Return `ImageFile` rows from SQLite.
- No UI redesign yet; add a simple "Create/Open Library" entry in settings or folder tree.

### Phase 2: Ratings, Tags, Albums

- Route rating/color/tag updates to DB when catalog mode is active, while still writing sidecars.
- Migrate JSON albums into DB.
- Replace catalog-mode album commands with DB-backed versions.
- Add smart album query storage.

### Phase 3: Fast Search And Metadata

- Add typed metadata extraction during indexing.
- Add filter/search queries using indexed columns.
- Add FTS table for captions, file names, tags, camera/lens, and paths.

### Phase 4: Maintenance And Relinking

- Add missing-file detection, root relink, cleanup, vacuum/optimize, and export library manifest.
- Add import from existing folder sidecars into a catalog.
- Add backup reminders or one-click database copy.

### Phase 5: Advanced digiKam-like Features

- Similarity/fingerprint table for duplicates and visual search.
- Face/person tables.
- Timeline/calendar views.
- Collections statistics and maintenance dashboard.

## First Pull Request Recommendation

The first PR should be backend-only:

- Add `rusqlite`.
- Add `library_db` module and migrations.
- Add commands: `create_library`, `open_library`, `add_library_root`, `scan_library_root`, `list_catalog_folder`.
- Add unit tests for migrations and path/root-relative resolution.

This proves the architecture without entangling it with current unresolved UI and file-management changes.
