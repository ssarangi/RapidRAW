use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{Datelike, NaiveDateTime};
use rusqlite::{Connection, OptionalExtension, params, params_from_iter, types::Value as SqlValue};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};
use uuid::Uuid;
use walkdir::WalkDir;

use crate::app_settings::load_settings;
use crate::app_state::CatalogScanControl;
use crate::file_management::{ImageFile, parse_virtual_path};
use crate::file_management::{assign_group_ids, read_file_mapped};
use crate::formats::{is_raw_file, is_supported_image_file};

const SCHEMA_VERSION: i64 = 7;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct LibraryInfo {
    pub id: String,
    pub name: String,
    pub db_path: String,
    pub schema_version: i64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CatalogRoot {
    pub id: i64,
    pub label: Option<String>,
    pub absolute_path: String,
    pub is_available: bool,
    pub last_scan_at: Option<i64>,
    pub image_count: i64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CatalogFolderNode {
    pub children: Vec<CatalogFolderNode>,
    pub is_dir: bool,
    pub name: String,
    pub path: String,
    pub image_count: i64,
    pub has_subdirs: bool,
    pub modified: Option<u64>,
    pub created: Option<u64>,
    pub root_id: i64,
    pub relative_path: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ScanResult {
    pub root_id: i64,
    pub scanned: usize,
    pub inserted_or_updated: usize,
    pub missing_marked: usize,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CatalogScanProgress {
    pub root_id: i64,
    pub root_path: String,
    pub current: usize,
    pub total: usize,
    pub current_path: Option<String>,
    pub camera: Option<String>,
    pub lens: Option<String>,
    pub year: Option<i64>,
    pub message: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct CatalogSearchQuery {
    pub root_id: Option<i64>,
    pub text: Option<String>,
    pub rating: Option<i64>,
    pub min_rating: Option<i64>,
    pub tags: Option<Vec<String>>,
    pub ai_tags: Option<Vec<String>>,
    pub tag_mode: Option<String>,
    pub year: Option<i64>,
    pub camera: Option<String>,
    pub lens: Option<String>,
    pub person: Option<String>,
    pub color: Option<String>,
    pub is_raw: Option<bool>,
    pub is_edited: Option<bool>,
    pub limit: Option<i64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CatalogFacetValue {
    pub value: String,
    pub count: i64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SmartCollection {
    pub id: i64,
    pub name: String,
    pub query_json: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CullSessionSummary {
    pub id: i64,
    pub root_id: Option<i64>,
    pub scope_path: String,
    pub state: String,
    pub total_count: i64,
    pub rejected_count: i64,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CullSessionDecision {
    pub id: i64,
    pub representative_path: String,
    pub proposed_status: String,
    pub final_status: String,
    pub quality_score: f64,
    pub reason: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CatalogMetrics {
    pub total_images: i64,
    pub edited_images: i64,
    pub rated_images: i64,
    pub missing_images: i64,
    pub ai_tags_suggested: i64,
    pub ai_tags_accepted: i64,
    pub ram_plus_analyzed: i64,
    pub ram_plus_pending: i64,
    pub ram_plus_failed: i64,
    pub cull_sessions: i64,
    pub cull_overrides: i64,
    pub years: Vec<CatalogFacetValue>,
    pub cameras: Vec<CatalogFacetValue>,
    pub lenses: Vec<CatalogFacetValue>,
    pub people: Vec<CatalogFacetValue>,
    pub tags: Vec<CatalogFacetValue>,
    pub ai_tags: Vec<CatalogFacetValue>,
    pub ratings: Vec<CatalogFacetValue>,
}

/// A human-maintained identity. Detection and clustering may propose links to
/// a person, but the person record is deliberately independent from any
/// specific model or embedding.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CatalogPerson {
    pub id: i64,
    pub display_name: String,
    pub state: String,
    pub face_count: i64,
}

/// A normalized face observation from one source image. Bounding-box values
/// are stored as fractions of the original image dimensions so they remain
/// valid when previews are regenerated at different sizes.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CatalogFace {
    pub id: i64,
    pub image_id: i64,
    pub person_id: Option<i64>,
    pub model_pack_id: String,
    pub confidence: f64,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub review_state: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CatalogFaceReviewItem {
    pub face: CatalogFace,
    pub image_path: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CatalogFaceCluster {
    pub id: i64,
    pub face_count: i64,
    pub representative_image_path: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CatalogAiTagReviewItem {
    pub id: i64,
    pub image_path: String,
    pub tag: String,
    pub confidence: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BackgroundJob {
    pub id: String,
    pub kind: String,
    pub state: String,
    pub root_id: Option<i64>,
    pub current: i64,
    pub total: i64,
    pub current_item: Option<String>,
    pub message: String,
    pub error: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BackgroundJobEvent {
    pub id: i64,
    pub job_id: String,
    pub state: String,
    pub message: String,
    pub current: i64,
    pub total: i64,
    pub created_at: i64,
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn can_pause_job(state: &str) -> bool {
    matches!(state, "queued" | "running")
}

fn can_resume_job(state: &str) -> bool {
    state == "paused"
}

pub(crate) fn active_library_path(
    state: &tauri::State<'_, crate::AppState>,
) -> Result<PathBuf, String> {
    state
        .active_library_path
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| "No active RapidRAW library is open".to_string())
}

fn record_job_event(
    conn: &Connection,
    job_id: &str,
    state: &str,
    message: &str,
    current: i64,
    total: i64,
) -> Result<(), String> {
    conn.execute(
        "INSERT INTO background_job_events(job_id, state, message, current, total, created_at)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
        params![job_id, state, message, current, total, now_secs()],
    )
    .map_err(|error| error.to_string())?;
    Ok(())
}

pub(crate) fn update_job(
    db_path: &Path,
    job_id: &str,
    state: &str,
    message: &str,
    current: i64,
    total: i64,
    current_item: Option<&str>,
    error: Option<&str>,
) -> Result<(), String> {
    let conn = open_connection(db_path)?;
    let now = now_secs();
    conn.execute(
        "UPDATE background_jobs
         SET state = ?2, message = ?3, current = ?4, total = ?5, current_item = ?6, error = ?7,
             updated_at = ?8, completed_at = CASE WHEN ?2 IN ('completed', 'cancelled', 'failed') THEN ?8 ELSE NULL END
         WHERE id = ?1",
        params![job_id, state, message, current, total, current_item, error, now],
    )
    .map_err(|error| error.to_string())?;
    record_job_event(&conn, job_id, state, message, current, total)
}

fn create_catalog_scan_job(
    conn: &Connection,
    root_id: i64,
    recursive: bool,
) -> Result<String, String> {
    let id = Uuid::new_v4().to_string();
    let now = now_secs();
    conn.execute(
        "INSERT INTO background_jobs(id, kind, state, root_id, payload_json, message, created_at, updated_at)
         VALUES(?1, 'catalog_scan', 'queued', ?2, ?3, 'Queued catalog scan', ?4, ?4)",
        params![id, root_id, serde_json::json!({ "recursive": recursive }).to_string(), now],
    )
    .map_err(|error| error.to_string())?;
    record_job_event(conn, &id, "queued", "Queued catalog scan", 0, 0)?;
    Ok(id)
}

pub(crate) fn create_background_job(
    db_path: &Path,
    kind: &str,
    payload: serde_json::Value,
) -> Result<String, String> {
    let conn = open_connection(db_path)?;
    let id = Uuid::new_v4().to_string();
    let now = now_secs();
    conn.execute(
        "INSERT INTO background_jobs(id, kind, state, payload_json, message, created_at, updated_at)
         VALUES(?1, ?2, 'queued', ?3, 'Queued background job', ?4, ?4)",
        params![id, kind, payload.to_string(), now],
    )
    .map_err(|error| error.to_string())?;
    record_job_event(&conn, &id, "queued", "Queued background job", 0, 0)?;
    Ok(id)
}

pub(crate) fn list_ai_tag_candidates(db_path: &Path) -> Result<Vec<(i64, String, i64)>, String> {
    list_ai_tag_candidates_for_model(db_path, "clip", "rapidraw-clip-v1")
}

pub(crate) fn list_ai_tag_candidates_for_model(
    db_path: &Path,
    model_id: &str,
    model_revision: &str,
) -> Result<Vec<(i64, String, i64)>, String> {
    let conn = open_connection(db_path)?;
    let mut statement = conn.prepare("SELECT i.id, r.absolute_path || '/' || i.relative_path, i.modified_at FROM images i JOIN collection_roots r ON r.id = i.root_id WHERE i.status = 'present' AND NOT EXISTS (SELECT 1 FROM image_ai_analysis_state s WHERE s.image_id = i.id AND s.analysis_kind = 'tagging' AND s.model_id = ?1 AND s.model_revision = ?2 AND s.image_modified_at = i.modified_at AND s.state = 'completed') ORDER BY i.id").map_err(|error| error.to_string())?;
    statement
        .query_map(params![model_id, model_revision], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

pub(crate) fn mark_ai_tag_analysis_state(
    db_path: &Path,
    image_id: i64,
    image_modified_at: i64,
    state: &str,
    error_message: Option<&str>,
) -> Result<(), String> {
    mark_ai_tag_analysis_state_for_model(db_path, image_id, image_modified_at, "clip", "rapidraw-clip-v1", state, error_message)
}

pub(crate) fn mark_ai_tag_analysis_state_for_model(
    db_path: &Path,
    image_id: i64,
    image_modified_at: i64,
    model_id: &str,
    model_revision: &str,
    state: &str,
    error_message: Option<&str>,
) -> Result<(), String> {
    let conn = open_connection(db_path)?;
    conn.execute("INSERT INTO image_ai_analysis_state(image_id, analysis_kind, model_id, model_revision, image_modified_at, state, error_message, processed_at, updated_at) VALUES(?1, 'tagging', ?2, ?3, ?4, ?5, ?6, CASE WHEN ?5 IN ('completed', 'failed') THEN strftime('%s','now') ELSE NULL END, strftime('%s','now')) ON CONFLICT(image_id, analysis_kind, model_id, model_revision, image_modified_at) DO UPDATE SET state = excluded.state, error_message = excluded.error_message, processed_at = excluded.processed_at, updated_at = excluded.updated_at", params![image_id, model_id, model_revision, image_modified_at, state, error_message]).map_err(|error| error.to_string())?;
    Ok(())
}

pub(crate) fn replace_clip_ai_tags(
    db_path: &Path,
    image_id: i64,
    tags: &[crate::tagging::ScoredTag],
) -> Result<(), String> {
    replace_ai_tags_for_model(db_path, image_id, "clip", "rapidraw-clip-v1", tags)
}

pub(crate) fn replace_ai_tags_for_model(
    db_path: &Path,
    image_id: i64,
    model_id: &str,
    model_revision: &str,
    tags: &[crate::tagging::ScoredTag],
) -> Result<(), String> {
    let mut conn = open_connection(db_path)?;
    let tx = conn.transaction().map_err(|error| error.to_string())?;
    tx.execute("DELETE FROM image_ai_tags WHERE image_id = ?1 AND model_id = ?2 AND model_revision = ?3", params![image_id, model_id, model_revision]).map_err(|error| error.to_string())?;
    for tag in tags {
        tx.execute(
            "INSERT OR IGNORE INTO tags(name, kind) VALUES(?1, 'ai')",
            [&tag.name],
        )
        .map_err(|error| error.to_string())?;
        let tag_id: i64 = tx
            .query_row(
                "SELECT id FROM tags WHERE name = ?1 AND kind = 'ai'",
                [&tag.name],
                |row| row.get(0),
            )
            .map_err(|error| error.to_string())?;
        tx.execute("INSERT INTO image_ai_tags(image_id, tag_id, model_id, model_revision, confidence, review_state, source, created_at, updated_at) VALUES(?1, ?2, ?3, ?4, ?5, 'suggested', 'local', strftime('%s','now'), strftime('%s','now'))", params![image_id, tag_id, model_id, model_revision, tag.confidence]).map_err(|error| error.to_string())?;
    }
    tx.commit().map_err(|error| error.to_string())
}

fn open_connection(path: &Path) -> Result<Connection, String> {
    let conn = Connection::open(path).map_err(|e| e.to_string())?;
    conn.busy_timeout(Duration::from_secs(30))
        .map_err(|e| e.to_string())?;
    conn.pragma_update(None, "foreign_keys", "ON")
        .map_err(|e| e.to_string())?;
    conn.pragma_update(None, "journal_mode", "WAL")
        .map_err(|e| e.to_string())?;
    conn.pragma_update(None, "synchronous", "NORMAL")
        .map_err(|e| e.to_string())?;
    Ok(conn)
}

fn migrate(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version INTEGER PRIMARY KEY,
          applied_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS libraries (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          app_version TEXT
        );

        CREATE TABLE IF NOT EXISTS collection_roots (
          id INTEGER PRIMARY KEY,
          library_id TEXT NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
          label TEXT,
          absolute_path TEXT NOT NULL,
          canonical_path TEXT,
          is_available INTEGER NOT NULL DEFAULT 1,
          last_scan_at INTEGER,
          UNIQUE(library_id, absolute_path)
        );

        CREATE TABLE IF NOT EXISTS folders (
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

        CREATE TABLE IF NOT EXISTS images (
          id INTEGER PRIMARY KEY,
          root_id INTEGER NOT NULL REFERENCES collection_roots(id) ON DELETE CASCADE,
          folder_id INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
          file_name TEXT NOT NULL,
          relative_path TEXT NOT NULL,
          extension TEXT,
          file_size INTEGER,
          modified_at INTEGER NOT NULL,
          status TEXT NOT NULL DEFAULT 'present',
          is_raw INTEGER NOT NULL DEFAULT 0,
          is_cloud_placeholder INTEGER NOT NULL DEFAULT 0,
          imported_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          UNIQUE(root_id, relative_path)
        );

        CREATE INDEX IF NOT EXISTS idx_images_folder ON images(folder_id, file_name);
        CREATE INDEX IF NOT EXISTS idx_images_modified ON images(modified_at);
        CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);

        CREATE TABLE IF NOT EXISTS image_versions (
          id INTEGER PRIMARY KEY,
          image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
          copy_id TEXT NOT NULL DEFAULT '',
          display_name TEXT,
          sidecar_path TEXT,
          rating INTEGER NOT NULL DEFAULT 0,
          color_label TEXT,
          is_edited INTEGER NOT NULL DEFAULT 0,
          tags_json TEXT,
          sidecar_modified_at INTEGER,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          UNIQUE(image_id, copy_id)
        );

        CREATE INDEX IF NOT EXISTS idx_versions_rating ON image_versions(rating);
        CREATE INDEX IF NOT EXISTS idx_versions_color ON image_versions(color_label);
        CREATE INDEX IF NOT EXISTS idx_versions_edited ON image_versions(is_edited);

        CREATE TABLE IF NOT EXISTS image_metadata (
          image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
          date_taken INTEGER,
          year INTEGER,
          width INTEGER,
          height INTEGER,
          camera_make TEXT,
          camera_model TEXT,
          lens_model TEXT,
          focal_length REAL,
          aperture REAL,
          shutter TEXT,
          iso INTEGER,
          title TEXT,
          caption TEXT,
          exif_json TEXT,
          updated_at INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_metadata_year ON image_metadata(year);
        CREATE INDEX IF NOT EXISTS idx_metadata_camera ON image_metadata(camera_make, camera_model);
        CREATE INDEX IF NOT EXISTS idx_metadata_lens ON image_metadata(lens_model);
        CREATE INDEX IF NOT EXISTS idx_metadata_iso ON image_metadata(iso);

        CREATE TABLE IF NOT EXISTS tags (
          id INTEGER PRIMARY KEY,
          name TEXT NOT NULL,
          kind TEXT NOT NULL DEFAULT 'user',
          UNIQUE(name, kind)
        );

        CREATE TABLE IF NOT EXISTS image_tags (
          image_version_id INTEGER NOT NULL REFERENCES image_versions(id) ON DELETE CASCADE,
          tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
          source TEXT NOT NULL DEFAULT 'user',
          PRIMARY KEY(image_version_id, tag_id, source)
        );

        CREATE INDEX IF NOT EXISTS idx_image_tags_tag ON image_tags(tag_id, image_version_id);

        CREATE TABLE IF NOT EXISTS people (
          id INTEGER PRIMARY KEY,
          display_name TEXT NOT NULL COLLATE NOCASE,
          state TEXT NOT NULL DEFAULT 'active' CHECK(state IN ('active', 'ignored', 'merged')),
          merged_into_person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          UNIQUE(display_name)
        );

        CREATE TABLE IF NOT EXISTS face_embeddings (
          id INTEGER PRIMARY KEY,
          model_pack_id TEXT NOT NULL,
          dimensions INTEGER NOT NULL CHECK(dimensions > 0),
          vector BLOB NOT NULL,
          norm REAL,
          created_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS faces (
          id INTEGER PRIMARY KEY,
          image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
          person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,
          embedding_id INTEGER REFERENCES face_embeddings(id) ON DELETE SET NULL,
          model_pack_id TEXT NOT NULL,
          detector_confidence REAL NOT NULL,
          bbox_x REAL NOT NULL CHECK(bbox_x >= 0 AND bbox_x <= 1),
          bbox_y REAL NOT NULL CHECK(bbox_y >= 0 AND bbox_y <= 1),
          bbox_width REAL NOT NULL CHECK(bbox_width > 0 AND bbox_width <= 1),
          bbox_height REAL NOT NULL CHECK(bbox_height > 0 AND bbox_height <= 1),
          landmarks_json TEXT,
          review_state TEXT NOT NULL DEFAULT 'unreviewed' CHECK(review_state IN ('unreviewed', 'confirmed', 'rejected')),
          source TEXT NOT NULL DEFAULT 'local' CHECK(source IN ('local', 'imported')),
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_faces_image ON faces(image_id);
        CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id, review_state);
        CREATE INDEX IF NOT EXISTS idx_faces_review ON faces(review_state, model_pack_id);

        CREATE TABLE IF NOT EXISTS face_clusters (
          id INTEGER PRIMARY KEY,
          model_pack_id TEXT NOT NULL,
          state TEXT NOT NULL DEFAULT 'unreviewed' CHECK(state IN ('unreviewed', 'accepted', 'rejected')),
          representative_face_id INTEGER REFERENCES faces(id) ON DELETE SET NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS face_cluster_members (
          cluster_id INTEGER NOT NULL REFERENCES face_clusters(id) ON DELETE CASCADE,
          face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
          similarity REAL,
          PRIMARY KEY(cluster_id, face_id)
        );

        CREATE INDEX IF NOT EXISTS idx_face_cluster_members_face ON face_cluster_members(face_id);

        CREATE TABLE IF NOT EXISTS face_scan_state (
          image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
          model_pack_id TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'complete', 'failed', 'skipped')),
          model_revision TEXT,
          error_message TEXT,
          processed_at INTEGER,
          updated_at INTEGER NOT NULL,
          PRIMARY KEY(image_id, model_pack_id)
        );

        CREATE INDEX IF NOT EXISTS idx_face_scan_state_status ON face_scan_state(status, model_pack_id);

        CREATE TABLE IF NOT EXISTS background_jobs (
          id TEXT PRIMARY KEY,
          kind TEXT NOT NULL,
          state TEXT NOT NULL CHECK(state IN ('queued', 'running', 'paused', 'cancelling', 'cancelled', 'completed', 'failed')),
          root_id INTEGER REFERENCES collection_roots(id) ON DELETE SET NULL,
          payload_json TEXT NOT NULL DEFAULT '{}',
          current INTEGER NOT NULL DEFAULT 0,
          total INTEGER NOT NULL DEFAULT 0,
          current_item TEXT,
          message TEXT NOT NULL DEFAULT '',
          error TEXT,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          completed_at INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_background_jobs_state ON background_jobs(state, updated_at DESC);

        CREATE TABLE IF NOT EXISTS background_job_events (
          id INTEGER PRIMARY KEY,
          job_id TEXT NOT NULL REFERENCES background_jobs(id) ON DELETE CASCADE,
          state TEXT NOT NULL,
          message TEXT NOT NULL,
          current INTEGER NOT NULL DEFAULT 0,
          total INTEGER NOT NULL DEFAULT 0,
          created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_background_job_events_job ON background_job_events(job_id, id DESC);

        CREATE TABLE IF NOT EXISTS image_ai_tags (
          image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
          tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
          model_id TEXT NOT NULL,
          model_revision TEXT NOT NULL,
          confidence REAL NOT NULL,
          review_state TEXT NOT NULL DEFAULT 'suggested' CHECK(review_state IN ('suggested', 'accepted', 'rejected')),
          source TEXT NOT NULL DEFAULT 'local',
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          PRIMARY KEY(image_id, tag_id, model_id, model_revision)
        );
        CREATE INDEX IF NOT EXISTS idx_image_ai_tags_tag ON image_ai_tags(tag_id, review_state, confidence DESC);

        CREATE TABLE IF NOT EXISTS smart_collections (
          id INTEGER PRIMARY KEY,
          name TEXT NOT NULL COLLATE NOCASE UNIQUE,
          query_json TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cull_sessions (
          id INTEGER PRIMARY KEY,
          root_id INTEGER REFERENCES collection_roots(id) ON DELETE SET NULL,
          scope_path TEXT NOT NULL,
          mode TEXT NOT NULL DEFAULT 'assisted',
          state TEXT NOT NULL CHECK(state IN ('planned', 'applied', 'cancelled', 'failed')),
          settings_json TEXT NOT NULL,
          feature_set_version TEXT NOT NULL,
          total_count INTEGER NOT NULL DEFAULT 0,
          rejected_count INTEGER NOT NULL DEFAULT 0,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cull_sessions_root ON cull_sessions(root_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS cull_decisions (
          id INTEGER PRIMARY KEY,
          session_id INTEGER NOT NULL REFERENCES cull_sessions(id) ON DELETE CASCADE,
          image_id INTEGER REFERENCES images(id) ON DELETE SET NULL,
          representative_path TEXT NOT NULL,
          proposed_status TEXT NOT NULL CHECK(proposed_status IN ('keep', 'reject')),
          final_status TEXT NOT NULL DEFAULT 'pending' CHECK(final_status IN ('pending', 'keep', 'reject', 'skipped')),
          quality_score REAL NOT NULL,
          reason TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          UNIQUE(session_id, representative_path)
        );
        CREATE INDEX IF NOT EXISTS idx_cull_decisions_session ON cull_decisions(session_id, proposed_status);

        CREATE TABLE IF NOT EXISTS cull_decision_events (
          id INTEGER PRIMARY KEY,
          decision_id INTEGER NOT NULL REFERENCES cull_decisions(id) ON DELETE CASCADE,
          previous_status TEXT NOT NULL CHECK(previous_status IN ('keep', 'reject')),
          next_status TEXT NOT NULL CHECK(next_status IN ('keep', 'reject')),
          source TEXT NOT NULL DEFAULT 'user',
          created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cull_decision_events_decision ON cull_decision_events(decision_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS image_ai_analysis_state (
          image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
          analysis_kind TEXT NOT NULL,
          model_id TEXT NOT NULL,
          model_revision TEXT NOT NULL,
          image_modified_at INTEGER NOT NULL,
          state TEXT NOT NULL CHECK(state IN ('pending', 'processing', 'completed', 'failed', 'skipped')),
          error_message TEXT,
          processed_at INTEGER,
          updated_at INTEGER NOT NULL,
          PRIMARY KEY(image_id, analysis_kind, model_id, model_revision, image_modified_at)
        );
        CREATE INDEX IF NOT EXISTS idx_image_ai_analysis_pending ON image_ai_analysis_state(analysis_kind, model_id, state);
        ",
    )
    .map_err(|e| e.to_string())?;

    conn.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES(?1, ?2)",
        params![SCHEMA_VERSION, now_secs()],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert_test_image(connection: &Connection) {
        connection
            .execute(
                "INSERT INTO libraries(id, name, created_at, updated_at) VALUES('library', 'Test', 0, 0)",
                [],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO collection_roots(id, library_id, absolute_path) VALUES(1, 'library', '/photos')",
                [],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO folders(id, root_id, relative_path, name) VALUES(1, 1, '', 'photos')",
                [],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO images(id, root_id, folder_id, file_name, relative_path, modified_at, imported_at, updated_at) VALUES(1, 1, 1, 'photo.jpg', 'photo.jpg', 0, 0, 0)",
                [],
            )
            .unwrap();
    }

    #[test]
    fn migration_creates_face_storage_and_records_current_version() {
        let connection = Connection::open_in_memory().unwrap();
        migrate(&connection).unwrap();

        for table in [
            "people",
            "face_embeddings",
            "faces",
            "face_scan_state",
            "face_clusters",
            "face_cluster_members",
            "smart_collections",
            "cull_sessions",
            "cull_decisions",
            "cull_decision_events",
        ] {
            let exists: i64 = connection
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
                    [table],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(exists, 1, "missing {table} table");
        }

        let version: i64 = connection
            .query_row("SELECT MAX(version) FROM schema_migrations", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
    }

    #[test]
    fn face_schema_enforces_normalized_bounding_boxes() {
        let connection = Connection::open_in_memory().unwrap();
        migrate(&connection).unwrap();
        insert_test_image(&connection);

        let error = connection
            .execute(
                "INSERT INTO faces(image_id, model_pack_id, detector_confidence, bbox_x, bbox_y, bbox_width, bbox_height, created_at, updated_at) VALUES(1, 'test', 0.9, 1.1, 0.1, 0.2, 0.2, 0, 0)",
                [],
            )
            .unwrap_err();
        assert!(error.to_string().contains("CHECK constraint failed"));
    }

    #[test]
    fn confirmed_face_links_are_available_for_person_facets() {
        let connection = Connection::open_in_memory().unwrap();
        migrate(&connection).unwrap();
        insert_test_image(&connection);
        connection
            .execute(
                "INSERT INTO people(id, display_name, created_at, updated_at) VALUES(1, 'Ada', 0, 0)",
                [],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO faces(image_id, person_id, model_pack_id, detector_confidence, bbox_x, bbox_y, bbox_width, bbox_height, review_state, created_at, updated_at) VALUES(1, 1, 'opencv-yunet-sface', 0.9, 0.1, 0.1, 0.2, 0.2, 'confirmed', 0, 0)",
                [],
            )
            .unwrap();

        let face_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM faces f JOIN people p ON p.id = f.person_id WHERE p.display_name = 'Ada' AND f.review_state = 'confirmed'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(face_count, 1);
    }

    #[test]
    fn recovery_marks_incomplete_jobs_failed() {
        let connection = Connection::open_in_memory().unwrap();
        migrate(&connection).unwrap();
        connection.execute("INSERT INTO background_jobs(id, kind, state, payload_json, message, created_at, updated_at) VALUES('job', 'catalog_scan', 'running', '{}', 'Scanning', 0, 0)", []).unwrap();
        recover_interrupted_jobs(&connection).unwrap();
        let state: String = connection
            .query_row(
                "SELECT state FROM background_jobs WHERE id = 'job'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(state, "failed");
    }

    #[test]
    fn job_pause_transitions_only_allow_active_jobs() {
        assert!(can_pause_job("queued"));
        assert!(can_pause_job("running"));
        assert!(!can_pause_job("paused"));
        assert!(!can_pause_job("completed"));
        assert!(can_resume_job("paused"));
        assert!(!can_resume_job("running"));
        assert!(!can_resume_job("cancelled"));
    }

    #[test]
    fn cull_sessions_persist_catalog_decisions_and_final_state() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("catalog.db");
        let connection = Connection::open(&path).unwrap();
        migrate(&connection).unwrap();
        insert_test_image(&connection);
        drop(connection);

        let session_id = record_cull_session(
            &path,
            "/photos",
            "{}",
            &[
                ("/photos/photo.jpg".to_string(), true, "unique".to_string(), 0.8),
                ("/photos/missing.jpg".to_string(), false, "blurry".to_string(), 0.2),
            ],
        )
        .unwrap()
        .unwrap();
        mark_cull_session_applied(&path, session_id).unwrap();

        let connection = Connection::open(&path).unwrap();
        let (state, decisions): (String, i64) = connection
            .query_row(
                "SELECT s.state, COUNT(d.id) FROM cull_sessions s JOIN cull_decisions d ON d.session_id = s.id WHERE s.id = ?1 GROUP BY s.id",
                [session_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        let rejected: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM cull_decisions WHERE session_id = ?1 AND final_status = 'reject'",
                [session_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(state, "applied");
        assert_eq!(decisions, 2);
        assert_eq!(rejected, 1);
    }

    #[test]
    fn ai_tag_storage_keeps_rejected_suggestions_out_of_searchable_results() {
        let connection = Connection::open_in_memory().unwrap();
        migrate(&connection).unwrap();
        insert_test_image(&connection);
        connection
            .execute(
                "INSERT INTO tags(id, name, kind) VALUES(1, 'bird', 'ai')",
                [],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO image_ai_tags(image_id, tag_id, model_id, model_revision, confidence, review_state, source, created_at, updated_at) VALUES(1, 1, 'clip', 'test', 0.9, 'suggested', 'local', 0, 0)",
                [],
            )
            .unwrap();

        let searchable: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM image_ai_tags WHERE image_id = 1 AND review_state <> 'rejected'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(searchable, 1);

        connection
            .execute(
                "UPDATE image_ai_tags SET review_state = 'rejected' WHERE image_id = 1",
                [],
            )
            .unwrap();
        let searchable: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM image_ai_tags WHERE image_id = 1 AND review_state <> 'rejected'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(searchable, 0);
    }
}

fn library_info(conn: &Connection, db_path: &Path) -> Result<LibraryInfo, String> {
    let (id, name): (String, String) = conn
        .query_row("SELECT id, name FROM libraries LIMIT 1", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .map_err(|e| e.to_string())?;
    Ok(LibraryInfo {
        id,
        name,
        db_path: db_path.to_string_lossy().into_owned(),
        schema_version: SCHEMA_VERSION,
    })
}

fn recover_interrupted_jobs(conn: &Connection) -> Result<(), String> {
    let now = now_secs();
    conn.execute(
        "UPDATE background_jobs SET state = 'failed', message = 'Interrupted by application restart', error = 'The application stopped before this job completed', updated_at = ?1, completed_at = ?1 WHERE state IN ('running', 'paused', 'cancelling')",
        [now],
    )
    .map_err(|error| error.to_string())?;
    Ok(())
}

fn default_library_dir(app_handle: &AppHandle, library_id: &str) -> Result<PathBuf, String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?;
    Ok(data_dir.join("libraries").join(library_id))
}

#[tauri::command]
pub fn create_library(
    name: String,
    directory: Option<String>,
    app_handle: AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<LibraryInfo, String> {
    let trimmed_name = name.trim();
    if trimmed_name.is_empty() {
        return Err("Library name cannot be empty".to_string());
    }

    let library_id = Uuid::new_v4().to_string();
    let dir = match directory {
        Some(path) if !path.trim().is_empty() => PathBuf::from(path),
        _ => default_library_dir(&app_handle, &library_id)?,
    };
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let db_path = dir.join("rapidraw-library.db");
    let conn = open_connection(&db_path)?;
    migrate(&conn)?;
    let now = now_secs();
    conn.execute(
        "INSERT INTO libraries(id, name, created_at, updated_at) VALUES(?1, ?2, ?3, ?3)",
        params![library_id, trimmed_name, now],
    )
    .map_err(|e| e.to_string())?;

    *state.active_library_path.lock().unwrap() = Some(db_path.clone());
    library_info(&conn, &db_path)
}

#[tauri::command]
pub fn open_library(
    path: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<LibraryInfo, String> {
    let db_path = PathBuf::from(path);
    if !db_path.exists() {
        return Err("Library database does not exist".to_string());
    }
    let conn = open_connection(&db_path)?;
    migrate(&conn)?;
    recover_interrupted_jobs(&conn)?;
    let info = library_info(&conn, &db_path)?;
    *state.active_library_path.lock().unwrap() = Some(db_path);
    Ok(info)
}

#[tauri::command]
pub fn close_library(state: tauri::State<'_, crate::AppState>) -> Result<(), String> {
    *state.active_library_path.lock().unwrap() = None;
    Ok(())
}

#[tauri::command]
pub fn delete_library(state: tauri::State<'_, crate::AppState>) -> Result<(), String> {
    let db_path = active_library_path(&state)?;
    *state.active_library_path.lock().unwrap() = None;
    fs::remove_file(&db_path).map_err(|e| format!("Failed to delete library database: {e}"))?;
    Ok(())
}

#[tauri::command]
pub fn get_active_library(
    state: tauri::State<'_, crate::AppState>,
) -> Result<Option<LibraryInfo>, String> {
    let Some(path) = state.active_library_path.lock().unwrap().clone() else {
        return Ok(None);
    };
    if !path.exists() {
        *state.active_library_path.lock().unwrap() = None;
        return Ok(None);
    }
    let conn = open_connection(&path)?;
    migrate(&conn)?;
    recover_interrupted_jobs(&conn)?;
    library_info(&conn, &path).map(Some)
}

fn current_library_id(conn: &Connection) -> Result<String, String> {
    conn.query_row("SELECT id FROM libraries LIMIT 1", [], |row| row.get(0))
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn add_library_root(
    path: String,
    label: Option<String>,
    state: tauri::State<'_, crate::AppState>,
) -> Result<CatalogRoot, String> {
    let db_path = active_library_path(&state)?;
    let conn = open_connection(&db_path)?;
    let library_id = current_library_id(&conn)?;
    let root_path = PathBuf::from(&path);
    let canonical_path = root_path
        .canonicalize()
        .ok()
        .map(|p| p.to_string_lossy().into_owned());
    let is_available = root_path.exists();
    conn.execute(
        "INSERT OR IGNORE INTO collection_roots(library_id, label, absolute_path, canonical_path, is_available)
         VALUES(?1, ?2, ?3, ?4, ?5)",
        params![
            library_id,
            label,
            path,
            canonical_path,
            if is_available { 1 } else { 0 }
        ],
    )
    .map_err(|e| e.to_string())?;

    let id = conn
        .query_row(
            "SELECT id FROM collection_roots WHERE library_id = ?1 AND absolute_path = ?2",
            params![library_id, path],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|e| e.to_string())?;

    get_root(&conn, id)
}

fn get_root(conn: &Connection, id: i64) -> Result<CatalogRoot, String> {
    conn.query_row(
        "
        SELECT r.id, r.label, r.absolute_path, r.is_available, r.last_scan_at, COUNT(i.id)
        FROM collection_roots r
        LEFT JOIN images i ON i.root_id = r.id AND i.status = 'present'
        WHERE r.id = ?1
        GROUP BY r.id
        ",
        params![id],
        |row| {
            Ok(CatalogRoot {
                id: row.get(0)?,
                label: row.get(1)?,
                absolute_path: row.get(2)?,
                is_available: row.get::<_, i64>(3)? != 0,
                last_scan_at: row.get(4)?,
                image_count: row.get(5)?,
            })
        },
    )
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn list_library_roots(
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<CatalogRoot>, String> {
    let db_path = active_library_path(&state)?;
    let conn = open_connection(&db_path)?;
    let mut stmt = conn
        .prepare(
            "
            SELECT r.id, r.label, r.absolute_path, r.is_available, r.last_scan_at, COUNT(i.id)
            FROM collection_roots r
            LEFT JOIN images i ON i.root_id = r.id AND i.status = 'present'
            GROUP BY r.id
            ORDER BY COALESCE(r.label, r.absolute_path) COLLATE NOCASE
            ",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            Ok(CatalogRoot {
                id: row.get(0)?,
                label: row.get(1)?,
                absolute_path: row.get(2)?,
                is_available: row.get::<_, i64>(3)? != 0,
                last_scan_at: row.get(4)?,
                image_count: row.get(5)?,
            })
        })
        .map_err(|e| e.to_string())?;

    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
}

fn catalog_folder_virtual_path(root_id: i64, relative_path: &str) -> String {
    if relative_path == "." {
        format!("LibraryFolder:{root_id}:.")
    } else {
        format!("LibraryFolder:{root_id}:{relative_path}")
    }
}

#[tauri::command]
pub fn list_catalog_folder_tree(
    root_id: i64,
    state: tauri::State<'_, crate::AppState>,
) -> Result<CatalogFolderNode, String> {
    let db_path = active_library_path(&state)?;
    let conn = open_connection(&db_path)?;
    let root = get_root(&conn, root_id)?;

    let rows = {
        let mut stmt = conn
            .prepare(
                "
                SELECT relative_path, name, image_count, modified_at
                FROM folders
                WHERE root_id = ?1
                ORDER BY relative_path COLLATE NOCASE
                ",
            )
            .map_err(|e| e.to_string())?;
        stmt.query_map(params![root_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, Option<i64>>(3)?,
            ))
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?
    };

    let mut child_paths: HashMap<String, Vec<String>> = HashMap::new();
    let mut folder_info: HashMap<String, (String, i64, Option<i64>)> = HashMap::new();
    for (relative_path, name, image_count, modified_at) in rows {
        let parent = if relative_path == "." {
            None
        } else {
            Path::new(&relative_path)
                .parent()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .filter(|p| !p.is_empty())
                .or_else(|| Some(".".to_string()))
        };
        if let Some(parent) = parent {
            child_paths
                .entry(parent)
                .or_default()
                .push(relative_path.clone());
        }
        folder_info.insert(relative_path, (name, image_count, modified_at));
    }

    folder_info.entry(".".to_string()).or_insert_with(|| {
        (
            root.label
                .clone()
                .or_else(|| {
                    Path::new(&root.absolute_path)
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                })
                .unwrap_or_else(|| root.absolute_path.clone()),
            root.image_count,
            None,
        )
    });

    fn build_node(
        root_id: i64,
        relative_path: &str,
        folder_info: &HashMap<String, (String, i64, Option<i64>)>,
        child_paths: &HashMap<String, Vec<String>>,
    ) -> CatalogFolderNode {
        let (name, image_count, modified_at) = folder_info
            .get(relative_path)
            .cloned()
            .unwrap_or_else(|| (relative_path.to_string(), 0, None));
        let mut children = child_paths
            .get(relative_path)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|child| build_node(root_id, &child, folder_info, child_paths))
            .collect::<Vec<_>>();
        children.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        let has_subdirs = !children.is_empty();
        CatalogFolderNode {
            children,
            is_dir: true,
            name,
            path: catalog_folder_virtual_path(root_id, relative_path),
            image_count,
            has_subdirs,
            modified: modified_at.map(|value| value as u64),
            created: None,
            root_id,
            relative_path: relative_path.to_string(),
        }
    }

    Ok(build_node(root_id, ".", &folder_info, &child_paths))
}

fn unix_modified(path: &Path) -> u64 {
    path.metadata()
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn sidecar_metadata(
    image_path: &Path,
    sidecar_path: &Path,
    app_handle: &AppHandle,
) -> (bool, u8, Option<Vec<String>>) {
    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    let metadata = crate::exif_processing::load_sidecar(sidecar_path);
    let is_raw = is_raw_file(image_path);
    let tm_override = crate::image_processing::resolve_tonemapper_override(&settings, is_raw);
    let is_edited =
        crate::image_processing::is_image_edited(&metadata.adjustments, is_raw, tm_override);
    (is_edited, metadata.rating, metadata.tags)
}

fn first_exif<'a>(exif: &'a HashMap<String, String>, keys: &[&str]) -> Option<&'a String> {
    keys.iter().find_map(|key| exif.get(*key))
}

fn parse_number(value: Option<&String>) -> Option<f64> {
    let value = value?;
    let cleaned = value
        .trim()
        .trim_start_matches("f/")
        .trim_end_matches("mm")
        .trim_end_matches(" s")
        .trim();
    if let Some((num, den)) = cleaned.split_once('/') {
        let num = num.trim().parse::<f64>().ok()?;
        let den = den.trim().parse::<f64>().ok()?;
        if den == 0.0 { None } else { Some(num / den) }
    } else {
        cleaned
            .split_whitespace()
            .next()
            .and_then(|part| part.parse::<f64>().ok())
    }
}

fn parse_i64(value: Option<&String>) -> Option<i64> {
    parse_number(value).map(|v| v.round() as i64)
}

fn parse_date_taken(value: Option<&String>) -> (Option<i64>, Option<i64>) {
    let Some(value) = value else {
        return (None, None);
    };
    let clean = value.trim().replace('T', " ");
    let normalized = clean.replace(':', "-").replacen('-', ":", 0);
    for candidate in [clean.as_str(), normalized.as_str()] {
        for format in [
            "%Y:%m:%d %H:%M:%S",
            "%Y:%m:%d %H:%M:%S%.f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%.f",
        ] {
            if let Ok(dt) = NaiveDateTime::parse_from_str(candidate, format) {
                return (Some(dt.and_utc().timestamp()), Some(dt.year() as i64));
            }
        }
    }
    (None, None)
}

fn read_catalog_exif(path: &Path) -> HashMap<String, String> {
    if let Some(existing) = crate::exif_processing::read_rrexif_sidecar(path) {
        return existing;
    }
    let path_str = path.to_string_lossy();
    if let Ok(mmap) = read_file_mapped(path) {
        crate::exif_processing::read_exif_data(&path_str, &mmap)
    } else if let Ok(bytes) = fs::read(path) {
        crate::exif_processing::read_exif_data(&path_str, &bytes)
    } else {
        HashMap::new()
    }
}

fn is_catalog_image_candidate(path: &Path) -> bool {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let lower_name = file_name.to_lowercase();
    if lower_name.ends_with(".rrdata") || lower_name.ends_with(".rrexif") {
        return false;
    }
    is_supported_image_file(path)
}

fn upsert_image_metadata(
    conn: &Connection,
    image_id: i64,
    path: &Path,
    exif: &HashMap<String, String>,
    updated_at: i64,
) -> Result<(), String> {
    let (width, height) = image::image_dimensions(path)
        .map(|(w, h)| (Some(w as i64), Some(h as i64)))
        .unwrap_or((None, None));
    let date_value = first_exif(
        exif,
        &[
            "DateTimeOriginal",
            "CreateDate",
            "DateTime",
            "DateTimeDigitized",
        ],
    );
    let (date_taken, year) = parse_date_taken(date_value);
    let camera_make = first_exif(exif, &["Make"]).cloned();
    let camera_model = first_exif(exif, &["Model", "UniqueCameraModel"]).cloned();
    let lens_model = first_exif(exif, &["LensModel", "Lens", "LensMake"]).cloned();
    let focal_length = parse_number(first_exif(exif, &["FocalLength", "FocalLengthIn35mmFilm"]));
    let aperture = parse_number(first_exif(exif, &["FNumber", "ApertureValue"]));
    let shutter = first_exif(exif, &["ExposureTime", "ShutterSpeedValue"]).cloned();
    let iso = parse_i64(first_exif(
        exif,
        &["PhotographicSensitivity", "ISOSpeedRatings", "ISO"],
    ));
    let title = first_exif(exif, &["Title", "ObjectName"]).cloned();
    let caption = first_exif(exif, &["ImageDescription", "Description", "Caption"]).cloned();
    let exif_json = serde_json::to_string(exif).ok();

    conn.execute(
        "
        INSERT INTO image_metadata(image_id, date_taken, year, width, height, camera_make, camera_model, lens_model, focal_length, aperture, shutter, iso, title, caption, exif_json, updated_at)
        VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
        ON CONFLICT(image_id) DO UPDATE SET
          date_taken = excluded.date_taken,
          year = excluded.year,
          width = excluded.width,
          height = excluded.height,
          camera_make = excluded.camera_make,
          camera_model = excluded.camera_model,
          lens_model = excluded.lens_model,
          focal_length = excluded.focal_length,
          aperture = excluded.aperture,
          shutter = excluded.shutter,
          iso = excluded.iso,
          title = excluded.title,
          caption = excluded.caption,
          exif_json = excluded.exif_json,
          updated_at = excluded.updated_at
        ",
        params![
            image_id,
            date_taken,
            year,
            width,
            height,
            camera_make,
            camera_model,
            lens_model,
            focal_length,
            aperture,
            shutter,
            iso,
            title,
            caption,
            exif_json,
            updated_at
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn normalize_tag(tag: &str) -> (String, String) {
    if let Some(user) = tag.strip_prefix("user:") {
        (user.to_string(), "user".to_string())
    } else if let Some(color) = tag.strip_prefix("color:") {
        (color.to_string(), "color".to_string())
    } else if let Some(person) = tag.strip_prefix("person:") {
        (person.to_string(), "person".to_string())
    } else {
        (tag.to_string(), "ai".to_string())
    }
}

fn sync_image_tags(
    conn: &Connection,
    version_id: i64,
    tags: Option<&Vec<String>>,
) -> Result<(), String> {
    conn.execute(
        "DELETE FROM image_tags WHERE image_version_id = ?1",
        params![version_id],
    )
    .map_err(|e| e.to_string())?;
    let Some(tags) = tags else {
        return Ok(());
    };
    for tag in tags {
        let (name, kind) = normalize_tag(tag);
        if name.trim().is_empty() {
            continue;
        }
        conn.execute(
            "INSERT OR IGNORE INTO tags(name, kind) VALUES(?1, ?2)",
            params![name, kind],
        )
        .map_err(|e| e.to_string())?;
        let tag_id: i64 = conn
            .query_row(
                "SELECT id FROM tags WHERE name = ?1 AND kind = ?2",
                params![name, kind],
                |row| row.get(0),
            )
            .map_err(|e| e.to_string())?;
        conn.execute(
            "INSERT OR IGNORE INTO image_tags(image_version_id, tag_id, source) VALUES(?1, ?2, ?3)",
            params![version_id, tag_id, kind],
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn sync_image_tags_conn(
    conn: &Connection,
    version_id: i64,
    tags: Option<&Vec<String>>,
) -> Result<(), String> {
    conn.execute(
        "DELETE FROM image_tags WHERE image_version_id = ?1",
        params![version_id],
    )
    .map_err(|e| e.to_string())?;
    let Some(tags) = tags else {
        return Ok(());
    };
    for tag in tags {
        let (name, kind) = normalize_tag(tag);
        if name.trim().is_empty() {
            continue;
        }
        conn.execute(
            "INSERT OR IGNORE INTO tags(name, kind) VALUES(?1, ?2)",
            params![name, kind],
        )
        .map_err(|e| e.to_string())?;
        let tag_id: i64 = conn
            .query_row(
                "SELECT id FROM tags WHERE name = ?1 AND kind = ?2",
                params![name, kind],
                |row| row.get(0),
            )
            .map_err(|e| e.to_string())?;
        conn.execute(
            "INSERT OR IGNORE INTO image_tags(image_version_id, tag_id, source) VALUES(?1, ?2, ?3)",
            params![version_id, tag_id, kind],
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn upsert_folder(
    conn: &Connection,
    root_id: i64,
    root_path: &Path,
    folder_path: &Path,
    indexed_at: i64,
) -> Result<i64, String> {
    let relative = folder_path
        .strip_prefix(root_path)
        .unwrap_or(folder_path)
        .to_string_lossy()
        .replace('\\', "/");
    let relative = if relative.is_empty() {
        ".".to_string()
    } else {
        relative
    };
    let name = if relative == "." {
        root_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| root_path.to_string_lossy().into_owned())
    } else {
        folder_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| relative.clone())
    };
    let modified = unix_modified(folder_path) as i64;
    conn.execute(
        "
        INSERT INTO folders(root_id, relative_path, name, modified_at, indexed_at)
        VALUES(?1, ?2, ?3, ?4, ?5)
        ON CONFLICT(root_id, relative_path) DO UPDATE SET
          name = excluded.name,
          modified_at = excluded.modified_at,
          indexed_at = excluded.indexed_at
        ",
        params![root_id, relative, name, modified, indexed_at],
    )
    .map_err(|e| e.to_string())?;
    conn.query_row(
        "SELECT id FROM folders WHERE root_id = ?1 AND relative_path = ?2",
        params![root_id, relative],
        |row| row.get(0),
    )
    .map_err(|e| e.to_string())
}

fn wait_for_catalog_scan_control(
    control: &Arc<CatalogScanControl>,
    root_id: i64,
    root_path: &str,
    current: usize,
    total: usize,
    on_progress: &mut impl FnMut(CatalogScanProgress),
) -> Result<(), String> {
    if control.cancelled.load(Ordering::SeqCst) {
        return Err("Catalog scan cancelled".to_string());
    }

    let mut paused = control.paused.lock().unwrap();
    let mut emitted_pause = false;
    while *paused {
        if control.cancelled.load(Ordering::SeqCst) {
            return Err("Catalog scan cancelled".to_string());
        }
        if !emitted_pause {
            on_progress(CatalogScanProgress {
                root_id,
                root_path: root_path.to_string(),
                current,
                total,
                current_path: None,
                camera: None,
                lens: None,
                year: None,
                message: "Indexing paused".to_string(),
            });
            emitted_pause = true;
        }
        paused = control.cvar.wait(paused).unwrap();
    }

    Ok(())
}

fn scan_library_root_impl<F>(
    root_id: i64,
    recursive: bool,
    app_handle: &AppHandle,
    db_path: PathBuf,
    control: Arc<CatalogScanControl>,
    mut on_progress: F,
) -> Result<ScanResult, String>
where
    F: FnMut(CatalogScanProgress),
{
    let mut conn = open_connection(&db_path)?;
    let root_path_str: String = conn
        .query_row(
            "SELECT absolute_path FROM collection_roots WHERE id = ?1",
            params![root_id],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;
    let root_path = PathBuf::from(&root_path_str);
    if !root_path.exists() {
        conn.execute(
            "UPDATE collection_roots SET is_available = 0 WHERE id = ?1",
            params![root_id],
        )
        .map_err(|e| e.to_string())?;
        return Err("Library root is not available".to_string());
    }

    let indexed_at = now_secs();
    let mut scanned = 0usize;
    let mut inserted_or_updated = 0usize;

    let paths: Vec<PathBuf> = if recursive {
        WalkDir::new(&root_path)
            .into_iter()
            .filter_map(Result::ok)
            .map(|e| e.into_path())
            .filter(|p| p.is_file())
            .filter(|p| is_catalog_image_candidate(p))
            .collect()
    } else {
        fs::read_dir(&root_path)
            .map_err(|e| e.to_string())?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .filter(|p| is_catalog_image_candidate(p))
            .collect()
    };
    let total = paths.len();

    on_progress(CatalogScanProgress {
        root_id,
        root_path: root_path_str.clone(),
        current: 0,
        total,
        current_path: None,
        camera: None,
        lens: None,
        year: None,
        message: "Catalog scan prepared".to_string(),
    });

    conn.execute(
        "UPDATE collection_roots SET is_available = 1, last_scan_at = ?2 WHERE id = ?1",
        params![root_id, indexed_at],
    )
    .map_err(|e| e.to_string())?;

    for path in paths {
        wait_for_catalog_scan_control(
            &control,
            root_id,
            &root_path_str,
            scanned,
            total,
            &mut on_progress,
        )?;
        scanned += 1;
        let parent = path.parent().unwrap_or(root_path.as_path());
        let image_tx = conn.transaction().map_err(|e| e.to_string())?;
        let folder_id = upsert_folder(&image_tx, root_id, &root_path, parent, indexed_at)?;
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let relative_path = path
            .strip_prefix(&root_path)
            .unwrap_or(path.as_path())
            .to_string_lossy()
            .replace('\\', "/");
        let metadata = path.metadata().ok();
        let file_size = metadata.as_ref().map(|m| m.len() as i64);
        let modified_at = metadata
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let extension = path.extension().map(|e| e.to_string_lossy().to_lowercase());
        let is_raw = is_raw_file(&path);

        image_tx.execute(
            "
            INSERT INTO images(root_id, folder_id, file_name, relative_path, extension, file_size, modified_at, status, is_raw, is_cloud_placeholder, imported_at, updated_at)
            VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, 'present', ?8, 0, ?9, ?9)
            ON CONFLICT(root_id, relative_path) DO UPDATE SET
              folder_id = excluded.folder_id,
              file_name = excluded.file_name,
              extension = excluded.extension,
              file_size = excluded.file_size,
              modified_at = excluded.modified_at,
              status = 'present',
              is_raw = excluded.is_raw,
              updated_at = excluded.updated_at
            ",
            params![
                root_id,
                folder_id,
                file_name,
                relative_path,
                extension,
                file_size,
                modified_at,
                if is_raw { 1 } else { 0 },
                indexed_at
            ],
        )
        .map_err(|e| e.to_string())?;
        let image_id: i64 = image_tx
            .query_row(
                "SELECT id FROM images WHERE root_id = ?1 AND relative_path = ?2",
                params![root_id, relative_path],
                |row| row.get(0),
            )
            .map_err(|e| e.to_string())?;

        let sidecar_path = parse_virtual_path(&path.to_string_lossy()).1;
        let (is_edited, rating, tags) = sidecar_metadata(&path, &sidecar_path, &app_handle);
        let tags_json = tags.as_ref().and_then(|t| serde_json::to_string(t).ok());
        let sidecar_modified = if sidecar_path.exists() {
            Some(unix_modified(&sidecar_path) as i64)
        } else {
            None
        };
        image_tx.execute(
            "
            INSERT INTO image_versions(image_id, copy_id, display_name, sidecar_path, rating, color_label, is_edited, tags_json, sidecar_modified_at, created_at, updated_at)
            VALUES(?1, '', ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?9)
            ON CONFLICT(image_id, copy_id) DO UPDATE SET
              display_name = excluded.display_name,
              sidecar_path = excluded.sidecar_path,
              rating = excluded.rating,
              color_label = excluded.color_label,
              is_edited = excluded.is_edited,
              tags_json = excluded.tags_json,
              sidecar_modified_at = excluded.sidecar_modified_at,
              updated_at = excluded.updated_at
            ",
            params![
                image_id,
                file_name,
                sidecar_path.to_string_lossy().into_owned(),
                rating,
                color_label_from_tags(tags.as_ref()),
                if is_edited { 1 } else { 0 },
                tags_json,
                sidecar_modified,
                indexed_at
            ],
        )
        .map_err(|e| e.to_string())?;
        let version_id: i64 = image_tx
            .query_row(
                "SELECT id FROM image_versions WHERE image_id = ?1 AND copy_id = ''",
                params![image_id],
                |row| row.get(0),
            )
            .map_err(|e| e.to_string())?;
        let exif = read_catalog_exif(&path);
        upsert_image_metadata(&image_tx, image_id, &path, &exif, indexed_at)?;
        sync_image_tags(&image_tx, version_id, tags.as_ref())?;
        image_tx.commit().map_err(|e| e.to_string())?;
        inserted_or_updated += 1;

        let camera = match (
            first_exif(&exif, &["Make"]),
            first_exif(&exif, &["Model", "UniqueCameraModel"]),
        ) {
            (Some(make), Some(model)) if !make.trim().is_empty() => {
                Some(format!("{} {}", make.trim(), model.trim()))
            }
            (_, Some(model)) => Some(model.trim().to_string()),
            _ => None,
        };
        let lens = first_exif(&exif, &["LensModel", "Lens", "LensMake"])
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let (_, year) = parse_date_taken(first_exif(
            &exif,
            &[
                "DateTimeOriginal",
                "CreateDate",
                "DateTime",
                "DateTimeDigitized",
            ],
        ));
        on_progress(CatalogScanProgress {
            root_id,
            root_path: root_path_str.clone(),
            current: scanned,
            total,
            current_path: Some(path.to_string_lossy().into_owned()),
            camera,
            lens,
            year,
            message: "Indexing image metadata".to_string(),
        });
    }

    wait_for_catalog_scan_control(
        &control,
        root_id,
        &root_path_str,
        scanned,
        total,
        &mut on_progress,
    )?;

    let maintenance_tx = conn.transaction().map_err(|e| e.to_string())?;
    let missing_marked = maintenance_tx
        .execute(
            "UPDATE images SET status = 'missing', updated_at = ?2 WHERE root_id = ?1 AND updated_at < ?2",
            params![root_id, indexed_at],
        )
        .map_err(|e| e.to_string())?;

    maintenance_tx.execute(
        "
        UPDATE folders
        SET image_count = (
          SELECT COUNT(*) FROM images WHERE images.folder_id = folders.id AND images.status = 'present'
        )
        WHERE root_id = ?1
        ",
        params![root_id],
    )
    .map_err(|e| e.to_string())?;
    maintenance_tx.commit().map_err(|e| e.to_string())?;

    Ok(ScanResult {
        root_id,
        scanned,
        inserted_or_updated,
        missing_marked,
    })
}

#[tauri::command]
pub fn scan_library_root(
    root_id: i64,
    recursive: bool,
    app_handle: AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<ScanResult, String> {
    let db_path = active_library_path(&state)?;
    let control = state.catalog_scan_control.clone();
    control.begin(root_id)?;
    let result = scan_library_root_impl(
        root_id,
        recursive,
        &app_handle,
        db_path,
        control.clone(),
        |_| {},
    );
    control.finish(root_id);
    result
}

#[tauri::command]
pub fn start_catalog_scan(
    root_id: i64,
    recursive: bool,
    app_handle: AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    let db_path = active_library_path(&state)?;
    let control = state.catalog_scan_control.clone();
    let root_path = open_connection(&db_path)?
        .query_row(
            "SELECT absolute_path FROM collection_roots WHERE id = ?1",
            params![root_id],
            |row| row.get::<_, String>(0),
        )
        .map_err(|e| e.to_string())?;
    let job_id = {
        let conn = open_connection(&db_path)?;
        create_catalog_scan_job(&conn, root_id, recursive)?
    };
    control.begin(root_id)?;
    *control.active_job_id.lock().unwrap() = Some(job_id.clone());
    update_job(
        &db_path,
        &job_id,
        "running",
        "Starting catalog scan",
        0,
        0,
        None,
        None,
    )?;
    let _ = app_handle.emit(
        "catalog-scan-progress",
        CatalogScanProgress {
            root_id,
            root_path: root_path.clone(),
            current: 0,
            total: 0,
            current_path: None,
            camera: None,
            lens: None,
            year: None,
            message: "Starting catalog scan".to_string(),
        },
    );
    let app_for_task = app_handle.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let result = scan_library_root_impl(
            root_id,
            recursive,
            &app_for_task,
            db_path.clone(),
            control.clone(),
            |progress| {
                let state = if progress.message == "Indexing paused" {
                    "paused"
                } else {
                    "running"
                };
                let _ = update_job(
                    &db_path,
                    &job_id,
                    state,
                    &progress.message,
                    progress.current as i64,
                    progress.total as i64,
                    progress.current_path.as_deref(),
                    None,
                );
                let _ = app_for_task.emit("catalog-scan-progress", progress);
            },
        );
        control.finish(root_id);
        match result {
            Ok(scan) => {
                let _ = update_job(
                    &db_path,
                    &job_id,
                    "completed",
                    "Catalog scan complete",
                    scan.scanned as i64,
                    scan.scanned as i64,
                    None,
                    None,
                );
                let _ = app_for_task.emit("catalog-scan-complete", scan);
            }
            Err(err) => {
                let state = if err == "Catalog scan cancelled" {
                    "cancelled"
                } else {
                    "failed"
                };
                let _ = update_job(&db_path, &job_id, state, &err, 0, 0, None, Some(&err));
                let _ = app_for_task.emit(
                    "catalog-scan-error",
                    serde_json::json!({ "rootId": root_id, "error": err }),
                );
            }
        }
    });
    Ok(())
}

#[tauri::command]
pub fn pause_catalog_scan(state: tauri::State<'_, crate::AppState>) -> Result<(), String> {
    if state
        .catalog_scan_control
        .active_root_id
        .lock()
        .unwrap()
        .is_none()
    {
        return Err("No catalog scan is running".to_string());
    }
    *state.catalog_scan_control.paused.lock().unwrap() = true;
    if let Some(job_id) = state
        .catalog_scan_control
        .active_job_id
        .lock()
        .unwrap()
        .clone()
    {
        update_job(
            &active_library_path(&state)?,
            &job_id,
            "paused",
            "Indexing paused",
            0,
            0,
            None,
            None,
        )?;
    }
    Ok(())
}

#[tauri::command]
pub fn resume_catalog_scan(state: tauri::State<'_, crate::AppState>) -> Result<(), String> {
    if state
        .catalog_scan_control
        .active_root_id
        .lock()
        .unwrap()
        .is_none()
    {
        return Err("No catalog scan is running".to_string());
    }
    *state.catalog_scan_control.paused.lock().unwrap() = false;
    state.catalog_scan_control.cvar.notify_all();
    if let Some(job_id) = state
        .catalog_scan_control
        .active_job_id
        .lock()
        .unwrap()
        .clone()
    {
        update_job(
            &active_library_path(&state)?,
            &job_id,
            "running",
            "Resuming catalog scan",
            0,
            0,
            None,
            None,
        )?;
    }
    Ok(())
}

#[tauri::command]
pub fn cancel_catalog_scan(state: tauri::State<'_, crate::AppState>) -> Result<(), String> {
    if state
        .catalog_scan_control
        .active_root_id
        .lock()
        .unwrap()
        .is_none()
    {
        return Err("No catalog scan is running".to_string());
    }
    state
        .catalog_scan_control
        .cancelled
        .store(true, Ordering::SeqCst);
    *state.catalog_scan_control.paused.lock().unwrap() = false;
    state.catalog_scan_control.cvar.notify_all();
    if let Some(job_id) = state
        .catalog_scan_control
        .active_job_id
        .lock()
        .unwrap()
        .clone()
    {
        update_job(
            &active_library_path(&state)?,
            &job_id,
            "cancelling",
            "Cancelling catalog scan",
            0,
            0,
            None,
            None,
        )?;
    }
    Ok(())
}

fn read_background_job(row: &rusqlite::Row<'_>) -> rusqlite::Result<BackgroundJob> {
    Ok(BackgroundJob {
        id: row.get(0)?,
        kind: row.get(1)?,
        state: row.get(2)?,
        root_id: row.get(3)?,
        current: row.get(4)?,
        total: row.get(5)?,
        current_item: row.get(6)?,
        message: row.get(7)?,
        error: row.get(8)?,
        created_at: row.get(9)?,
        updated_at: row.get(10)?,
    })
}

#[tauri::command]
pub fn list_background_jobs(
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<BackgroundJob>, String> {
    let conn = open_connection(&active_library_path(&state)?)?;
    let mut statement = conn.prepare("SELECT id, kind, state, root_id, current, total, current_item, message, error, created_at, updated_at FROM background_jobs ORDER BY CASE WHEN state IN ('running', 'paused', 'cancelling') THEN 0 ELSE 1 END, updated_at DESC LIMIT 100").map_err(|error| error.to_string())?;
    statement
        .query_map([], read_background_job)
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn list_background_job_events(
    job_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<BackgroundJobEvent>, String> {
    let conn = open_connection(&active_library_path(&state)?)?;
    let mut statement = conn.prepare("SELECT id, job_id, state, message, current, total, created_at FROM background_job_events WHERE job_id = ?1 ORDER BY id DESC LIMIT 200").map_err(|error| error.to_string())?;
    statement
        .query_map([job_id], |row| {
            Ok(BackgroundJobEvent {
                id: row.get(0)?,
                job_id: row.get(1)?,
                state: row.get(2)?,
                message: row.get(3)?,
                current: row.get(4)?,
                total: row.get(5)?,
                created_at: row.get(6)?,
            })
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn cancel_background_job(
    job_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    let db_path = active_library_path(&state)?;
    let conn = open_connection(&db_path)?;
    let (kind, job_state): (String, String) = conn
        .query_row(
            "SELECT kind, state FROM background_jobs WHERE id = ?1",
            [&job_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|error| error.to_string())?;
    if !matches!(
        job_state.as_str(),
        "queued" | "running" | "paused" | "cancelling"
    ) {
        return Err("This job is no longer cancellable".to_string());
    }

    if kind == "catalog_scan" {
        if state
            .catalog_scan_control
            .active_job_id
            .lock()
            .unwrap()
            .as_deref()
            != Some(&job_id)
        {
            return Err("The catalog scan is not active in this application session".to_string());
        }
        state
            .catalog_scan_control
            .cancelled
            .store(true, Ordering::SeqCst);
        *state.catalog_scan_control.paused.lock().unwrap() = false;
        state.catalog_scan_control.cvar.notify_all();
    } else if let Some(token) = state
        .background_job_cancellations
        .lock()
        .unwrap()
        .get(&job_id)
        .cloned()
    {
        token.store(true, Ordering::SeqCst);
    } else {
        return Err("This job cannot be cancelled after an application restart".to_string());
    }

    update_job(
        &db_path,
        &job_id,
        "cancelling",
        "Cancellation requested",
        0,
        0,
        None,
        None,
    )
}

#[tauri::command]
pub fn pause_background_job(
    job_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    let db_path = active_library_path(&state)?;
    let (kind, job_state): (String, String) = open_connection(&db_path)?
        .query_row(
            "SELECT kind, state FROM background_jobs WHERE id = ?1",
            [&job_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|error| error.to_string())?;
    if !can_pause_job(&job_state) {
        return Err("This job is no longer pausable".to_string());
    }
    if kind == "catalog_scan" {
        if state
            .catalog_scan_control
            .active_job_id
            .lock()
            .unwrap()
            .as_deref()
            != Some(&job_id)
        {
            return Err("The catalog scan is not active in this application session".to_string());
        }
        *state.catalog_scan_control.paused.lock().unwrap() = true;
        return update_job(
            &db_path,
            &job_id,
            "paused",
            "Indexing paused",
            0,
            0,
            None,
            None,
        );
    }
    let token = state
        .background_job_pauses
        .lock()
        .unwrap()
        .get(&job_id)
        .cloned()
        .ok_or_else(|| "This job cannot be paused after an application restart".to_string())?;
    token.store(true, Ordering::SeqCst);
    update_job(
        &db_path,
        &job_id,
        "paused",
        "Pause requested",
        0,
        0,
        None,
        None,
    )
}

#[tauri::command]
pub fn resume_background_job(
    job_id: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    let db_path = active_library_path(&state)?;
    let (kind, job_state): (String, String) = open_connection(&db_path)?
        .query_row(
            "SELECT kind, state FROM background_jobs WHERE id = ?1",
            [&job_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|error| error.to_string())?;
    if !can_resume_job(&job_state) {
        return Err("This job is not paused".to_string());
    }
    if kind == "catalog_scan" {
        if state
            .catalog_scan_control
            .active_job_id
            .lock()
            .unwrap()
            .as_deref()
            != Some(&job_id)
        {
            return Err("The catalog scan is not active in this application session".to_string());
        }
        *state.catalog_scan_control.paused.lock().unwrap() = false;
        state.catalog_scan_control.cvar.notify_all();
        return update_job(
            &db_path,
            &job_id,
            "running",
            "Resuming catalog scan",
            0,
            0,
            None,
            None,
        );
    }
    let token = state
        .background_job_pauses
        .lock()
        .unwrap()
        .get(&job_id)
        .cloned()
        .ok_or_else(|| "This job cannot be resumed after an application restart".to_string())?;
    token.store(false, Ordering::SeqCst);
    update_job(
        &db_path,
        &job_id,
        "running",
        "Resume requested",
        0,
        0,
        None,
        None,
    )
}

fn color_label_from_tags(tags: Option<&Vec<String>>) -> Option<String> {
    tags.and_then(|items| {
        items
            .iter()
            .find_map(|tag| tag.strip_prefix("color:").map(|s| s.to_string()))
    })
}

fn update_catalog_version_for_path(
    conn: &Connection,
    app_handle: &AppHandle,
    path: &str,
) -> Result<bool, String> {
    let source_path = parse_virtual_path(path).0;
    let roots = {
        let mut stmt = conn
            .prepare("SELECT id, absolute_path FROM collection_roots")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?
    };

    for (root_id, root_path_str) in roots {
        let root_path = PathBuf::from(&root_path_str);
        let Ok(relative) = source_path.strip_prefix(&root_path) else {
            continue;
        };
        let relative_path = relative.to_string_lossy().replace('\\', "/");
        let Some(image_id): Option<i64> = conn
            .query_row(
                "SELECT id FROM images WHERE root_id = ?1 AND relative_path = ?2",
                params![root_id, relative_path],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| e.to_string())?
        else {
            continue;
        };

        let sidecar_path = parse_virtual_path(path).1;
        let (is_edited, rating, tags) = sidecar_metadata(&source_path, &sidecar_path, app_handle);
        let tags_json = tags
            .as_ref()
            .and_then(|items| serde_json::to_string(items).ok());
        let sidecar_modified = if sidecar_path.exists() {
            Some(unix_modified(&sidecar_path) as i64)
        } else {
            None
        };
        let version_id: i64 = conn
            .query_row(
                "SELECT id FROM image_versions WHERE image_id = ?1 AND copy_id = ''",
                params![image_id],
                |row| row.get(0),
            )
            .map_err(|e| e.to_string())?;

        conn.execute(
            "
            UPDATE image_versions
            SET rating = ?2,
                color_label = ?3,
                is_edited = ?4,
                tags_json = ?5,
                sidecar_modified_at = ?6,
                updated_at = ?7
            WHERE image_id = ?1 AND copy_id = ''
            ",
            params![
                image_id,
                rating,
                color_label_from_tags(tags.as_ref()),
                if is_edited { 1 } else { 0 },
                tags_json,
                sidecar_modified,
                now_secs()
            ],
        )
        .map_err(|e| e.to_string())?;
        sync_image_tags_conn(conn, version_id, tags.as_ref())?;
        return Ok(true);
    }

    Ok(false)
}

#[tauri::command]
pub fn sync_catalog_paths(
    paths: Vec<String>,
    app_handle: AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<usize, String> {
    let db_path = active_library_path(&state)?;
    let conn = open_connection(&db_path)?;
    let mut updated = 0usize;
    for path in paths {
        if update_catalog_version_for_path(&conn, &app_handle, &path)? {
            updated += 1;
        }
    }
    Ok(updated)
}

#[tauri::command]
pub fn list_catalog_images(
    root_id: Option<i64>,
    recursive: Option<bool>,
    folder_path: Option<String>,
    app_handle: AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<ImageFile>, String> {
    let db_path = active_library_path(&state)?;
    let conn = open_connection(&db_path)?;
    let mut sql = "
        SELECT r.absolute_path, i.relative_path, i.modified_at, v.is_edited, v.rating, v.tags_json, i.is_raw, m.exif_json
        FROM images i
        JOIN collection_roots r ON r.id = i.root_id
        JOIN image_versions v ON v.image_id = i.id AND v.copy_id = ''
        LEFT JOIN image_metadata m ON m.image_id = i.id
        WHERE i.status = 'present'
    "
    .to_string();
    if root_id.is_some() {
        sql.push_str(" AND i.root_id = ?1");
    }
    sql.push_str(" ORDER BY i.relative_path COLLATE NOCASE");

    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let mut rows = if let Some(root_id) = root_id {
        stmt.query(params![root_id]).map_err(|e| e.to_string())?
    } else {
        stmt.query([]).map_err(|e| e.to_string())?
    };

    let mut result = Vec::new();
    while let Some(row) = rows.next().map_err(|e| e.to_string())? {
        let root_path: String = row.get(0).map_err(|e: rusqlite::Error| e.to_string())?;
        let relative_path: String = row.get(1).map_err(|e: rusqlite::Error| e.to_string())?;
        let folder_path = folder_path.as_deref().unwrap_or(".");
        if folder_path != "." {
            let prefix = format!("{folder_path}/");
            if recursive.unwrap_or(false) {
                if !relative_path.starts_with(&prefix) {
                    continue;
                }
            } else if Path::new(&relative_path)
                .parent()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .unwrap_or_default()
                != folder_path
            {
                continue;
            }
        } else if !recursive.unwrap_or(false) && relative_path.contains('/') {
            continue;
        }
        let full_path = Path::new(&root_path).join(&relative_path);
        let tags_json: Option<String> = row.get(5).map_err(|e: rusqlite::Error| e.to_string())?;
        let tags: Option<Vec<String>> = tags_json.and_then(|json| serde_json::from_str(&json).ok());
        let exif_json: Option<String> = row.get(7).map_err(|e: rusqlite::Error| e.to_string())?;
        let exif: Option<HashMap<String, String>> =
            exif_json.and_then(|json| serde_json::from_str(&json).ok());
        result.push(ImageFile {
            path: full_path.to_string_lossy().into_owned(),
            modified: row.get::<_, i64>(2).map_err(|e| e.to_string())? as u64,
            is_edited: row.get::<_, i64>(3).map_err(|e| e.to_string())? != 0,
            rating: row.get::<_, i64>(4).map_err(|e| e.to_string())? as u8,
            tags,
            exif,
            is_virtual_copy: false,
            is_cloud_placeholder: false,
            is_raw: row.get::<_, i64>(6).map_err(|e| e.to_string())? != 0,
            group_id: None,
        });
    }

    let settings = load_settings(app_handle).unwrap_or_default();
    assign_group_ids(&mut result, &settings);
    Ok(result)
}

fn query_catalog_images(
    conn: &Connection,
    sql: &str,
    values: Vec<SqlValue>,
    app_handle: AppHandle,
) -> Result<Vec<ImageFile>, String> {
    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;
    let mut rows = stmt
        .query(params_from_iter(values.iter()))
        .map_err(|e| e.to_string())?;
    let mut result = Vec::new();
    while let Some(row) = rows.next().map_err(|e| e.to_string())? {
        let root_path: String = row.get(0).map_err(|e: rusqlite::Error| e.to_string())?;
        let relative_path: String = row.get(1).map_err(|e: rusqlite::Error| e.to_string())?;
        let full_path = Path::new(&root_path).join(&relative_path);
        let tags_json: Option<String> = row.get(5).map_err(|e: rusqlite::Error| e.to_string())?;
        let tags: Option<Vec<String>> = tags_json.and_then(|json| serde_json::from_str(&json).ok());
        let exif_json: Option<String> = row.get(7).map_err(|e: rusqlite::Error| e.to_string())?;
        let exif: Option<HashMap<String, String>> =
            exif_json.and_then(|json| serde_json::from_str(&json).ok());
        result.push(ImageFile {
            path: full_path.to_string_lossy().into_owned(),
            modified: row.get::<_, i64>(2).map_err(|e| e.to_string())? as u64,
            is_edited: row.get::<_, i64>(3).map_err(|e| e.to_string())? != 0,
            rating: row.get::<_, i64>(4).map_err(|e| e.to_string())? as u8,
            tags,
            exif,
            is_virtual_copy: false,
            is_cloud_placeholder: false,
            is_raw: row.get::<_, i64>(6).map_err(|e| e.to_string())? != 0,
            group_id: None,
        });
    }
    let settings = load_settings(app_handle).unwrap_or_default();
    assign_group_ids(&mut result, &settings);
    Ok(result)
}

#[tauri::command]
pub fn search_catalog_images(
    query: CatalogSearchQuery,
    app_handle: AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<ImageFile>, String> {
    let db_path = active_library_path(&state)?;
    let conn = open_connection(&db_path)?;
    let mut sql = "
        SELECT r.absolute_path, i.relative_path, i.modified_at, v.is_edited, v.rating, v.tags_json, i.is_raw, m.exif_json
        FROM images i
        JOIN collection_roots r ON r.id = i.root_id
        JOIN image_versions v ON v.image_id = i.id AND v.copy_id = ''
        LEFT JOIN image_metadata m ON m.image_id = i.id
        WHERE i.status = 'present'
    "
    .to_string();
    let mut values: Vec<SqlValue> = Vec::new();

    if let Some(root_id) = query.root_id {
        sql.push_str(" AND i.root_id = ?");
        values.push(SqlValue::Integer(root_id));
    }
    if let Some(rating) = query.rating {
        sql.push_str(" AND v.rating = ?");
        values.push(SqlValue::Integer(rating));
    }
    if let Some(min_rating) = query.min_rating {
        sql.push_str(" AND v.rating >= ?");
        values.push(SqlValue::Integer(min_rating));
    }
    if let Some(year) = query.year {
        sql.push_str(" AND m.year = ?");
        values.push(SqlValue::Integer(year));
    }
    if let Some(is_raw) = query.is_raw {
        sql.push_str(" AND i.is_raw = ?");
        values.push(SqlValue::Integer(if is_raw { 1 } else { 0 }));
    }
    if let Some(is_edited) = query.is_edited {
        sql.push_str(" AND v.is_edited = ?");
        values.push(SqlValue::Integer(if is_edited { 1 } else { 0 }));
    }
    if let Some(color) = query.color.filter(|s| !s.trim().is_empty()) {
        sql.push_str(" AND LOWER(v.color_label) = LOWER(?)");
        values.push(SqlValue::Text(color));
    }
    if let Some(lens) = query.lens.filter(|s| !s.trim().is_empty()) {
        sql.push_str(" AND LOWER(COALESCE(m.lens_model, '')) LIKE LOWER(?)");
        values.push(SqlValue::Text(format!("%{}%", lens.trim())));
    }
    if let Some(camera) = query.camera.filter(|s| !s.trim().is_empty()) {
        sql.push_str(" AND LOWER(COALESCE(m.camera_make, '') || ' ' || COALESCE(m.camera_model, '')) LIKE LOWER(?)");
        values.push(SqlValue::Text(format!("%{}%", camera.trim())));
    }
    if let Some(person) = query.person.filter(|s| !s.trim().is_empty()) {
        sql.push_str(
            " AND EXISTS (
                SELECT 1
                FROM image_tags it
                JOIN tags t ON t.id = it.tag_id
                WHERE it.image_version_id = v.id AND t.kind = 'person' AND LOWER(t.name) LIKE LOWER(?)
                UNION ALL
                SELECT 1
                FROM faces f
                JOIN people p ON p.id = f.person_id
                WHERE f.image_id = i.id
                  AND f.review_state = 'confirmed'
                  AND p.state = 'active'
                  AND LOWER(p.display_name) LIKE LOWER(?)
            )",
        );
        let like = SqlValue::Text(format!("%{}%", person.trim()));
        values.push(like.clone());
        values.push(like);
    }
    if let Some(tags) = query.tags {
        let cleaned: Vec<String> = tags
            .into_iter()
            .map(|t| t.trim().trim_start_matches("user:").to_string())
            .filter(|t| !t.is_empty())
            .collect();
        if !cleaned.is_empty() {
            let use_and = query
                .tag_mode
                .as_deref()
                .map(|m| m.eq_ignore_ascii_case("AND"))
                .unwrap_or(true);
            if use_and {
                for tag in cleaned {
                    sql.push_str(
                        " AND EXISTS (
                            SELECT 1 FROM image_tags it
                            JOIN tags t ON t.id = it.tag_id
                            WHERE it.image_version_id = v.id AND LOWER(t.name) LIKE LOWER(?)
                        )",
                    );
                    values.push(SqlValue::Text(format!("%{}%", tag)));
                }
            } else {
                sql.push_str(
                    " AND EXISTS (
                        SELECT 1 FROM image_tags it
                        JOIN tags t ON t.id = it.tag_id
                        WHERE it.image_version_id = v.id AND (",
                );
                for (index, tag) in cleaned.into_iter().enumerate() {
                    if index > 0 {
                        sql.push_str(" OR ");
                    }
                    sql.push_str("LOWER(t.name) LIKE LOWER(?)");
                    values.push(SqlValue::Text(format!("%{}%", tag)));
                }
                sql.push_str("))");
            }
        }
    }
    if let Some(tags) = query.ai_tags {
        let cleaned: Vec<String> = tags
            .into_iter()
            .map(|tag| tag.trim().to_string())
            .filter(|tag| !tag.is_empty())
            .collect();
        if !cleaned.is_empty() {
            let use_and = query
                .tag_mode
                .as_deref()
                .map(|mode| mode.eq_ignore_ascii_case("AND"))
                .unwrap_or(true);
            if use_and {
                for tag in cleaned {
                    sql.push_str(
                        " AND EXISTS (
                            SELECT 1 FROM image_ai_tags iat
                            JOIN tags t ON t.id = iat.tag_id
                            WHERE iat.image_id = i.id
                              AND iat.review_state <> 'rejected'
                              AND LOWER(t.name) LIKE LOWER(?)
                        )",
                    );
                    values.push(SqlValue::Text(format!("%{}%", tag)));
                }
            } else {
                sql.push_str(
                    " AND EXISTS (
                        SELECT 1 FROM image_ai_tags iat
                        JOIN tags t ON t.id = iat.tag_id
                        WHERE iat.image_id = i.id
                          AND iat.review_state <> 'rejected'
                          AND (",
                );
                for (index, tag) in cleaned.into_iter().enumerate() {
                    if index > 0 {
                        sql.push_str(" OR ");
                    }
                    sql.push_str("LOWER(t.name) LIKE LOWER(?)");
                    values.push(SqlValue::Text(format!("%{}%", tag)));
                }
                sql.push_str("))");
            }
        }
    }
    if let Some(text) = query.text.filter(|s| !s.trim().is_empty()) {
        sql.push_str(
            " AND (
                LOWER(i.file_name) LIKE LOWER(?)
                OR LOWER(i.relative_path) LIKE LOWER(?)
                OR LOWER(COALESCE(m.title, '')) LIKE LOWER(?)
                OR LOWER(COALESCE(m.caption, '')) LIKE LOWER(?)
                OR LOWER(COALESCE(m.camera_make, '') || ' ' || COALESCE(m.camera_model, '')) LIKE LOWER(?)
                OR LOWER(COALESCE(m.lens_model, '')) LIKE LOWER(?)
                OR LOWER(COALESCE(v.tags_json, '')) LIKE LOWER(?)
            )",
        );
        let like = SqlValue::Text(format!("%{}%", text.trim()));
        for _ in 0..7 {
            values.push(like.clone());
        }
    }

    sql.push_str(
        " ORDER BY COALESCE(m.date_taken, i.modified_at) DESC, i.relative_path COLLATE NOCASE",
    );
    if let Some(limit) = query.limit {
        sql.push_str(" LIMIT ?");
        values.push(SqlValue::Integer(limit.clamp(1, 20_000)));
    }

    query_catalog_images(&conn, &sql, values, app_handle)
}

#[tauri::command]
pub fn list_catalog_people(
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<CatalogPerson>, String> {
    let db_path = active_library_path(&state)?;
    let conn = open_connection(&db_path)?;
    let mut statement = conn
        .prepare(
            "SELECT p.id, p.display_name, p.state, COUNT(f.id)
             FROM people p
             LEFT JOIN faces f ON f.person_id = p.id AND f.review_state = 'confirmed'
             WHERE p.state = 'active'
             GROUP BY p.id
             ORDER BY COUNT(f.id) DESC, p.display_name COLLATE NOCASE",
        )
        .map_err(|error| error.to_string())?;
    statement
        .query_map([], |row| {
            Ok(CatalogPerson {
                id: row.get(0)?,
                display_name: row.get(1)?,
                state: row.get(2)?,
                face_count: row.get(3)?,
            })
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn list_smart_collections(
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<SmartCollection>, String> {
    let conn = open_connection(&active_library_path(&state)?)?;
    let mut statement = conn
        .prepare("SELECT id, name, query_json FROM smart_collections ORDER BY name COLLATE NOCASE")
        .map_err(|error| error.to_string())?;
    statement
        .query_map([], |row| {
            Ok(SmartCollection {
                id: row.get(0)?,
                name: row.get(1)?,
                query_json: row.get(2)?,
            })
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn save_smart_collection(
    name: String,
    query_json: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<SmartCollection, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("Collection name cannot be empty".to_string());
    }
    serde_json::from_str::<CatalogSearchQuery>(&query_json)
        .map_err(|error| format!("Invalid collection query: {error}"))?;
    let conn = open_connection(&active_library_path(&state)?)?;
    conn.execute("INSERT INTO smart_collections(name, query_json, created_at, updated_at) VALUES(?1, ?2, strftime('%s','now'), strftime('%s','now')) ON CONFLICT(name) DO UPDATE SET query_json = excluded.query_json, updated_at = excluded.updated_at", params![name, query_json]).map_err(|error| error.to_string())?;
    conn.query_row(
        "SELECT id, name, query_json FROM smart_collections WHERE name = ?1",
        [name],
        |row| {
            Ok(SmartCollection {
                id: row.get(0)?,
                name: row.get(1)?,
                query_json: row.get(2)?,
            })
        },
    )
    .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn delete_smart_collection(
    id: i64,
    state: tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    let changed = open_connection(&active_library_path(&state)?)?
        .execute("DELETE FROM smart_collections WHERE id = ?1", [id])
        .map_err(|error| error.to_string())?;
    if changed == 0 {
        return Err("Smart collection was not found".to_string());
    }
    Ok(())
}

pub(crate) fn record_cull_session(
    db_path: &Path,
    scope_path: &str,
    settings_json: &str,
    decisions: &[(String, bool, String, f64)],
) -> Result<Option<i64>, String> {
    let mut connection = open_connection(db_path)?;
    let roots = {
        let mut statement = connection
            .prepare("SELECT id, absolute_path FROM collection_roots")
            .map_err(|error| error.to_string())?;
        statement
            .query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)))
            .map_err(|error| error.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?
    };
    let scope = Path::new(scope_path);
    let Some((root_id, root_path)) = roots.into_iter().find(|(_, root_path)| scope.starts_with(root_path)) else {
        return Ok(None);
    };
    let transaction = connection.transaction().map_err(|error| error.to_string())?;
    transaction
        .execute(
            "INSERT INTO cull_sessions(root_id, scope_path, state, settings_json, feature_set_version, total_count, rejected_count, created_at, updated_at) VALUES(?1, ?2, 'planned', ?3, 'culling-v1', ?4, ?5, strftime('%s','now'), strftime('%s','now'))",
            params![root_id, scope_path, settings_json, decisions.len() as i64, decisions.iter().filter(|(_, keep, _, _)| !*keep).count() as i64],
        )
        .map_err(|error| error.to_string())?;
    let session_id = transaction.last_insert_rowid();
    let image_ids = {
        let mut statement = transaction
            .prepare("SELECT id, relative_path FROM images WHERE root_id = ?1")
            .map_err(|error| error.to_string())?;
        statement
            .query_map([root_id], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)))
            .map_err(|error| error.to_string())?
            .filter_map(Result::ok)
            .map(|(id, relative_path)| (Path::new(&root_path).join(relative_path).to_string_lossy().into_owned(), id))
            .collect::<HashMap<_, _>>()
    };
    for (path, keep, reason, quality_score) in decisions {
        transaction
            .execute(
                "INSERT INTO cull_decisions(session_id, image_id, representative_path, proposed_status, quality_score, reason, created_at, updated_at) VALUES(?1, ?2, ?3, ?4, ?5, ?6, strftime('%s','now'), strftime('%s','now'))",
                params![session_id, image_ids.get(path), path, if *keep { "keep" } else { "reject" }, quality_score, reason],
            )
            .map_err(|error| error.to_string())?;
    }
    transaction.commit().map_err(|error| error.to_string())?;
    Ok(Some(session_id))
}

pub(crate) fn mark_cull_session_applied(db_path: &Path, session_id: i64) -> Result<(), String> {
    let connection = open_connection(db_path)?;
    connection
        .execute(
            "UPDATE cull_decisions SET final_status = proposed_status, updated_at = strftime('%s','now') WHERE session_id = ?1 AND final_status = 'pending'",
            [session_id],
        )
        .map_err(|error| error.to_string())?;
    connection
        .execute(
            "UPDATE cull_sessions SET state = 'applied', updated_at = strftime('%s','now') WHERE id = ?1",
            [session_id],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn list_cull_sessions(
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<CullSessionSummary>, String> {
    let connection = open_connection(&active_library_path(&state)?)?;
    let mut statement = connection
        .prepare("SELECT id, root_id, scope_path, state, total_count, rejected_count, created_at, updated_at FROM cull_sessions ORDER BY updated_at DESC LIMIT 100")
        .map_err(|error| error.to_string())?;
    statement
        .query_map([], |row| Ok(CullSessionSummary { id: row.get(0)?, root_id: row.get(1)?, scope_path: row.get(2)?, state: row.get(3)?, total_count: row.get(4)?, rejected_count: row.get(5)?, created_at: row.get(6)?, updated_at: row.get(7)? }))
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn list_cull_session_decisions(
    session_id: i64,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<CullSessionDecision>, String> {
    let connection = open_connection(&active_library_path(&state)?)?;
    let mut statement = connection
        .prepare("SELECT id, representative_path, proposed_status, final_status, quality_score, reason FROM cull_decisions WHERE session_id = ?1 ORDER BY quality_score DESC")
        .map_err(|error| error.to_string())?;
    statement
        .query_map([session_id], |row| Ok(CullSessionDecision { id: row.get(0)?, representative_path: row.get(1)?, proposed_status: row.get(2)?, final_status: row.get(3)?, quality_score: row.get(4)?, reason: row.get(5)? }))
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn update_cull_session_decision(
    session_id: i64,
    representative_path: String,
    keep: bool,
    state: tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    let mut connection = open_connection(&active_library_path(&state)?)?;
    let transaction = connection.transaction().map_err(|error| error.to_string())?;
    let decision: Option<(i64, String)> = transaction.query_row(
        "SELECT id, proposed_status FROM cull_decisions WHERE session_id = ?1 AND representative_path = ?2",
        params![session_id, representative_path],
        |row| Ok((row.get(0)?, row.get(1)?)),
    ).optional().map_err(|error| error.to_string())?;
    let Some((decision_id, previous_status)) = decision else { return Err("Culling decision was not found".to_string()); };
    let next_status = if keep { "keep" } else { "reject" };
    if previous_status != next_status {
        transaction.execute("INSERT INTO cull_decision_events(decision_id, previous_status, next_status, created_at) VALUES(?1, ?2, ?3, strftime('%s','now'))", params![decision_id, previous_status, next_status]).map_err(|error| error.to_string())?;
        transaction.execute("UPDATE cull_decisions SET proposed_status = ?1, updated_at = strftime('%s','now') WHERE id = ?2", params![next_status, decision_id]).map_err(|error| error.to_string())?;
    }
    transaction.execute(
        "UPDATE cull_sessions SET rejected_count = (SELECT COUNT(*) FROM cull_decisions WHERE session_id = ?1 AND proposed_status = 'reject'), updated_at = strftime('%s','now') WHERE id = ?1",
        [session_id],
    ).map_err(|error| error.to_string())?;
    transaction.commit().map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn create_catalog_person(
    display_name: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<CatalogPerson, String> {
    let display_name = display_name.trim();
    if display_name.is_empty() {
        return Err("Person name cannot be empty".to_string());
    }
    let db_path = active_library_path(&state)?;
    let conn = open_connection(&db_path)?;
    let now = now_secs();
    conn.execute(
        "INSERT INTO people(display_name, state, created_at, updated_at)
         VALUES(?1, 'active', ?2, ?2)
         ON CONFLICT(display_name) DO UPDATE SET updated_at = excluded.updated_at",
        params![display_name, now],
    )
    .map_err(|error| error.to_string())?;
    conn.query_row(
        "SELECT p.id, p.display_name, p.state, COUNT(f.id)
         FROM people p
         LEFT JOIN faces f ON f.person_id = p.id AND f.review_state = 'confirmed'
         WHERE p.display_name = ?1
         GROUP BY p.id",
        [display_name],
        |row| {
            Ok(CatalogPerson {
                id: row.get(0)?,
                display_name: row.get(1)?,
                state: row.get(2)?,
                face_count: row.get(3)?,
            })
        },
    )
    .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn merge_catalog_people(
    source_person_id: i64,
    target_person_id: i64,
    state: tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    if source_person_id == target_person_id {
        return Err("Choose two different people to merge".to_string());
    }
    let mut connection = open_connection(&active_library_path(&state)?)?;
    let transaction = connection.transaction().map_err(|error| error.to_string())?;
    let source_exists: Option<i64> = transaction
        .query_row("SELECT id FROM people WHERE id = ?1 AND state = 'active'", [source_person_id], |row| row.get(0))
        .optional()
        .map_err(|error| error.to_string())?;
    let target_exists: Option<i64> = transaction
        .query_row("SELECT id FROM people WHERE id = ?1 AND state = 'active'", [target_person_id], |row| row.get(0))
        .optional()
        .map_err(|error| error.to_string())?;
    if source_exists.is_none() || target_exists.is_none() {
        return Err("Both people must be active catalog people".to_string());
    }
    transaction.execute("UPDATE faces SET person_id = ?1, updated_at = strftime('%s','now') WHERE person_id = ?2", params![target_person_id, source_person_id]).map_err(|error| error.to_string())?;
    transaction.execute("UPDATE people SET state = 'merged', merged_into_person_id = ?1, updated_at = strftime('%s','now') WHERE id = ?2", params![target_person_id, source_person_id]).map_err(|error| error.to_string())?;
    transaction.commit().map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn list_catalog_faces(
    image_id: i64,
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<CatalogFace>, String> {
    let db_path = active_library_path(&state)?;
    let conn = open_connection(&db_path)?;
    let mut statement = conn
        .prepare(
            "SELECT id, image_id, person_id, model_pack_id, detector_confidence,
                    bbox_x, bbox_y, bbox_width, bbox_height, review_state
             FROM faces
             WHERE image_id = ?1
             ORDER BY detector_confidence DESC, id",
        )
        .map_err(|error| error.to_string())?;
    statement
        .query_map([image_id], |row| {
            Ok(CatalogFace {
                id: row.get(0)?,
                image_id: row.get(1)?,
                person_id: row.get(2)?,
                model_pack_id: row.get(3)?,
                confidence: row.get(4)?,
                x: row.get(5)?,
                y: row.get(6)?,
                width: row.get(7)?,
                height: row.get(8)?,
                review_state: row.get(9)?,
            })
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn list_unreviewed_catalog_faces(
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<CatalogFaceReviewItem>, String> {
    let conn = open_connection(&active_library_path(&state)?)?;
    let mut statement = conn.prepare("SELECT f.id, f.image_id, f.person_id, f.model_pack_id, f.detector_confidence, f.bbox_x, f.bbox_y, f.bbox_width, f.bbox_height, f.review_state, r.absolute_path || '/' || i.relative_path FROM faces f JOIN images i ON i.id = f.image_id JOIN collection_roots r ON r.id = i.root_id WHERE f.review_state = 'unreviewed' AND i.status = 'present' ORDER BY f.detector_confidence DESC LIMIT 500").map_err(|error| error.to_string())?;
    statement
        .query_map([], |row| {
            Ok(CatalogFaceReviewItem {
                face: CatalogFace {
                    id: row.get(0)?,
                    image_id: row.get(1)?,
                    person_id: row.get(2)?,
                    model_pack_id: row.get(3)?,
                    confidence: row.get(4)?,
                    x: row.get(5)?,
                    y: row.get(6)?,
                    width: row.get(7)?,
                    height: row.get(8)?,
                    review_state: row.get(9)?,
                },
                image_path: row.get(10)?,
            })
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn list_unreviewed_face_clusters(
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<CatalogFaceCluster>, String> {
    let conn = open_connection(&active_library_path(&state)?)?;
    let mut statement = conn.prepare("SELECT c.id, COUNT(m.face_id), r.absolute_path || '/' || i.relative_path FROM face_clusters c JOIN face_cluster_members m ON m.cluster_id = c.id JOIN faces f ON f.id = c.representative_face_id JOIN images i ON i.id = f.image_id JOIN collection_roots r ON r.id = i.root_id WHERE c.state = 'unreviewed' GROUP BY c.id ORDER BY COUNT(m.face_id) DESC, c.id").map_err(|error| error.to_string())?;
    statement
        .query_map([], |row| {
            Ok(CatalogFaceCluster {
                id: row.get(0)?,
                face_count: row.get(1)?,
                representative_image_path: row.get(2)?,
            })
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn confirm_face_cluster(
    cluster_id: i64,
    person_id: i64,
    state: tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    let conn = open_connection(&active_library_path(&state)?)?;
    let active: Option<i64> = conn
        .query_row(
            "SELECT id FROM people WHERE id = ?1 AND state = 'active'",
            [person_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| error.to_string())?;
    if active.is_none() {
        return Err("Selected person is not available".to_string());
    }
    let changed = conn.execute("UPDATE faces SET person_id = ?1, review_state = 'confirmed', updated_at = strftime('%s','now') WHERE review_state = 'unreviewed' AND id IN (SELECT face_id FROM face_cluster_members WHERE cluster_id = ?2)", params![person_id, cluster_id]).map_err(|error| error.to_string())?;
    if changed == 0 {
        return Err("Face cluster has no reviewable faces".to_string());
    }
    conn.execute("UPDATE face_clusters SET state = 'accepted', updated_at = strftime('%s','now') WHERE id = ?1", [cluster_id]).map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn list_suggested_ai_tags(
    state: tauri::State<'_, crate::AppState>,
) -> Result<Vec<CatalogAiTagReviewItem>, String> {
    let conn = open_connection(&active_library_path(&state)?)?;
    let mut statement = conn.prepare("SELECT iat.id, r.absolute_path || '/' || i.relative_path, t.name, iat.confidence FROM image_ai_tags iat JOIN images i ON i.id = iat.image_id JOIN collection_roots r ON r.id = i.root_id JOIN tags t ON t.id = iat.tag_id WHERE i.status = 'present' AND iat.review_state = 'suggested' ORDER BY iat.confidence DESC, iat.id LIMIT 500").map_err(|error| error.to_string())?;
    statement
        .query_map([], |row| {
            Ok(CatalogAiTagReviewItem {
                id: row.get(0)?,
                image_path: row.get(1)?,
                tag: row.get(2)?,
                confidence: row.get(3)?,
            })
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn review_ai_tag(
    id: i64,
    review_state: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    if !matches!(review_state.as_str(), "accepted" | "rejected") {
        return Err("Invalid AI tag review state".to_string());
    }
    let conn = open_connection(&active_library_path(&state)?)?;
    let changed = conn.execute("UPDATE image_ai_tags SET review_state = ?1, updated_at = strftime('%s','now') WHERE id = ?2", params![review_state, id]).map_err(|error| error.to_string())?;
    if changed == 0 {
        return Err("AI tag suggestion was not found".to_string());
    }
    Ok(())
}

#[tauri::command]
pub fn review_catalog_face(
    face_id: i64,
    person_id: Option<i64>,
    review_state: String,
    state: tauri::State<'_, crate::AppState>,
) -> Result<(), String> {
    if !matches!(
        review_state.as_str(),
        "unreviewed" | "confirmed" | "rejected"
    ) {
        return Err("Invalid face review state".to_string());
    }
    if review_state == "confirmed" && person_id.is_none() {
        return Err("A confirmed face must be assigned to a person".to_string());
    }

    let db_path = active_library_path(&state)?;
    let conn = open_connection(&db_path)?;
    if let Some(person_id) = person_id {
        let is_active: Option<i64> = conn
            .query_row(
                "SELECT id FROM people WHERE id = ?1 AND state = 'active'",
                [person_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| error.to_string())?;
        if is_active.is_none() {
            return Err("Selected person is not available".to_string());
        }
    }
    let changed = conn
        .execute(
            "UPDATE faces
             SET person_id = ?1, review_state = ?2, updated_at = ?3
             WHERE id = ?4",
            params![person_id, review_state, now_secs(), face_id],
        )
        .map_err(|error| error.to_string())?;
    if changed == 0 {
        return Err("Face observation was not found".to_string());
    }
    Ok(())
}

fn facet_query(conn: &Connection, sql: &str) -> Result<Vec<CatalogFacetValue>, String> {
    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            Ok(CatalogFacetValue {
                value: row.get(0)?,
                count: row.get(1)?,
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_catalog_metrics(
    state: tauri::State<'_, crate::AppState>,
) -> Result<CatalogMetrics, String> {
    let db_path = active_library_path(&state)?;
    let conn = open_connection(&db_path)?;
    let total_images = conn
        .query_row(
            "SELECT COUNT(*) FROM images WHERE status = 'present'",
            [],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;
    let edited_images = conn
        .query_row(
            "SELECT COUNT(*) FROM images i JOIN image_versions v ON v.image_id = i.id AND v.copy_id = '' WHERE i.status = 'present' AND v.is_edited = 1",
            [],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;
    let rated_images = conn
        .query_row(
            "SELECT COUNT(*) FROM images i JOIN image_versions v ON v.image_id = i.id AND v.copy_id = '' WHERE i.status = 'present' AND v.rating > 0",
            [],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;
    let missing_images = conn
        .query_row(
            "SELECT COUNT(*) FROM images WHERE status = 'missing'",
            [],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;
    let ai_tags_suggested = conn
        .query_row(
            "SELECT COUNT(*) FROM image_ai_tags WHERE review_state = 'suggested'",
            [],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;
    let ai_tags_accepted = conn
        .query_row(
            "SELECT COUNT(*) FROM image_ai_tags WHERE review_state = 'accepted'",
            [],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;
    let ram_plus_analyzed = conn
        .query_row(
            "SELECT COUNT(*) FROM images i WHERE i.status = 'present' AND EXISTS (SELECT 1 FROM image_ai_analysis_state s WHERE s.image_id = i.id AND s.analysis_kind = 'tagging' AND s.model_id = 'ram-plus' AND s.model_revision = 'onnx-v1' AND s.image_modified_at = i.modified_at AND s.state = 'completed')",
            [],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;
    let ram_plus_pending = conn
        .query_row(
            "SELECT COUNT(*) FROM images i WHERE i.status = 'present' AND NOT EXISTS (SELECT 1 FROM image_ai_analysis_state s WHERE s.image_id = i.id AND s.analysis_kind = 'tagging' AND s.model_id = 'ram-plus' AND s.model_revision = 'onnx-v1' AND s.image_modified_at = i.modified_at AND s.state = 'completed')",
            [],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;
    let ram_plus_failed = conn
        .query_row(
            "SELECT COUNT(*) FROM images i WHERE i.status = 'present' AND EXISTS (SELECT 1 FROM image_ai_analysis_state s WHERE s.image_id = i.id AND s.analysis_kind = 'tagging' AND s.model_id = 'ram-plus' AND s.model_revision = 'onnx-v1' AND s.image_modified_at = i.modified_at AND s.state = 'failed')",
            [],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;
    let cull_sessions = conn
        .query_row("SELECT COUNT(*) FROM cull_sessions", [], |row| row.get(0))
        .map_err(|e| e.to_string())?;
    let cull_overrides = conn
        .query_row("SELECT COUNT(*) FROM cull_decision_events", [], |row| row.get(0))
        .map_err(|e| e.to_string())?;

    Ok(CatalogMetrics {
        total_images,
        edited_images,
        rated_images,
        missing_images,
        ai_tags_suggested,
        ai_tags_accepted,
        ram_plus_analyzed,
        ram_plus_pending,
        ram_plus_failed,
        cull_sessions,
        cull_overrides,
        years: facet_query(
            &conn,
            "SELECT CAST(m.year AS TEXT), COUNT(*) FROM image_metadata m JOIN images i ON i.id = m.image_id WHERE i.status = 'present' AND m.year IS NOT NULL GROUP BY m.year ORDER BY m.year DESC",
        )?,
        cameras: facet_query(
            &conn,
            "SELECT TRIM(COALESCE(m.camera_make, '') || ' ' || COALESCE(m.camera_model, '')), COUNT(*) FROM image_metadata m JOIN images i ON i.id = m.image_id WHERE i.status = 'present' AND TRIM(COALESCE(m.camera_make, '') || ' ' || COALESCE(m.camera_model, '')) <> '' GROUP BY 1 ORDER BY COUNT(*) DESC, 1 COLLATE NOCASE LIMIT 50",
        )?,
        lenses: facet_query(
            &conn,
            "SELECT m.lens_model, COUNT(*) FROM image_metadata m JOIN images i ON i.id = m.image_id WHERE i.status = 'present' AND m.lens_model IS NOT NULL AND m.lens_model <> '' GROUP BY m.lens_model ORDER BY COUNT(*) DESC, m.lens_model COLLATE NOCASE LIMIT 50",
        )?,
        people: facet_query(
            &conn,
            "SELECT name, COUNT(DISTINCT image_id)
             FROM (
               SELECT p.display_name AS name, f.image_id AS image_id
               FROM faces f
               JOIN people p ON p.id = f.person_id
               JOIN images i ON i.id = f.image_id
               WHERE i.status = 'present' AND f.review_state = 'confirmed' AND p.state = 'active'
               UNION ALL
               SELECT t.name AS name, i.id AS image_id
               FROM image_tags it
               JOIN tags t ON t.id = it.tag_id
               JOIN image_versions v ON v.id = it.image_version_id
               JOIN images i ON i.id = v.image_id
               WHERE i.status = 'present' AND t.kind = 'person'
             )
             GROUP BY name
             ORDER BY COUNT(DISTINCT image_id) DESC, name COLLATE NOCASE
             LIMIT 75",
        )?,
        tags: facet_query(
            &conn,
            "SELECT t.name, COUNT(*) FROM image_tags it JOIN tags t ON t.id = it.tag_id JOIN image_versions v ON v.id = it.image_version_id JOIN images i ON i.id = v.image_id WHERE i.status = 'present' AND t.kind <> 'person' GROUP BY t.id ORDER BY COUNT(*) DESC, t.name COLLATE NOCASE LIMIT 75",
        )?,
        ai_tags: facet_query(
            &conn,
            "SELECT t.name, COUNT(DISTINCT iat.image_id)
             FROM image_ai_tags iat
             JOIN tags t ON t.id = iat.tag_id
             JOIN images i ON i.id = iat.image_id
             WHERE i.status = 'present'
               AND t.kind = 'ai'
               AND iat.review_state <> 'rejected'
             GROUP BY t.id
             ORDER BY COUNT(DISTINCT iat.image_id) DESC, t.name COLLATE NOCASE
             LIMIT 75",
        )?,
        ratings: facet_query(
            &conn,
            "SELECT CAST(v.rating AS TEXT), COUNT(*) FROM image_versions v JOIN images i ON i.id = v.image_id WHERE i.status = 'present' GROUP BY v.rating ORDER BY v.rating DESC",
        )?,
    })
}
