use std::env;
use std::fs;
use std::path::PathBuf;

use rapidraw_lib::BackgroundJobControl;
use rapidraw_lib::face_detection::{
    run_face_detection_headless_for_pack, run_face_recognition_headless_for_pack,
};
use rapidraw_lib::face_model_registry::{
    FaceModelPackId, FaceModelRuntimeSupport, FaceModelSelectionPolicy, InstalledFaceModelPack,
    face_model_packs, installed_face_runtime_paths_for_pack_in_dir,
    installed_face_runtime_paths_in_dir, runtime_file_names,
};
use rapidraw_lib::image_restoration::{
    RestorationRecipe, run_restoration_worker, validate_restoration_recipe,
};
use rapidraw_lib::resolve_auto_cull_path_candidates;
use rapidraw_lib::scan_library_root_headless;
use rapidraw_lib::tagging::run_catalog_ram_plus_tagging_headless;
use rapidraw_lib::visual_model_registry::{
    verified_visual_model_pack_dir, visual_model_pack_revision_in_dir, visual_model_packs,
};
use rapidraw_lib::{CullingSettings, cull_images_headless};
use rapidraw_lib::{
    add_library_root_headless, create_library_headless, open_library_headless,
    remove_library_root_headless,
};
use rusqlite::Connection;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;

fn database_argument(arguments: &[String]) -> Result<PathBuf, String> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == "--database")
        .map(|pair| PathBuf::from(&pair[1]))
        .ok_or_else(|| "--database <path> is required".to_string())
}

fn named_argument(arguments: &[String], flag: &str) -> Result<String, String> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
        .ok_or_else(|| format!("{flag} <value> is required"))
}

fn numeric_argument(arguments: &[String], flag: &str) -> Result<i64, String> {
    named_argument(arguments, flag)?
        .parse::<i64>()
        .map_err(|_| format!("{flag} must be an integer"))
}

fn optional_numeric_argument(arguments: &[String], flag: &str) -> Result<Option<i64>, String> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| {
            pair[1]
                .parse::<i64>()
                .map_err(|_| format!("{flag} must be an integer"))
        })
        .transpose()
}

fn optional_face_model_policy(arguments: &[String]) -> Result<FaceModelSelectionPolicy, String> {
    let Some(index) = arguments.iter().position(|argument| argument == "--policy") else {
        return Ok(FaceModelSelectionPolicy::Accuracy);
    };
    let value = arguments
        .get(index + 1)
        .ok_or_else(|| "Missing value for --policy".to_string())?;
    match value.as_str() {
        "accuracy" => Ok(FaceModelSelectionPolicy::Accuracy),
        "balanced" => Ok(FaceModelSelectionPolicy::Balanced),
        "speed" => Ok(FaceModelSelectionPolicy::Speed),
        "automatic" => Ok(FaceModelSelectionPolicy::Automatic),
        _ => Err("--policy must be accuracy, balanced, speed, or automatic".to_string()),
    }
}

fn optional_face_model_pack(arguments: &[String]) -> Result<Option<FaceModelPackId>, String> {
    let Some(index) = arguments.iter().position(|argument| argument == "--pack") else {
        return Ok(None);
    };
    let value = arguments
        .get(index + 1)
        .ok_or_else(|| "Missing value for --pack".to_string())?;
    FaceModelPackId::try_from(value.as_str()).map(Some)
}

fn resolve_face_runtime_for_cli(
    arguments: &[String],
    face_models_dir: &std::path::Path,
) -> Result<rapidraw_lib::face_model_registry::FaceRuntimePaths, String> {
    let pack = optional_face_model_pack(arguments)?;
    let has_policy = arguments.iter().any(|argument| argument == "--policy");
    if pack.is_some() && has_policy {
        return Err("Use either --pack or --policy, not both".to_string());
    }
    match pack {
        Some(pack_id) => installed_face_runtime_paths_for_pack_in_dir(face_models_dir, pack_id),
        None => installed_face_runtime_paths_in_dir(
            face_models_dir,
            optional_face_model_policy(arguments)?,
        ),
    }
}

fn exit_code_for_error(error: &str) -> i32 {
    let normalized = error.to_ascii_lowercase();
    if normalized.contains("cancelled") {
        3
    } else if error.starts_with("Usage:")
        || error.starts_with("--")
        || normalized.contains(" is required")
        || normalized.contains("must be an integer")
        || normalized.contains("must be a number")
        || normalized.starts_with("unknown model")
    {
        2
    } else {
        1
    }
}

fn main() {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let result = match arguments.first().map(String::as_str) {
        Some("library") if arguments.get(1).map(String::as_str) == Some("inspect") => inspect(&arguments),
        Some("library") if arguments.get(1).map(String::as_str) == Some("create") => create_library_cli(&arguments),
        Some("library") if arguments.get(1).map(String::as_str) == Some("open") => open_library_cli(&arguments),
        Some("library") if arguments.get(1).map(String::as_str) == Some("add-root") => add_library_root_cli(&arguments),
        Some("library") if arguments.get(1).map(String::as_str) == Some("remove-root") => remove_library_root_cli(&arguments),
        Some("library") if arguments.get(1).map(String::as_str) == Some("roots") => list_roots(&arguments),
        Some("library") if arguments.get(1).map(String::as_str) == Some("metrics") => metrics(&arguments),
        Some("library") if arguments.get(1).map(String::as_str) == Some("scan") => run_catalog_scan_cli(&arguments),
        Some("library") if arguments.get(1).map(String::as_str) == Some("thumbnails") => run_catalog_thumbnails_cli(&arguments),
        Some("library") if arguments.get(1).map(String::as_str) == Some("metadata") => run_catalog_metadata_cli(&arguments),
        Some("jobs") if arguments.get(1).map(String::as_str) == Some("list") => list_jobs(&arguments),
        Some("jobs") if arguments.get(1).map(String::as_str) == Some("show") => show_job(&arguments),
        Some("faces") if arguments.get(1).map(String::as_str) == Some("status") => face_status(&arguments),
        Some("faces") if arguments.get(1).map(String::as_str) == Some("clusters") => face_clusters(&arguments),
        Some("faces") if arguments.get(1).map(String::as_str) == Some("detect") => run_face_detection_cli(&arguments),
        Some("faces") if arguments.get(1).map(String::as_str) == Some("recognize") => run_face_recognition_cli(&arguments),
        Some("people") if arguments.get(1).map(String::as_str) == Some("list") => list_people(&arguments),
        Some("people") if arguments.get(1).map(String::as_str) == Some("images") => list_person_images(&arguments),
        Some("tags") if arguments.get(1).map(String::as_str) == Some("status") => tag_status(&arguments),
        Some("tags") if arguments.get(1).map(String::as_str) == Some("top") => top_tags(&arguments),
        Some("tags") if arguments.get(1).map(String::as_str) == Some("export-suggestions") => export_tag_suggestions(&arguments),
        Some("tags") if arguments.get(1).map(String::as_str) == Some("review") => review_tag_suggestion(&arguments),
        Some("tags") if arguments.get(1).map(String::as_str) == Some("run") => run_ram_plus_tagging_cli(&arguments),
        Some("species") if arguments.get(1).map(String::as_str) == Some("list") => list_species_suggestions(&arguments),
        Some("species") if arguments.get(1).map(String::as_str) == Some("review") => review_species_suggestion(&arguments),
        Some("collections") if arguments.get(1).map(String::as_str) == Some("list") => list_collections(&arguments),
        Some("collections") if arguments.get(1).map(String::as_str) == Some("show") => show_collection(&arguments),
        Some("cull") if arguments.get(1).map(String::as_str) == Some("sessions") => cull_sessions(&arguments),
        Some("cull") if arguments.get(1).map(String::as_str) == Some("decisions") => cull_decisions(&arguments),
        Some("cull") if arguments.get(1).map(String::as_str) == Some("analyze") => run_cull_analysis_cli(&arguments),
        Some("models") if arguments.get(1).map(String::as_str) == Some("list") => list_models(),
        Some("models") if arguments.get(1).map(String::as_str) == Some("info") => verify_model(&arguments),
        Some("models") if arguments.get(1).map(String::as_str) == Some("verify") => verify_installed_model(&arguments),
        Some("restore") if arguments.get(1).map(String::as_str) == Some("list") => list_derivatives(&arguments),
        Some("restore") if arguments.get(1).map(String::as_str) == Some("run") => run_restore_cli(&arguments),
        Some("raw") if arguments.get(1).map(String::as_str) == Some("develop") => run_raw_develop_cli(&arguments),
        Some("raw") if arguments.get(1).map(String::as_str) == Some("inspect") => run_raw_inspect_cli(&arguments),
        _ => Err("Usage: rapidraw-cli library create --name <name> --database <catalog.db> | rapidraw-cli library open --database <catalog.db> | rapidraw-cli library add-root --database <catalog.db> --path <folder> [--label <name>] | rapidraw-cli library remove-root --database <catalog.db> --root <id> | rapidraw-cli library inspect|roots|metrics|scan|thumbnails|metadata --database <catalog.db> | rapidraw-cli library scan --database <catalog.db> --root <id> [--non-recursive] | rapidraw-cli library thumbnails --database <catalog.db> [--root <id>] [--force] | rapidraw-cli library metadata --database <catalog.db> [--root <id>] | rapidraw-cli jobs list --database <catalog.db> | rapidraw-cli jobs show --database <catalog.db> --id <job-id> | rapidraw-cli faces status|clusters --database <catalog.db> | rapidraw-cli faces detect|recognize --database <catalog.db> --face-models-dir <models/face> [--root <id>] [--policy accuracy|balanced|speed|automatic | --pack <pack-id>] | rapidraw-cli people list|images --database <catalog.db> | rapidraw-cli people images --database <catalog.db> --person <id> | rapidraw-cli tags status|top|export-suggestions|review|run --database <catalog.db> | rapidraw-cli tags review --database <catalog.db> --id <rowid> --state accepted|rejected | rapidraw-cli tags run --database <catalog.db> --models-dir <models/visual> [--max-tags <1-100>] [--with-bioclip] | rapidraw-cli species list|review --database <catalog.db> | rapidraw-cli species review --database <catalog.db> --id <id> --state accepted|rejected | rapidraw-cli collections list|show --database <catalog.db> | rapidraw-cli cull sessions|decisions|analyze --database <catalog.db> | rapidraw-cli cull analyze --database <catalog.db> --root <id> [--similarity-threshold <n>] [--blur-threshold <n>] | rapidraw-cli models list|info|verify | rapidraw-cli restore list|run --database <catalog.db> | rapidraw-cli raw inspect|develop".to_string()),
    };
    match result {
        Ok(value) => println!("{}", value),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(exit_code_for_error(&error));
        }
    }
}

fn list_models() -> Result<serde_json::Value, String> {
    serde_json::to_value(visual_model_packs()).map_err(|error| error.to_string())
}

fn create_library_cli(arguments: &[String]) -> Result<serde_json::Value, String> {
    let db_path = database_argument(arguments)?;
    let name = named_argument(arguments, "--name")?;
    serde_json::to_value(create_library_headless(&name, &db_path)?)
        .map_err(|error| error.to_string())
}

fn open_library_cli(arguments: &[String]) -> Result<serde_json::Value, String> {
    serde_json::to_value(open_library_headless(&database_argument(arguments)?)?)
        .map_err(|error| error.to_string())
}

fn add_library_root_cli(arguments: &[String]) -> Result<serde_json::Value, String> {
    let db_path = database_argument(arguments)?;
    let path = named_argument(arguments, "--path")?;
    let label = arguments
        .windows(2)
        .find(|pair| pair[0] == "--label")
        .map(|pair| pair[1].clone());
    serde_json::to_value(add_library_root_headless(&db_path, &path, label)?)
        .map_err(|error| error.to_string())
}

fn remove_library_root_cli(arguments: &[String]) -> Result<serde_json::Value, String> {
    let db_path = database_argument(arguments)?;
    let root_id = numeric_argument(arguments, "--root")?;
    remove_library_root_headless(&db_path, root_id)?;
    Ok(json!({ "rootId": root_id, "removed": true }))
}

fn inspect(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let name: String = connection
        .query_row("SELECT name FROM libraries LIMIT 1", [], |row| row.get(0))
        .map_err(|error| error.to_string())?;
    let images: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM images WHERE status = 'present'",
            [],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    Ok(json!({ "name": name, "images": images }))
}

fn list_roots(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let mut statement = connection
        .prepare("SELECT r.id, r.label, r.absolute_path, r.last_scan_at, COUNT(i.id), SUM(CASE WHEN i.status = 'missing' THEN 1 ELSE 0 END) FROM collection_roots r LEFT JOIN images i ON i.root_id = r.id GROUP BY r.id ORDER BY r.id")
        .map_err(|error| error.to_string())?;
    let roots = statement.query_map([], |row| Ok(json!({ "id": row.get::<_, i64>(0)?, "label": row.get::<_, Option<String>>(1)?, "path": row.get::<_, String>(2)?, "lastScanAt": row.get::<_, Option<i64>>(3)?, "images": row.get::<_, i64>(4)?, "missingImages": row.get::<_, i64>(5)? }))).map_err(|error| error.to_string())?.collect::<Result<Vec<_>, _>>().map_err(|error| error.to_string())?;
    Ok(json!(roots))
}

fn metrics(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let count = |sql| {
        connection
            .query_row(sql, [], |row| row.get::<_, i64>(0))
            .map_err(|error| error.to_string())
    };
    Ok(json!({
        "images": count("SELECT COUNT(*) FROM images WHERE status = 'present'")?,
        "missing": count("SELECT COUNT(*) FROM images WHERE status = 'missing'")?,
        "aiSuggested": count("SELECT COUNT(*) FROM image_ai_tags WHERE review_state = 'suggested'")?,
        "aiAccepted": count("SELECT COUNT(*) FROM image_ai_tags WHERE review_state = 'accepted'")?,
        "ramPlusAnalyzed": count("SELECT COUNT(*) FROM images i WHERE i.status = 'present' AND EXISTS (SELECT 1 FROM image_ai_analysis_state s WHERE s.image_id = i.id AND s.analysis_kind = 'tagging' AND s.model_id = 'ram-plus' AND s.image_modified_at = i.modified_at AND s.state = 'completed')")?,
        "ramPlusPending": count("SELECT COUNT(*) FROM images i WHERE i.status = 'present' AND NOT EXISTS (SELECT 1 FROM image_ai_analysis_state s WHERE s.image_id = i.id AND s.analysis_kind = 'tagging' AND s.model_id = 'ram-plus' AND s.image_modified_at = i.modified_at AND s.state = 'completed')")?,
        "ramPlusFailed": count("SELECT COUNT(*) FROM images i WHERE i.status = 'present' AND EXISTS (SELECT 1 FROM image_ai_analysis_state s WHERE s.image_id = i.id AND s.analysis_kind = 'tagging' AND s.model_id = 'ram-plus' AND s.image_modified_at = i.modified_at AND s.state = 'failed') AND NOT EXISTS (SELECT 1 FROM image_ai_analysis_state s WHERE s.image_id = i.id AND s.analysis_kind = 'tagging' AND s.model_id = 'ram-plus' AND s.image_modified_at = i.modified_at AND s.state = 'completed')")?,
        "cullSessions": count("SELECT COUNT(*) FROM cull_sessions")?,
        "cullOverrides": count("SELECT COUNT(*) FROM cull_decision_events")?
    }))
}

fn run_catalog_scan_cli(arguments: &[String]) -> Result<serde_json::Value, String> {
    let db_path = database_argument(arguments)?;
    let root_id = numeric_argument(arguments, "--root")?;
    let recursive = !arguments
        .iter()
        .any(|argument| argument == "--non-recursive");
    let job_id = rapidraw_lib::create_background_job(
        &db_path,
        "catalog_scan",
        json!({ "rootId": root_id, "recursive": recursive, "headless": true }),
    )?;
    let control = BackgroundJobControl::new();
    let progress_db_path = db_path.clone();
    let progress_job_id = job_id.clone();
    let result = scan_library_root_headless(
        root_id,
        recursive,
        db_path.clone(),
        rapidraw_lib::AppSettings::default(),
        control,
        move |progress| {
            let state = if progress.message == "Indexing paused" {
                "paused"
            } else {
                "running"
            };
            let _ = rapidraw_lib::update_job(
                &progress_db_path,
                &progress_job_id,
                state,
                &progress.message,
                progress.current as i64,
                progress.total as i64,
                progress.current_path.as_deref(),
                None,
            );
        },
    );
    match result {
        Ok(scan) => {
            rapidraw_lib::update_job(
                &db_path,
                &job_id,
                "completed",
                "Catalog scan complete",
                scan.scanned as i64,
                scan.scanned as i64,
                None,
                None,
            )?;
            Ok(json!({
                "jobId": job_id,
                "state": "completed",
                "rootId": scan.root_id,
                "scanned": scan.scanned,
                "updated": scan.inserted_or_updated,
                "missingMarked": scan.missing_marked,
            }))
        }
        Err(error) => {
            let state = if error == "Catalog scan cancelled" {
                "cancelled"
            } else {
                "failed"
            };
            let _ = rapidraw_lib::update_job(
                &db_path,
                &job_id,
                state,
                &error,
                0,
                0,
                None,
                Some(&error),
            );
            Err(error)
        }
    }
}

fn run_catalog_thumbnails_cli(arguments: &[String]) -> Result<serde_json::Value, String> {
    let db_path = database_argument(arguments)?;
    let root_id = optional_numeric_argument(arguments, "--root")?;
    let force = arguments.iter().any(|arg| arg == "--force");
    let thumb_cache_dir = arguments
        .windows(2)
        .find(|pair| pair[0] == "--thumb-dir")
        .map(|pair| PathBuf::from(&pair[1]))
        .unwrap_or_else(|| {
            db_path
                .parent()
                .map(|p| p.join("thumbnails"))
                .unwrap_or_else(|| PathBuf::from("thumbnails"))
        });
    let job_id = rapidraw_lib::create_background_job(
        &db_path,
        "thumbnail_generation",
        json!({ "rootId": root_id, "forceRegenerate": force, "headless": true }),
    )?;
    let control = BackgroundJobControl::new();
    let progress_db_path = db_path.clone();
    let progress_job_id = job_id.clone();
    let result = rapidraw_lib::run_catalog_thumbnail_generation_headless(
        db_path.clone(),
        root_id,
        thumb_cache_dir,
        force,
        control,
        move |current, total, current_item| {
            let _ = rapidraw_lib::update_job(
                &progress_db_path,
                &progress_job_id,
                "running",
                "Generating thumbnail",
                current as i64,
                total as i64,
                current_item,
                None,
            );
        },
    );
    match result {
        Ok(stats) => {
            let summary_message = format!(
                "Thumbnail generation complete: {} generated, {} skipped, {} failed",
                stats.generated, stats.skipped, stats.failed
            );
            let state = if stats.total > 0 && stats.generated + stats.skipped == 0 {
                "failed"
            } else {
                "completed"
            };
            let error_str = if !stats.failure_reasons.is_empty() {
                Some(stats.failure_reasons.join("; "))
            } else {
                None
            };
            rapidraw_lib::update_job(
                &db_path,
                &job_id,
                state,
                &summary_message,
                (stats.generated + stats.skipped) as i64,
                stats.total as i64,
                None,
                error_str.as_deref(),
            )?;
            Ok(json!({
                "jobId": job_id,
                "state": state,
                "rootId": root_id,
                "total": stats.total,
                "generated": stats.generated,
                "skipped": stats.skipped,
                "failed": stats.failed,
            }))
        }
        Err(error) => {
            let state = if error == "Thumbnail generation cancelled" {
                "cancelled"
            } else {
                "failed"
            };
            let (current, total, current_item) = Connection::open(&db_path)
                .ok()
                .and_then(|conn| rapidraw_lib::read_job_progress(&conn, &job_id).ok())
                .unwrap_or((0, 0, None));
            let _ = rapidraw_lib::update_job(
                &db_path,
                &job_id,
                state,
                &error,
                current,
                total,
                current_item.as_deref(),
                Some(&error),
            );
            Err(error)
        }
    }
}

fn run_catalog_metadata_cli(arguments: &[String]) -> Result<serde_json::Value, String> {
    let db_path = database_argument(arguments)?;
    let root_id = optional_numeric_argument(arguments, "--root")?;
    let job_id = rapidraw_lib::create_background_job(
        &db_path,
        "metadata_extraction",
        json!({ "rootId": root_id, "headless": true }),
    )?;
    let control = BackgroundJobControl::new();
    let progress_db_path = db_path.clone();
    let progress_job_id = job_id.clone();
    let result = rapidraw_lib::run_catalog_metadata_extraction_headless(
        db_path.clone(),
        root_id,
        rapidraw_lib::AppSettings::default(),
        control,
        move |current, total, current_item| {
            let _ = rapidraw_lib::update_job(
                &progress_db_path,
                &progress_job_id,
                "running",
                "Extracting metadata",
                current as i64,
                total as i64,
                current_item,
                None,
            );
        },
    );
    match result {
        Ok(stats) => {
            let summary_message = format!(
                "Metadata extraction complete: {} processed, {} failed",
                stats.processed, stats.failed
            );
            let state = if stats.total > 0 && stats.processed == 0 {
                "failed"
            } else {
                "completed"
            };
            let error_str = if !stats.failure_reasons.is_empty() {
                Some(stats.failure_reasons.join("; "))
            } else {
                None
            };
            rapidraw_lib::update_job(
                &db_path,
                &job_id,
                state,
                &summary_message,
                stats.processed as i64,
                stats.total as i64,
                None,
                error_str.as_deref(),
            )?;
            Ok(json!({
                "jobId": job_id,
                "state": state,
                "rootId": root_id,
                "total": stats.total,
                "processed": stats.processed,
                "failed": stats.failed,
            }))
        }
        Err(error) => {
            let state = if error == "Metadata extraction cancelled" {
                "cancelled"
            } else {
                "failed"
            };
            let (current, total, current_item) = Connection::open(&db_path)
                .ok()
                .and_then(|conn| rapidraw_lib::read_job_progress(&conn, &job_id).ok())
                .unwrap_or((0, 0, None));
            let _ = rapidraw_lib::update_job(
                &db_path,
                &job_id,
                state,
                &error,
                current,
                total,
                current_item.as_deref(),
                Some(&error),
            );
            Err(error)
        }
    }
}

fn list_jobs(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let state = arguments
        .windows(2)
        .find(|pair| pair[0] == "--state")
        .map(|pair| pair[1].as_str());
    let sql = if state.is_some() {
        "SELECT id, kind, state, current, total, message FROM background_jobs WHERE state = ?1 ORDER BY created_at DESC LIMIT 100"
    } else {
        "SELECT id, kind, state, current, total, message FROM background_jobs ORDER BY created_at DESC LIMIT 100"
    };
    let mut statement = connection.prepare(sql).map_err(|error| error.to_string())?;
    let jobs = if let Some(state) = state {
        statement.query_map([state], |row| Ok(json!({ "id": row.get::<_, String>(0)?, "kind": row.get::<_, String>(1)?, "state": row.get::<_, String>(2)?, "current": row.get::<_, i64>(3)?, "total": row.get::<_, i64>(4)?, "message": row.get::<_, String>(5)? }))).map_err(|error| error.to_string())?.collect::<Result<Vec<_>, _>>().map_err(|error| error.to_string())?
    } else {
        statement.query_map([], |row| Ok(json!({ "id": row.get::<_, String>(0)?, "kind": row.get::<_, String>(1)?, "state": row.get::<_, String>(2)?, "current": row.get::<_, i64>(3)?, "total": row.get::<_, i64>(4)?, "message": row.get::<_, String>(5)? }))).map_err(|error| error.to_string())?.collect::<Result<Vec<_>, _>>().map_err(|error| error.to_string())?
    };
    Ok(json!(jobs))
}

fn show_job(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let job_id = named_argument(arguments, "--id")?;
    let job = connection
        .query_row(
            "SELECT id, kind, state, root_id, payload_json, current, total, current_item, message, error, created_at, updated_at, completed_at FROM background_jobs WHERE id = ?1",
            [&job_id],
            |row| {
                let payload_json: String = row.get(4)?;
                let payload = serde_json::from_str::<serde_json::Value>(&payload_json)
                    .unwrap_or_else(|_| json!({ "invalidPayload": true, "raw": payload_json }));
                Ok(json!({
                    "id": row.get::<_, String>(0)?,
                    "kind": row.get::<_, String>(1)?,
                    "state": row.get::<_, String>(2)?,
                    "rootId": row.get::<_, Option<i64>>(3)?,
                    "payload": payload,
                    "current": row.get::<_, i64>(5)?,
                    "total": row.get::<_, i64>(6)?,
                    "currentItem": row.get::<_, Option<String>>(7)?,
                    "message": row.get::<_, String>(8)?,
                    "error": row.get::<_, Option<String>>(9)?,
                    "createdAt": row.get::<_, i64>(10)?,
                    "updatedAt": row.get::<_, i64>(11)?,
                    "completedAt": row.get::<_, Option<i64>>(12)?,
                }))
            },
        )
        .map_err(|error| format!("Could not load job {job_id}: {error}"))?;
    let mut statement = connection
        .prepare(
            "SELECT id, state, message, current, total, created_at FROM background_job_events WHERE job_id = ?1 ORDER BY id DESC LIMIT 200",
        )
        .map_err(|error| error.to_string())?;
    let events = statement
        .query_map([&job_id], |row| {
            Ok(json!({
                "id": row.get::<_, i64>(0)?,
                "state": row.get::<_, String>(1)?,
                "message": row.get::<_, String>(2)?,
                "current": row.get::<_, i64>(3)?,
                "total": row.get::<_, i64>(4)?,
                "createdAt": row.get::<_, i64>(5)?,
            }))
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let mut job = job;
    job["events"] = json!(events);
    Ok(job)
}

fn face_status(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let detected: i64 = connection
        .query_row("SELECT COUNT(*) FROM faces", [], |row| row.get(0))
        .map_err(|error| error.to_string())?;
    let embedded: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM faces WHERE embedding_id IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    let confirmed: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM faces WHERE review_state = 'confirmed'",
            [],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    let clusters: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM face_clusters WHERE state = 'unreviewed'",
            [],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    Ok(
        json!({ "detected": detected, "embedded": embedded, "confirmed": confirmed, "unknownClusters": clusters }),
    )
}

fn face_clusters(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let mut statement = connection.prepare("SELECT c.id, COUNT(m.face_id) FROM face_clusters c JOIN face_cluster_members m ON m.cluster_id = c.id WHERE c.state = 'unreviewed' GROUP BY c.id ORDER BY COUNT(m.face_id) DESC").map_err(|error| error.to_string())?;
    let clusters = statement
        .query_map([], |row| {
            Ok(json!({ "id": row.get::<_, i64>(0)?, "faces": row.get::<_, i64>(1)? }))
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    Ok(json!(clusters))
}

fn list_people(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let mut statement = connection
        .prepare(
            "SELECT p.id, p.display_name, p.state, p.merged_into_person_id, COUNT(f.id)
             FROM people p
             LEFT JOIN faces f ON f.person_id = p.id AND f.review_state = 'confirmed'
             GROUP BY p.id
             ORDER BY p.display_name COLLATE NOCASE",
        )
        .map_err(|error| error.to_string())?;
    let people = statement
        .query_map([], |row| {
            Ok(json!({
                "id": row.get::<_, i64>(0)?,
                "name": row.get::<_, String>(1)?,
                "state": row.get::<_, String>(2)?,
                "mergedIntoPersonId": row.get::<_, Option<i64>>(3)?,
                "confirmedFaces": row.get::<_, i64>(4)?,
            }))
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    Ok(json!(people))
}

fn list_person_images(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let person_id = numeric_argument(arguments, "--person")?;
    let mut statement = connection
        .prepare(
            "SELECT DISTINCT i.id, r.absolute_path || '/' || i.relative_path, i.file_name, f.detector_confidence
             FROM faces f
             JOIN images i ON i.id = f.image_id
             JOIN collection_roots r ON r.id = i.root_id
             WHERE f.person_id = ?1 AND f.review_state = 'confirmed' AND i.status = 'present'
             ORDER BY i.relative_path",
        )
        .map_err(|error| error.to_string())?;
    let images = statement
        .query_map([person_id], |row| {
            Ok(json!({
                "imageId": row.get::<_, i64>(0)?,
                "path": row.get::<_, String>(1)?,
                "fileName": row.get::<_, String>(2)?,
                "faceConfidence": row.get::<_, f64>(3)?,
            }))
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    Ok(json!({ "personId": person_id, "images": images }))
}

fn run_face_detection_cli(arguments: &[String]) -> Result<serde_json::Value, String> {
    let db_path = database_argument(arguments)?;
    let face_models_dir = PathBuf::from(named_argument(arguments, "--face-models-dir")?);
    let root_id = optional_numeric_argument(arguments, "--root")?;
    let policy = optional_face_model_policy(arguments)?;
    let runtime = resolve_face_runtime_for_cli(arguments, &face_models_dir)?;
    let job_id = rapidraw_lib::create_background_job(
        &db_path,
        "face_detection",
        json!({ "rootId": root_id, "modelPackId": runtime.pack_id.as_str(), "policy": policy, "headless": true }),
    )?;
    let control = BackgroundJobControl::new();
    if let Err(error) = run_face_detection_headless_for_pack(
        &db_path,
        root_id,
        runtime.pack_id.as_str(),
        &runtime.detector,
        &job_id,
        &control,
        Arc::new(tokio::sync::Semaphore::new(1)),
    ) {
        let state = if error == "Face detection cancelled" {
            "cancelled"
        } else {
            "failed"
        };
        let _ =
            rapidraw_lib::update_job(&db_path, &job_id, state, &error, 0, 0, None, Some(&error));
        return Err(error);
    }
    Ok(json!({ "jobId": job_id, "state": "completed" }))
}

fn run_face_recognition_cli(arguments: &[String]) -> Result<serde_json::Value, String> {
    let db_path = database_argument(arguments)?;
    let face_models_dir = PathBuf::from(named_argument(arguments, "--face-models-dir")?);
    let root_id = optional_numeric_argument(arguments, "--root")?;
    let policy = optional_face_model_policy(arguments)?;
    let runtime = resolve_face_runtime_for_cli(arguments, &face_models_dir)?;
    let job_id = rapidraw_lib::create_background_job(
        &db_path,
        "face_recognition",
        json!({ "rootId": root_id, "modelPackId": runtime.pack_id.as_str(), "policy": policy, "headless": true }),
    )?;
    let control = BackgroundJobControl::new();
    if let Err(error) = run_face_recognition_headless_for_pack(
        &db_path,
        root_id,
        runtime.pack_id.as_str(),
        &runtime.recognizer,
        &job_id,
        &control,
        Arc::new(tokio::sync::Semaphore::new(1)),
    ) {
        let state = if error == "Face recognition cancelled" {
            "cancelled"
        } else {
            "failed"
        };
        let _ =
            rapidraw_lib::update_job(&db_path, &job_id, state, &error, 0, 0, None, Some(&error));
        return Err(error);
    }
    Ok(json!({ "jobId": job_id, "state": "completed" }))
}

fn tag_status(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let count = |state: &str| {
        connection
            .query_row(
                "SELECT COUNT(*) FROM image_ai_tags WHERE review_state = ?1",
                [state],
                |row| row.get::<_, i64>(0),
            )
            .map_err(|error| error.to_string())
    };
    Ok(
        json!({ "suggested": count("suggested")?, "accepted": count("accepted")?, "rejected": count("rejected")? }),
    )
}

fn top_tags(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let mut statement = connection.prepare("SELECT t.name, COUNT(DISTINCT iat.image_id) FROM image_ai_tags iat JOIN tags t ON t.id = iat.tag_id WHERE iat.review_state <> 'rejected' GROUP BY t.id ORDER BY COUNT(DISTINCT iat.image_id) DESC, t.name LIMIT 50").map_err(|error| error.to_string())?;
    let tags = statement
        .query_map([], |row| {
            Ok(json!({ "tag": row.get::<_, String>(0)?, "images": row.get::<_, i64>(1)? }))
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    Ok(json!(tags))
}

fn review_tag_suggestion(arguments: &[String]) -> Result<serde_json::Value, String> {
    let db_path = database_argument(arguments)?;
    let id = numeric_argument(arguments, "--id")?;
    let state = named_argument(arguments, "--state")?;
    rapidraw_lib::review_ai_tag_headless(&db_path, id, &state)?;
    Ok(json!({ "id": id, "state": state }))
}

fn list_species_suggestions(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let state = arguments
        .windows(2)
        .find(|pair| pair[0] == "--state")
        .map(|pair| pair[1].as_str())
        .unwrap_or("suggested");
    if !matches!(state, "suggested" | "accepted" | "rejected") {
        return Err("--state must be suggested, accepted, or rejected".to_string());
    }
    let mut statement = connection
        .prepare(
            "SELECT s.id, s.image_id, r.absolute_path || '/' || i.relative_path,
                    s.scientific_name, s.common_name, s.taxon_rank, s.confidence, s.model_id,
                    s.review_state
             FROM species_classifications s
             JOIN images i ON i.id = s.image_id
             JOIN collection_roots r ON r.id = i.root_id
             WHERE s.review_state = ?1 AND i.status = 'present'
             ORDER BY s.confidence DESC, s.id
             LIMIT 500",
        )
        .map_err(|error| error.to_string())?;
    let suggestions = statement
        .query_map([state], |row| {
            Ok(json!({
                "id": row.get::<_, i64>(0)?,
                "imageId": row.get::<_, i64>(1)?,
                "path": row.get::<_, String>(2)?,
                "scientificName": row.get::<_, String>(3)?,
                "commonName": row.get::<_, Option<String>>(4)?,
                "taxonRank": row.get::<_, Option<String>>(5)?,
                "confidence": row.get::<_, f64>(6)?,
                "modelId": row.get::<_, String>(7)?,
                "state": row.get::<_, String>(8)?,
            }))
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    Ok(json!(suggestions))
}

fn review_species_suggestion(arguments: &[String]) -> Result<serde_json::Value, String> {
    let db_path = database_argument(arguments)?;
    let id = numeric_argument(arguments, "--id")?;
    let state = named_argument(arguments, "--state")?;
    rapidraw_lib::review_species_headless(&db_path, id, &state)?;
    Ok(json!({ "id": id, "state": state }))
}

fn list_collections(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let mut statement = connection
        .prepare("SELECT id, name, query_json, created_at, updated_at FROM smart_collections ORDER BY name COLLATE NOCASE")
        .map_err(|error| error.to_string())?;
    let collections = statement
        .query_map([], |row| {
            let query_json: String = row.get(2)?;
            let query = serde_json::from_str::<serde_json::Value>(&query_json)
                .unwrap_or_else(|_| json!({ "invalidQuery": true, "raw": query_json }));
            Ok(json!({
                "id": row.get::<_, i64>(0)?,
                "name": row.get::<_, String>(1)?,
                "query": query,
                "createdAt": row.get::<_, i64>(3)?,
                "updatedAt": row.get::<_, i64>(4)?,
            }))
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    Ok(json!(collections))
}

fn show_collection(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let name = named_argument(arguments, "--name")?;
    connection
        .query_row(
            "SELECT id, name, query_json, created_at, updated_at FROM smart_collections WHERE name = ?1 COLLATE NOCASE",
            [name],
            |row| {
                let query_json: String = row.get(2)?;
                let query = serde_json::from_str::<serde_json::Value>(&query_json).map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(
                        query_json.len(),
                        rusqlite::types::Type::Text,
                        Box::new(error),
                    )
                })?;
                Ok(json!({
                    "id": row.get::<_, i64>(0)?,
                    "name": row.get::<_, String>(1)?,
                    "query": query,
                    "createdAt": row.get::<_, i64>(3)?,
                    "updatedAt": row.get::<_, i64>(4)?,
                }))
            },
        )
        .map_err(|error| error.to_string())
}

fn cull_sessions(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let mut statement = connection
        .prepare("SELECT id, root_id, scope_path, state, total_count, rejected_count, created_at, updated_at FROM cull_sessions ORDER BY updated_at DESC LIMIT 100")
        .map_err(|error| error.to_string())?;
    let sessions = statement
        .query_map([], |row| {
            Ok(json!({
                "id": row.get::<_, i64>(0)?,
                "rootId": row.get::<_, Option<i64>>(1)?,
                "scopePath": row.get::<_, String>(2)?,
                "state": row.get::<_, String>(3)?,
                "total": row.get::<_, i64>(4)?,
                "rejected": row.get::<_, i64>(5)?,
                "createdAt": row.get::<_, i64>(6)?,
                "updatedAt": row.get::<_, i64>(7)?,
            }))
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    Ok(json!(sessions))
}

fn cull_decisions(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let session_id = numeric_argument(arguments, "--session")?;
    let mut statement = connection
        .prepare("SELECT representative_path, proposed_status, final_status, quality_score, reason FROM cull_decisions WHERE session_id = ?1 ORDER BY quality_score DESC")
        .map_err(|error| error.to_string())?;
    let decisions = statement
        .query_map([session_id], |row| {
            Ok(json!({
                "path": row.get::<_, String>(0)?,
                "proposed": row.get::<_, String>(1)?,
                "final": row.get::<_, String>(2)?,
                "qualityScore": row.get::<_, f64>(3)?,
                "reason": row.get::<_, String>(4)?,
            }))
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    Ok(json!(decisions))
}

fn run_cull_analysis_cli(arguments: &[String]) -> Result<serde_json::Value, String> {
    let db_path = database_argument(arguments)?;
    let root_id = numeric_argument(arguments, "--root")?;
    let similarity_threshold = arguments
        .windows(2)
        .find(|pair| pair[0] == "--similarity-threshold")
        .map(|pair| {
            pair[1]
                .parse::<u32>()
                .map_err(|_| "--similarity-threshold must be an integer".to_string())
        })
        .transpose()?
        .unwrap_or(CullingSettings::default().similarity_threshold);
    let blur_threshold = arguments
        .windows(2)
        .find(|pair| pair[0] == "--blur-threshold")
        .map(|pair| {
            pair[1]
                .parse::<f64>()
                .map_err(|_| "--blur-threshold must be a number".to_string())
        })
        .transpose()?
        .unwrap_or(CullingSettings::default().blur_threshold);
    let settings = CullingSettings {
        similarity_threshold,
        blur_threshold,
        group_similar: !arguments
            .iter()
            .any(|argument| argument == "--no-group-similar"),
        filter_blurry: !arguments
            .iter()
            .any(|argument| argument == "--no-filter-blurry"),
        ..CullingSettings::default()
    };

    let connection = Connection::open(&db_path).map_err(|error| error.to_string())?;
    let root_path: String = connection
        .query_row(
            "SELECT absolute_path FROM collection_roots WHERE id = ?1",
            [root_id],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    let mut statement = connection
        .prepare(
            "SELECT ?1 || '/' || relative_path FROM images WHERE root_id = ?2 AND status = 'present' ORDER BY relative_path",
        )
        .map_err(|error| error.to_string())?;
    let paths = statement
        .query_map(rusqlite::params![root_path, root_id], |row| {
            row.get::<_, String>(0)
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    drop(statement);
    drop(connection);

    let candidates =
        resolve_auto_cull_path_candidates(paths, &rapidraw_lib::AppSettings::default());
    let representative_paths = candidates
        .iter()
        .map(|candidate| candidate.representative_path.clone())
        .collect::<Vec<_>>();
    let job_id = rapidraw_lib::create_background_job(
        &db_path,
        "cull_analysis",
        json!({ "rootId": root_id, "settings": settings.clone() }),
    )?;
    rapidraw_lib::update_job(
        &db_path,
        &job_id,
        "running",
        "Starting technical culling analysis",
        0,
        representative_paths.len() as i64,
        None,
        None,
    )?;
    let report = match cull_images_headless(
        representative_paths,
        settings.clone(),
        rapidraw_lib::AppSettings::default(),
        |current, total, path| {
            let _ = rapidraw_lib::update_job(
                &db_path,
                &job_id,
                "running",
                "Analyzing images",
                current as i64,
                total as i64,
                path,
                None,
            );
        },
    ) {
        Ok(report) => report,
        Err(error) => {
            let _ = rapidraw_lib::update_job(
                &db_path,
                &job_id,
                "failed",
                &error,
                0,
                0,
                None,
                Some(&error),
            );
            return Err(error);
        }
    };

    let mut duplicate_of = HashMap::new();
    for group in &report.suggestions.similar_groups {
        for duplicate in &group.duplicates {
            duplicate_of.insert(duplicate.path.clone(), group.representative.path.clone());
        }
    }
    let blurry = report
        .suggestions
        .blurry_images
        .iter()
        .map(|image| image.path.as_str())
        .collect::<std::collections::HashSet<_>>();
    let decisions = report
        .analyses
        .iter()
        .map(|analysis| {
            let (keep, reason, factors) = if let Some(representative) = duplicate_of.get(&analysis.path) {
                (
                    false,
                    format!("duplicate_of:{representative}"),
                    json!([{ "id": "duplicate", "label": "Near duplicate", "impact": "reject", "detail": format!("Lower-ranked than {representative}") }]),
                )
            } else if blurry.contains(analysis.path.as_str()) {
                (
                    false,
                    "blurry".to_string(),
                    json!([{ "id": "sharpness", "label": "Low sharpness", "impact": "reject", "detail": format!("Laplacian sharpness {:.1}; threshold {:.1}", analysis.sharpness_metric, settings.blur_threshold) }]),
                )
            } else {
                (
                    true,
                    "unique".to_string(),
                    json!([{ "id": "technical_quality", "label": "Technical quality", "impact": "context", "detail": format!("Score {:.2}", analysis.quality_score) }]),
                )
            };
            (
                analysis.path.clone(),
                keep,
                reason,
                analysis.quality_score,
                factors.to_string(),
            )
        })
        .collect::<Vec<_>>();
    let session_id = rapidraw_lib::record_cull_session(
        &db_path,
        &root_path,
        &serde_json::to_string(&settings).map_err(|error| error.to_string())?,
        &decisions,
    )?;
    rapidraw_lib::update_job(
        &db_path,
        &job_id,
        "completed",
        "Technical culling analysis complete",
        report.analyses.len() as i64,
        candidates.len() as i64,
        None,
        None,
    )?;
    Ok(json!({
        "jobId": job_id,
        "sessionId": session_id,
        "state": "completed",
        "logicalCaptures": candidates,
        "report": report,
    }))
}

fn export_tag_suggestions(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let mut statement = connection
        .prepare("SELECT iat.rowid, r.absolute_path || '/' || i.relative_path, t.name, iat.confidence, iat.model_id FROM image_ai_tags iat JOIN images i ON i.id = iat.image_id JOIN collection_roots r ON r.id = i.root_id JOIN tags t ON t.id = iat.tag_id WHERE i.status = 'present' AND iat.review_state = 'suggested' ORDER BY iat.confidence DESC")
        .map_err(|error| error.to_string())?;
    let suggestions = statement
        .query_map([], |row| {
            Ok(json!({
                "id": row.get::<_, i64>(0)?,
                "path": row.get::<_, String>(1)?,
                "tag": row.get::<_, String>(2)?,
                "confidence": row.get::<_, f64>(3)?,
                "modelId": row.get::<_, String>(4)?,
            }))
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    Ok(json!(suggestions))
}

fn run_ram_plus_tagging_cli(arguments: &[String]) -> Result<serde_json::Value, String> {
    let db_path = database_argument(arguments)?;
    let models_dir = PathBuf::from(named_argument(arguments, "--models-dir")?);
    let max_tags = arguments
        .windows(2)
        .find(|pair| pair[0] == "--max-tags")
        .map(|pair| {
            pair[1]
                .parse::<usize>()
                .map_err(|_| "--max-tags must be an integer between 1 and 100".to_string())
        })
        .transpose()?
        .unwrap_or(20);
    if !(1..=100).contains(&max_tags) {
        return Err("--max-tags must be between 1 and 100".to_string());
    }
    let include_bioclip = arguments
        .iter()
        .any(|argument| argument == "--with-bioclip");
    let job_id = rapidraw_lib::create_background_job(
        &db_path,
        "ram_plus_tagging",
        json!({
            "modelId": "ram-plus",
            "modelRevision": "onnx-v1",
            "headless": true,
            "includeBioClip": include_bioclip,
        }),
    )?;
    let control = BackgroundJobControl::new();
    if let Err(error) = run_catalog_ram_plus_tagging_headless(
        &db_path,
        &models_dir,
        max_tags,
        include_bioclip,
        &job_id,
        &control,
        Arc::new(tokio::sync::Semaphore::new(1)),
    ) {
        let state = if error == "RAM++ tagging cancelled" {
            "cancelled"
        } else {
            "failed"
        };
        let _ =
            rapidraw_lib::update_job(&db_path, &job_id, state, &error, 0, 0, None, Some(&error));
        return Err(error);
    }
    Ok(json!({ "jobId": job_id, "state": "completed" }))
}

fn verify_model(arguments: &[String]) -> Result<serde_json::Value, String> {
    let model_id = named_argument(arguments, "--id")?;
    let visual_pack = visual_model_packs().into_iter().find(|p| p.id == model_id);
    let face_pack = rapidraw_lib::face_model_registry::face_model_packs()
        .into_iter()
        .find(|p| p.id == model_id);
    match (visual_pack, face_pack) {
        (Some(pack), _) => Ok(json!({
            "id": pack.id,
            "displayName": pack.display_name,
            "task": pack.task,
            "runnable": pack.id == "ram-plus-onnx",
            "license": pack.license_name,
            "sourceUrl": pack.model_source_url,
        })),
        (_, Some(pack)) => Ok(json!({
            "id": pack.id,
            "displayName": pack.display_name,
            "detector": pack.detector,
            "recognizer": pack.recognizer,
            "runnable": pack.runtime_support == FaceModelRuntimeSupport::Supported,
            "license": pack.license_name,
            "sourceUrl": pack.model_source_url,
        })),
        (None, None) => Err(format!("Unknown model: {model_id}")),
    }
}

fn sha256_file(path: &std::path::Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(hex::encode(hasher.finalize()))
}

fn verify_visual_runtime(directory: &std::path::Path, model_id: &str) -> Result<(), String> {
    let model_name = match model_id {
        "ram-plus-onnx" => "model.onnx",
        rapidraw_lib::visual_model_registry::BIOCLIP_V2_MODEL_ID => "vision_encoder.onnx",
        rapidraw_lib::visual_model_registry::RAWNIND_MODEL_ID => {
            rapidraw_lib::visual_model_registry::RAWNIND_MODEL_FILE_NAME
        }
        rapidraw_lib::visual_model_registry::NAFNET_MODEL_ID => {
            rapidraw_lib::visual_model_registry::NAFNET_MODEL_FILE_NAME
        }
        _ => return Err("No runtime adapter is available in this build".to_string()),
    };
    let model_path = directory.join(model_name);
    ort::session::Session::builder()
        .and_then(|builder| builder.commit_from_file(&model_path))
        .map_err(|error| format!("ONNX session could not be created: {error}"))?;

    if model_id == rapidraw_lib::visual_model_registry::BIOCLIP_V2_MODEL_ID {
        let labels = directory.join("species_labels.json");
        let manifest = directory.join("taxonomy_manifest.json");
        let labels_json = fs::read_to_string(labels)
            .map_err(|error| format!("BioCLIP taxonomy cannot be read: {error}"))?;
        let taxonomy = serde_json::from_str::<serde_json::Value>(&labels_json)
            .map_err(|error| format!("BioCLIP taxonomy is invalid JSON: {error}"))?;
        if taxonomy.as_array().is_none_or(Vec::is_empty) {
            return Err("BioCLIP taxonomy has no labels".to_string());
        }
        let manifest_json = fs::read_to_string(manifest)
            .map_err(|error| format!("BioCLIP taxonomy manifest cannot be read: {error}"))?;
        let manifest_value = serde_json::from_str::<serde_json::Value>(&manifest_json)
            .map_err(|error| format!("BioCLIP taxonomy manifest is invalid JSON: {error}"))?;
        let parts = manifest_value
            .get("embeddingParts")
            .and_then(serde_json::Value::as_array)
            .ok_or("BioCLIP taxonomy manifest has no embedding parts")?;
        for part in parts {
            let name = part
                .as_str()
                .ok_or("BioCLIP taxonomy manifest has an invalid embedding part")?;
            let embedding_bytes = fs::metadata(directory.join(name))
                .map_err(|error| format!("BioCLIP embeddings cannot be read: {error}"))?
                .len();
            if embedding_bytes == 0 || embedding_bytes % 4 != 0 {
                return Err("BioCLIP embeddings are not a packed f32 array".to_string());
            }
        }
    }
    Ok(())
}

fn verify_face_runtime(directory: &std::path::Path, model_id: &str) -> Result<(), String> {
    let (detector, recognizer) = runtime_file_names(FaceModelPackId::try_from(model_id)?);
    for model_name in [detector, recognizer] {
        let model_path = directory.join(model_name);
        if !model_path.is_file() {
            return Err(format!(
                "{model_name} is not installed in {}",
                directory.display()
            ));
        }
        ort::session::Session::builder()
            .and_then(|builder| builder.commit_from_file(&model_path))
            .map_err(|error| format!("{model_name} could not create an ONNX session: {error}"))?;
    }
    Ok(())
}

fn verify_installed_model(arguments: &[String]) -> Result<serde_json::Value, String> {
    let model_id = named_argument(arguments, "--id")?;
    if let Some(pack) = visual_model_packs()
        .into_iter()
        .find(|pack| pack.id == model_id)
    {
        let models_dir = PathBuf::from(named_argument(arguments, "--models-dir")?);
        let directory = models_dir.join(&pack.id);
        let manifest_path = directory.join("manifest.json");
        let manifest_pack_id = fs::read_to_string(&manifest_path)
            .ok()
            .and_then(|contents| serde_json::from_str::<serde_json::Value>(&contents).ok())
            .and_then(|manifest| {
                manifest
                    .get("packId")
                    .and_then(|id| id.as_str())
                    .map(str::to_string)
            });
        let artifacts = pack
            .artifacts
            .iter()
            .map(|artifact| {
                let path = directory.join(&artifact.file_name);
                let metadata = path.metadata().ok();
                json!({
                    "fileName": artifact.file_name,
                    "exists": metadata.as_ref().is_some_and(|metadata| metadata.is_file()),
                    "bytes": metadata.map(|metadata| metadata.len()),
                })
            })
            .collect::<Vec<_>>();
        let integrity_check = verified_visual_model_pack_dir(&models_dir, &pack.id);
        let installed = integrity_check.is_ok();
        let runtime_check = if installed {
            verify_visual_runtime(&directory, &pack.id)
        } else {
            Err(integrity_check
                .as_ref()
                .err()
                .cloned()
                .unwrap_or_else(|| "Pack is not installed".to_string()))
        };
        return Ok(json!({
            "id": pack.id,
            "type": "visual",
            "installed": installed,
            "runnable": runtime_check.is_ok(),
            "integrity": "manifest-artifact-digest-and-runtime-session",
            "integrityValidationError": integrity_check.err(),
            "runtimeValidationError": runtime_check.err(),
            "manifestPackId": manifest_pack_id,
            "artifacts": artifacts,
        }));
    }

    if let Some(pack) = face_model_packs()
        .into_iter()
        .find(|pack| pack.id == model_id)
    {
        let models_dir = PathBuf::from(named_argument(arguments, "--face-models-dir")?);
        let directory = models_dir.join(&pack.id);
        let manifest_path = directory.join("rapidraw-face-model.json");
        let manifest = fs::read(&manifest_path)
            .ok()
            .and_then(|contents| serde_json::from_slice::<InstalledFaceModelPack>(&contents).ok());
        let artifacts = manifest
            .as_ref()
            .map(|manifest| {
                manifest
                    .artifacts
                    .iter()
                    .map(|artifact| {
                        let path = directory.join(&artifact.file_name);
                        let actual_sha256 = path.is_file().then(|| sha256_file(&path)).transpose();
                        let matches = actual_sha256
                            .as_ref()
                            .ok()
                            .and_then(|actual| actual.as_ref())
                            .is_some_and(|actual| actual == &artifact.sha256);
                        json!({
                            "fileName": artifact.file_name,
                            "exists": path.is_file(),
                            "expectedSha256": artifact.sha256,
                            "actualSha256": actual_sha256.ok().flatten(),
                            "matches": matches,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let installed = manifest
            .as_ref()
            .is_some_and(|manifest| manifest.pack_id == pack.id)
            && !artifacts.is_empty()
            && artifacts
                .iter()
                .all(|artifact| artifact["matches"].as_bool() == Some(true));
        let runtime_check = if installed {
            verify_face_runtime(&directory, &pack.id)
        } else {
            Err("Pack is not installed".to_string())
        };
        return Ok(json!({
            "id": pack.id,
            "type": "face",
            "installed": installed,
            "runnable": runtime_check.is_ok(),
            "integrity": "manifest-sha256-and-runtime-session",
            "runtimeValidationError": runtime_check.err(),
            "manifestPresent": manifest.is_some(),
            "artifacts": artifacts,
        }));
    }

    Err(format!("Unknown model: {model_id}"))
}

fn list_derivatives(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let image_id = numeric_argument(arguments, "--image")?;
    let mut stmt = connection
        .prepare("SELECT id, operation_kind, model_id, model_revision, output_path, output_format, width, height, state, created_at FROM image_derivatives WHERE source_image_id = ?1 ORDER BY created_at DESC")
        .map_err(|e| e.to_string())?;
    let derivatives = stmt
        .query_map([image_id], |row| {
            Ok(json!({
                "id": row.get::<_, i64>(0)?,
                "operationKind": row.get::<_, String>(1)?,
                "modelId": row.get::<_, String>(2)?,
                "modelRevision": row.get::<_, String>(3)?,
                "outputPath": row.get::<_, String>(4)?,
                "outputFormat": row.get::<_, String>(5)?,
                "width": row.get::<_, Option<i64>>(6)?,
                "height": row.get::<_, Option<i64>>(7)?,
                "state": row.get::<_, String>(8)?,
                "createdAt": row.get::<_, i64>(9)?,
            }))
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(json!(derivatives))
}

fn run_restore_cli(arguments: &[String]) -> Result<serde_json::Value, String> {
    let db_path = database_argument(arguments)?;
    let image_id = numeric_argument(arguments, "--image")?;
    let visual_models_dir = PathBuf::from(named_argument(arguments, "--models-dir")?);
    let operation_kind = arguments
        .windows(2)
        .find(|pair| pair[0] == "--operation")
        .map(|pair| pair[1].clone())
        .unwrap_or_else(|| "raw_denoise".to_string());
    let model_id = arguments
        .windows(2)
        .find(|pair| pair[0] == "--model")
        .map(|pair| pair[1].clone())
        .unwrap_or_else(|| {
            if operation_kind == "rgb_denoise" {
                rapidraw_lib::visual_model_registry::NAFNET_MODEL_ID.to_string()
            } else {
                rapidraw_lib::visual_model_registry::RAWNIND_MODEL_ID.to_string()
            }
        });
    let mut recipe = RestorationRecipe {
        operation_kind,
        model_id,
        ..RestorationRecipe::default()
    };
    validate_restoration_recipe(&recipe)?;
    recipe.model_revision =
        visual_model_pack_revision_in_dir(&visual_models_dir, &recipe.model_id)?;
    let job_id = rapidraw_lib::create_background_job(
        &db_path,
        &recipe.operation_kind,
        json!({ "imageId": image_id, "recipe": recipe }),
    )?;
    let control = BackgroundJobControl::new();
    if let Err(error) = run_restoration_worker(
        &db_path,
        image_id,
        &recipe,
        &job_id,
        &control,
        &visual_models_dir,
        Arc::new(tokio::sync::Semaphore::new(1)),
    ) {
        let state = if error == "Restoration cancelled" {
            "cancelled"
        } else {
            "failed"
        };
        let _ =
            rapidraw_lib::update_job(&db_path, &job_id, state, &error, 0, 100, None, Some(&error));
        return Err(error);
    }
    Ok(json!({ "jobId": job_id, "imageId": image_id, "state": "completed" }))
}

/// Reports what our own demosaic pipeline sees for a RAW file, without
/// developing anything - CFA eligibility, ISO, orientation, and which
/// algorithm ISO auto-selection would pick. Use this first when a file
/// unexpectedly falls back to rawler's PPG pipeline.
fn run_raw_inspect_cli(arguments: &[String]) -> Result<serde_json::Value, String> {
    let input_path = named_argument(arguments, "--input")?;
    let bytes = fs::read(&input_path).map_err(|error| error.to_string())?;
    let sensor = rapidraw_lib::custom_raw_pipeline::decode_raw_sensor_data(&bytes)
        .map_err(|error| error.to_string())?;
    let exposure_gain = rapidraw_lib::custom_raw_pipeline::estimate_exposure_gain(&sensor);
    let effective_iso = ((sensor.iso as f32) * exposure_gain).round() as u32;
    let selected = rapidraw_lib::demosaic_algorithms::select_by_iso(effective_iso);
    let pdaf_known = rapidraw_lib::raw_pdaf_data::lookup(&sensor.camera_name).is_some();
    Ok(json!({
        "width": sensor.width,
        "height": sensor.height,
        "iso": sensor.iso,
        "orientation": format!("{:?}", sensor.orientation),
        "isStandardBayer": sensor.is_standard_bayer,
        "activeArea": sensor.active_area,
        "cropArea": sensor.crop_area,
        "autoSelectedAlgorithm": rapidraw_lib::demosaic_algorithms::algorithm_name(selected),
        "cameraName": sensor.camera_name,
        "pdafPatternKnown": pdaf_known,
        "exposureGain": exposure_gain,
        "effectiveIso": effective_iso,
    }))
}

/// Develops a single RAW file through our own demosaic pipeline
/// (AMaZE/IGV/LMMSE/bilinear) and writes a PNG, so it can be diffed
/// pixel-by-pixel or eyeballed against ART/RawTherapee reference output
/// while debugging the pipeline outside the full Tauri app. This is the
/// intended debug entry point for the upcoming raw-denoise, raw-sharpen,
/// and tone-curve work as well - those stages will hang off this same
/// command as additional flags once implemented.
///
/// By default the output is display-ready sRGB (gamma-encoded, like a
/// typical raw-converter preview). `--linear` instead writes the same
/// LINEAR pre-tonemap intermediate the live app pipeline consumes
/// (raw_processing::develop_raw_image's contract), re-encoded to sRGB only
/// for PNG viewability - useful when debugging exposure/highlight-rolloff
/// behavior that happens downstream of demosaic.
///
/// `--denoise` and `--sharpen` accept `auto` (ISO-based suggestion, same
/// pattern as `--demosaic auto`) or an explicit 0..1 amount; `0` (the
/// implicit default) disables the stage entirely. `--sharpen-method`
/// selects `unsharp` (default, classic unsharp mask) or `rld` (Richardson-
/// Lucy deconvolution). `--no-preprocess` disables the raw-domain
/// hot/dead-pixel + CFA-line-banding correction that otherwise always runs
/// before demosaic.
fn run_raw_develop_cli(arguments: &[String]) -> Result<serde_json::Value, String> {
    let input_path = named_argument(arguments, "--input")?;
    let output_path = named_argument(arguments, "--output")?;
    let demosaic_arg = arguments
        .windows(2)
        .find(|pair| pair[0] == "--demosaic")
        .map(|pair| pair[1].clone())
        .unwrap_or_else(|| "auto".to_string());
    let highlight_compression = arguments
        .windows(2)
        .find(|pair| pair[0] == "--highlight-compression")
        .map(|pair| {
            pair[1]
                .parse::<f32>()
                .map_err(|_| "--highlight-compression must be a number".to_string())
        })
        .transpose()?
        .unwrap_or(2.5);
    let linear_intermediate = arguments.iter().any(|argument| argument == "--linear");
    let preprocess = !arguments
        .iter()
        .any(|argument| argument == "--no-preprocess");
    let auto_exposure = !arguments
        .iter()
        .any(|argument| argument == "--no-auto-exposure");

    let bytes = fs::read(&input_path).map_err(|error| error.to_string())?;
    let sensor = rapidraw_lib::custom_raw_pipeline::decode_raw_sensor_data(&bytes)
        .map_err(|error| error.to_string())?;

    let exposure_gain = if auto_exposure {
        rapidraw_lib::custom_raw_pipeline::estimate_exposure_gain(&sensor)
    } else {
        1.0
    };
    let effective_iso = ((sensor.iso as f32) * exposure_gain).round() as u32;

    let algo = if demosaic_arg.eq_ignore_ascii_case("auto") {
        rapidraw_lib::demosaic_algorithms::select_by_iso(effective_iso)
    } else {
        rapidraw_lib::demosaic_algorithms::parse_algorithm_name(&demosaic_arg).ok_or_else(|| {
            format!(
                "unknown --demosaic value '{demosaic_arg}': expected auto|amaze|igv|lmmse|bilinear"
            )
        })?
    };

    let parse_stage_arg = |flag: &str, auto_default: f32| -> Result<f32, String> {
        match arguments
            .windows(2)
            .find(|pair| pair[0] == flag)
            .map(|pair| pair[1].as_str())
        {
            None => Ok(0.0),
            Some("auto") => Ok(auto_default),
            Some(value) => value
                .parse::<f32>()
                .map_err(|_| format!("{flag} must be 'auto' or a number between 0 and 1"))
                .map(|v| v.clamp(0.0, 1.0)),
        }
    };
    let denoise_strength = parse_stage_arg(
        "--denoise",
        rapidraw_lib::raw_denoise::suggest_strength_for_iso(effective_iso),
    )?;
    let sharpen_amount = parse_stage_arg(
        "--sharpen",
        rapidraw_lib::raw_sharpen::suggest_amount_for_iso(effective_iso),
    )?;
    let sharpen_method_arg = arguments
        .windows(2)
        .find(|pair| pair[0] == "--sharpen-method")
        .map(|pair| pair[1].clone())
        .unwrap_or_else(|| "unsharp".to_string());
    let sharpen_method = rapidraw_lib::raw_sharpen::parse_method_name(&sharpen_method_arg)
        .ok_or_else(|| {
            format!("unknown --sharpen-method value '{sharpen_method_arg}': expected unsharp|rld")
        })?;
    let options = rapidraw_lib::custom_raw_pipeline::DevelopOptions {
        preprocess,
        denoise_strength,
        sharpen_amount,
        sharpen_method,
        exposure_gain,
    };

    let image = if linear_intermediate {
        let linear = rapidraw_lib::custom_raw_pipeline::develop_raw_image_custom_with_algorithm(
            &bytes,
            algo,
            highlight_compression,
            &options,
        )
        .map_err(|error| error.to_string())?;
        // Naive gamma+clamp re-encode purely for PNG viewability - the real
        // linear values (which can exceed 1.0 up to highlight_compression)
        // are what the live app's GPU tonemap stage actually consumes.
        let linear_f32 = linear.to_rgba32f();
        image::DynamicImage::ImageRgba8(image::ImageBuffer::from_fn(
            linear.width(),
            linear.height(),
            |x, y| {
                let p = linear_f32.get_pixel(x, y).0;
                let gamma = |v: f32| {
                    let v = v.clamp(0.0, 1.0);
                    if v <= 0.0031308 {
                        v * 12.92
                    } else {
                        1.055 * v.powf(1.0 / 2.4) - 0.055
                    }
                };
                image::Rgba([
                    (gamma(p[0]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    (gamma(p[1]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    (gamma(p[2]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    255,
                ])
            },
        ))
    } else {
        rapidraw_lib::custom_raw_pipeline::develop_raw_custom_with_options(&bytes, algo, &options)
            .map_err(|error| error.to_string())?
    };

    image
        .save(&output_path)
        .map_err(|error| error.to_string())?;

    Ok(json!({
        "input": input_path,
        "output": output_path,
        "width": image.width(),
        "height": image.height(),
        "preprocess": preprocess,
        "denoiseStrength": denoise_strength,
        "sharpenAmount": sharpen_amount,
        "sharpenMethod": rapidraw_lib::raw_sharpen::method_name(sharpen_method),
        "iso": sensor.iso,
        "demosaic": rapidraw_lib::demosaic_algorithms::algorithm_name(algo),
        "linear": linear_intermediate,
        "exposureGain": exposure_gain,
        "effectiveIso": effective_iso,
    }))
}

#[cfg(test)]
mod tests {
    use super::{
        exit_code_for_error, optional_face_model_pack, verify_face_runtime, verify_visual_runtime,
    };
    use rapidraw_lib::face_model_registry::FaceModelPackId;
    use std::path::Path;

    #[test]
    fn cli_errors_use_stable_automation_exit_codes() {
        assert_eq!(exit_code_for_error("--database <path> is required"), 2);
        assert_eq!(exit_code_for_error("Catalog scan cancelled"), 3);
        assert_eq!(exit_code_for_error("database is locked"), 1);
    }

    #[test]
    fn explicit_face_pack_arguments_are_typed_and_reject_unknown_ids() {
        let arguments = vec![
            "faces".to_string(),
            "detect".to_string(),
            "--pack".to_string(),
            FaceModelPackId::InsightFaceAntelopeV2.as_str().to_string(),
        ];
        assert_eq!(
            optional_face_model_pack(&arguments).unwrap(),
            Some(FaceModelPackId::InsightFaceAntelopeV2)
        );
        assert!(
            optional_face_model_pack(&[
                "faces".to_string(),
                "detect".to_string(),
                "--pack".to_string(),
                "not-a-face-pack".to_string(),
            ])
            .is_err()
        );
    }

    #[test]
    fn visual_runtime_verification_never_claims_unknown_models_are_runnable() {
        assert!(verify_visual_runtime(Path::new("/tmp"), "unknown-model").is_err());
    }

    #[test]
    fn face_runtime_verification_never_claims_conversion_packs_are_runnable() {
        assert!(
            verify_face_runtime(
                Path::new("/tmp"),
                FaceModelPackId::InsightFaceBuffaloSc.as_str()
            )
            .is_err()
        );
    }

    #[test]
    fn cli_metrics_reports_artifact_revision_tagging_counts() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("catalog.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute(
            "CREATE TABLE images (id INTEGER PRIMARY KEY, root_id INTEGER, folder_id INTEGER, file_name TEXT, relative_path TEXT, modified_at INTEGER, imported_at INTEGER, updated_at INTEGER, status TEXT)",
            [],
        ).unwrap();
        conn.execute(
            "CREATE TABLE image_ai_tags (id INTEGER PRIMARY KEY, review_state TEXT)",
            [],
        )
        .unwrap();
        conn.execute(
            "CREATE TABLE image_ai_analysis_state (image_id INTEGER, analysis_kind TEXT, model_id TEXT, model_revision TEXT, image_modified_at INTEGER, state TEXT, error_message TEXT, processed_at INTEGER, updated_at INTEGER)",
            [],
        ).unwrap();
        conn.execute("CREATE TABLE cull_sessions (id TEXT)", [])
            .unwrap();
        conn.execute("CREATE TABLE cull_decision_events (id TEXT)", [])
            .unwrap();

        conn.execute("INSERT INTO images(id, root_id, folder_id, file_name, relative_path, modified_at, imported_at, updated_at, status) VALUES(1, 1, 1, 'img1.jpg', 'img1.jpg', 100, 0, 0, 'present')", []).unwrap();
        conn.execute("INSERT INTO images(id, root_id, folder_id, file_name, relative_path, modified_at, imported_at, updated_at, status) VALUES(2, 1, 1, 'img2.jpg', 'img2.jpg', 200, 0, 0, 'present')", []).unwrap();
        // Insert tagging state with sha256:... revision
        conn.execute("INSERT INTO image_ai_analysis_state(image_id, analysis_kind, model_id, model_revision, image_modified_at, state, processed_at, updated_at) VALUES(1, 'tagging', 'ram-plus', 'sha256:abcd1234efgh5678', 100, 'completed', 100, 100)", []).unwrap();
        conn.execute("INSERT INTO image_ai_analysis_state(image_id, analysis_kind, model_id, model_revision, image_modified_at, state, processed_at, updated_at) VALUES(2, 'tagging', 'ram-plus', 'sha256:abcd1234efgh5678', 200, 'failed', 200, 200)", []).unwrap();
        drop(conn);

        let db_path_str = db_path.to_str().unwrap().to_string();
        let result = super::metrics(&["--database".to_string(), db_path_str]).unwrap();
        assert_eq!(result["images"], 2);
        assert_eq!(result["ramPlusAnalyzed"], 1);
        assert_eq!(result["ramPlusPending"], 1);
        assert_eq!(result["ramPlusFailed"], 1);
    }
}
