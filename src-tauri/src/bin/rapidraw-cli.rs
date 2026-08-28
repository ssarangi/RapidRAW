use std::env;
use std::fs;
use std::path::PathBuf;

use rapidraw_lib::BackgroundJobControl;
use rapidraw_lib::face_detection::{run_face_detection_headless, run_face_recognition_headless};
use rapidraw_lib::face_model_registry::installed_face_model_path_in_dir;
use rapidraw_lib::face_model_registry::{InstalledFaceModelPack, face_model_packs};
use rapidraw_lib::image_restoration::{
    RestorationRecipe, run_restoration_worker, validate_restoration_recipe,
};
use rapidraw_lib::resolve_auto_cull_path_candidates;
use rapidraw_lib::scan_library_root_headless;
use rapidraw_lib::tagging::run_catalog_ram_plus_tagging_headless;
use rapidraw_lib::visual_model_registry::visual_model_packs;
use rapidraw_lib::{CullingSettings, cull_images_headless};
use rapidraw_lib::{add_library_root_headless, create_library_headless};
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

fn main() {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let result = match arguments.first().map(String::as_str) {
        Some("library") if arguments.get(1).map(String::as_str) == Some("inspect") => inspect(&arguments),
        Some("library") if arguments.get(1).map(String::as_str) == Some("create") => create_library_cli(&arguments),
        Some("library") if arguments.get(1).map(String::as_str) == Some("add-root") => add_library_root_cli(&arguments),
        Some("library") if arguments.get(1).map(String::as_str) == Some("roots") => list_roots(&arguments),
        Some("library") if arguments.get(1).map(String::as_str) == Some("metrics") => metrics(&arguments),
        Some("library") if arguments.get(1).map(String::as_str) == Some("scan") => run_catalog_scan_cli(&arguments),
        Some("jobs") if arguments.get(1).map(String::as_str) == Some("list") => list_jobs(&arguments),
        Some("jobs") if arguments.get(1).map(String::as_str) == Some("show") => show_job(&arguments),
        Some("faces") if arguments.get(1).map(String::as_str) == Some("status") => face_status(&arguments),
        Some("faces") if arguments.get(1).map(String::as_str) == Some("clusters") => face_clusters(&arguments),
        Some("faces") if arguments.get(1).map(String::as_str) == Some("detect") => run_face_detection_cli(&arguments),
        Some("faces") if arguments.get(1).map(String::as_str) == Some("recognize") => run_face_recognition_cli(&arguments),
        Some("tags") if arguments.get(1).map(String::as_str) == Some("status") => tag_status(&arguments),
        Some("tags") if arguments.get(1).map(String::as_str) == Some("top") => top_tags(&arguments),
        Some("tags") if arguments.get(1).map(String::as_str) == Some("export-suggestions") => export_tag_suggestions(&arguments),
        Some("tags") if arguments.get(1).map(String::as_str) == Some("run") => run_ram_plus_tagging_cli(&arguments),
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
        _ => Err("Usage: rapidraw-cli library create --name <name> --database <catalog.db> | rapidraw-cli library add-root --database <catalog.db> --path <folder> [--label <name>] | rapidraw-cli library inspect|roots|metrics|scan --database <catalog.db> | rapidraw-cli library scan --database <catalog.db> --root <id> [--non-recursive] | rapidraw-cli jobs list --database <catalog.db> | rapidraw-cli jobs show --database <catalog.db> --id <job-id> | rapidraw-cli faces status|clusters --database <catalog.db> | rapidraw-cli faces detect|recognize --database <catalog.db> --face-models-dir <models/face> [--root <id>] | rapidraw-cli tags status|top|export-suggestions|run --database <catalog.db> | rapidraw-cli tags run --database <catalog.db> --models-dir <models/visual> [--max-tags <1-100>] [--with-bioclip] | rapidraw-cli collections list --database <catalog.db> | rapidraw-cli collections show --database <catalog.db> --name <name> | rapidraw-cli cull sessions|decisions|analyze --database <catalog.db> | rapidraw-cli cull analyze --database <catalog.db> --root <id> [--similarity-threshold <n>] [--blur-threshold <n>] | rapidraw-cli models list | rapidraw-cli models info --id <model-id> | rapidraw-cli models verify --id <model-id> [--models-dir <models/visual>|--face-models-dir <models/face>] | rapidraw-cli restore list --database <catalog.db> --image <id> | rapidraw-cli restore run --database <catalog.db> --image <id> --models-dir <models/visual> [--operation raw_denoise|rgb_denoise] [--model <model-id>]".to_string()),
    };
    match result {
        Ok(value) => println!("{}", value),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
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
    Ok(
        json!({ "images": count("SELECT COUNT(*) FROM images WHERE status = 'present'")?, "missing": count("SELECT COUNT(*) FROM images WHERE status = 'missing'")?, "aiSuggested": count("SELECT COUNT(*) FROM image_ai_tags WHERE review_state = 'suggested'")?, "aiAccepted": count("SELECT COUNT(*) FROM image_ai_tags WHERE review_state = 'accepted'")?, "ramPlusAnalyzed": count("SELECT COUNT(*) FROM images i WHERE i.status = 'present' AND EXISTS (SELECT 1 FROM image_ai_analysis_state s WHERE s.image_id = i.id AND s.analysis_kind = 'tagging' AND s.model_id = 'ram-plus' AND s.model_revision = 'onnx-v1' AND s.image_modified_at = i.modified_at AND s.state = 'completed')")?, "ramPlusPending": count("SELECT COUNT(*) FROM images i WHERE i.status = 'present' AND NOT EXISTS (SELECT 1 FROM image_ai_analysis_state s WHERE s.image_id = i.id AND s.analysis_kind = 'tagging' AND s.model_id = 'ram-plus' AND s.model_revision = 'onnx-v1' AND s.image_modified_at = i.modified_at AND s.state = 'completed')")?, "ramPlusFailed": count("SELECT COUNT(*) FROM images i WHERE i.status = 'present' AND EXISTS (SELECT 1 FROM image_ai_analysis_state s WHERE s.image_id = i.id AND s.analysis_kind = 'tagging' AND s.model_id = 'ram-plus' AND s.model_revision = 'onnx-v1' AND s.image_modified_at = i.modified_at AND s.state = 'failed')")?, "cullSessions": count("SELECT COUNT(*) FROM cull_sessions")?, "cullOverrides": count("SELECT COUNT(*) FROM cull_decision_events")? }),
    )
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

fn run_face_detection_cli(arguments: &[String]) -> Result<serde_json::Value, String> {
    let db_path = database_argument(arguments)?;
    let face_models_dir = PathBuf::from(named_argument(arguments, "--face-models-dir")?);
    let root_id = optional_numeric_argument(arguments, "--root")?;
    let model_path = installed_face_model_path_in_dir(
        &face_models_dir,
        "opencv-yunet-sface",
        "face_detection_yunet_2023mar.onnx",
    )?;
    let job_id = rapidraw_lib::create_background_job(
        &db_path,
        "face_detection",
        json!({ "rootId": root_id, "modelPackId": "opencv-yunet-sface", "headless": true }),
    )?;
    let control = BackgroundJobControl::new();
    if let Err(error) = run_face_detection_headless(
        &db_path,
        root_id,
        &model_path,
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
    let model_path = installed_face_model_path_in_dir(
        &face_models_dir,
        "opencv-yunet-sface",
        "face_recognition_sface_2021dec.onnx",
    )?;
    let job_id = rapidraw_lib::create_background_job(
        &db_path,
        "face_recognition",
        json!({ "rootId": root_id, "modelPackId": "opencv-yunet-sface", "headless": true }),
    )?;
    let control = BackgroundJobControl::new();
    if let Err(error) = run_face_recognition_headless(
        &db_path,
        root_id,
        &model_path,
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
        .prepare("SELECT iat.id, r.absolute_path || '/' || i.relative_path, t.name, iat.confidence, iat.model_id FROM image_ai_tags iat JOIN images i ON i.id = iat.image_id JOIN collection_roots r ON r.id = i.root_id JOIN tags t ON t.id = iat.tag_id WHERE i.status = 'present' AND iat.review_state = 'suggested' ORDER BY iat.confidence DESC")
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
            "runnable": pack.id == "opencv-yunet-sface",
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
        let installed = manifest_pack_id.as_deref() == Some(pack.id.as_str())
            && artifacts
                .iter()
                .all(|artifact| artifact["exists"].as_bool() == Some(true));
        let runtime_supported = matches!(
            pack.id.as_str(),
            "ram-plus-onnx" | "bioclip-v1" | "rawnind-utnet2-bayer" | "nafnet-sidd-rgb"
        );
        return Ok(json!({
            "id": pack.id,
            "type": "visual",
            "installed": installed,
            "runnable": installed && runtime_supported,
            "integrity": "manifest-and-artifact-presence",
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
        return Ok(json!({
            "id": pack.id,
            "type": "face",
            "installed": installed,
            "runnable": installed && pack.id == "opencv-yunet-sface",
            "integrity": "manifest-and-sha256",
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
                "nafnet-sidd-rgb".to_string()
            } else {
                "rawnind-utnet2-bayer".to_string()
            }
        });
    let recipe = RestorationRecipe {
        operation_kind,
        model_id,
        ..RestorationRecipe::default()
    };
    validate_restoration_recipe(&recipe)?;
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
