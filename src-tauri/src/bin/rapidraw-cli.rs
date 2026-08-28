use std::env;
use std::path::PathBuf;

use rusqlite::Connection;
use rapidraw_lib::visual_model_registry::visual_model_packs;
use serde_json::json;


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

fn main() {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let result = match arguments.first().map(String::as_str) {
        Some("library") if arguments.get(1).map(String::as_str) == Some("inspect") => inspect(&arguments),
        Some("library") if arguments.get(1).map(String::as_str) == Some("roots") => list_roots(&arguments),
        Some("library") if arguments.get(1).map(String::as_str) == Some("metrics") => metrics(&arguments),
        Some("jobs") if arguments.get(1).map(String::as_str) == Some("list") => list_jobs(&arguments),
        Some("faces") if arguments.get(1).map(String::as_str) == Some("status") => face_status(&arguments),
        Some("faces") if arguments.get(1).map(String::as_str) == Some("clusters") => face_clusters(&arguments),
        Some("tags") if arguments.get(1).map(String::as_str) == Some("status") => tag_status(&arguments),
        Some("tags") if arguments.get(1).map(String::as_str) == Some("top") => top_tags(&arguments),
        Some("tags") if arguments.get(1).map(String::as_str) == Some("export-suggestions") => export_tag_suggestions(&arguments),
        Some("collections") if arguments.get(1).map(String::as_str) == Some("list") => list_collections(&arguments),
        Some("collections") if arguments.get(1).map(String::as_str) == Some("show") => show_collection(&arguments),
        Some("cull") if arguments.get(1).map(String::as_str) == Some("sessions") => cull_sessions(&arguments),
        Some("cull") if arguments.get(1).map(String::as_str) == Some("decisions") => cull_decisions(&arguments),
        Some("models") if arguments.get(1).map(String::as_str) == Some("list") => list_models(),
        Some("models") if arguments.get(1).map(String::as_str) == Some("info") => verify_model(&arguments),
        Some("restore") if arguments.get(1).map(String::as_str) == Some("list") => list_derivatives(&arguments),
        Some("restore") if arguments.get(1).map(String::as_str) == Some("run") => run_restore_cli(&arguments),
        _ => Err("Usage: rapidraw-cli library inspect|roots|metrics --database <catalog.db> | rapidraw-cli jobs list --database <catalog.db> | rapidraw-cli faces status|clusters --database <catalog.db> | rapidraw-cli tags status|top|export-suggestions --database <catalog.db> | rapidraw-cli collections list --database <catalog.db> | rapidraw-cli collections show --database <catalog.db> --name <name> | rapidraw-cli cull sessions --database <catalog.db> | rapidraw-cli cull decisions --database <catalog.db> --session <id> | rapidraw-cli models list | rapidraw-cli models info --id <model-id> | rapidraw-cli restore list --database <catalog.db> --image <id> | rapidraw-cli restore run --database <catalog.db> --image <id>".to_string()),
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
        .query_map([], |row| Ok(json!({
            "id": row.get::<_, i64>(0)?,
            "rootId": row.get::<_, Option<i64>>(1)?,
            "scopePath": row.get::<_, String>(2)?,
            "state": row.get::<_, String>(3)?,
            "total": row.get::<_, i64>(4)?,
            "rejected": row.get::<_, i64>(5)?,
            "createdAt": row.get::<_, i64>(6)?,
            "updatedAt": row.get::<_, i64>(7)?,
        })))
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
        .query_map([session_id], |row| Ok(json!({
            "path": row.get::<_, String>(0)?,
            "proposed": row.get::<_, String>(1)?,
            "final": row.get::<_, String>(2)?,
            "qualityScore": row.get::<_, f64>(3)?,
            "reason": row.get::<_, String>(4)?,
        })))
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    Ok(json!(decisions))
}

fn export_tag_suggestions(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection = Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let mut statement = connection
        .prepare("SELECT iat.id, r.absolute_path || '/' || i.relative_path, t.name, iat.confidence, iat.model_id FROM image_ai_tags iat JOIN images i ON i.id = iat.image_id JOIN collection_roots r ON r.id = i.root_id JOIN tags t ON t.id = iat.tag_id WHERE i.status = 'present' AND iat.review_state = 'suggested' ORDER BY iat.confidence DESC")
        .map_err(|error| error.to_string())?;
    let suggestions = statement
        .query_map([], |row| Ok(json!({
            "id": row.get::<_, i64>(0)?,
            "path": row.get::<_, String>(1)?,
            "tag": row.get::<_, String>(2)?,
            "confidence": row.get::<_, f64>(3)?,
            "modelId": row.get::<_, String>(4)?,
        })))
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    Ok(json!(suggestions))
}

fn verify_model(arguments: &[String]) -> Result<serde_json::Value, String> {
    let model_id = named_argument(arguments, "--id")?;
    let visual_pack = visual_model_packs().into_iter().find(|p| p.id == model_id);
    let face_pack = rapidraw_lib::face_model_registry::face_model_packs().into_iter().find(|p| p.id == model_id);
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

fn list_derivatives(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection = Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let image_id = numeric_argument(arguments, "--image")?;
    let mut stmt = connection
        .prepare("SELECT id, operation_kind, model_id, model_revision, output_path, output_format, width, height, state, created_at FROM image_derivatives WHERE source_image_id = ?1 ORDER BY created_at DESC")
        .map_err(|e| e.to_string())?;
    let derivatives = stmt
        .query_map([image_id], |row| Ok(json!({
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
        })))
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(json!(derivatives))
}

fn run_restore_cli(arguments: &[String]) -> Result<serde_json::Value, String> {
    let db_path = database_argument(arguments)?;
    let image_id = numeric_argument(arguments, "--image")?;
    let _ = (db_path, image_id);
    Err(
        "restore run is temporarily unavailable: it previously wrote a display-referred microcontrast image while recording it as a RawNIND derivative. Use the catalog RAW Restore/RGB Denoise jobs until the CLI uses the same model-backed restoration service."
            .to_string(),
    )
}
