use std::env;
use std::path::PathBuf;

use rusqlite::Connection;
use serde_json::json;

fn database_argument(arguments: &[String]) -> Result<PathBuf, String> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == "--database")
        .map(|pair| PathBuf::from(&pair[1]))
        .ok_or_else(|| "--database <path> is required".to_string())
}

fn main() {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let result = match arguments.first().map(String::as_str) {
        Some("library") if arguments.get(1).map(String::as_str) == Some("inspect") => inspect(&arguments),
        Some("jobs") if arguments.get(1).map(String::as_str) == Some("list") => list_jobs(&arguments),
        Some("faces") if arguments.get(1).map(String::as_str) == Some("status") => face_status(&arguments),
        Some("tags") if arguments.get(1).map(String::as_str) == Some("status") => tag_status(&arguments),
        _ => Err("Usage: rapidraw-cli library inspect --database <catalog.db> | rapidraw-cli jobs list --database <catalog.db> | rapidraw-cli faces status --database <catalog.db> | rapidraw-cli tags status --database <catalog.db>".to_string()),
    };
    match result {
        Ok(value) => println!("{}", value),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
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

fn list_jobs(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection =
        Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let mut statement = connection.prepare("SELECT id, kind, state, current, total, message FROM background_jobs ORDER BY created_at DESC LIMIT 100").map_err(|error| error.to_string())?;
    let jobs = statement.query_map([], |row| Ok(json!({ "id": row.get::<_, String>(0)?, "kind": row.get::<_, String>(1)?, "state": row.get::<_, String>(2)?, "current": row.get::<_, i64>(3)?, "total": row.get::<_, i64>(4)?, "message": row.get::<_, String>(5)? }))).map_err(|error| error.to_string())?.collect::<Result<Vec<_>, _>>().map_err(|error| error.to_string())?;
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
