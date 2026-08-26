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
        _ => Err("Usage: rapidraw-cli library inspect --database <catalog.db> | rapidraw-cli jobs list --database <catalog.db>".to_string()),
    };
    match result {
        Ok(value) => println!("{}", value),
        Err(error) => { eprintln!("{error}"); std::process::exit(2); }
    }
}

fn inspect(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection = Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let name: String = connection.query_row("SELECT name FROM libraries LIMIT 1", [], |row| row.get(0)).map_err(|error| error.to_string())?;
    let images: i64 = connection.query_row("SELECT COUNT(*) FROM images WHERE status = 'present'", [], |row| row.get(0)).map_err(|error| error.to_string())?;
    Ok(json!({ "name": name, "images": images }))
}

fn list_jobs(arguments: &[String]) -> Result<serde_json::Value, String> {
    let connection = Connection::open(database_argument(arguments)?).map_err(|error| error.to_string())?;
    let mut statement = connection.prepare("SELECT id, kind, state, current, total, message FROM background_jobs ORDER BY created_at DESC LIMIT 100").map_err(|error| error.to_string())?;
    let jobs = statement.query_map([], |row| Ok(json!({ "id": row.get::<_, String>(0)?, "kind": row.get::<_, String>(1)?, "state": row.get::<_, String>(2)?, "current": row.get::<_, i64>(3)?, "total": row.get::<_, i64>(4)?, "message": row.get::<_, String>(5)? }))).map_err(|error| error.to_string())?.collect::<Result<Vec<_>, _>>().map_err(|error| error.to_string())?;
    Ok(json!(jobs))
}
