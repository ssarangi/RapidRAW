use std::io::Cursor;

use base64::{Engine as _, engine::general_purpose};
use image::codecs::jpeg::JpegEncoder;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use tauri::AppHandle;

use crate::library_db::{active_library_path, open_connection};

/// Long edge a photo is downsampled to before it's sent to Gemini. Vision
/// token cost scales with image tile count, and none of the critique
/// categories (exposure, sharpness, framing, distracting elements) need
/// full-resolution detail to assess - this keeps a single critique cheap
/// enough to run per-photo on demand.
const MAX_CRITIQUE_DIMENSION: u32 = 1024;
const JPEG_QUALITY: u8 = 82;

/// Parses a candidate flash/lite model name (e.g. "models/gemini-3.7-flash" or
/// "models/gemini-2.5-flash-lite-preview-04-17") into a sort key so the newest
/// stable flash-tier model can be picked automatically instead of a hardcoded
/// id that silently 404s once Google retires it. Returns None for anything
/// that isn't a flash/lite text+vision model (embeddings, TTS, image-gen, etc).
fn parse_flash_candidate(model_name: &str) -> Option<(u32, u32, bool, bool)> {
    let short = model_name.rsplit('/').next().unwrap_or(model_name);
    if !short.contains("flash") {
        return None;
    }
    if short.contains("tts") || short.contains("embedding") || short.contains("image-generation") {
        return None;
    }
    let is_lite = short.contains("flash-lite");
    let is_preview = short.contains("preview") || short.contains("exp");
    let version_part = short.strip_prefix("gemini-")?.split("-flash").next()?;
    let mut segments = version_part.splitn(2, '.');
    let major: u32 = segments.next()?.parse().ok()?;
    let minor: u32 = segments.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    // Sort key: newest version first, then prefer the cheaper "flash-lite"
    // over full "flash" (this is a one-shot critique call, not a task that
    // needs the bigger model), then prefer a stable release over preview/exp.
    Some((major, minor, is_lite, !is_preview))
}

/// Looks up available flash/lite Gemini models instead of hardcoding an id
/// (so this doesn't 404 the moment Google renames or retires whatever was
/// current when this was written), sorted best-first. Callers try them in
/// order and fall through to the next one on a retryable error (e.g. a 503
/// while Google's infra is overloaded), rather than failing the whole
/// critique because one specific model instance is temporarily down.
async fn resolve_flash_models(api_key: &str) -> Result<Vec<String>, String> {
    let client = reqwest::Client::new();
    let url = format!("https://generativelanguage.googleapis.com/v1beta/models?key={api_key}");
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|error| format!("Could not reach Gemini: {error}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!(
            "Gemini returned HTTP {status} while listing models: {body}"
        ));
    }
    let payload: serde_json::Value = response
        .json()
        .await
        .map_err(|error| format!("Could not parse Gemini's model list: {error}"))?;
    let models = payload["models"]
        .as_array()
        .ok_or_else(|| "Gemini's model list response was missing 'models'".to_string())?;

    let mut candidates: Vec<(String, (u32, u32, bool, bool))> = Vec::new();
    for model in models {
        let Some(name) = model["name"].as_str() else {
            continue;
        };
        let supports_generate =
            model["supportedGenerationMethods"]
                .as_array()
                .is_some_and(|methods| {
                    methods
                        .iter()
                        .any(|m| m.as_str() == Some("generateContent"))
                });
        if !supports_generate {
            continue;
        }
        let Some(key) = parse_flash_candidate(name) else {
            continue;
        };
        candidates.push((name.rsplit('/').next().unwrap_or(name).to_string(), key));
    }

    if candidates.is_empty() {
        return Err("No flash/lite Gemini model available for this API key".to_string());
    }
    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    Ok(candidates.into_iter().map(|(name, _)| name).collect())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiCritiqueRegion {
    pub label: String,
    pub positive: bool,
    pub note: String,
    /// [x, y, width, height], each normalized 0-1 with origin at top-left.
    #[serde(rename = "box")]
    pub bbox: [f32; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiCritiqueResponse {
    pub overall_summary: String,
    #[serde(default)]
    pub regions: Vec<GeminiCritiqueRegion>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiCritique {
    pub image_id: i64,
    pub overall_summary: String,
    pub regions: Vec<GeminiCritiqueRegion>,
    pub cached: bool,
}

fn resolve_image_path(conn: &rusqlite::Connection, image_id: i64) -> Result<String, String> {
    conn.query_row(
        "SELECT r.absolute_path || '/' || i.relative_path
         FROM images i
         JOIN collection_roots r ON r.id = i.root_id
         WHERE i.id = ?1",
        params![image_id],
        |row| row.get(0),
    )
    .map_err(|error| format!("Image not found: {error}"))
}

fn build_request_body(image_base64: &str) -> serde_json::Value {
    let prompt = "You are a photography mentor helping someone cull a shoot. Look at this photo \
        and identify up to 4 notable regions - things that are either genuine strengths (sharp eye, \
        good light, clean background) or weaknesses (obstructing branch, blown highlight, distracting \
        element, soft focus area). For each region give a short 2-4 word label, whether it's positive, \
        a normalized bounding box [x, y, width, height] in 0-1 range with origin at the top-left, and a \
        one-sentence note. Also give a 1-2 sentence overall critique. Be specific to what is actually in \
        this image - do not give generic photography advice.";

    serde_json::json!({
        "contents": [{
            "parts": [
                { "inline_data": { "mime_type": "image/jpeg", "data": image_base64 } },
                { "text": prompt }
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "overallSummary": { "type": "STRING" },
                    "regions": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "label": { "type": "STRING" },
                                "positive": { "type": "BOOLEAN" },
                                "note": { "type": "STRING" },
                                "box": {
                                    "type": "ARRAY",
                                    "items": { "type": "NUMBER" }
                                }
                            },
                            "required": ["label", "positive", "note", "box"]
                        }
                    }
                },
                "required": ["overallSummary", "regions"]
            }
        }
    })
}

/// A failed call against one specific model. Retryable means the model
/// itself is likely fine but temporarily unavailable/overloaded (503, 429,
/// other 5xx) or otherwise worth trying a different model for - the caller
/// should fall through to the next candidate rather than giving up. Fatal
/// covers things another model won't fix (bad request, invalid key).
enum GeminiCallError {
    Retryable(String),
    Fatal(String),
}

async fn call_gemini(
    api_key: &str,
    model: &str,
    image_base64: &str,
) -> Result<GeminiCritiqueResponse, GeminiCallError> {
    let client = reqwest::Client::new();
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    );
    let response = client
        .post(&url)
        .json(&build_request_body(image_base64))
        .send()
        .await
        .map_err(|error| GeminiCallError::Retryable(format!("Could not reach Gemini: {error}")))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        let message = format!("Gemini ({model}) returned HTTP {status}: {body}");
        // 429 (rate limited/quota) and any 5xx (503 overloaded, 500 internal
        // error, etc.) are about this model instance's current state, not a
        // reason to believe every flash/lite model is unusable.
        return Err(if status.as_u16() == 429 || status.is_server_error() {
            GeminiCallError::Retryable(message)
        } else {
            GeminiCallError::Fatal(message)
        });
    }

    let payload: serde_json::Value = response.json().await.map_err(|error| {
        GeminiCallError::Fatal(format!("Could not parse Gemini's response: {error}"))
    })?;

    let text = payload["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .ok_or_else(|| {
            GeminiCallError::Fatal(
                "Gemini's response did not contain the expected text part".to_string(),
            )
        })?;

    serde_json::from_str(text).map_err(|error| {
        GeminiCallError::Fatal(format!("Could not parse Gemini's critique JSON: {error}"))
    })
}

#[tauri::command]
pub async fn get_or_generate_gemini_critique(
    image_id: i64,
    app_handle: AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<GeminiCritique, String> {
    let db_path = active_library_path(&state)?;
    {
        let conn = open_connection(&db_path)?;
        let cached: Option<String> = conn
            .query_row(
                "SELECT response_json FROM gemini_critiques WHERE image_id = ?1",
                params![image_id],
                |row| row.get(0),
            )
            .ok();
        if let Some(response_json) = cached {
            let parsed: GeminiCritiqueResponse = serde_json::from_str(&response_json)
                .map_err(|error| format!("Stored critique is corrupted: {error}"))?;
            return Ok(GeminiCritique {
                image_id,
                overall_summary: parsed.overall_summary,
                regions: parsed.regions,
                cached: true,
            });
        }
    }

    let settings = crate::app_settings::load_settings(app_handle).unwrap_or_default();
    let api_key = settings
        .gemini_api_key
        .clone()
        .filter(|key| !key.trim().is_empty())
        .ok_or_else(|| "Add a Gemini API key in Settings > General > Tagging first".to_string())?;

    let image_path = {
        let conn = open_connection(&db_path)?;
        resolve_image_path(&conn, image_id)?
    };

    let file_bytes = std::fs::read(&image_path).map_err(|error| error.to_string())?;
    let image = crate::image_loader::load_base_image_from_bytes(
        &file_bytes,
        &image_path,
        true,
        &settings,
        None,
        None,
    )
    .map_err(|error| error.to_string())?;
    let resized = image.thumbnail(MAX_CRITIQUE_DIMENSION, MAX_CRITIQUE_DIMENSION);

    let mut buf = Cursor::new(Vec::new());
    resized
        .write_with_encoder(JpegEncoder::new_with_quality(&mut buf, JPEG_QUALITY))
        .map_err(|error| error.to_string())?;
    let image_base64 = general_purpose::STANDARD.encode(buf.into_inner());

    let candidate_models = resolve_flash_models(&api_key).await?;
    let mut last_error: Option<String> = None;
    let mut succeeded: Option<(String, GeminiCritiqueResponse)> = None;
    for model in &candidate_models {
        match call_gemini(&api_key, model, &image_base64).await {
            Ok(critique) => {
                succeeded = Some((model.clone(), critique));
                break;
            }
            Err(GeminiCallError::Retryable(message)) => {
                last_error = Some(message);
            }
            Err(GeminiCallError::Fatal(message)) => return Err(message),
        }
    }
    let (model, critique) = succeeded.ok_or_else(|| {
        last_error.unwrap_or_else(|| "No Gemini model produced a critique".to_string())
    })?;

    let response_json = serde_json::to_string(&critique).map_err(|error| error.to_string())?;
    {
        let conn = open_connection(&db_path)?;
        conn.execute(
            "INSERT INTO gemini_critiques(image_id, model, response_json, created_at) VALUES(?1, ?2, ?3, strftime('%s','now'))
             ON CONFLICT(image_id) DO UPDATE SET model=excluded.model, response_json=excluded.response_json, created_at=excluded.created_at",
            params![image_id, model, response_json],
        )
        .map_err(|error| error.to_string())?;
    }

    Ok(GeminiCritique {
        image_id,
        overall_summary: critique.overall_summary,
        regions: critique.regions,
        cached: false,
    })
}
