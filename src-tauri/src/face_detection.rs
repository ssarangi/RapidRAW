use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::Ordering;

use image::{DynamicImage, imageops::FilterType};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;
use rusqlite::params;
use tauri::{AppHandle, Emitter, Manager};

use crate::library_db::{active_library_path, create_background_job, update_job};

const YUNET_MODEL: &str = "face_detection_yunet_2023mar.onnx";
const SFACE_MODEL: &str = "face_recognition_sface_2021dec.onnx";
const INPUT_SIZE: u32 = 320;
const SFACE_INPUT_SIZE: u32 = 112;

#[derive(Debug, Clone)]
struct Detection {
    confidence: f32,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    landmarks: [[f32; 2]; 5],
}

fn intersection_over_union(a: &Detection, b: &Detection) -> f32 {
    let left = a.x.max(b.x);
    let top = a.y.max(b.y);
    let right = (a.x + a.width).min(b.x + b.width);
    let bottom = (a.y + a.height).min(b.y + b.height);
    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    intersection / (a.width * a.height + b.width * b.height - intersection).max(f32::EPSILON)
}

fn detect_yunet(image: DynamicImage, session: &mut Session) -> Result<Vec<Detection>, String> {
    let (original_width, original_height) = (image.width(), image.height());
    if original_width == 0 || original_height == 0 {
        return Ok(Vec::new());
    }
    let resized = image
        .resize_exact(INPUT_SIZE, INPUT_SIZE, FilterType::Triangle)
        .to_rgb8();
    let mut input = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    for (x, y, pixel) in resized.enumerate_pixels() {
        input[[0, 0, y as usize, x as usize]] = pixel[2] as f32;
        input[[0, 1, y as usize, x as usize]] = pixel[1] as f32;
        input[[0, 2, y as usize, x as usize]] = pixel[0] as f32;
    }
    let outputs = session
        .run(ort::inputs![
            Tensor::from_array(input).map_err(|error| error.to_string())?
        ])
        .map_err(|error| error.to_string())?;
    let values = outputs[0]
        .try_extract_array::<f32>()
        .map_err(|error| error.to_string())?
        .to_owned();
    let rows = values
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|error| format!("Unexpected YuNet output shape: {error}"))?;
    let mut candidates = Vec::new();
    for row in rows.index_axis(ndarray::Axis(0), 0).outer_iter() {
        if row.len() < 15 || row[14] < 0.7 {
            continue;
        }
        let scale_x = original_width as f32 / INPUT_SIZE as f32;
        let scale_y = original_height as f32 / INPUT_SIZE as f32;
        let x = (row[0] * scale_x).max(0.0);
        let y = (row[1] * scale_y).max(0.0);
        let width = (row[2] * scale_x).min(original_width as f32 - x).max(0.0);
        let height = (row[3] * scale_y).min(original_height as f32 - y).max(0.0);
        if width < 4.0 || height < 4.0 {
            continue;
        }
        let mut landmarks = [[0.0; 2]; 5];
        for index in 0..5 {
            landmarks[index] = [
                row[4 + index * 2] * scale_x / original_width as f32,
                row[5 + index * 2] * scale_y / original_height as f32,
            ];
        }
        candidates.push(Detection {
            confidence: row[14],
            x: x / original_width as f32,
            y: y / original_height as f32,
            width: width / original_width as f32,
            height: height / original_height as f32,
            landmarks,
        });
    }
    candidates.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut accepted = Vec::new();
    for candidate in candidates {
        if accepted
            .iter()
            .all(|existing| intersection_over_union(existing, &candidate) < 0.3)
        {
            accepted.push(candidate);
        }
    }
    Ok(accepted)
}

fn normalize_embedding(mut values: Vec<f32>) -> Result<Vec<f32>, String> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        return Err("SFace returned an invalid embedding".to_string());
    }
    for value in &mut values {
        *value /= norm;
    }
    Ok(values)
}

fn decode_embedding(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("exact four-byte chunks")))
            .collect(),
    )
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() {
        return -1.0;
    }
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn suggest_people(conn: &rusqlite::Connection) -> Result<(), String> {
    let mut centroids: HashMap<i64, (Vec<f32>, usize)> = HashMap::new();
    let mut references = conn.prepare("SELECT f.person_id, e.vector FROM faces f JOIN face_embeddings e ON e.id = f.embedding_id WHERE f.review_state = 'confirmed' AND f.person_id IS NOT NULL AND e.model_pack_id = 'opencv-yunet-sface'").map_err(|error| error.to_string())?;
    for row in references
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
        .map_err(|error| error.to_string())?
    {
        let (person_id, bytes) = row.map_err(|error| error.to_string())?;
        let Some(vector) = decode_embedding(&bytes) else {
            continue;
        };
        let entry = centroids
            .entry(person_id)
            .or_insert_with(|| (vec![0.0; vector.len()], 0));
        if entry.0.len() != vector.len() {
            continue;
        }
        for (target, value) in entry.0.iter_mut().zip(vector) {
            *target += value;
        }
        entry.1 += 1;
    }
    for (vector, count) in centroids.values_mut() {
        if *count > 0 {
            for value in vector {
                *value /= *count as f32;
            }
        }
    }
    let mut candidates = conn.prepare("SELECT f.id, e.vector FROM faces f JOIN face_embeddings e ON e.id = f.embedding_id WHERE f.review_state = 'unreviewed' AND e.model_pack_id = 'opencv-yunet-sface'").map_err(|error| error.to_string())?;
    for row in candidates
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
        .map_err(|error| error.to_string())?
    {
        let (face_id, bytes) = row.map_err(|error| error.to_string())?;
        let Some(vector) = decode_embedding(&bytes) else {
            continue;
        };
        let best = centroids
            .iter()
            .map(|(person_id, (centroid, _))| (*person_id, cosine_similarity(&vector, centroid)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((person_id, similarity)) = best {
            if similarity >= 0.55 {
                conn.execute("UPDATE faces SET person_id = ?1, updated_at = strftime('%s','now') WHERE id = ?2", params![person_id, face_id]).map_err(|error| error.to_string())?;
            }
        }
    }
    Ok(())
}

fn extract_sface_embedding(
    image: &DynamicImage,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    session: &mut Session,
) -> Result<Vec<f32>, String> {
    let image_width = image.width() as f64;
    let image_height = image.height() as f64;
    let left = (x * image_width).clamp(0.0, image_width - 1.0) as u32;
    let top = (y * image_height).clamp(0.0, image_height - 1.0) as u32;
    let crop_width = (width * image_width)
        .max(1.0)
        .min(image_width - left as f64) as u32;
    let crop_height = (height * image_height)
        .max(1.0)
        .min(image_height - top as f64) as u32;
    let face = image
        .crop_imm(left, top, crop_width, crop_height)
        .resize_exact(SFACE_INPUT_SIZE, SFACE_INPUT_SIZE, FilterType::Triangle)
        .to_rgb8();
    let mut input =
        Array4::<f32>::zeros((1, 3, SFACE_INPUT_SIZE as usize, SFACE_INPUT_SIZE as usize));
    for (pixel_x, pixel_y, pixel) in face.enumerate_pixels() {
        input[[0, 0, pixel_y as usize, pixel_x as usize]] = pixel[2] as f32;
        input[[0, 1, pixel_y as usize, pixel_x as usize]] = pixel[1] as f32;
        input[[0, 2, pixel_y as usize, pixel_x as usize]] = pixel[0] as f32;
    }
    let outputs = session
        .run(ort::inputs![
            Tensor::from_array(input).map_err(|error| error.to_string())?
        ])
        .map_err(|error| error.to_string())?;
    let embedding = outputs[0]
        .try_extract_array::<f32>()
        .map_err(|error| error.to_string())?
        .iter()
        .copied()
        .collect::<Vec<_>>();
    normalize_embedding(embedding)
}

#[tauri::command]
pub fn start_face_detection(
    root_id: Option<i64>,
    app_handle: AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<String, String> {
    let db_path = active_library_path(&state)?;
    let model_path = crate::face_model_registry::installed_face_model_path(
        &app_handle,
        "opencv-yunet-sface",
        YUNET_MODEL,
    )?;
    let job_id = create_background_job(
        &db_path,
        "face_detection",
        serde_json::json!({ "rootId": root_id, "modelPackId": "opencv-yunet-sface" }),
    )?;
    let cancellation = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    state
        .background_job_cancellations
        .lock()
        .unwrap()
        .insert(job_id.clone(), cancellation.clone());
    let app = app_handle.clone();
    let event_job_id = job_id.clone();
    let worker_job_id = job_id.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let result = run_face_detection(
            &db_path,
            root_id,
            &model_path,
            &worker_job_id,
            &cancellation,
        );
        if let Err(error) = result {
            let job_state = if error == "Face detection cancelled" {
                "cancelled"
            } else {
                "failed"
            };
            let _ = update_job(
                &db_path,
                &worker_job_id,
                job_state,
                &error,
                0,
                0,
                None,
                Some(&error),
            );
        }
        let _ = app.emit(
            "face-detection-complete",
            serde_json::json!({ "jobId": event_job_id }),
        );
    });
    Ok(job_id)
}

#[tauri::command]
pub fn start_face_recognition(
    root_id: Option<i64>,
    app_handle: AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<String, String> {
    let db_path = active_library_path(&state)?;
    let model_path = crate::face_model_registry::installed_face_model_path(
        &app_handle,
        "opencv-yunet-sface",
        SFACE_MODEL,
    )?;
    let job_id = create_background_job(
        &db_path,
        "face_recognition",
        serde_json::json!({ "rootId": root_id, "modelPackId": "opencv-yunet-sface" }),
    )?;
    let cancellation = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    state
        .background_job_cancellations
        .lock()
        .unwrap()
        .insert(job_id.clone(), cancellation.clone());
    let app = app_handle.clone();
    let worker_job_id = job_id.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let result = run_face_recognition(
            &db_path,
            root_id,
            &model_path,
            &worker_job_id,
            &cancellation,
        );
        if let Err(error) = result {
            let job_state = if error == "Face recognition cancelled" {
                "cancelled"
            } else {
                "failed"
            };
            let _ = update_job(
                &db_path,
                &worker_job_id,
                job_state,
                &error,
                0,
                0,
                None,
                Some(&error),
            );
        }
        app.state::<crate::AppState>()
            .background_job_cancellations
            .lock()
            .unwrap()
            .remove(&worker_job_id);
        let _ = app.emit(
            "face-recognition-complete",
            serde_json::json!({ "jobId": worker_job_id }),
        );
    });
    Ok(job_id)
}

fn run_face_recognition(
    db_path: &Path,
    root_id: Option<i64>,
    model_path: &Path,
    job_id: &str,
    cancellation: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<(), String> {
    let conn = rusqlite::Connection::open(db_path).map_err(|error| error.to_string())?;
    let mut sql = "SELECT f.id, r.absolute_path, i.relative_path, f.bbox_x, f.bbox_y, f.bbox_width, f.bbox_height FROM faces f JOIN images i ON i.id = f.image_id JOIN collection_roots r ON r.id = i.root_id WHERE i.status = 'present' AND f.model_pack_id = 'opencv-yunet-sface' AND f.review_state <> 'rejected' AND f.embedding_id IS NULL".to_string();
    if root_id.is_some() {
        sql.push_str(" AND i.root_id = ?1");
    }
    let mut statement = conn.prepare(&sql).map_err(|error| error.to_string())?;
    let faces: Vec<(i64, String, String, f64, f64, f64, f64)> = if let Some(root_id) = root_id {
        statement
            .query_map([root_id], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            })
            .map_err(|error| error.to_string())?
            .collect::<Result<_, _>>()
            .map_err(|error| error.to_string())?
    } else {
        statement
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            })
            .map_err(|error| error.to_string())?
            .collect::<Result<_, _>>()
            .map_err(|error| error.to_string())?
    };
    let total = faces.len() as i64;
    update_job(
        db_path,
        job_id,
        "running",
        "Loading SFace recognizer",
        0,
        total,
        None,
        None,
    )?;
    if faces.is_empty() {
        suggest_people(&conn)?;
        return update_job(
            db_path,
            job_id,
            "completed",
            "Person suggestions refreshed",
            0,
            0,
            None,
            None,
        );
    }
    let _ = ort::init().with_name("RapidRAW-FaceRecognition").commit();
    let mut session = Session::builder()
        .map_err(|error| error.to_string())?
        .commit_from_file(model_path)
        .map_err(|error| error.to_string())?;
    for (index, (face_id, root, relative, x, y, width, height)) in faces.into_iter().enumerate() {
        if cancellation.load(Ordering::SeqCst) {
            return Err("Face recognition cancelled".to_string());
        }
        let path = Path::new(&root).join(&relative);
        let current = index as i64 + 1;
        let _ = update_job(
            db_path,
            job_id,
            "running",
            "Embedding face",
            current,
            total,
            Some(&path.to_string_lossy()),
            None,
        );
        let image = match image::open(&path) {
            Ok(image) => image,
            Err(_) => continue,
        };
        let embedding = match extract_sface_embedding(&image, x, y, width, height, &mut session) {
            Ok(embedding) => embedding,
            Err(_) => continue,
        };
        let bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        conn.execute("INSERT INTO face_embeddings(model_pack_id, dimensions, vector, norm, created_at) VALUES('opencv-yunet-sface', ?1, ?2, 1.0, strftime('%s','now'))", params![embedding.len() as i64, bytes]).map_err(|error| error.to_string())?;
        let embedding_id = conn.last_insert_rowid();
        conn.execute(
            "UPDATE faces SET embedding_id = ?1, updated_at = strftime('%s','now') WHERE id = ?2",
            params![embedding_id, face_id],
        )
        .map_err(|error| error.to_string())?;
    }
    suggest_people(&conn)?;
    update_job(
        db_path,
        job_id,
        "completed",
        "Face recognition embeddings complete",
        total,
        total,
        None,
        None,
    )
}

fn run_face_detection(
    db_path: &Path,
    root_id: Option<i64>,
    model_path: &Path,
    job_id: &str,
    cancellation: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<(), String> {
    let conn = rusqlite::Connection::open(db_path).map_err(|error| error.to_string())?;
    let mut sql = "SELECT i.id, r.absolute_path, i.relative_path FROM images i JOIN collection_roots r ON r.id = i.root_id WHERE i.status = 'present'".to_string();
    if root_id.is_some() {
        sql.push_str(" AND i.root_id = ?1");
    }
    let mut statement = conn.prepare(&sql).map_err(|error| error.to_string())?;
    let images: Vec<(i64, String, String)> = if let Some(root_id) = root_id {
        statement
            .query_map([root_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|error| error.to_string())?
            .collect::<Result<_, _>>()
            .map_err(|error| error.to_string())?
    } else {
        statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|error| error.to_string())?
            .collect::<Result<_, _>>()
            .map_err(|error| error.to_string())?
    };
    let total = images.len() as i64;
    update_job(
        db_path,
        job_id,
        "running",
        "Loading YuNet face detector",
        0,
        total,
        None,
        None,
    )?;
    let _ = ort::init().with_name("RapidRAW-FaceDetection").commit();
    let mut session = Session::builder()
        .map_err(|error| error.to_string())?
        .commit_from_file(model_path)
        .map_err(|error| error.to_string())?;
    for (index, (image_id, root, relative)) in images.into_iter().enumerate() {
        if cancellation.load(Ordering::SeqCst) {
            return Err("Face detection cancelled".to_string());
        }
        let path = Path::new(&root).join(&relative);
        let current = index as i64 + 1;
        let _ = update_job(
            db_path,
            job_id,
            "running",
            "Detecting faces",
            current,
            total,
            Some(&path.to_string_lossy()),
            None,
        );
        let detections = match image::open(&path) {
            Ok(image) => detect_yunet(image, &mut session)?,
            Err(_) => continue,
        };
        conn.execute("DELETE FROM faces WHERE image_id = ?1 AND model_pack_id = 'opencv-yunet-sface' AND source = 'local' AND review_state = 'unreviewed'", [image_id]).map_err(|error| error.to_string())?;
        for detection in detections {
            conn.execute("INSERT INTO faces(image_id, model_pack_id, detector_confidence, bbox_x, bbox_y, bbox_width, bbox_height, landmarks_json, review_state, source, created_at, updated_at) VALUES(?1, 'opencv-yunet-sface', ?2, ?3, ?4, ?5, ?6, ?7, 'unreviewed', 'local', strftime('%s','now'), strftime('%s','now'))", params![image_id, detection.confidence, detection.x, detection.y, detection.width, detection.height, serde_json::to_string(&detection.landmarks).map_err(|error| error.to_string())?]).map_err(|error| error.to_string())?;
        }
    }
    update_job(
        db_path,
        job_id,
        "completed",
        "Face detection complete",
        total,
        total,
        None,
        None,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_measure_is_high_for_same_face() {
        let face = Detection {
            confidence: 0.9,
            x: 0.1,
            y: 0.1,
            width: 0.3,
            height: 0.3,
            landmarks: [[0.0; 2]; 5],
        };
        let overlapping = Detection {
            confidence: 0.8,
            x: 0.12,
            y: 0.12,
            width: 0.3,
            height: 0.3,
            landmarks: [[0.0; 2]; 5],
        };
        assert!(intersection_over_union(&face, &overlapping) > 0.3);
    }
}
