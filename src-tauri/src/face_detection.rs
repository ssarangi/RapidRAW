use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use image::{imageops::FilterType, DynamicImage};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;
use rusqlite::params;
use tauri::{AppHandle, Emitter, Manager};

use crate::library_db::{active_library_path, create_background_job, update_job};

const YUNET_MODEL: &str = "face_detection_yunet_2023mar.onnx";
const SFACE_MODEL: &str = "face_recognition_sface_2021dec.onnx";
const INPUT_SIZE: u32 = 640;
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

/// Geometry available from YuNet's five landmarks. This is deliberately a
/// pose/framing estimate, not an eye-state or expression classifier.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FacePoseEstimate {
    pub frontal_score: f32,
    pub roll_degrees: f32,
    pub frame_fraction: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct CullingFaceAnalysis {
    pub face_count: usize,
    pub best_pose: Option<FacePoseEstimate>,
}

fn estimate_face_pose(
    landmarks: [[f32; 2]; 5],
    bbox_width: f32,
    bbox_height: f32,
) -> Option<FacePoseEstimate> {
    let right_eye = landmarks[0];
    let left_eye = landmarks[1];
    let nose = landmarks[2];
    let eye_dx = left_eye[0] - right_eye[0];
    let eye_dy = left_eye[1] - right_eye[1];
    let eye_distance = (eye_dx.powi(2) + eye_dy.powi(2)).sqrt();
    if !eye_distance.is_finite() || eye_distance <= f32::EPSILON {
        return None;
    }

    let eye_midpoint_x = (left_eye[0] + right_eye[0]) * 0.5;
    // A face is most frontal when the nose lies near the eye midpoint. This
    // cannot measure 3D head pose, but it is a useful conservative ranking
    // signal for otherwise near-identical people frames.
    let normalized_nose_offset = ((nose[0] - eye_midpoint_x).abs() / eye_distance).min(1.0);
    let frontal_score = (1.0 - normalized_nose_offset * 1.35).clamp(0.0, 1.0);
    let roll_degrees = eye_dy.atan2(eye_dx).to_degrees().abs();
    let frame_fraction = (bbox_width.max(0.0) * bbox_height.max(0.0)).clamp(0.0, 1.0);
    Some(FacePoseEstimate {
        frontal_score,
        roll_degrees,
        frame_fraction,
    })
}

pub(crate) fn estimate_stored_face_pose(
    landmarks_json: &str,
    bbox_width: f64,
    bbox_height: f64,
) -> Option<FacePoseEstimate> {
    let landmarks = serde_json::from_str::<[[f32; 2]; 5]>(landmarks_json).ok()?;
    estimate_face_pose(landmarks, bbox_width as f32, bbox_height as f32)
}

fn intersection_over_union(a: &Detection, b: &Detection) -> f32 {
    let left = a.x.max(b.x);
    let top = a.y.max(b.y);
    let right = (a.x + a.width).min(b.x + b.width);
    let bottom = (a.y + a.height).min(b.y + b.height);
    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    intersection / (a.width * a.height + b.width * b.height - intersection).max(f32::EPSILON)
}

fn detect_yunet(image: &DynamicImage, session: &mut Session) -> Result<Vec<Detection>, String> {
    let (original_width, original_height) = (image.width(), image.height());
    if original_width == 0 || original_height == 0 {
        return Ok(Vec::new());
    }
    let scale =
        (INPUT_SIZE as f32 / original_width as f32).min(INPUT_SIZE as f32 / original_height as f32);
    let scaled_w = ((original_width as f32 * scale).round() as u32).clamp(1, INPUT_SIZE);
    let scaled_h = ((original_height as f32 * scale).round() as u32).clamp(1, INPUT_SIZE);
    let pad_x = (INPUT_SIZE - scaled_w) / 2;
    let pad_y = (INPUT_SIZE - scaled_h) / 2;
    let resized = image
        .resize_exact(scaled_w, scaled_h, FilterType::Triangle)
        .to_rgb8();
    let mut input = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    for (x, y, pixel) in resized.enumerate_pixels() {
        input[[0, 0, (y + pad_y) as usize, (x + pad_x) as usize]] = pixel[2] as f32;
        input[[0, 1, (y + pad_y) as usize, (x + pad_x) as usize]] = pixel[1] as f32;
        input[[0, 2, (y + pad_y) as usize, (x + pad_x) as usize]] = pixel[0] as f32;
    }
    let outputs = session
        .run(ort::inputs![
            Tensor::from_array(input).map_err(|error| error.to_string())?
        ])
        .map_err(|error| error.to_string())?;

    // YuNet output indices:
    // Stride 8:  cls=0, obj=3, bbox=6, kps=9
    // Stride 16: cls=1, obj=4, bbox=7, kps=10
    // Stride 32: cls=2, obj=5, bbox=8, kps=11
    let strides = [
        (8u32, 0, 3, 6, 9),
        (16u32, 1, 4, 7, 10),
        (32u32, 2, 5, 8, 11),
    ];
    let mut candidates = Vec::new();

    for (stride, cls_idx, obj_idx, bbox_idx, kps_idx) in strides {
        let cls_arr = outputs[cls_idx]
            .try_extract_array::<f32>()
            .map_err(|e| e.to_string())?;
        let obj_arr = outputs[obj_idx]
            .try_extract_array::<f32>()
            .map_err(|e| e.to_string())?;
        let bbox_arr = outputs[bbox_idx]
            .try_extract_array::<f32>()
            .map_err(|e| e.to_string())?;
        let kps_arr = outputs[kps_idx]
            .try_extract_array::<f32>()
            .map_err(|e| e.to_string())?;

        let cols = INPUT_SIZE / stride;
        let count = cls_arr.shape()[1];

        for i in 0..count {
            let cls = cls_arr[[0, i, 0]].clamp(0.0, 1.0);
            let obj = obj_arr[[0, i, 0]].clamp(0.0, 1.0);
            let score = (cls * obj).sqrt();

            // DigiKam / OpenCV Zoo recommended score threshold (0.80) to avoid false detections on hands/patterns
            if score < 0.80 {
                continue;
            }

            let gy = (i as u32) / cols;
            let gx = (i as u32) % cols;

            let cx = ((gx as f32) + bbox_arr[[0, i, 0]]) * (stride as f32);
            let cy = ((gy as f32) + bbox_arr[[0, i, 1]]) * (stride as f32);
            let w = bbox_arr[[0, i, 2]].exp() * (stride as f32);
            let h = bbox_arr[[0, i, 3]].exp() * (stride as f32);

            let x = cx - w / 2.0;
            let y = cy - h / 2.0;

            let orig_x = ((x - pad_x as f32) / scale).max(0.0);
            let orig_y = ((y - pad_y as f32) / scale).max(0.0);
            let orig_w = (w / scale).min(original_width as f32 - orig_x).max(0.0);
            let orig_h = (h / scale).min(original_height as f32 - orig_y).max(0.0);

            let norm_x = orig_x / original_width as f32;
            let norm_y = orig_y / original_height as f32;
            let norm_w = orig_w / original_width as f32;
            let norm_h = orig_h / original_height as f32;

            // Filter out tiny noise and unnatural aspect ratios (faces are between 0.45 and 1.8 aspect ratio)
            if orig_w < 16.0 || orig_h < 16.0 || norm_w / norm_h < 0.40 || norm_w / norm_h > 1.90 {
                continue;
            }

            let mut landmarks = [[0.0; 2]; 5];
            for k in 0..5 {
                let lx = ((gx as f32) + kps_arr[[0, i, 2 * k]]) * (stride as f32);
                let ly = ((gy as f32) + kps_arr[[0, i, 2 * k + 1]]) * (stride as f32);
                let orig_lx = ((lx - pad_x as f32) / scale).max(0.0) / original_width as f32;
                let orig_ly = ((ly - pad_y as f32) / scale).max(0.0) / original_height as f32;
                landmarks[k] = [orig_lx, orig_ly];
            }

            // Geometric landmark validation:
            // 0: right eye, 1: left eye, 2: nose tip, 3: right mouth, 4: left mouth
            let eye_dist = ((landmarks[0][0] - landmarks[1][0]).powi(2)
                + (landmarks[0][1] - landmarks[1][1]).powi(2))
            .sqrt();
            if eye_dist < norm_w * 0.10 {
                continue;
            }

            candidates.push(Detection {
                confidence: score,
                x: norm_x,
                y: norm_y,
                width: norm_w,
                height: norm_h,
                landmarks,
            });
        }
    }

    candidates.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut accepted = Vec::new();
    for candidate in candidates {
        if accepted
            .iter()
            .all(|existing| intersection_over_union(existing, &candidate) < 0.30)
        {
            accepted.push(candidate);
        }
    }
    Ok(accepted)
}

/// Lightweight detector used by filesystem culling. It deliberately returns
/// only a count: no catalog rows, embeddings, or person identities are
/// created for an ad-hoc folder scan.
pub(crate) struct LocalFaceDetector {
    session: std::sync::Mutex<Session>,
}

pub(crate) fn load_local_face_detector(
    app_handle: &AppHandle,
) -> Result<LocalFaceDetector, String> {
    let model_path = crate::face_model_registry::installed_face_model_path(
        app_handle,
        "opencv-yunet-sface",
        YUNET_MODEL,
    )?;
    let session = Session::builder()
        .map_err(|error| error.to_string())?
        .commit_from_file(model_path)
        .map_err(|error| error.to_string())?;
    Ok(LocalFaceDetector {
        session: std::sync::Mutex::new(session),
    })
}

pub(crate) fn analyze_faces_for_culling(
    detector: &LocalFaceDetector,
    path: &Path,
    app_handle: &AppHandle,
) -> Result<CullingFaceAnalysis, String> {
    let image = load_image_for_face_ai(path, app_handle)?;
    let mut session = detector
        .session
        .lock()
        .map_err(|_| "Face detector session lock was poisoned".to_string())?;
    let detections = detect_yunet(&image, &mut session)?;
    let best_pose = detections
        .iter()
        .filter_map(|detection| {
            estimate_face_pose(detection.landmarks, detection.width, detection.height)
        })
        .max_by(|left, right| {
            (left.frontal_score * 0.8 + left.frame_fraction * 0.2)
                .total_cmp(&(right.frontal_score * 0.8 + right.frame_fraction * 0.2))
        });
    Ok(CullingFaceAnalysis {
        face_count: detections.len(),
        best_pose,
    })
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
            if similarity >= 0.45 {
                conn.execute("UPDATE faces SET person_id = ?1, updated_at = strftime('%s','now') WHERE id = ?2", params![person_id, face_id]).map_err(|error| error.to_string())?;
            }
        }
    }
    Ok(())
}

fn cluster_unknown_faces(conn: &rusqlite::Connection) -> Result<usize, String> {
    let mut statement = conn.prepare("SELECT f.id, e.vector FROM faces f JOIN face_embeddings e ON e.id = f.embedding_id WHERE f.review_state = 'unreviewed' AND e.model_pack_id = 'opencv-yunet-sface' ORDER BY f.id").map_err(|error| error.to_string())?;
    let faces: Vec<(i64, Vec<f32>)> = statement
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .filter_map(|(id, bytes)| decode_embedding(&bytes).map(|vector| (id, vector)))
        .collect();
    conn.execute("DELETE FROM face_clusters WHERE model_pack_id = 'opencv-yunet-sface' AND state = 'unreviewed'", []).map_err(|error| error.to_string())?;
    let mut assigned = vec![false; faces.len()];
    let mut clusters = 0;
    for index in 0..faces.len() {
        if assigned[index] {
            continue;
        }
        let (representative_id, representative) = &faces[index];
        let members: Vec<(i64, f32)> = faces
            .iter()
            .enumerate()
            .filter_map(|(candidate_index, (id, vector))| {
                if !assigned[candidate_index] && cosine_similarity(representative, vector) >= 0.48 {
                    assigned[candidate_index] = true;
                    Some((*id, cosine_similarity(representative, vector)))
                } else {
                    None
                }
            })
            .collect();
        if members.len() < 2 {
            continue;
        }
        conn.execute("INSERT INTO face_clusters(model_pack_id, representative_face_id, created_at, updated_at) VALUES('opencv-yunet-sface', ?1, strftime('%s','now'), strftime('%s','now'))", [representative_id]).map_err(|error| error.to_string())?;
        let cluster_id = conn.last_insert_rowid();
        for (face_id, similarity) in members {
            conn.execute("INSERT INTO face_cluster_members(cluster_id, face_id, similarity) VALUES(?1, ?2, ?3)", params![cluster_id, face_id, similarity]).map_err(|error| error.to_string())?;
        }
        clusters += 1;
    }
    Ok(clusters)
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

    // Add 10% context margin around face bounding box for better embedding alignment
    let margin_x = width * 0.10;
    let margin_y = height * 0.10;
    let norm_left = (x - margin_x).max(0.0);
    let norm_top = (y - margin_y).max(0.0);
    let norm_width = (width + margin_x * 2.0).min(1.0 - norm_left);
    let norm_height = (height + margin_y * 2.0).min(1.0 - norm_top);

    let left = (norm_left * image_width).clamp(0.0, image_width - 1.0) as u32;
    let top = (norm_top * image_height).clamp(0.0, image_height - 1.0) as u32;
    let crop_width = (norm_width * image_width)
        .max(1.0)
        .min(image_width - left as f64) as u32;
    let crop_height = (norm_height * image_height)
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

pub fn get_face_crops_dir(app_handle: &AppHandle) -> Result<PathBuf, String> {
    let cache_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|e| e.to_string())?;
    let face_crops_dir = cache_dir.join("face_crops");
    if !face_crops_dir.exists() {
        fs::create_dir_all(&face_crops_dir).map_err(|e| e.to_string())?;
    }
    Ok(face_crops_dir)
}

pub fn extract_square_face_crop(
    image: &DynamicImage,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    target_size: u32,
) -> DynamicImage {
    let img_w = image.width() as f64;
    let img_h = image.height() as f64;

    let cx = (x + width / 2.0) * img_w;
    let cy = (y + height / 2.0) * img_h;

    // 25% padding on each side (side length = 1.5x max face dimension)
    let face_dim_px = (width * img_w).max(height * img_h);
    let mut crop_side = face_dim_px * 1.5;

    crop_side = crop_side.min(img_w).min(img_h).max(16.0);

    let mut left = cx - crop_side / 2.0;
    let mut top = cy - crop_side / 2.0;

    if left < 0.0 {
        left = 0.0;
    } else if left + crop_side > img_w {
        left = img_w - crop_side;
    }

    if top < 0.0 {
        top = 0.0;
    } else if top + crop_side > img_h {
        top = img_h - crop_side;
    }

    let left = left.max(0.0) as u32;
    let top = top.max(0.0) as u32;
    let crop_side = (crop_side as u32)
        .min(image.width() - left)
        .min(image.height() - top);

    let cropped = image.crop_imm(left, top, crop_side, crop_side);
    cropped.resize_exact(target_size, target_size, FilterType::Lanczos3)
}

pub fn extract_square_face_crop_jpeg(
    image: &DynamicImage,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    target_size: u32,
) -> Result<Vec<u8>, String> {
    let crop = extract_square_face_crop(image, x, y, width, height, target_size);
    let mut bytes = std::io::Cursor::new(Vec::new());
    crop.write_to(&mut bytes, image::ImageFormat::Jpeg)
        .map_err(|e| e.to_string())?;
    Ok(bytes.into_inner())
}

pub fn save_face_crop_image(
    image: &DynamicImage,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    face_id: i64,
    app_handle: &AppHandle,
) -> Result<PathBuf, String> {
    let face_crops_dir = get_face_crops_dir(app_handle)?;
    let crop_path = face_crops_dir.join(format!("{face_id}.jpg"));
    let crop_image = extract_square_face_crop(image, x, y, width, height, 320);
    crop_image
        .save_with_format(&crop_path, image::ImageFormat::Jpeg)
        .map_err(|e| e.to_string())?;
    Ok(crop_path)
}

pub fn load_image_for_face_ai(path: &Path, app_handle: &AppHandle) -> Result<DynamicImage, String> {
    let path_str = path.to_string_lossy();
    if crate::formats::is_raw_file(path) {
        // 1. Comprehensive multi-IFD TIFF/RAF/ARW embedded JPEG extraction
        if let Ok(file_bytes) = std::fs::read(path) {
            if let Some(img) =
                crate::image_loader::safe_embedded_preview_fallback(&file_bytes, &path_str)
            {
                if img.width() >= 800 || img.height() >= 800 {
                    return Ok(img);
                }
            }
        }
        // 2. Check if a companion JPG/JPEG exists in the same directory (e.g. RAW+JPG shooting)
        for ext in &["JPG", "jpg", "JPEG", "jpeg"] {
            let companion = path.with_extension(ext);
            if companion.exists() && companion != path {
                if let Ok(img) = image::open(&companion) {
                    return Ok(img);
                }
            }
        }
        // 3. Extract rawler preview
        if let Ok(img) = rawler::analyze::extract_preview_pixels(
            path,
            &rawler::decoders::RawDecodeParams::default(),
        ) {
            return Ok(img);
        }
        // 4. Try embedded preview with lower res threshold
        if let Some(img) = crate::file_management::try_load_embedded_raw_preview(path, 720) {
            return Ok(img);
        }
    }
    if let Ok(img) = image::open(path) {
        return Ok(img);
    }
    crate::file_management::get_cached_or_generate_thumbnail_image(&path_str, app_handle, None)
        .map_err(|e| e.to_string())
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
    let job_control = crate::app_state::BackgroundJobControl::new();
    state
        .background_job_controls
        .lock()
        .unwrap()
        .insert(job_id.clone(), job_control.clone());
    let app = app_handle.clone();
    let event_job_id = job_id.clone();
    let worker_job_id = job_id.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let ai_semaphore = app.state::<crate::AppState>().ai_job_semaphore.clone();
        let result = run_face_detection(
            &app,
            &db_path,
            root_id,
            &model_path,
            &worker_job_id,
            &job_control,
            ai_semaphore,
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
        app.state::<crate::AppState>()
            .background_job_controls
            .lock()
            .unwrap()
            .remove(&worker_job_id);
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
    let job_control = crate::app_state::BackgroundJobControl::new();
    state
        .background_job_controls
        .lock()
        .unwrap()
        .insert(job_id.clone(), job_control.clone());
    let app = app_handle.clone();
    let worker_job_id = job_id.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let ai_semaphore = app.state::<crate::AppState>().ai_job_semaphore.clone();
        let result = run_face_recognition(
            &app,
            &db_path,
            root_id,
            &model_path,
            &worker_job_id,
            &job_control,
            ai_semaphore,
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
            .background_job_controls
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
    app_handle: &AppHandle,
    db_path: &Path,
    root_id: Option<i64>,
    model_path: &Path,
    job_id: &str,
    job_control: &std::sync::Arc<crate::app_state::BackgroundJobControl>,
    ai_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
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
        cluster_unknown_faces(&conn)?;
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
        if !tauri::async_runtime::block_on(job_control.wait_until_runnable()) {
            return Err("Face recognition cancelled".to_string());
        }
        if *job_control.cancellation_receiver().borrow() {
            return Err("Face recognition cancelled".to_string());
        }
        let path = Path::new(&root).join(&relative);
        let current = index as i64 + 1;
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("image");
        let image = match load_image_for_face_ai(&path, app_handle) {
            Ok(image) => image,
            Err(e) => {
                let _ = update_job(
                    db_path,
                    job_id,
                    "running",
                    &format!("{file_name}: Skipped ({e})"),
                    current,
                    total,
                    Some(&path.to_string_lossy()),
                    None,
                );
                continue;
            }
        };
        let _permit = tauri::async_runtime::block_on(ai_semaphore.clone().acquire_owned()).ok();
        let embedding_result = extract_sface_embedding(&image, x, y, width, height, &mut session);
        drop(_permit);
        let embedding = match embedding_result {
            Ok(embedding) => {
                let _ = update_job(
                    db_path,
                    job_id,
                    "running",
                    &format!("{file_name}: Extracted face embedding"),
                    current,
                    total,
                    Some(&path.to_string_lossy()),
                    None,
                );
                embedding
            }
            Err(e) => {
                let _ = update_job(
                    db_path,
                    job_id,
                    "running",
                    &format!("{file_name}: Embedding failed ({e})"),
                    current,
                    total,
                    Some(&path.to_string_lossy()),
                    None,
                );
                continue;
            }
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

        // Mirror embedding to companion face in RAW+JPG pair
        let _ = conn.execute(
            "UPDATE faces SET embedding_id = ?1, updated_at = strftime('%s','now')
             WHERE id IN (
                 SELECT f2.id FROM faces f2
                 JOIN images i2 ON i2.id = f2.image_id
                 JOIN faces f1 ON f1.id = ?2
                 JOIN images i1 ON i1.id = f1.image_id
                 WHERE i2.folder_id = i1.folder_id
                   AND i2.id != i1.id
                   AND substr(i2.file_name, 1, instr(i2.file_name || '.', '.') - 1) = substr(i1.file_name, 1, instr(i1.file_name || '.', '.') - 1)
                   AND abs(f2.bbox_x - f1.bbox_x) < 0.05
                   AND abs(f2.bbox_y - f1.bbox_y) < 0.05
             )",
            params![embedding_id, face_id],
        );
    }
    suggest_people(&conn)?;
    cluster_unknown_faces(&conn)?;
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

struct ScannedImageRecord {
    id: i64,
    root: String,
    relative: String,
    file_name: String,
    folder_id: i64,
    is_raw: bool,
}

fn run_face_detection(
    app_handle: &AppHandle,
    db_path: &Path,
    root_id: Option<i64>,
    model_path: &Path,
    job_id: &str,
    job_control: &std::sync::Arc<crate::app_state::BackgroundJobControl>,
    ai_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
) -> Result<(), String> {
    let conn = rusqlite::Connection::open(db_path).map_err(|error| error.to_string())?;
    let mut sql = "SELECT i.id, r.absolute_path, i.relative_path, i.folder_id, i.file_name, i.is_raw FROM images i JOIN collection_roots r ON r.id = i.root_id WHERE i.status = 'present'".to_string();
    if root_id.is_some() {
        sql.push_str(" AND i.root_id = ?1");
    }
    let mut statement = conn.prepare(&sql).map_err(|error| error.to_string())?;
    let raw_images: Vec<ScannedImageRecord> = if let Some(root_id) = root_id {
        statement
            .query_map([root_id], |row| {
                Ok(ScannedImageRecord {
                    id: row.get(0)?,
                    root: row.get(1)?,
                    relative: row.get(2)?,
                    folder_id: row.get(3)?,
                    file_name: row.get(4)?,
                    is_raw: row.get::<_, i64>(5)? != 0,
                })
            })
            .map_err(|error| error.to_string())?
            .collect::<Result<_, _>>()
            .map_err(|error| error.to_string())?
    } else {
        statement
            .query_map([], |row| {
                Ok(ScannedImageRecord {
                    id: row.get(0)?,
                    root: row.get(1)?,
                    relative: row.get(2)?,
                    folder_id: row.get(3)?,
                    file_name: row.get(4)?,
                    is_raw: row.get::<_, i64>(5)? != 0,
                })
            })
            .map_err(|error| error.to_string())?
            .collect::<Result<_, _>>()
            .map_err(|error| error.to_string())?
    };

    // Group images by (folder_id, lower(file_stem)) to identify RAW+JPG pairs
    let mut paired_groups: std::collections::BTreeMap<(i64, String), Vec<ScannedImageRecord>> =
        std::collections::BTreeMap::new();
    for item in raw_images {
        let stem = Path::new(&item.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&item.file_name)
            .to_lowercase();
        paired_groups
            .entry((item.folder_id, stem))
            .or_default()
            .push(item);
    }

    // For each group, prioritize JPG (raster) over RAW as the primary scan target
    let mut scan_tasks: Vec<(ScannedImageRecord, Vec<i64>)> = Vec::new();
    for (_key, mut items) in paired_groups {
        items.sort_by_key(|img| {
            let lower = img.file_name.to_lowercase();
            let is_jpg = lower.ends_with(".jpg") || lower.ends_with(".jpeg");
            (!is_jpg, img.is_raw)
        });
        let primary = items.remove(0);
        let companion_ids: Vec<i64> = items.into_iter().map(|it| it.id).collect();
        scan_tasks.push((primary, companion_ids));
    }

    let total = scan_tasks.len() as i64;
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
    for (index, (primary, companion_ids)) in scan_tasks.into_iter().enumerate() {
        if !tauri::async_runtime::block_on(job_control.wait_until_runnable()) {
            return Err("Face detection cancelled".to_string());
        }
        if *job_control.cancellation_receiver().borrow() {
            return Err("Face detection cancelled".to_string());
        }
        let path = Path::new(&primary.root).join(&primary.relative);
        let current = index as i64 + 1;
        let file_name = primary.file_name.as_str();
        let (detections, loaded_image) = match load_image_for_face_ai(&path, app_handle) {
            Ok(image) => {
                let _permit =
                    tauri::async_runtime::block_on(ai_semaphore.clone().acquire_owned()).ok();
                let res = detect_yunet(&image, &mut session);
                drop(_permit);
                match res {
                    Ok(detections) => (detections, Some(image)),
                    Err(e) => return Err(e),
                }
            }
            Err(e) => {
                let _ = update_job(
                    db_path,
                    job_id,
                    "running",
                    &format!("{file_name}: Skipped ({e})"),
                    current,
                    total,
                    Some(&path.to_string_lossy()),
                    None,
                );
                continue;
            }
        };
        let face_count = detections.len();
        let detection_msg = match face_count {
            0 => format!("{file_name}: No faces detected"),
            1 => format!(
                "{file_name}: 1 face detected ({:.0}% confidence)",
                detections[0].confidence * 100.0
            ),
            n => format!("{file_name}: {n} faces detected"),
        };
        let _ = update_job(
            db_path,
            job_id,
            "running",
            &detection_msg,
            current,
            total,
            Some(&path.to_string_lossy()),
            None,
        );

        // Delete unreviewed faces for primary and all companions in the pair
        conn.execute("DELETE FROM faces WHERE image_id = ?1 AND model_pack_id = 'opencv-yunet-sface' AND source = 'local' AND review_state = 'unreviewed'", [primary.id]).map_err(|error| error.to_string())?;
        for &comp_id in &companion_ids {
            let _ = conn.execute("DELETE FROM faces WHERE image_id = ?1 AND model_pack_id = 'opencv-yunet-sface' AND source = 'local' AND review_state = 'unreviewed'", [comp_id]);
        }

        for detection in detections {
            let thumb_bytes = if let Some(ref img) = loaded_image {
                extract_square_face_crop_jpeg(
                    img,
                    detection.x as f64,
                    detection.y as f64,
                    detection.width as f64,
                    detection.height as f64,
                    256,
                )
                .ok()
            } else {
                None
            };
            conn.execute(
                "INSERT INTO faces(image_id, model_pack_id, detector_confidence, bbox_x, bbox_y, bbox_width, bbox_height, landmarks_json, thumbnail_jpeg, review_state, source, created_at, updated_at) VALUES(?1, 'opencv-yunet-sface', ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'unreviewed', 'local', strftime('%s','now'), strftime('%s','now'))",
                params![
                    primary.id,
                    detection.confidence,
                    detection.x,
                    detection.y,
                    detection.width,
                    detection.height,
                    serde_json::to_string(&detection.landmarks).map_err(|error| error.to_string())?,
                    thumb_bytes.as_ref(),
                ],
            ).map_err(|error| error.to_string())?;
            let face_id = conn.last_insert_rowid();
            if let Some(ref img) = loaded_image {
                let _ = save_face_crop_image(
                    img,
                    detection.x as f64,
                    detection.y as f64,
                    detection.width as f64,
                    detection.height as f64,
                    face_id,
                    app_handle,
                );
            }

            // Mirror face detection to companion RAW images in the same pair
            for &comp_id in &companion_ids {
                let _ = conn.execute(
                    "INSERT INTO faces(image_id, model_pack_id, detector_confidence, bbox_x, bbox_y, bbox_width, bbox_height, landmarks_json, thumbnail_jpeg, review_state, source, created_at, updated_at) VALUES(?1, 'opencv-yunet-sface', ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'unreviewed', 'local', strftime('%s','now'), strftime('%s','now'))",
                    params![
                        comp_id,
                        detection.confidence,
                        detection.x,
                        detection.y,
                        detection.width,
                        detection.height,
                        serde_json::to_string(&detection.landmarks).map_err(|error| error.to_string())?,
                        thumb_bytes.as_ref(),
                    ],
                );
                let comp_face_id = conn.last_insert_rowid();
                if let Some(ref img) = loaded_image {
                    let _ = save_face_crop_image(
                        img,
                        detection.x as f64,
                        detection.y as f64,
                        detection.width as f64,
                        detection.height as f64,
                        comp_face_id,
                        app_handle,
                    );
                }
            }
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

    #[test]
    fn normalized_embeddings_have_unit_similarity_with_themselves() {
        let embedding = normalize_embedding(vec![3.0, 4.0]).unwrap();
        assert!((cosine_similarity(&embedding, &embedding) - 1.0).abs() < 0.000_1);
    }

    #[test]
    fn cosine_similarity_rejects_mismatched_dimensions() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0]), -1.0);
    }

    #[test]
    fn orthogonal_embeddings_have_zero_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn opposite_embeddings_have_negative_unit_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), -1.0);
    }

    #[test]
    fn face_pose_estimate_rewards_centered_nose_and_small_roll() {
        let frontal = estimate_face_pose(
            [
                [0.40, 0.40],
                [0.60, 0.40],
                [0.50, 0.52],
                [0.43, 0.63],
                [0.57, 0.63],
            ],
            0.30,
            0.30,
        )
        .unwrap();
        let turned = estimate_face_pose(
            [
                [0.40, 0.40],
                [0.60, 0.40],
                [0.57, 0.52],
                [0.43, 0.63],
                [0.57, 0.63],
            ],
            0.30,
            0.30,
        )
        .unwrap();
        assert!(frontal.frontal_score > turned.frontal_score);
        assert!(frontal.roll_degrees < 0.01);
    }

    #[test]
    fn stored_face_pose_rejects_invalid_landmarks() {
        assert!(estimate_stored_face_pose("not-json", 0.2, 0.2).is_none());
    }

    #[test]
    fn test_yunet_detection() {
        let model_path = Path::new(
            "/home/ssarangi/.local/share/io.github.CyberTimon.RapidRAW/models/face/opencv-yunet-sface/face_detection_yunet_2023mar.onnx",
        );
        if !model_path.exists() {
            return;
        }
        let _ = ort::init().commit();
        let mut session = Session::builder()
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();

        let test_img_path = "/home/ssarangi/Pictures/Shadow & Simba & Sachit & Soumya/DSCF8273.JPG";
        if let Ok(img) = image::open(test_img_path) {
            let detections = detect_yunet(&img, &mut session).unwrap();
            println!("test_yunet_detection found {} faces:", detections.len());
            for (i, det) in detections.iter().enumerate() {
                println!(
                    "  Face {i}: conf={:.2}%, bbox=({:.3}, {:.3}, {:.3}, {:.3})",
                    det.confidence * 100.0,
                    det.x,
                    det.y,
                    det.width,
                    det.height
                );
            }
            assert_eq!(
                detections.len(),
                1,
                "Expected exactly 1 face detected in test photo"
            );
        }
    }
}
