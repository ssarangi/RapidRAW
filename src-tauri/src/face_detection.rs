use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use image::{DynamicImage, Rgb, RgbImage, imageops::FilterType};
use ndarray::{Array4, ArrayViewD};
use ort::session::Session;
use ort::value::Tensor;
use rusqlite::{params, types::Value};
use tauri::{AppHandle, Emitter, Manager};

use crate::face_model_registry::FaceModelPackId;
use crate::library_db::{active_library_path, clear_face_index, create_background_job, update_job};

const INPUT_SIZE: u32 = 640;
const SFACE_INPUT_SIZE: u32 = 112;
const INSIGHTFACE_INPUT_SIZE: u32 = 640;
const YUNET_REVIEW_THRESHOLD: f32 = 0.65;
const YUNET_MIN_PROFILE_ASPECT_RATIO: f32 = 0.30;
const YUNET_MAX_FACE_ASPECT_RATIO: f32 = 1.90;
const YUNET_MIN_PROFILE_EYE_DISTANCE_RATIO: f32 = 0.07;
const ARCFACE_FIVE_POINT_TEMPLATE: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

fn configured_face_model_policy(
    db_path: &Path,
) -> crate::face_model_registry::FaceModelSelectionPolicy {
    crate::library_db::face_model_policy_for_database(db_path)
        .unwrap_or(crate::face_model_registry::FaceModelSelectionPolicy::Accuracy)
}

/// Ad-hoc folder culling has no catalog in which to persist a face-index
/// choice, so it continues to use the application default.
fn configured_local_face_model_policy(
    app_handle: &AppHandle,
) -> crate::face_model_registry::FaceModelSelectionPolicy {
    crate::load_settings(app_handle.clone())
        .ok()
        .and_then(|settings| settings.face_model_policy)
        .unwrap_or(crate::face_model_registry::FaceModelSelectionPolicy::Accuracy)
}

#[derive(Debug, Clone)]
struct Detection {
    confidence: f32,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    landmarks: [[f32; 2]; 5],
}

fn landmarks_support_alignment(landmarks: &[[f32; 2]; 5], bbox_width: f64) -> bool {
    let eye_distance = ((landmarks[0][0] - landmarks[1][0]).powi(2)
        + (landmarks[0][1] - landmarks[1][1]).powi(2))
    .sqrt();
    landmarks
        .iter()
        .flatten()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        && eye_distance >= (bbox_width as f32 * 0.12)
}

fn bilinear_pixel(image: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
    let x = x.clamp(0.0, image.width().saturating_sub(1) as f32);
    let y = y.clamp(0.0, image.height().saturating_sub(1) as f32);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width() - 1);
    let y1 = (y0 + 1).min(image.height() - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let p00 = image.get_pixel(x0, y0);
    let p10 = image.get_pixel(x1, y0);
    let p01 = image.get_pixel(x0, y1);
    let p11 = image.get_pixel(x1, y1);
    Rgb(std::array::from_fn(|channel| {
        let top = p00[channel] as f32 * (1.0 - fx) + p10[channel] as f32 * fx;
        let bottom = p01[channel] as f32 * (1.0 - fx) + p11[channel] as f32 * fx;
        (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8
    }))
}

/// Five-point similarity alignment shared by SFace and ArcFace. Returning
/// `None` for compressed/occluded profile landmarks is intentional: forcing
/// an affine transform in that case creates a worse embedding than a padded,
/// aspect-preserving fallback crop.
fn align_five_point_face(image: &DynamicImage, landmarks: [[f32; 2]; 5]) -> Option<RgbImage> {
    let source = image.to_rgb8();
    let src_points: [[f32; 2]; 5] =
        landmarks.map(|[x, y]| [x * source.width() as f32, y * source.height() as f32]);
    let src_mean = src_points.iter().fold([0.0; 2], |mut total, point| {
        total[0] += point[0] / 5.0;
        total[1] += point[1] / 5.0;
        total
    });
    let dst_mean = ARCFACE_FIVE_POINT_TEMPLATE
        .iter()
        .fold([0.0; 2], |mut total, point| {
            total[0] += point[0] / 5.0;
            total[1] += point[1] / 5.0;
            total
        });
    let (mut scale_cos, mut scale_sin, mut denominator) = (0.0, 0.0, 0.0);
    for (source_point, destination_point) in src_points.iter().zip(ARCFACE_FIVE_POINT_TEMPLATE) {
        let sx = source_point[0] - src_mean[0];
        let sy = source_point[1] - src_mean[1];
        let dx = destination_point[0] - dst_mean[0];
        let dy = destination_point[1] - dst_mean[1];
        scale_cos += sx * dx + sy * dy;
        scale_sin += sx * dy - sy * dx;
        denominator += sx * sx + sy * sy;
    }
    if denominator <= f32::EPSILON {
        return None;
    }
    let a = scale_cos / denominator;
    let b = scale_sin / denominator;
    let magnitude = a * a + b * b;
    if !magnitude.is_finite() || magnitude <= f32::EPSILON {
        return None;
    }
    let tx = dst_mean[0] - a * src_mean[0] + b * src_mean[1];
    let ty = dst_mean[1] - b * src_mean[0] - a * src_mean[1];
    let mut aligned = RgbImage::new(SFACE_INPUT_SIZE, SFACE_INPUT_SIZE);
    for y in 0..SFACE_INPUT_SIZE {
        for x in 0..SFACE_INPUT_SIZE {
            let dx = x as f32 - tx;
            let dy = y as f32 - ty;
            aligned.put_pixel(
                x,
                y,
                bilinear_pixel(
                    &source,
                    (a * dx + b * dy) / magnitude,
                    (-b * dx + a * dy) / magnitude,
                ),
            );
        }
    }
    Some(aligned)
}

/// Geometry available from YuNet's five landmarks. This is deliberately a
/// pose/framing estimate, not an eye-state or expression classifier.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FacePoseEstimate {
    pub frontal_score: f32,
    pub roll_degrees: f32,
    pub frame_fraction: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CullingFaceAnalysis {
    pub face_count: usize,
    pub best_pose: Option<FacePoseEstimate>,
    /// Eye openness for the SAME face `best_pose` was computed from (the
    /// most frontal/prominent detected face), not an average across every
    /// face in frame - a background bystander blinking shouldn't affect the
    /// primary subject's verdict.
    pub eye_openness: Option<crate::eye_state::EyeOpennessEstimate>,
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

fn scrfd_tensor_value(tensor: &ArrayViewD<'_, f32>, row: usize, column: usize) -> Option<f32> {
    match tensor.ndim() {
        2 => Some(tensor[[row, column]]),
        3 => Some(tensor[[0, row, column]]),
        _ => None,
    }
}

fn scrfd_tensor_rows(tensor: &ArrayViewD<'_, f32>) -> Option<usize> {
    match tensor.ndim() {
        2 => Some(tensor.shape()[0]),
        3 if tensor.shape()[0] == 1 => Some(tensor.shape()[1]),
        _ => None,
    }
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

            // This is a review-first catalog detector. Profiles consistently
            // score below the conservative OpenCV Zoo recommendation, so keep
            // candidates at 0.65 and rely on landmarks/face review downstream.
            if score < YUNET_REVIEW_THRESHOLD {
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

            // A turned face is narrower in image space. Keep thin profile
            // boxes while retaining a cap that rejects elongated patterns.
            if orig_w < 16.0
                || orig_h < 16.0
                || norm_w / norm_h < YUNET_MIN_PROFILE_ASPECT_RATIO
                || norm_w / norm_h > YUNET_MAX_FACE_ASPECT_RATIO
            {
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
            if eye_dist < norm_w * YUNET_MIN_PROFILE_EYE_DISTANCE_RATIO {
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

/// Decode the SCRFD ONNX contract used by the InsightFace packs.  SCRFD emits
/// score, LTRB distance and five-landmark tensors for strides 8/16/32, with
/// two anchors per spatial location.  It is intentionally a separate decoder
/// from YuNet: output tensors and box parameterisation are not compatible.
fn detect_scrfd(image: &DynamicImage, session: &mut Session) -> Result<Vec<Detection>, String> {
    let (original_width, original_height) = (image.width(), image.height());
    if original_width == 0 || original_height == 0 {
        return Ok(Vec::new());
    }
    let scale = (INSIGHTFACE_INPUT_SIZE as f32 / original_width as f32)
        .min(INSIGHTFACE_INPUT_SIZE as f32 / original_height as f32);
    let scaled_w =
        ((original_width as f32 * scale).round() as u32).clamp(1, INSIGHTFACE_INPUT_SIZE);
    let scaled_h =
        ((original_height as f32 * scale).round() as u32).clamp(1, INSIGHTFACE_INPUT_SIZE);
    let pad_x = (INSIGHTFACE_INPUT_SIZE - scaled_w) / 2;
    let pad_y = (INSIGHTFACE_INPUT_SIZE - scaled_h) / 2;
    let resized = image
        .resize_exact(scaled_w, scaled_h, FilterType::Triangle)
        .to_rgb8();
    // The source letterbox is black (0 in BGR); initialize after applying
    // SCRFD's normalization so padded pixels remain black rather than gray.
    let mut input = Array4::<f32>::from_elem(
        (
            1,
            3,
            INSIGHTFACE_INPUT_SIZE as usize,
            INSIGHTFACE_INPUT_SIZE as usize,
        ),
        -127.5 / 128.0,
    );
    for (x, y, pixel) in resized.enumerate_pixels() {
        input[[0, 0, (y + pad_y) as usize, (x + pad_x) as usize]] =
            (pixel[2] as f32 - 127.5) / 128.0;
        input[[0, 1, (y + pad_y) as usize, (x + pad_x) as usize]] =
            (pixel[1] as f32 - 127.5) / 128.0;
        input[[0, 2, (y + pad_y) as usize, (x + pad_x) as usize]] =
            (pixel[0] as f32 - 127.5) / 128.0;
    }
    let outputs = session
        .run(ort::inputs![
            Tensor::from_array(input).map_err(|e| e.to_string())?
        ])
        .map_err(|e| e.to_string())?;
    if outputs.len() != 9 {
        return Err(format!(
            "SCRFD expected 9 output tensors, received {}",
            outputs.len()
        ));
    }
    let mut candidates = Vec::new();
    // SCRFD orders all score tensors first, then all box tensors, then all
    // landmark tensors. The exported Buffalo SC model uses rank-2 tensors;
    // other ORT builds may add a batch dimension, which the helpers accept.
    for stride_index in 0..3 {
        let scores = outputs[stride_index]
            .try_extract_array::<f32>()
            .map_err(|e| e.to_string())?;
        let boxes = outputs[stride_index + 3]
            .try_extract_array::<f32>()
            .map_err(|e| e.to_string())?;
        let keypoints = outputs[stride_index + 6]
            .try_extract_array::<f32>()
            .map_err(|e| e.to_string())?;
        let Some(count) = scrfd_tensor_rows(&scores) else {
            return Err("SCRFD score tensor contract is invalid".to_string());
        };
        if scrfd_tensor_rows(&boxes) != Some(count)
            || scrfd_tensor_rows(&keypoints) != Some(count)
            || boxes.shape().last() != Some(&4)
            || keypoints.shape().last() != Some(&10)
        {
            return Err("SCRFD output tensor contract is invalid".to_string());
        }
        let grid = ((count / 2) as f32).sqrt().round() as u32;
        if grid == 0 || grid * grid * 2 != count as u32 || INSIGHTFACE_INPUT_SIZE % grid != 0 {
            return Err(format!(
                "SCRFD cannot infer anchor grid from {count} candidates"
            ));
        }
        let stride = INSIGHTFACE_INPUT_SIZE / grid;
        for index in 0..count {
            let score = scrfd_tensor_value(&scores, index, 0)
                .expect("validated SCRFD score tensor")
                .clamp(0.0, 1.0);
            // Recovery-friendly threshold; matching stays review-first.
            if score < 0.55 {
                continue;
            }
            let location = index as u32 / 2;
            let cx = (location % grid) as f32 * stride as f32;
            let cy = (location / grid) as f32 * stride as f32;
            let left = cx
                - scrfd_tensor_value(&boxes, index, 0).expect("validated SCRFD box tensor")
                    * stride as f32;
            let top = cy
                - scrfd_tensor_value(&boxes, index, 1).expect("validated SCRFD box tensor")
                    * stride as f32;
            let right = cx
                + scrfd_tensor_value(&boxes, index, 2).expect("validated SCRFD box tensor")
                    * stride as f32;
            let bottom = cy
                + scrfd_tensor_value(&boxes, index, 3).expect("validated SCRFD box tensor")
                    * stride as f32;
            let x = ((left - pad_x as f32) / scale).max(0.0);
            let y = ((top - pad_y as f32) / scale).max(0.0);
            let width = ((right - left) / scale)
                .min(original_width as f32 - x)
                .max(0.0);
            let height = ((bottom - top) / scale)
                .min(original_height as f32 - y)
                .max(0.0);
            if width < 12.0 || height < 12.0 {
                continue;
            }
            let mut landmarks = [[0.0; 2]; 5];
            for point in 0..5 {
                landmarks[point] = [
                    (((cx
                        + scrfd_tensor_value(&keypoints, index, point * 2)
                            .expect("validated SCRFD keypoint tensor")
                            * stride as f32
                        - pad_x as f32)
                        / scale)
                        / original_width as f32)
                        .clamp(0.0, 1.0),
                    (((cy
                        + scrfd_tensor_value(&keypoints, index, point * 2 + 1)
                            .expect("validated SCRFD keypoint tensor")
                            * stride as f32
                        - pad_y as f32)
                        / scale)
                        / original_height as f32)
                        .clamp(0.0, 1.0),
                ];
            }
            candidates.push(Detection {
                confidence: score,
                x: x / original_width as f32,
                y: y / original_height as f32,
                width: width / original_width as f32,
                height: height / original_height as f32,
                landmarks,
            });
        }
    }
    candidates.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut accepted = Vec::new();
    for candidate in candidates {
        if accepted
            .iter()
            .all(|existing| intersection_over_union(existing, &candidate) < 0.40)
        {
            accepted.push(candidate);
        }
    }
    Ok(accepted)
}

fn detect_for_pack(
    image: &DynamicImage,
    pack_id: FaceModelPackId,
    session: &mut Session,
) -> Result<Vec<Detection>, String> {
    if pack_id == FaceModelPackId::YuNetSFace {
        detect_yunet(image, session)
    } else {
        detect_scrfd(image, session)
    }
}

/// Lightweight detector used by filesystem culling. It deliberately returns
/// only a count: no catalog rows, embeddings, or person identities are
/// created for an ad-hoc folder scan.
pub(crate) struct LocalFaceDetector {
    pack_id: FaceModelPackId,
    session: std::sync::Mutex<Session>,
}

pub(crate) fn load_local_face_detector(
    app_handle: &AppHandle,
) -> Result<LocalFaceDetector, String> {
    let runtime = crate::face_model_registry::installed_face_runtime_paths(
        app_handle,
        configured_local_face_model_policy(app_handle),
    )?;
    let session = Session::builder()
        .map_err(|error| error.to_string())?
        .commit_from_file(runtime.detector)
        .map_err(|error| error.to_string())?;
    Ok(LocalFaceDetector {
        pack_id: runtime.pack_id,
        session: std::sync::Mutex::new(session),
    })
}

pub(crate) fn analyze_faces_for_culling(
    detector: &LocalFaceDetector,
    eye_classifier: Option<&crate::eye_state::EyeStateClassifier>,
    path: &Path,
    app_handle: &AppHandle,
) -> Result<CullingFaceAnalysis, String> {
    let image = load_image_for_face_ai(path, app_handle)?;
    let mut session = detector
        .session
        .lock()
        .map_err(|_| "Face detector session lock was poisoned".to_string())?;
    let detections = detect_for_pack(&image, detector.pack_id, &mut session)?;
    let best = detections
        .iter()
        .filter_map(|detection| {
            estimate_face_pose(detection.landmarks, detection.width, detection.height)
                .map(|pose| (detection, pose))
        })
        .max_by(|(_, left), (_, right)| {
            (left.frontal_score * 0.8 + left.frame_fraction * 0.2)
                .total_cmp(&(right.frontal_score * 0.8 + right.frame_fraction * 0.2))
        });
    let best_pose = best.map(|(_, pose)| pose);
    let eye_openness = match (best, eye_classifier) {
        (Some((detection, _)), Some(classifier)) => {
            crate::eye_state::classify_face_eye_state(classifier, &image, detection.landmarks)
        }
        _ => None,
    };
    Ok(CullingFaceAnalysis {
        face_count: detections.len(),
        best_pose,
        eye_openness,
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

const SFACE_SUGGESTION_THRESHOLD: f32 = 0.45;
const ARCFACE_SUGGESTION_THRESHOLD: f32 = 0.42;
const SFACE_CLUSTER_THRESHOLD: f32 = 0.48;
const ARCFACE_CLUSTER_THRESHOLD: f32 = 0.48;

fn suggestion_threshold(pack_id: FaceModelPackId) -> f32 {
    if pack_id == FaceModelPackId::YuNetSFace {
        SFACE_SUGGESTION_THRESHOLD
    } else {
        ARCFACE_SUGGESTION_THRESHOLD
    }
}

fn cluster_threshold(pack_id: FaceModelPackId) -> f32 {
    if pack_id == FaceModelPackId::YuNetSFace {
        SFACE_CLUSTER_THRESHOLD
    } else {
        ARCFACE_CLUSTER_THRESHOLD
    }
}

fn component_root(parents: &mut [usize], node: usize) -> usize {
    if parents[node] != node {
        parents[node] = component_root(parents, parents[node]);
    }
    parents[node]
}

fn merge_components(parents: &mut [usize], left: usize, right: usize) {
    let left_root = component_root(parents, left);
    let right_root = component_root(parents, right);
    if left_root != right_root {
        parents[right_root] = left_root;
    }
}

fn suggest_people(conn: &rusqlite::Connection, model_pack_id: &str) -> Result<(), String> {
    let pack_id = FaceModelPackId::try_from(model_pack_id)?;
    let mut exemplars: HashMap<i64, Vec<Vec<f32>>> = HashMap::new();
    let mut references = conn.prepare("SELECT f.person_id, e.vector FROM faces f JOIN face_embeddings e ON e.id = f.embedding_id WHERE f.review_state = 'confirmed' AND f.person_id IS NOT NULL AND e.model_pack_id = ?1").map_err(|error| error.to_string())?;
    for row in references
        .query_map([model_pack_id], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
        .map_err(|error| error.to_string())?
    {
        let (person_id, bytes) = row.map_err(|error| error.to_string())?;
        let Some(vector) = decode_embedding(&bytes) else {
            continue;
        };
        exemplars.entry(person_id).or_default().push(vector);
    }
    let mut candidates = conn.prepare("SELECT f.id, e.vector FROM faces f JOIN face_embeddings e ON e.id = f.embedding_id WHERE f.review_state = 'unreviewed' AND e.model_pack_id = ?1").map_err(|error| error.to_string())?;
    for row in candidates
        .query_map([model_pack_id], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
        .map_err(|error| error.to_string())?
    {
        let (face_id, bytes) = row.map_err(|error| error.to_string())?;
        let Some(vector) = decode_embedding(&bytes) else {
            continue;
        };
        let best = exemplars
            .iter()
            // A profile should be compared with every confirmed appearance of
            // a person; a centroid often erases the profile appearance mode.
            .filter_map(|(person_id, person_exemplars)| {
                person_exemplars
                    .iter()
                    .map(|reference| cosine_similarity(&vector, reference))
                    .max_by(f32::total_cmp)
                    .map(|score| (*person_id, score))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((person_id, similarity)) = best {
            if similarity >= suggestion_threshold(pack_id) {
                conn.execute("UPDATE faces SET person_id = ?1, updated_at = strftime('%s','now') WHERE id = ?2", params![person_id, face_id]).map_err(|error| error.to_string())?;
            }
        }
    }
    Ok(())
}

fn cluster_unknown_faces(
    conn: &rusqlite::Connection,
    model_pack_id: &str,
) -> Result<usize, String> {
    let pack_id = FaceModelPackId::try_from(model_pack_id)?;
    let mut statement = conn.prepare("SELECT f.id, e.vector FROM faces f JOIN face_embeddings e ON e.id = f.embedding_id WHERE f.review_state = 'unreviewed' AND e.model_pack_id = ?1 ORDER BY f.id").map_err(|error| error.to_string())?;
    let faces: Vec<(i64, Vec<f32>)> = statement
        .query_map([model_pack_id], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .filter_map(|(id, bytes)| decode_embedding(&bytes).map(|vector| (id, vector)))
        .collect();
    conn.execute(
        "DELETE FROM face_clusters WHERE model_pack_id = ?1 AND state = 'unreviewed'",
        [model_pack_id],
    )
    .map_err(|error| error.to_string())?;
    let mut parents: Vec<usize> = (0..faces.len()).collect();
    let threshold = cluster_threshold(pack_id);
    for left in 0..faces.len() {
        for right in (left + 1)..faces.len() {
            if cosine_similarity(&faces[left].1, &faces[right].1) >= threshold {
                merge_components(&mut parents, left, right);
            }
        }
    }
    let mut components: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for index in 0..faces.len() {
        let root = component_root(&mut parents, index);
        components.entry(root).or_default().push(index);
    }
    let mut clusters = 0;
    for member_indexes in components.into_values() {
        if member_indexes.len() < 2 {
            continue;
        }
        // Choose the medoid: it is less sensitive than file order when a
        // cluster contains frontal and profile appearances.
        let representative_index = *member_indexes
            .iter()
            .max_by(|left, right| {
                let average_similarity = |index: &usize| {
                    member_indexes
                        .iter()
                        .map(|other| cosine_similarity(&faces[*index].1, &faces[*other].1))
                        .sum::<f32>()
                        / member_indexes.len() as f32
                };
                average_similarity(left).total_cmp(&average_similarity(right))
            })
            .expect("cluster has at least two members");
        let (representative_id, representative) = &faces[representative_index];
        conn.execute("INSERT INTO face_clusters(model_pack_id, representative_face_id, created_at, updated_at) VALUES(?1, ?2, strftime('%s','now'), strftime('%s','now'))", params![model_pack_id, representative_id]).map_err(|error| error.to_string())?;
        let cluster_id = conn.last_insert_rowid();
        for member_index in member_indexes {
            let (face_id, member) = &faces[member_index];
            let similarity = cosine_similarity(representative, member);
            conn.execute("INSERT INTO face_cluster_members(cluster_id, face_id, similarity) VALUES(?1, ?2, ?3)", params![cluster_id, *face_id, similarity]).map_err(|error| error.to_string())?;
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
    landmarks: Option<[[f32; 2]; 5]>,
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
    let face = landmarks
        .filter(|points| landmarks_support_alignment(points, width))
        .and_then(|points| align_five_point_face(image, points))
        .unwrap_or_else(|| {
            image
                .crop_imm(left, top, crop_width, crop_height)
                .resize_exact(SFACE_INPUT_SIZE, SFACE_INPUT_SIZE, FilterType::Triangle)
                .to_rgb8()
        });
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

fn extract_arcface_embedding(
    image: &DynamicImage,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    landmarks: Option<[[f32; 2]; 5]>,
    session: &mut Session,
) -> Result<Vec<f32>, String> {
    let image_width = image.width() as f64;
    let image_height = image.height() as f64;
    let margin_x = width * 0.18;
    let margin_y = height * 0.18;
    let left = ((x - margin_x).max(0.0) * image_width).clamp(0.0, image_width - 1.0) as u32;
    let top = ((y - margin_y).max(0.0) * image_height).clamp(0.0, image_height - 1.0) as u32;
    let crop_width = ((width + margin_x * 2.0) * image_width)
        .max(1.0)
        .min(image_width - left as f64) as u32;
    let crop_height = ((height + margin_y * 2.0) * image_height)
        .max(1.0)
        .min(image_height - top as f64) as u32;
    let face = landmarks
        .filter(|points| landmarks_support_alignment(points, width))
        .and_then(|points| align_five_point_face(image, points))
        .unwrap_or_else(|| {
            image
                .crop_imm(left, top, crop_width, crop_height)
                .resize_exact(SFACE_INPUT_SIZE, SFACE_INPUT_SIZE, FilterType::Triangle)
                .to_rgb8()
        });
    let mut input =
        Array4::<f32>::zeros((1, 3, SFACE_INPUT_SIZE as usize, SFACE_INPUT_SIZE as usize));
    for (pixel_x, pixel_y, pixel) in face.enumerate_pixels() {
        input[[0, 0, pixel_y as usize, pixel_x as usize]] = (pixel[2] as f32 - 127.5) / 127.5;
        input[[0, 1, pixel_y as usize, pixel_x as usize]] = (pixel[1] as f32 - 127.5) / 127.5;
        input[[0, 2, pixel_y as usize, pixel_x as usize]] = (pixel[0] as f32 - 127.5) / 127.5;
    }
    let outputs = session
        .run(ort::inputs![
            Tensor::from_array(input).map_err(|e| e.to_string())?
        ])
        .map_err(|e| e.to_string())?;
    normalize_embedding(
        outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| e.to_string())?
            .iter()
            .copied()
            .collect(),
    )
}

fn extract_embedding_for_pack(
    image: &DynamicImage,
    pack_id: FaceModelPackId,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    landmarks: Option<[[f32; 2]; 5]>,
    session: &mut Session,
) -> Result<Vec<f32>, String> {
    if pack_id == FaceModelPackId::YuNetSFace {
        extract_sface_embedding(image, x, y, width, height, landmarks, session)
    } else {
        extract_arcface_embedding(image, x, y, width, height, landmarks, session)
    }
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

/// Loads an image suitable for local AI inference without requiring a Tauri
/// window. RAW files use embedded previews, a JPEG companion, or rawler.
/// Callers with an application handle may use `load_image_for_face_ai` for an
/// additional thumbnail-cache fallback.
fn load_oriented_raster_for_local_ai(path: &Path) -> Result<DynamicImage, String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    crate::image_loader::load_image_with_orientation(&bytes, None)
        .map_err(|error| error.to_string())
}

pub fn load_image_for_local_ai(path: &Path) -> Result<DynamicImage, String> {
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
                if let Ok(img) = load_oriented_raster_for_local_ai(&companion) {
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
    if let Ok(img) = load_oriented_raster_for_local_ai(path) {
        return Ok(img);
    }
    Err(format!(
        "Could not load an AI preview for {}",
        path.display()
    ))
}

pub fn load_image_for_face_ai(path: &Path, app_handle: &AppHandle) -> Result<DynamicImage, String> {
    match load_image_for_local_ai(path) {
        Ok(image) => Ok(image),
        Err(_) => crate::file_management::get_cached_or_generate_thumbnail_image(
            &path.to_string_lossy(),
            app_handle,
            None,
        )
        .map_err(|error| error.to_string()),
    }
}

#[tauri::command]
pub fn start_face_detection(
    root_id: Option<i64>,
    relative_path: Option<String>,
    only_pending: Option<bool>,
    app_handle: AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<String, String> {
    let db_path = active_library_path(&state)?;
    let job_id = create_background_job(
        &db_path,
        "face_detection",
        serde_json::json!({ "rootId": root_id, "relativePath": relative_path, "onlyPending": only_pending.unwrap_or(false) }),
    )?;
    crate::library_db::set_background_job_root_id(&db_path, &job_id, root_id)?;
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
        // Resolving an installed pack verifies its model artifacts. Keep that
        // disk work in the worker so the modal can close immediately after the
        // job has been durably queued.
        let result = crate::face_model_registry::installed_face_runtime_paths(
            &app,
            configured_face_model_policy(&db_path),
        )
        .and_then(|runtime| {
            run_face_detection(
                &app,
                &db_path,
                root_id,
                relative_path.as_deref(),
                only_pending.unwrap_or(false),
                runtime.pack_id.as_str(),
                &runtime.detector,
                &worker_job_id,
                &job_control,
                ai_semaphore.clone(),
            )
            .and_then(|_| {
                update_job(
                    &db_path,
                    &worker_job_id,
                    "running",
                    "Face scan complete; identifying faces",
                    0,
                    0,
                    None,
                    None,
                )?;
                // Detection and recognition are one catalog operation. The
                // recognizer selects every unembedded, already-detected face
                // in this scope as well as any face found by this scan.
                run_face_recognition(
                    &app,
                    &db_path,
                    root_id,
                    relative_path.as_deref(),
                    runtime.pack_id.as_str(),
                    &runtime.recognizer,
                    &worker_job_id,
                    &job_control,
                    ai_semaphore,
                )
            })
        });
        if let Err(error) = result {
            let job_state =
                if error == "Face detection cancelled" || error == "Face recognition cancelled" {
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
    let runtime = crate::face_model_registry::installed_face_runtime_paths(
        &app_handle,
        configured_face_model_policy(&db_path),
    )?;
    let job_id = create_background_job(
        &db_path,
        "face_recognition",
        serde_json::json!({ "rootId": root_id, "modelPackId": runtime.pack_id.as_str() }),
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
            None,
            runtime.pack_id.as_str(),
            &runtime.recognizer,
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

/// Rebuild the catalog's complete face index after the user deliberately
/// changes the catalog-scoped processing mode. Existing face rows and
/// embeddings are cleared as a single transaction before detector and
/// recognizer passes run with the newly selected compatible pair.
#[tauri::command]
pub fn reprocess_face_index(
    app_handle: AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<String, String> {
    let db_path = active_library_path(&state)?;
    let runtime = crate::face_model_registry::installed_face_runtime_paths(
        &app_handle,
        configured_face_model_policy(&db_path),
    )?;
    clear_face_index(&db_path)?;
    let job_id = create_background_job(
        &db_path,
        "face_reindex",
        serde_json::json!({ "modelPackId": runtime.pack_id.as_str(), "fullReprocess": true }),
    )?;
    let job_control = crate::app_state::BackgroundJobControl::new();
    state
        .background_job_controls
        .lock()
        .unwrap()
        .insert(job_id.clone(), job_control.clone());
    let app = app_handle.clone();
    let worker_job_id = job_id.clone();
    let event_job_id = job_id.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let ai_semaphore = app.state::<crate::AppState>().ai_job_semaphore.clone();
        let result = run_face_detection(
            &app,
            &db_path,
            None,
            None,
            false,
            runtime.pack_id.as_str(),
            &runtime.detector,
            &worker_job_id,
            &job_control,
            ai_semaphore.clone(),
        )
        .and_then(|_| {
            update_job(
                &db_path,
                &worker_job_id,
                "running",
                "Face detection complete; identifying faces",
                0,
                0,
                None,
                None,
            )?;
            run_face_recognition(
                &app,
                &db_path,
                None,
                None,
                runtime.pack_id.as_str(),
                &runtime.recognizer,
                &worker_job_id,
                &job_control,
                ai_semaphore,
            )
        });
        if let Err(error) = result {
            let job_state =
                if error == "Face detection cancelled" || error == "Face recognition cancelled" {
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
            "face-reindex-complete",
            serde_json::json!({ "jobId": event_job_id }),
        );
    });
    Ok(job_id)
}

fn run_face_recognition(
    app_handle: &AppHandle,
    db_path: &Path,
    root_id: Option<i64>,
    relative_path: Option<&str>,
    model_pack_id: &str,
    model_path: &Path,
    job_id: &str,
    job_control: &std::sync::Arc<crate::app_state::BackgroundJobControl>,
    ai_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
) -> Result<(), String> {
    run_face_recognition_with_loader(
        db_path,
        root_id,
        relative_path,
        model_pack_id,
        model_path,
        job_id,
        job_control,
        ai_semaphore,
        |path| load_image_for_face_ai(path, app_handle),
    )
}

/// Runs SFace embedding extraction without a Tauri window. Face crops are
/// already persisted as database thumbnails by detection, so recognition only
/// needs an app-independent image loader.
pub fn run_face_recognition_headless(
    db_path: &Path,
    root_id: Option<i64>,
    model_path: &Path,
    job_id: &str,
    job_control: &std::sync::Arc<crate::app_state::BackgroundJobControl>,
    ai_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
) -> Result<(), String> {
    run_face_recognition_headless_for_pack(
        db_path,
        root_id,
        FaceModelPackId::YuNetSFace.as_str(),
        model_path,
        job_id,
        job_control,
        ai_semaphore,
    )
}

pub fn run_face_recognition_headless_for_pack(
    db_path: &Path,
    root_id: Option<i64>,
    model_pack_id: &str,
    model_path: &Path,
    job_id: &str,
    job_control: &std::sync::Arc<crate::app_state::BackgroundJobControl>,
    ai_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
) -> Result<(), String> {
    run_face_recognition_with_loader(
        db_path,
        root_id,
        None,
        model_pack_id,
        model_path,
        job_id,
        job_control,
        ai_semaphore,
        load_image_for_local_ai,
    )
}

fn run_face_recognition_with_loader<F>(
    db_path: &Path,
    root_id: Option<i64>,
    relative_path: Option<&str>,
    model_pack_id: &str,
    model_path: &Path,
    job_id: &str,
    job_control: &std::sync::Arc<crate::app_state::BackgroundJobControl>,
    ai_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
    load_image: F,
) -> Result<(), String>
where
    F: Fn(&Path) -> Result<DynamicImage, String>,
{
    let conn = rusqlite::Connection::open(db_path).map_err(|error| error.to_string())?;
    let pack_id = FaceModelPackId::try_from(model_pack_id)?;
    let recognizer_model_id = pack_id.recognizer_model_id();
    // Embeddings are only comparable inside a single detector/recognizer
    // pack. This also repairs catalogs produced before pack-aware runtime
    // selection, where a paired RAW/JPEG face could inherit an embedding from
    // another model pack.
    conn.execute(
        "UPDATE faces
         SET embedding_id = NULL, recognizer_model_id = ?2, updated_at = strftime('%s','now')
         WHERE model_pack_id = ?1
           AND embedding_id IS NOT NULL
           AND EXISTS (
               SELECT 1 FROM face_embeddings e
               WHERE e.id = faces.embedding_id AND e.model_pack_id <> ?1
           )",
        [model_pack_id, recognizer_model_id],
    )
    .map_err(|error| error.to_string())?;
    let mut sql = "SELECT f.id, r.absolute_path, i.relative_path, f.bbox_x, f.bbox_y, f.bbox_width, f.bbox_height, f.landmarks_json FROM faces f JOIN images i ON i.id = f.image_id JOIN collection_roots r ON r.id = i.root_id WHERE i.status = 'present' AND f.model_pack_id = ? AND f.review_state <> 'rejected' AND f.embedding_id IS NULL".to_string();
    let mut query_params = vec![Value::Text(model_pack_id.to_string())];
    if let Some(root_id) = root_id {
        sql.push_str(" AND i.root_id = ?");
        query_params.push(Value::Integer(root_id));
    }
    if let Some(relative_path) = relative_path.filter(|path| *path != ".") {
        sql.push_str(" AND (i.relative_path = ? OR i.relative_path LIKE ?)");
        query_params.push(Value::Text(relative_path.to_string()));
        query_params.push(Value::Text(format!("{relative_path}/%")));
    }
    let mut statement = conn.prepare(&sql).map_err(|error| error.to_string())?;
    let faces: Vec<(i64, String, String, f64, f64, f64, f64, Option<String>)> = statement
        .query_map(rusqlite::params_from_iter(query_params.iter()), |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
                row.get(7)?,
            ))
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<_, _>>()
        .map_err(|error| error.to_string())?;
    let total = faces.len() as i64;
    update_job(
        db_path,
        job_id,
        "running",
        "Loading face recognizer",
        0,
        total,
        None,
        None,
    )?;
    if faces.is_empty() {
        suggest_people(&conn, model_pack_id)?;
        cluster_unknown_faces(&conn, model_pack_id)?;
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
    for (index, (face_id, root, relative, x, y, width, height, landmarks_json)) in
        faces.into_iter().enumerate()
    {
        if !tauri::async_runtime::block_on(job_control.wait_until_runnable()) {
            return Err("Face recognition cancelled".to_string());
        }
        if *job_control.cancellation_receiver().borrow() {
            return Err("Face recognition cancelled".to_string());
        }
        let path = Path::new(&root).join(&relative);
        let current = index as i64 + 1;
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("image");
        let image = match load_image(&path) {
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
        let landmarks =
            landmarks_json.and_then(|json| serde_json::from_str::<[[f32; 2]; 5]>(&json).ok());
        let embedding_result = extract_embedding_for_pack(
            &image,
            pack_id,
            x,
            y,
            width,
            height,
            landmarks,
            &mut session,
        );
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
        conn.execute("INSERT INTO face_embeddings(model_pack_id, recognizer_model_id, dimensions, vector, norm, created_at) VALUES(?1, ?2, ?3, ?4, 1.0, strftime('%s','now'))", params![model_pack_id, recognizer_model_id, embedding.len() as i64, bytes]).map_err(|error| error.to_string())?;
        let embedding_id = conn.last_insert_rowid();
        conn.execute(
            "UPDATE faces
             SET embedding_id = ?1, recognizer_model_id = ?2, updated_at = strftime('%s','now')
             WHERE id = ?3",
            params![embedding_id, recognizer_model_id, face_id],
        )
        .map_err(|error| error.to_string())?;

        // Mirror embedding to companion face in RAW+JPG pair
        let _ = conn.execute(
            "UPDATE faces
             SET embedding_id = ?1, recognizer_model_id = ?2, updated_at = strftime('%s','now')
             WHERE id IN (
                 SELECT f2.id FROM faces f2
                 JOIN images i2 ON i2.id = f2.image_id
                 JOIN faces f1 ON f1.id = ?3
                 JOIN images i1 ON i1.id = f1.image_id
                 WHERE i2.folder_id = i1.folder_id
                   AND i2.id != i1.id
                   AND f2.model_pack_id = f1.model_pack_id
                   AND substr(i2.file_name, 1, instr(i2.file_name || '.', '.') - 1) = substr(i1.file_name, 1, instr(i1.file_name || '.', '.') - 1)
                   AND abs(f2.bbox_x - f1.bbox_x) < 0.05
                   AND abs(f2.bbox_y - f1.bbox_y) < 0.05
             )",
            params![embedding_id, recognizer_model_id, face_id],
        );
    }
    suggest_people(&conn, model_pack_id)?;
    cluster_unknown_faces(&conn, model_pack_id)?;
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
    relative_path: Option<&str>,
    only_pending: bool,
    model_pack_id: &str,
    model_path: &Path,
    job_id: &str,
    job_control: &std::sync::Arc<crate::app_state::BackgroundJobControl>,
    ai_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
) -> Result<(), String> {
    run_face_detection_with_loader(
        db_path,
        root_id,
        relative_path,
        only_pending,
        model_pack_id,
        model_path,
        job_id,
        job_control,
        ai_semaphore,
        |path| load_image_for_face_ai(path, app_handle),
        |image, x, y, width, height, face_id| {
            let _ = save_face_crop_image(image, x, y, width, height, face_id, app_handle);
        },
    )
}

/// Runs YuNet detection without a Tauri window. Database face thumbnails are
/// retained; filesystem face crops are UI cache data and are intentionally not
/// written by a headless command.
pub fn run_face_detection_headless(
    db_path: &Path,
    root_id: Option<i64>,
    model_path: &Path,
    job_id: &str,
    job_control: &std::sync::Arc<crate::app_state::BackgroundJobControl>,
    ai_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
) -> Result<(), String> {
    run_face_detection_headless_for_pack(
        db_path,
        root_id,
        FaceModelPackId::YuNetSFace.as_str(),
        model_path,
        job_id,
        job_control,
        ai_semaphore,
    )
}

pub fn run_face_detection_headless_for_pack(
    db_path: &Path,
    root_id: Option<i64>,
    model_pack_id: &str,
    model_path: &Path,
    job_id: &str,
    job_control: &std::sync::Arc<crate::app_state::BackgroundJobControl>,
    ai_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
) -> Result<(), String> {
    run_face_detection_with_loader(
        db_path,
        root_id,
        None,
        false,
        model_pack_id,
        model_path,
        job_id,
        job_control,
        ai_semaphore,
        load_image_for_local_ai,
        |_, _, _, _, _, _| {},
    )
}

fn run_face_detection_with_loader<F, S>(
    db_path: &Path,
    root_id: Option<i64>,
    relative_path: Option<&str>,
    only_pending: bool,
    model_pack_id: &str,
    model_path: &Path,
    job_id: &str,
    job_control: &std::sync::Arc<crate::app_state::BackgroundJobControl>,
    ai_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
    load_image: F,
    save_crop: S,
) -> Result<(), String>
where
    F: Fn(&Path) -> Result<DynamicImage, String>,
    S: Fn(&DynamicImage, f64, f64, f64, f64, i64),
{
    let pack_id = FaceModelPackId::try_from(model_pack_id)?;
    let detector_model_id = pack_id.detector_model_id();
    let recognizer_model_id = pack_id.recognizer_model_id();
    let conn = rusqlite::Connection::open(db_path).map_err(|error| error.to_string())?;
    let mut sql = "SELECT i.id, r.absolute_path, i.relative_path, i.folder_id, i.file_name, i.is_raw FROM images i JOIN collection_roots r ON r.id = i.root_id WHERE i.status = 'present'".to_string();
    if root_id.is_some() {
        sql.push_str(" AND i.root_id = ?1");
    }
    let mut statement = conn.prepare(&sql).map_err(|error| error.to_string())?;
    let mut raw_images: Vec<ScannedImageRecord> = if let Some(root_id) = root_id {
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
    if let Some(relative_path) = relative_path.filter(|path| *path != ".") {
        let prefix = format!("{relative_path}/");
        raw_images.retain(|image| image.relative.starts_with(&prefix));
    }
    if only_pending {
        let mut state = conn
            .prepare(
                "SELECT image_id FROM face_scan_state WHERE status = 'complete'
                 UNION
                 SELECT image_id FROM faces",
            )
            .map_err(|error| error.to_string())?;
        let completed_ids = state
            .query_map([], |row| row.get::<_, i64>(0))
            .map_err(|error| error.to_string())?
            .collect::<Result<HashSet<_>, _>>()
            .map_err(|error| error.to_string())?;
        raw_images.retain(|image| !completed_ids.contains(&image.id));
    }

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
        "Loading face detector",
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
        let mut processed_ids = companion_ids.clone();
        processed_ids.push(primary.id);
        for image_id in &processed_ids {
            conn.execute(
                "INSERT INTO face_scan_state(image_id, model_pack_id, status, model_revision, error_message, processed_at, updated_at) VALUES(?1, ?2, 'processing', NULL, NULL, NULL, strftime('%s','now')) ON CONFLICT(image_id, model_pack_id) DO UPDATE SET status = 'processing', error_message = NULL, processed_at = NULL, updated_at = excluded.updated_at",
                params![image_id, model_pack_id],
            )
            .map_err(|error| error.to_string())?;
        }
        let (detections, loaded_image) = match load_image(&path) {
            Ok(image) => {
                let _permit =
                    tauri::async_runtime::block_on(ai_semaphore.clone().acquire_owned()).ok();
                let res = detect_for_pack(&image, pack_id, &mut session);
                drop(_permit);
                match res {
                    Ok(detections) => (detections, Some(image)),
                    Err(e) => return Err(e),
                }
            }
            Err(e) => {
                for image_id in &processed_ids {
                    let _ = conn.execute(
                        "UPDATE face_scan_state SET status = 'failed', error_message = ?3, processed_at = strftime('%s','now'), updated_at = strftime('%s','now') WHERE image_id = ?1 AND model_pack_id = ?2",
                        params![image_id, model_pack_id, &e],
                    );
                }
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
        conn.execute("DELETE FROM faces WHERE image_id = ?1 AND model_pack_id = ?2 AND source = 'local' AND review_state = 'unreviewed'", params![primary.id, model_pack_id]).map_err(|error| error.to_string())?;
        for &comp_id in &companion_ids {
            let _ = conn.execute("DELETE FROM faces WHERE image_id = ?1 AND model_pack_id = ?2 AND source = 'local' AND review_state = 'unreviewed'", params![comp_id, model_pack_id]);
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
                "INSERT INTO faces(image_id, model_pack_id, detector_model_id, recognizer_model_id, detector_confidence, bbox_x, bbox_y, bbox_width, bbox_height, landmarks_json, thumbnail_jpeg, review_state, source, created_at, updated_at) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 'unreviewed', 'local', strftime('%s','now'), strftime('%s','now'))",
                params![
                    primary.id,
                    model_pack_id,
                    detector_model_id,
                    recognizer_model_id,
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
                save_crop(
                    img,
                    detection.x as f64,
                    detection.y as f64,
                    detection.width as f64,
                    detection.height as f64,
                    face_id,
                );
            }

            // Mirror face detection to companion RAW images in the same pair
            for &comp_id in &companion_ids {
                let _ = conn.execute(
                    "INSERT INTO faces(image_id, model_pack_id, detector_model_id, recognizer_model_id, detector_confidence, bbox_x, bbox_y, bbox_width, bbox_height, landmarks_json, thumbnail_jpeg, review_state, source, created_at, updated_at) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 'unreviewed', 'local', strftime('%s','now'), strftime('%s','now'))",
                    params![
                        comp_id,
                        model_pack_id,
                        detector_model_id,
                        recognizer_model_id,
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
                    save_crop(
                        img,
                        detection.x as f64,
                        detection.y as f64,
                        detection.width as f64,
                        detection.height as f64,
                        comp_face_id,
                    );
                }
            }
        }
        for image_id in &processed_ids {
            conn.execute(
                "UPDATE face_scan_state SET status = 'complete', error_message = NULL, processed_at = strftime('%s','now'), updated_at = strftime('%s','now') WHERE image_id = ?1 AND model_pack_id = ?2",
                params![image_id, model_pack_id],
            )
            .map_err(|error| error.to_string())?;
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
    fn local_ai_loader_reads_regular_images_without_an_app_handle() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("fixture.png");
        image::RgbImage::from_pixel(12, 8, image::Rgb([12, 34, 56]))
            .save(&path)
            .unwrap();

        let image = load_image_for_local_ai(&path).unwrap();
        assert_eq!((image.width(), image.height()), (12, 8));
    }

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
    fn five_point_alignment_maps_a_valid_face_to_recognizer_size() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(160, 160, Rgb([50, 100, 150])));
        let landmarks = ARCFACE_FIVE_POINT_TEMPLATE.map(|[x, y]| [x / 112.0, y / 112.0]);
        let aligned = align_five_point_face(&image, landmarks).unwrap();
        assert_eq!(
            (aligned.width(), aligned.height()),
            (SFACE_INPUT_SIZE, SFACE_INPUT_SIZE)
        );
    }

    #[test]
    fn compressed_profile_landmarks_use_safe_crop_fallback() {
        let landmarks = [
            [0.50, 0.40],
            [0.51, 0.40],
            [0.52, 0.50],
            [0.50, 0.60],
            [0.53, 0.60],
        ];
        assert!(!landmarks_support_alignment(&landmarks, 0.25));
    }

    #[test]
    fn arcface_and_sface_keep_model_specific_suggestion_thresholds() {
        assert!(
            suggestion_threshold(FaceModelPackId::YuNetSFace)
                > suggestion_threshold(FaceModelPackId::InsightFaceAntelopeV2)
        );
    }

    #[test]
    fn connected_components_preserve_transitive_profile_links() {
        let mut parents: Vec<usize> = (0..3).collect();
        merge_components(&mut parents, 0, 1);
        merge_components(&mut parents, 1, 2);
        assert_eq!(
            component_root(&mut parents, 0),
            component_root(&mut parents, 2)
        );
    }

    #[test]
    #[ignore = "requires a locally installed ONNX Runtime and private developer fixture image"]
    fn test_yunet_detection() {
        let model_path =
            Path::new("/home/ssarangi/.local/share/io.github.CyberTimon.RapidRAW/models/face")
                .join(FaceModelPackId::YuNetSFace.as_str())
                .join(crate::face_model_registry::YUNET_DETECTOR_FILE);
        if !model_path.exists() {
            return;
        }
        let _ = ort::init().commit();
        let mut session = Session::builder()
            .unwrap()
            .commit_from_file(&model_path)
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
