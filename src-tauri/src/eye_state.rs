//! Open/semi-closed/closed eye classification for culling.
//!
//! Reuses the eye-center landmarks YuNet already produces during face
//! detection (see face_detection.rs) rather than running a second face
//! detector. YuNet only gives a point per eye, not a bounding box, so the
//! crop region here is a tuned heuristic sized off the inter-eye distance,
//! not something sourced from the classifier's own training data.
//!
//! Model: OCEC (https://github.com/PINTO0309/OCEC), MIT licensed, trained
//! on the "Open and Closed Eyes Dataset" (ODC-By) plus custom video data.
//! It outputs a single sigmoid `prob_open` score (not a hard label), which
//! is what lets this bucket results into three states instead of two.

use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::Array;
use ort::session::Session;
use ort::value::Tensor;
use std::sync::Mutex;
use tauri::{AppHandle, Manager};

const EYE_INPUT_WIDTH: u32 = 40;
const EYE_INPUT_HEIGHT: u32 = 24;

// Eye-region crop size relative to inter-eye distance. Tuned by inspection,
// not derived from OCEC's own training data (it was trained against a
// different upstream detector's own eye bounding boxes, which we don't
// have - see the module doc comment).
const EYE_CROP_WIDTH_RATIO: f32 = 0.55;

const OPEN_THRESHOLD: f32 = 0.7;
const CLOSED_THRESHOLD: f32 = 0.3;

pub(crate) struct EyeStateClassifier {
    session: Mutex<Session>,
}

pub(crate) fn load_eye_state_classifier(app_handle: &AppHandle) -> Result<EyeStateClassifier, String> {
    let visual_models_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("models")
        .join("visual");
    let model_path = crate::visual_model_registry::installed_visual_model_path_in_dir(
        &visual_models_dir,
        "ocec-eye-state",
        "ocec_s.onnx",
    )?;
    let session = Session::builder()
        .map_err(|error| error.to_string())?
        .commit_from_file(model_path)
        .map_err(|error| error.to_string())?;
    Ok(EyeStateClassifier {
        session: Mutex::new(session),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EyeState {
    Open,
    SemiClosed,
    Closed,
}

impl EyeState {
    fn from_prob_open(prob_open: f32) -> Self {
        if prob_open >= OPEN_THRESHOLD {
            EyeState::Open
        } else if prob_open >= CLOSED_THRESHOLD {
            EyeState::SemiClosed
        } else {
            EyeState::Closed
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            EyeState::Open => "Open",
            EyeState::SemiClosed => "Semi-closed",
            EyeState::Closed => "Closed",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EyeOpennessEstimate {
    /// The worse (more closed) of the two eyes - one closed eye still reads
    /// as "eyes closed" in a photo, so this deliberately isn't an average.
    pub prob_open: f32,
    pub state: EyeState,
}

fn crop_eye_region(
    image: &DynamicImage,
    eye_center: [f32; 2],
    inter_eye_distance: f32,
) -> Option<DynamicImage> {
    if !inter_eye_distance.is_finite() || inter_eye_distance <= 1.0 {
        return None;
    }
    let box_width = inter_eye_distance * EYE_CROP_WIDTH_RATIO;
    let box_height = box_width * (EYE_INPUT_HEIGHT as f32 / EYE_INPUT_WIDTH as f32);
    let (img_w, img_h) = image.dimensions();

    let x0 = (eye_center[0] - box_width / 2.0).max(0.0);
    let y0 = (eye_center[1] - box_height / 2.0).max(0.0);
    let x1 = (eye_center[0] + box_width / 2.0).min(img_w as f32);
    let y1 = (eye_center[1] + box_height / 2.0).min(img_h as f32);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(image.crop_imm(x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32))
}

/// Preprocessing matches OCEC's reference implementation exactly: resize to
/// (40, 24) with bilinear interpolation, scale to 0-1 (no mean/std
/// normalization), HWC -> CHW.
fn eye_input_tensor(crop: &DynamicImage) -> Array<f32, ndarray::Dim<[usize; 4]>> {
    let resized = crop
        .resize_exact(EYE_INPUT_WIDTH, EYE_INPUT_HEIGHT, FilterType::Triangle)
        .to_rgb8();
    let mut input = Array::zeros((1, 3, EYE_INPUT_HEIGHT as usize, EYE_INPUT_WIDTH as usize));
    for (x, y, pixel) in resized.enumerate_pixels() {
        for channel in 0..3 {
            input[[0, channel, y as usize, x as usize]] = pixel[channel] as f32 / 255.0;
        }
    }
    input
}

fn classify_eye_crop(session: &mut Session, crop: &DynamicImage) -> Result<f32, String> {
    let input = Tensor::from_array(eye_input_tensor(crop)).map_err(|error| error.to_string())?;
    let outputs = session
        .run(ort::inputs![input])
        .map_err(|error| error.to_string())?;
    let values = outputs[0]
        .try_extract_array::<f32>()
        .map_err(|error| error.to_string())?;
    let prob_open = values.iter().next().copied().unwrap_or(0.0);
    Ok(prob_open.clamp(0.0, 1.0))
}

/// Classifies eye openness for one detected face from its YuNet 5-point
/// landmarks (index 0 = right eye, index 1 = left eye). Returns `None` only
/// when neither eye could be cropped or classified (e.g. face too close to
/// the frame edge) - a single successfully-classified eye is enough to
/// return a result.
pub(crate) fn classify_face_eye_state(
    classifier: &EyeStateClassifier,
    image: &DynamicImage,
    landmarks: [[f32; 2]; 5],
) -> Option<EyeOpennessEstimate> {
    let right_eye = landmarks[0];
    let left_eye = landmarks[1];
    let inter_eye_distance = ((right_eye[0] - left_eye[0]).powi(2)
        + (right_eye[1] - left_eye[1]).powi(2))
    .sqrt();

    let mut session = classifier.session.lock().ok()?;
    let mut probs = Vec::new();
    for eye_center in [right_eye, left_eye] {
        if let Some(crop) = crop_eye_region(image, eye_center, inter_eye_distance) {
            if let Ok(prob_open) = classify_eye_crop(&mut session, &crop) {
                probs.push(prob_open);
            }
        }
    }
    let prob_open = probs.into_iter().fold(None, |worst: Option<f32>, prob| {
        Some(worst.map_or(prob, |w| w.min(prob)))
    })?;
    Some(EyeOpennessEstimate {
        prob_open,
        state: EyeState::from_prob_open(prob_open),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_state_buckets_at_the_documented_thresholds() {
        assert_eq!(EyeState::from_prob_open(0.95), EyeState::Open);
        assert_eq!(EyeState::from_prob_open(0.7), EyeState::Open);
        assert_eq!(EyeState::from_prob_open(0.69), EyeState::SemiClosed);
        assert_eq!(EyeState::from_prob_open(0.3), EyeState::SemiClosed);
        assert_eq!(EyeState::from_prob_open(0.29), EyeState::Closed);
        assert_eq!(EyeState::from_prob_open(0.0), EyeState::Closed);
    }

    #[test]
    fn crop_eye_region_rejects_degenerate_geometry() {
        let image = DynamicImage::new_rgb8(100, 100);
        assert!(crop_eye_region(&image, [50.0, 50.0], 0.0).is_none());
        assert!(crop_eye_region(&image, [50.0, 50.0], f32::NAN).is_none());
        // A real inter-eye distance produces an in-bounds crop even near an edge.
        assert!(crop_eye_region(&image, [2.0, 2.0], 20.0).is_some());
    }

    #[test]
    fn eye_input_tensor_has_the_shape_ocec_expects() {
        let image = DynamicImage::new_rgb8(64, 64);
        let tensor = eye_input_tensor(&image);
        assert_eq!(tensor.shape(), &[1, 3, EYE_INPUT_HEIGHT as usize, EYE_INPUT_WIDTH as usize]);
    }
}
