use serde::{Deserialize, Serialize};

/// The artifact can be fetched and prepared by RapidRAW without a conversion
/// step. Conversion-required models stay visible for evaluation, but are never
/// presented as ready to run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ModelAvailability {
    DirectDownload,
    ConversionRequired,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ModelArtifactFormat {
    Onnx,
    Zip,
    Checkpoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceModelArtifact {
    pub file_name: String,
    pub format: ModelArtifactFormat,
    pub source_url: String,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceModelPack {
    pub id: String,
    pub display_name: String,
    pub description: String,
    pub detector: String,
    pub recognizer: String,
    pub detector_landmarks: u8,
    pub embedding_dimensions: Option<u16>,
    pub availability: ModelAvailability,
    pub artifacts: Vec<FaceModelArtifact>,
    pub license_name: String,
    pub license_url: String,
    pub license_acknowledgement_required: bool,
    pub model_source_url: String,
}

fn artifact(file_name: &str, format: ModelArtifactFormat, source_url: &str) -> FaceModelArtifact {
    FaceModelArtifact {
        file_name: file_name.to_string(),
        format,
        source_url: source_url.to_string(),
        sha256: None,
    }
}

/// All face-model candidates are defined in one place. Direct-download packs
/// are the first runtime targets. Conversion-required packs remain visible so
/// an evaluator can compare them once a reproducible ONNX artifact is pinned.
pub fn face_model_packs() -> Vec<FaceModelPack> {
    vec![
        FaceModelPack {
            id: "opencv-yunet-sface".to_string(),
            display_name: "YuNet + SFace".to_string(),
            description: "Fast local baseline used by current digiKam releases.".to_string(),
            detector: "YuNet".to_string(),
            recognizer: "SFace".to_string(),
            detector_landmarks: 5,
            embedding_dimensions: Some(128),
            availability: ModelAvailability::DirectDownload,
            artifacts: vec![
                artifact(
                    "face_detection_yunet_2023mar.onnx",
                    ModelArtifactFormat::Onnx,
                    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                ),
                artifact(
                    "face_recognition_sface_2021dec.onnx",
                    ModelArtifactFormat::Onnx,
                    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
                ),
            ],
            license_name: "MIT (YuNet), Apache-2.0 (SFace)".to_string(),
            license_url: "https://github.com/opencv/opencv_zoo".to_string(),
            license_acknowledgement_required: false,
            model_source_url: "https://github.com/opencv/opencv_zoo".to_string(),
        },
        insightface_pack(
            "insightface-buffalo-sc",
            "InsightFace Buffalo SC",
            "Compact SCRFD + MobileFaceNet pack for fast experiments.",
            "buffalo_sc.zip",
            "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
        ),
        insightface_pack(
            "insightface-buffalo-s",
            "InsightFace Buffalo S",
            "Compact SCRFD + ArcFace pack for local experiments.",
            "buffalo_s.zip",
            "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
        ),
        insightface_pack(
            "insightface-buffalo-m",
            "InsightFace Buffalo M",
            "Balanced SCRFD + ArcFace pack for local experiments.",
            "buffalo_m.zip",
            "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip",
        ),
        insightface_pack(
            "insightface-buffalo-l",
            "InsightFace Buffalo L",
            "Large SCRFD + ArcFace pack for accuracy-oriented experiments.",
            "buffalo_l.zip",
            "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        ),
        insightface_pack(
            "insightface-antelopev2",
            "InsightFace AntelopeV2",
            "High-capacity SCRFD + ArcFace pack for accuracy comparisons.",
            "antelopev2.zip",
            "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
        ),
        conversion_pack(
            "retinaface-arcface",
            "RetinaFace + ArcFace",
            "Quality-oriented detector and recognition baseline awaiting a pinned ONNX artifact.",
            "RetinaFace",
            "ArcFace",
            "https://github.com/deepinsight/insightface",
        ),
        conversion_pack(
            "retinaface-adaface-ir50",
            "RetinaFace + AdaFace IR50",
            "Quality-adaptive recognition experiment awaiting a pinned ONNX artifact.",
            "RetinaFace",
            "AdaFace IR50",
            "https://github.com/mk-minchul/AdaFace",
        ),
        conversion_pack(
            "blazeface-facenet-128",
            "BlazeFace + FaceNet 128",
            "Mobile-oriented comparison baseline awaiting a pinned ONNX artifact.",
            "BlazeFace",
            "FaceNet 128",
            "https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector",
        ),
        conversion_pack(
            "yunet-openface-nn4",
            "YuNet + OpenFace nn4",
            "Legacy comparison baseline awaiting a pinned ONNX artifact.",
            "YuNet",
            "OpenFace nn4.small2",
            "https://github.com/cmusatyalab/openface",
        ),
    ]
}

fn insightface_pack(
    id: &str,
    display_name: &str,
    description: &str,
    file_name: &str,
    source_url: &str,
) -> FaceModelPack {
    FaceModelPack {
        id: id.to_string(),
        display_name: display_name.to_string(),
        description: description.to_string(),
        detector: "SCRFD".to_string(),
        recognizer: "ArcFace/MobileFaceNet".to_string(),
        detector_landmarks: 5,
        embedding_dimensions: Some(512),
        availability: ModelAvailability::DirectDownload,
        artifacts: vec![artifact(file_name, ModelArtifactFormat::Zip, source_url)],
        license_name: "InsightFace public pretrained model license".to_string(),
        license_url: "https://github.com/deepinsight/insightface/tree/master/model_zoo".to_string(),
        license_acknowledgement_required: true,
        model_source_url: "https://github.com/deepinsight/insightface".to_string(),
    }
}

fn conversion_pack(
    id: &str,
    display_name: &str,
    description: &str,
    detector: &str,
    recognizer: &str,
    source_url: &str,
) -> FaceModelPack {
    FaceModelPack {
        id: id.to_string(),
        display_name: display_name.to_string(),
        description: description.to_string(),
        detector: detector.to_string(),
        recognizer: recognizer.to_string(),
        detector_landmarks: 5,
        embedding_dimensions: None,
        availability: ModelAvailability::ConversionRequired,
        artifacts: vec![artifact(
            "upstream-model",
            ModelArtifactFormat::Checkpoint,
            source_url,
        )],
        license_name: "See upstream model source".to_string(),
        license_url: source_url.to_string(),
        license_acknowledgement_required: true,
        model_source_url: source_url.to_string(),
    }
}

#[tauri::command]
pub fn list_face_model_packs() -> Vec<FaceModelPack> {
    face_model_packs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn pack_ids_are_unique_and_non_empty() {
        let packs = face_model_packs();
        let ids: HashSet<_> = packs.iter().map(|pack| pack.id.as_str()).collect();

        assert_eq!(ids.len(), packs.len());
        assert!(packs.iter().all(|pack| !pack.id.is_empty()));
    }

    #[test]
    fn direct_downloads_have_https_artifacts() {
        for pack in face_model_packs()
            .iter()
            .filter(|pack| pack.availability == ModelAvailability::DirectDownload)
        {
            assert!(!pack.artifacts.is_empty(), "{} has no artifacts", pack.id);
            assert!(
                pack.artifacts
                    .iter()
                    .all(|artifact| artifact.source_url.starts_with("https://")),
                "{} includes a non-HTTPS artifact",
                pack.id
            );
        }
    }

    #[test]
    fn face_packs_provide_landmarks_for_alignment() {
        assert!(
            face_model_packs()
                .iter()
                .all(|pack| pack.detector_landmarks >= 5)
        );
    }
}
