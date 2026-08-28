use std::collections::HashSet;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager};
use zip::ZipArchive;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledFaceModelArtifact {
    pub file_name: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledFaceModelPack {
    pub pack_id: String,
    pub installed_at: i64,
    pub artifacts: Vec<InstalledFaceModelArtifact>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceModelPackStatus {
    #[serde(flatten)]
    pub pack: FaceModelPack,
    pub installed: bool,
    pub install_path: String,
    pub installed_artifacts: Vec<InstalledFaceModelArtifact>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FaceModelDownloadProgress {
    pack_id: String,
    display_name: String,
    current: usize,
    total: usize,
    stage: String,
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

#[tauri::command]
pub fn list_face_model_packs() -> Vec<FaceModelPack> {
    face_model_packs()
}

fn face_models_dir(app_handle: &AppHandle) -> Result<PathBuf, String> {
    let path = app_handle
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("models")
        .join("face");
    fs::create_dir_all(&path).map_err(|error| error.to_string())?;
    Ok(path)
}

fn pack_dir(app_handle: &AppHandle, pack_id: &str) -> Result<PathBuf, String> {
    Ok(face_models_dir(app_handle)?.join(pack_id))
}

pub(crate) fn installed_face_model_path(
    app_handle: &AppHandle,
    pack_id: &str,
    file_name: &str,
) -> Result<PathBuf, String> {
    let path = pack_dir(app_handle, pack_id)?.join(file_name);
    if path.exists() {
        Ok(path)
    } else {
        Err(format!("Face model {pack_id} is not installed"))
    }
}

fn installed_manifest_path(pack_dir: &Path) -> PathBuf {
    pack_dir.join("rapidraw-face-model.json")
}

fn read_installed_manifest(pack_dir: &Path) -> Result<Option<InstalledFaceModelPack>, String> {
    let path = installed_manifest_path(pack_dir);
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(path).map_err(|error| error.to_string())?;
    serde_json::from_str(&content)
        .map(Some)
        .map_err(|error| error.to_string())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|error| error.to_string())?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let bytes = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if bytes == 0 {
            break;
        }
        hasher.update(&buffer[..bytes]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("Cannot determine parent directory for {}", path.display()))?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("Invalid model filename {}", path.display()))?;
    let temporary = parent.join(format!(".{file_name}.download"));
    {
        let mut file = fs::File::create(&temporary).map_err(|error| error.to_string())?;
        file.write_all(bytes).map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
    }
    fs::rename(&temporary, path)
        .or_else(|rename_error| {
            if path.exists() {
                fs::remove_file(path)?;
                fs::rename(&temporary, path)
            } else {
                Err(rename_error)
            }
        })
        .map_err(|error| error.to_string())
}

fn verify_expected_hash(path: &Path, expected: Option<&str>) -> Result<(), String> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let actual = sha256_file(path)?;
    if actual.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        Err(format!("Hash mismatch for {}", path.display()))
    }
}

fn extract_onnx_archive(destination: &Path, archive: &[u8]) -> Result<Vec<PathBuf>, String> {
    let mut archive = ZipArchive::new(Cursor::new(archive)).map_err(|error| error.to_string())?;
    let mut installed = Vec::new();
    let mut names = HashSet::new();

    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(|error| error.to_string())?;
        if entry.is_dir() || !entry.name().to_ascii_lowercase().ends_with(".onnx") {
            continue;
        }
        let enclosed = entry
            .enclosed_name()
            .ok_or_else(|| format!("Unsafe archive path {}", entry.name()))?;
        let file_name = enclosed
            .file_name()
            .ok_or_else(|| format!("Invalid archive filename {}", entry.name()))?;
        let file_name = file_name
            .to_str()
            .ok_or_else(|| format!("Non-UTF8 archive filename {}", entry.name()))?;
        if !names.insert(file_name.to_string()) {
            return Err(format!(
                "Archive contains duplicate model filename {file_name}"
            ));
        }

        let mut bytes = Vec::new();
        entry
            .read_to_end(&mut bytes)
            .map_err(|error| error.to_string())?;
        let output = destination.join(file_name);
        write_atomically(&output, &bytes)?;
        installed.push(output);
    }

    if installed.is_empty() {
        return Err("Downloaded model archive contains no ONNX files".to_string());
    }
    Ok(installed)
}

fn find_pack(pack_id: &str) -> Result<FaceModelPack, String> {
    face_model_packs()
        .into_iter()
        .find(|pack| pack.id == pack_id)
        .ok_or_else(|| format!("Unknown face model pack: {pack_id}"))
}

#[tauri::command]
pub fn list_face_model_pack_statuses(
    app_handle: AppHandle,
) -> Result<Vec<FaceModelPackStatus>, String> {
    face_model_packs()
        .into_iter()
        .map(|pack| {
            let path = pack_dir(&app_handle, &pack.id)?;
            let manifest = read_installed_manifest(&path)?;
            Ok(FaceModelPackStatus {
                install_path: path.to_string_lossy().into_owned(),
                installed: manifest.is_some(),
                installed_artifacts: manifest.map(|item| item.artifacts).unwrap_or_default(),
                pack,
            })
        })
        .collect()
}

#[tauri::command]
pub async fn download_face_model_pack(
    pack_id: String,
    accept_restricted_license: bool,
    app_handle: AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<FaceModelPackStatus, String> {
    let pack = find_pack(&pack_id)?;
    if pack.availability != ModelAvailability::DirectDownload {
        return Err(format!(
            "{} requires a pinned ONNX conversion before RapidRAW can install it",
            pack.display_name
        ));
    }
    if pack.license_acknowledgement_required && !accept_restricted_license {
        return Err(format!(
            "{} requires license acknowledgement before download",
            pack.display_name
        ));
    }

    let destination = pack_dir(&app_handle, &pack.id)?;
    fs::create_dir_all(&destination).map_err(|error| error.to_string())?;
    let total = pack.artifacts.len();
    let mut installed_paths = Vec::new();
    let job = crate::library_db::active_library_path(&state)
        .ok()
        .and_then(|db_path| {
            crate::library_db::create_background_job(
                &db_path,
                "model_download",
                serde_json::json!({ "packId": pack.id, "displayName": pack.display_name }),
            )
            .map(|job_id| (db_path, job_id))
            .ok()
        });
    if let Some((db_path, job_id)) = job.as_ref() {
        let _ = crate::library_db::update_job(
            db_path,
            job_id,
            "running",
            "Starting model download",
            0,
            total as i64,
            None,
            None,
        );
    }
    let job_control = crate::app_state::BackgroundJobControl::new();
    if let Some((_, job_id)) = job.as_ref() {
        state.background_job_controls.lock().unwrap().insert(job_id.clone(), job_control.clone());
    }

    for (index, artifact) in pack.artifacts.iter().enumerate() {
        let is_runnable = job_control.wait_until_runnable().await;
        if !is_runnable || *job_control.cancellation_receiver().borrow() {
            if let Some((db_path, job_id)) = job.as_ref() {
                let _ = crate::library_db::update_job(
                    db_path,
                    job_id,
                    "cancelled",
                    "Model download cancelled",
                    index as i64,
                    total as i64,
                    None,
                    None,
                );
                state.background_job_controls.lock().unwrap().remove(job_id);
            }
            return Err("Model download cancelled".to_string());
        }

        if let Some((db_path, job_id)) = job.as_ref() {
            let _ = crate::library_db::update_job(
                db_path,
                job_id,
                "running",
                &format!("Downloading {}", artifact.file_name),
                index as i64,
                total as i64,
                Some(&artifact.file_name),
                None,
            );
        }
        let _ = app_handle.emit(
            "face-model-download-progress",
            FaceModelDownloadProgress {
                pack_id: pack.id.clone(),
                display_name: pack.display_name.clone(),
                current: index,
                total,
                stage: format!("Downloading {}", artifact.file_name),
            },
        );
        let response = tokio::select! {
            res = reqwest::get(&artifact.source_url) => {
                res.map_err(|error| error.to_string())?
                   .error_for_status()
                   .map_err(|error| error.to_string())?
            }
            _ = async {
                let mut rx = job_control.cancellation_receiver();
                rx.wait_for(|c| *c).await.unwrap();
            } => {
                if let Some((db_path, job_id)) = job.as_ref() {
                    let _ = crate::library_db::update_job(
                        db_path,
                        job_id,
                        "cancelled",
                        "Model download cancelled",
                        index as i64,
                        total as i64,
                        None,
                        None,
                    );
                    state.background_job_controls.lock().unwrap().remove(job_id);
                }
                return Err("Model download cancelled".to_string());
            }
        };

        let bytes = tokio::select! {
            res = response.bytes() => {
                res.map_err(|error| error.to_string())?
            }
            _ = async {
                let mut rx = job_control.cancellation_receiver();
                rx.wait_for(|c| *c).await.unwrap();
            } => {
                if let Some((db_path, job_id)) = job.as_ref() {
                    let _ = crate::library_db::update_job(
                        db_path,
                        job_id,
                        "cancelled",
                        "Model download cancelled",
                        index as i64,
                        total as i64,
                        None,
                        None,
                    );
                    state.background_job_controls.lock().unwrap().remove(job_id);
                }
                return Err("Model download cancelled".to_string());
            }
        };

        match artifact.format {
            ModelArtifactFormat::Onnx => {
                let target = destination.join(&artifact.file_name);
                write_atomically(&target, &bytes)?;
                verify_expected_hash(&target, artifact.sha256.as_deref())?;
                installed_paths.push(target);
            }
            ModelArtifactFormat::Zip => {
                installed_paths.extend(extract_onnx_archive(&destination, &bytes)?);
            }
            ModelArtifactFormat::Checkpoint => {
                return Err(format!(
                    "{} is not directly usable by ONNX Runtime",
                    artifact.file_name
                ));
            }
        }
    }

    let installed = InstalledFaceModelPack {
        pack_id: pack.id.clone(),
        installed_at: Utc::now().timestamp(),
        artifacts: installed_paths
            .iter()
            .map(|path| {
                Ok(InstalledFaceModelArtifact {
                    file_name: path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .ok_or_else(|| format!("Invalid model path {}", path.display()))?
                        .to_string(),
                    sha256: sha256_file(path)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
    };
    let manifest = serde_json::to_vec_pretty(&installed).map_err(|error| error.to_string())?;
    write_atomically(&installed_manifest_path(&destination), &manifest)?;

    if let Some((db_path, job_id)) = job.as_ref() {
        let _ = crate::library_db::update_job(
            db_path,
            job_id,
            "completed",
            "Model download complete",
            total as i64,
            total as i64,
            None,
            None,
        );
        state
            .background_job_controls
            .lock()
            .unwrap()
            .remove(job_id);
    }

    let _ = app_handle.emit(
        "face-model-download-complete",
        FaceModelDownloadProgress {
            pack_id: pack.id.clone(),
            display_name: pack.display_name.clone(),
            current: total,
            total,
            stage: "Ready".to_string(),
        },
    );

    Ok(FaceModelPackStatus {
        pack,
        installed: true,
        install_path: destination.to_string_lossy().into_owned(),
        installed_artifacts: installed.artifacts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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

    #[test]
    fn installed_manifest_round_trips() {
        let directory = tempdir().unwrap();
        let manifest = InstalledFaceModelPack {
            pack_id: "opencv-yunet-sface".to_string(),
            installed_at: 1,
            artifacts: vec![InstalledFaceModelArtifact {
                file_name: "model.onnx".to_string(),
                sha256: "abc".to_string(),
            }],
        };
        write_atomically(
            &installed_manifest_path(directory.path()),
            &serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let restored = read_installed_manifest(directory.path()).unwrap().unwrap();
        assert_eq!(restored.pack_id, manifest.pack_id);
        assert_eq!(restored.artifacts[0].file_name, "model.onnx");
    }

    #[test]
    fn archive_extraction_rejects_empty_archives() {
        let directory = tempdir().unwrap();
        let mut bytes = Cursor::new(Vec::new());
        {
            let writer = zip::ZipWriter::new(&mut bytes);
            writer.finish().unwrap();
        }

        let error = extract_onnx_archive(directory.path(), bytes.get_ref()).unwrap_err();
        assert!(error.contains("no ONNX files"));
    }
}
