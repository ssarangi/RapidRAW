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

fn artifact(
    file_name: &str,
    format: ModelArtifactFormat,
    source_url: &str,
    sha256: Option<&str>,
) -> FaceModelArtifact {
    FaceModelArtifact {
        file_name: file_name.to_string(),
        format,
        source_url: source_url.to_string(),
        sha256: sha256.map(ToString::to_string),
    }
}

/// All face-model candidates are defined in one place. Direct-download packs
/// are the first runtime targets with pinned immutable revisions. Conversion-required
/// packs remain visible so an evaluator can compare them once a reproducible ONNX artifact is pinned.
pub fn face_model_packs() -> Vec<FaceModelPack> {
    vec![
        // Pack: YuNet + SFace
        // Upstream Repository: https://github.com/opencv/opencv_zoo
        // Immutable Revision: 47534e27c9851bb1128ccc0102f1145e27f23f98 (commit on main)
        // Date Verified: 2026-08-28
        // Verified SHA-256:
        //   - face_detection_yunet_2023mar.onnx: 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
        //   - face_recognition_sface_2021dec.onnx: 0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79
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
                    "https://github.com/opencv/opencv_zoo/raw/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                    Some("8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
                ),
                artifact(
                    "face_recognition_sface_2021dec.onnx",
                    ModelArtifactFormat::Onnx,
                    "https://github.com/opencv/opencv_zoo/raw/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
                    Some("0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
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
        availability: ModelAvailability::ConversionRequired,
        artifacts: vec![artifact(
            file_name,
            ModelArtifactFormat::Zip,
            source_url,
            None,
        )],
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
    installed_face_model_path_in_dir(&face_models_dir(app_handle)?, pack_id, file_name)
}

/// Resolves a face-model artifact from an explicit face-model root. Headless
/// callers use this without constructing a Tauri application handle.
pub fn installed_face_model_path_in_dir(
    face_models_dir: &Path,
    pack_id: &str,
    file_name: &str,
) -> Result<PathBuf, String> {
    let pack = find_pack(pack_id)?;
    let directory = face_models_dir.join(pack_id);
    let manifest = read_installed_manifest(&directory)?;
    if !is_complete_install(&pack, &directory, manifest.as_ref()) {
        return Err(format!(
            "Face model {pack_id} is not installed or has been modified"
        ));
    }
    let path = directory.join(file_name);
    if path.is_file() {
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

fn is_complete_install(
    pack: &FaceModelPack,
    pack_dir: &Path,
    manifest: Option<&InstalledFaceModelPack>,
) -> bool {
    let Some(manifest) = manifest else {
        return false;
    };
    manifest.pack_id == pack.id
        && manifest.artifacts.len() == pack.artifacts.len()
        && pack.artifacts.iter().all(|artifact| {
            let matching_manifest_artifacts = manifest
                .artifacts
                .iter()
                .filter(|installed| installed.file_name == artifact.file_name)
                .collect::<Vec<_>>();
            let Some(installed) = matching_manifest_artifacts.first() else {
                return false;
            };
            if matching_manifest_artifacts.len() != 1 {
                return false;
            }

            let path = pack_dir.join(&artifact.file_name);
            let Ok(actual) = sha256_file(&path) else {
                return false;
            };
            actual.eq_ignore_ascii_case(&installed.sha256)
                && artifact
                    .sha256
                    .as_deref()
                    .map(|expected| actual.eq_ignore_ascii_case(expected))
                    .unwrap_or(true)
        })
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

fn verify_expected_hash(bytes: &[u8], expected: Option<&str>) -> Result<(), String> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let actual = hex::encode(Sha256::digest(bytes));
    if actual.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        Err(format!("Hash mismatch: expected {expected}, got {actual}"))
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
                installed: is_complete_install(&pack, &path, manifest.as_ref()),
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
                serde_json::json!({
                    "packId": pack.id,
                    "displayName": pack.display_name,
                    "registry": "face",
                }),
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
        state
            .background_job_controls
            .lock()
            .unwrap()
            .insert(job_id.clone(), job_control.clone());
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
                verify_expected_hash(&bytes, artifact.sha256.as_deref())?;
                write_atomically(&target, &bytes)?;
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
        state.background_job_controls.lock().unwrap().remove(job_id);
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
    fn direct_download_face_packs_require_pinned_sha256_digests() {
        for pack in face_model_packs()
            .iter()
            .filter(|pack| pack.availability == ModelAvailability::DirectDownload)
        {
            for artifact in &pack.artifacts {
                let digest = artifact.sha256.as_deref().unwrap_or("");
                assert_eq!(
                    digest.len(),
                    64,
                    "Direct-download face artifact {} in pack {} must have a 64-char hex SHA-256",
                    artifact.file_name,
                    pack.id
                );
                assert!(
                    digest.chars().all(|c| c.is_ascii_hexdigit()),
                    "SHA-256 for {} in pack {} must be valid hex",
                    artifact.file_name,
                    pack.id
                );
            }
        }
    }

    #[test]
    fn direct_download_face_packs_use_immutable_urls() {
        for pack in face_model_packs()
            .iter()
            .filter(|pack| pack.availability == ModelAvailability::DirectDownload)
        {
            for artifact in &pack.artifacts {
                assert!(
                    artifact.source_url.starts_with("https://"),
                    "{} artifact {} must be HTTPS",
                    pack.id,
                    artifact.file_name
                );
                assert!(
                    !artifact.source_url.contains("/raw/main/")
                        && !artifact.source_url.contains("/raw/master/")
                        && !artifact.source_url.contains("/resolve/main/")
                        && !artifact.source_url.contains("/resolve/master/"),
                    "{} artifact {} must use an immutable commit revision in URL: {}",
                    pack.id,
                    artifact.file_name,
                    artifact.source_url
                );
            }
        }
    }

    #[test]
    fn runtime_install_validation_rejects_mismatched_registry_pin_even_if_manifest_matches_disk() {
        let directory = tempdir().unwrap();
        let pack = face_model_packs()
            .into_iter()
            .find(|pack| pack.id == "opencv-yunet-sface")
            .unwrap();

        for artifact in &pack.artifacts {
            fs::write(directory.path().join(&artifact.file_name), b"forged_bytes").unwrap();
        }

        let manifest = InstalledFaceModelPack {
            pack_id: pack.id.clone(),
            installed_at: 1,
            artifacts: pack
                .artifacts
                .iter()
                .map(|artifact| {
                    let path = directory.path().join(&artifact.file_name);
                    InstalledFaceModelArtifact {
                        file_name: artifact.file_name.clone(),
                        sha256: sha256_file(&path).unwrap(),
                    }
                })
                .collect(),
        };

        assert!(
            !is_complete_install(&pack, directory.path(), Some(&manifest)),
            "Forged disk bytes matching manifest must be rejected when registry pin differs"
        );
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
    fn complete_install_requires_manifest_and_pinned_artifact_digests() {
        let directory = tempdir().unwrap();
        let pack = face_model_packs()
            .into_iter()
            .find(|pack| pack.id == "opencv-yunet-sface")
            .unwrap();
        let manifest = InstalledFaceModelPack {
            pack_id: pack.id.clone(),
            installed_at: 1,
            artifacts: pack
                .artifacts
                .iter()
                .map(|artifact| InstalledFaceModelArtifact {
                    file_name: artifact.file_name.clone(),
                    sha256: "test".to_string(),
                })
                .collect(),
        };
        assert!(!is_complete_install(
            &pack,
            directory.path(),
            Some(&manifest)
        ));
        for artifact in &pack.artifacts {
            fs::write(directory.path().join(&artifact.file_name), b"onnx").unwrap();
        }
        assert!(!is_complete_install(
            &pack,
            directory.path(),
            Some(&manifest)
        ));
        let manifest = InstalledFaceModelPack {
            artifacts: pack
                .artifacts
                .iter()
                .map(|artifact| {
                    let path = directory.path().join(&artifact.file_name);
                    InstalledFaceModelArtifact {
                        file_name: artifact.file_name.clone(),
                        sha256: sha256_file(&path).unwrap(),
                    }
                })
                .collect(),
            ..manifest
        };
        // A local manifest matching local files cannot override a registry pin.
        assert!(!is_complete_install(
            &pack,
            directory.path(),
            Some(&manifest)
        ));

        let mut unpinned_pack = pack.clone();
        for artifact in &mut unpinned_pack.artifacts {
            artifact.sha256 = None;
        }
        assert!(is_complete_install(
            &unpinned_pack,
            directory.path(),
            Some(&manifest)
        ));
        fs::write(
            directory.path().join(&pack.artifacts[0].file_name),
            b"modified",
        )
        .unwrap();
        assert!(!is_complete_install(
            &pack,
            directory.path(),
            Some(&manifest)
        ));
    }

    #[test]
    fn expected_hash_is_checked_before_an_artifact_is_written() {
        let bytes = b"trusted model bytes";
        let digest = hex::encode(Sha256::digest(bytes));
        assert!(verify_expected_hash(bytes, Some(&digest)).is_ok());
        assert!(verify_expected_hash(bytes, Some(&"0".repeat(64))).is_err());
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
