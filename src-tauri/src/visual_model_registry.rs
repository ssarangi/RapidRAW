use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::UNIX_EPOCH;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum VisualModelAvailability {
    DirectDownload,
    BundleRequired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualModelArtifact {
    pub file_name: String,
    pub source_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_sha256: Option<String>,
    /// When set, `source_url` points at a zip archive rather than the raw
    /// artifact bytes; this is the path of the member inside that archive to
    /// extract and save as `file_name`. `expected_sha256` then verifies the
    /// extracted member, not the archive itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub archive_member: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualModelPack {
    pub id: String,
    pub display_name: String,
    pub description: String,
    pub task: String,
    pub availability: VisualModelAvailability,
    pub artifacts: Vec<VisualModelArtifact>,
    pub license_name: String,
    pub license_url: String,
    pub model_source_url: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualModelPackStatus {
    #[serde(flatten)]
    pub pack: VisualModelPack,
    pub installed: bool,
    pub install_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InstalledVisualModelPack {
    pack_id: String,
    installed_at: i64,
    artifacts: Vec<InstalledVisualModelArtifact>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InstalledVisualModelArtifact {
    file_name: String,
    sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ArtifactFingerprint {
    file_name: String,
    bytes: u64,
    modified_nanos: u128,
}

static VERIFIED_PACKS: OnceLock<Mutex<HashMap<PathBuf, Vec<ArtifactFingerprint>>>> =
    OnceLock::new();

fn visual_artifact(
    file_name: &str,
    source_url: &str,
    expected_sha256: Option<&str>,
) -> VisualModelArtifact {
    VisualModelArtifact {
        file_name: file_name.to_string(),
        source_url: source_url.to_string(),
        expected_sha256: expected_sha256.map(ToString::to_string),
        archive_member: None,
    }
}

/// An artifact that is a single member of a zip archive - e.g. darktable-ai's
/// `.dtmodel` release assets, which bundle a `config.json` and one or more
/// `.onnx` files together. `expected_sha256` verifies the extracted member,
/// not the archive.
fn zip_artifact(
    file_name: &str,
    source_url: &str,
    archive_member: &str,
    expected_sha256: &str,
) -> VisualModelArtifact {
    VisualModelArtifact {
        file_name: file_name.to_string(),
        source_url: source_url.to_string(),
        expected_sha256: Some(expected_sha256.to_string()),
        archive_member: Some(archive_member.to_string()),
    }
}

fn ram_plus_artifact(file_name: &str, sha256: &str) -> VisualModelArtifact {
    visual_artifact(
        file_name,
        &format!(
            "https://huggingface.co/benjaminjonard/ram-plus-onnx/resolve/e22cfb689ba3a3e02f23925e1f581e3e07166c75/{file_name}"
        ),
        Some(sha256),
    )
}

pub fn visual_model_packs() -> Vec<VisualModelPack> {
    vec![
        // Pack: RAM++ (Recognize Anything Model Plus)
        // Upstream Repository: https://huggingface.co/benjaminjonard/ram-plus-onnx
        // Immutable Revision: e22cfb689ba3a3e02f23925e1f581e3e07166c75
        // Date Verified: 2026-08-28
        // Verified SHA-256:
        //   - model.onnx: 40ac94a3eb0c5fddfe26b1d6a047fb7afe529b3b7afe569facac8590a3d88773
        //   - tags.txt: 1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3
        //   - thresholds.txt: 81250c4b6a6eb5dc553a084b34d8354b4f2951de5c799a64d3f638d88fe8cf7b
        VisualModelPack {
            id: "ram-plus-onnx".to_string(),
            display_name: "RAM++".to_string(),
            description: "Broad multi-label tagging for scenes, objects, activities, and wildlife gates.".to_string(),
            task: "Broad visual tagging".to_string(),
            availability: VisualModelAvailability::DirectDownload,
            artifacts: vec![
                ram_plus_artifact(
                    "model.onnx",
                    "40ac94a3eb0c5fddfe26b1d6a047fb7afe529b3b7afe569facac8590a3d88773",
                ),
                ram_plus_artifact(
                    "tags.txt",
                    "1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3",
                ),
                ram_plus_artifact(
                    "thresholds.txt",
                    "81250c4b6a6eb5dc553a084b34d8354b4f2951de5c799a64d3f638d88fe8cf7b",
                ),
            ],
            license_name: "Apache-2.0".to_string(),
            license_url: "https://huggingface.co/benjaminjonard/ram-plus-onnx".to_string(),
            model_source_url: "https://github.com/xinyu1205/recognize-anything".to_string(),
        },
        // Pack: BioCLIP (Imageomics ViT-B/16 with Tree-of-Life taxonomy)
        // Upstream Repository: https://github.com/ssarangi/RapidRAW / https://huggingface.co/imageomics/bioclip
        // Immutable Release Tag: v0.1.0-models
        // Date Verified: 2026-08-28
        // Verified SHA-256:
        //   - vision_encoder.onnx: 3ed2c2ad149851297481727463c3085c030bafcf278c070bd9617d810beed5a4
        //   - vision_encoder.onnx.data: c0cdb287d84c0e66dcf58f5f4c1e8ba75b5f42bf2b701addfff60b0879ed5bf1
        //   - species_embeddings.bin: 5c4383ca3cd4cfb33ba3166d935a521a2034ae20f0c2045bdab5561331bf81cd
        //   - species_labels.json: 57fa63a8bf67c0fbcb9329400939d7352da576283684361fece1fde04c8b9eda
        VisualModelPack {
            id: "bioclip-v1".to_string(),
            display_name: "BioCLIP".to_string(),
            description: "Taxonomy-aware organism classification, including birds. Uses an ONNX ViT encoder with Tree-of-Life taxonomy embeddings.".to_string(),
            task: "Wildlife and species classification".to_string(),
            availability: VisualModelAvailability::DirectDownload,
            artifacts: vec![
                visual_artifact(
                    "vision_encoder.onnx",
                    "https://github.com/ssarangi/RapidRAW/releases/download/v0.1.0-models/vision_encoder.onnx",
                    Some("3ed2c2ad149851297481727463c3085c030bafcf278c070bd9617d810beed5a4"),
                ),
                visual_artifact(
                    "vision_encoder.onnx.data",
                    "https://github.com/ssarangi/RapidRAW/releases/download/v0.1.0-models/vision_encoder.onnx.data",
                    Some("c0cdb287d84c0e66dcf58f5f4c1e8ba75b5f42bf2b701addfff60b0879ed5bf1"),
                ),
                visual_artifact(
                    "species_embeddings.bin",
                    "https://github.com/ssarangi/RapidRAW/releases/download/v0.1.0-models/species_embeddings.bin",
                    Some("5c4383ca3cd4cfb33ba3166d935a521a2034ae20f0c2045bdab5561331bf81cd"),
                ),
                visual_artifact(
                    "species_labels.json",
                    "https://github.com/ssarangi/RapidRAW/releases/download/v0.1.0-models/species_labels.json",
                    Some("57fa63a8bf67c0fbcb9329400939d7352da576283684361fece1fde04c8b9eda"),
                ),
            ],
            license_name: "MIT".to_string(),
            license_url: "https://huggingface.co/imageomics/bioclip".to_string(),
            model_source_url: "https://github.com/Imageomics/bioclip".to_string(),
        },
        // Pack: OCEC (Open/Closed Eye Classifier)
        // Upstream Repository: https://github.com/PINTO0309/OCEC
        // Immutable Release Tag: onnx
        // Date Verified: 2026-08-29
        // Verified SHA-256:
        //   - ocec_s.onnx: 9a346a08b256ad70725044cd2aa582858e108c6f45d42a9c3415afc604ba9b64
        // Trained on the "Open and Closed Eyes Dataset" (ODC-By) plus custom
        // recorded video; outputs a single sigmoid prob_open score rather
        // than a hard open/closed label, which is what lets the culling
        // pipeline bucket it into open/semi-closed/closed instead of just
        // a binary verdict.
        VisualModelPack {
            id: "ocec-eye-state".to_string(),
            display_name: "OCEC Eye State".to_string(),
            description: "Classifies a cropped eye region as open or closed, used to flag closed-eye/blink rejections during culling.".to_string(),
            task: "Eye openness classification".to_string(),
            availability: VisualModelAvailability::DirectDownload,
            artifacts: vec![
                visual_artifact(
                    "ocec_s.onnx",
                    "https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_s.onnx",
                    Some("9a346a08b256ad70725044cd2aa582858e108c6f45d42a9c3415afc604ba9b64"),
                ),
            ],
            license_name: "MIT".to_string(),
            license_url: "https://github.com/PINTO0309/OCEC/blob/main/LICENSE".to_string(),
            model_source_url: "https://github.com/PINTO0309/OCEC".to_string(),
        },
        VisualModelPack {
            // Pack: RawNIND UtNet2 Bayer RAW denoiser
            // Upstream Release: darktable-org/darktable-ai release-5.6.0 (rawdenoise-nind.dtmodel)
            // Date Verified: 2026-08-30
            // Verified SHA-256 (extracted model_bayer.onnx):
            //   da27509dab6a2915da67e988acd86cf71f9d5bbc8d1aa0ed32933578a887b901
            id: "rawnind-utnet2-bayer".to_string(),
            display_name: "RawNIND Bayer RAW Denoise".to_string(),
            description: "Deep sensor Bayer joint denoising and demosaicing directly into linear Rec.2020.".to_string(),
            task: "RAW sensor denoising".to_string(),
            availability: VisualModelAvailability::DirectDownload,
            artifacts: vec![
                zip_artifact(
                    "rawnind_bayer.onnx",
                    "https://github.com/darktable-org/darktable-ai/releases/download/release-5.6.0/rawdenoise-nind.dtmodel",
                    "rawdenoise-nind/model_bayer.onnx",
                    "da27509dab6a2915da67e988acd86cf71f9d5bbc8d1aa0ed32933578a887b901",
                ),
            ],
            license_name: "GPL-3.0".to_string(),
            license_url: "https://arxiv.org/abs/2501.08924".to_string(),
            model_source_url: "https://github.com/darktable-org/darktable-ai/blob/master/models/rawdenoise-nind/README.md".to_string(),
        },
        VisualModelPack {
            // Pack: NAFNet SIDD RGB denoiser
            // Upstream Release: darktable-org/darktable-ai release-5.6.0 (denoise-nafnet.dtmodel)
            // Date Verified: 2026-08-30
            // Verified SHA-256 (extracted model.onnx):
            //   8b437280db2f9f0ef5c733fa0fec70fc10012a62d19dac24a25b80ec8c529230
            id: "nafnet-sidd-rgb".to_string(),
            display_name: "NAFNet SIDD RGB Denoise".to_string(),
            description: "Fast high-fidelity nonlinear activation-free network for developed linear RGB images.".to_string(),
            task: "RGB image denoising".to_string(),
            availability: VisualModelAvailability::DirectDownload,
            artifacts: vec![
                zip_artifact(
                    "nafnet_sidd.onnx",
                    "https://github.com/darktable-org/darktable-ai/releases/download/release-5.6.0/denoise-nafnet.dtmodel",
                    "denoise-nafnet/model.onnx",
                    "8b437280db2f9f0ef5c733fa0fec70fc10012a62d19dac24a25b80ec8c529230",
                ),
            ],
            license_name: "MIT".to_string(),
            license_url: "https://github.com/megvii-research/NAFNet".to_string(),
            model_source_url: "https://github.com/darktable-org/darktable-ai/blob/master/models/denoise-nafnet/README.md".to_string(),
        },
    ]
}

fn models_dir(app_handle: &AppHandle) -> Result<PathBuf, String> {
    let path = app_handle
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("models")
        .join("visual");
    fs::create_dir_all(&path).map_err(|error| error.to_string())?;
    Ok(path)
}

fn pack_dir(app_handle: &AppHandle, pack_id: &str) -> Result<PathBuf, String> {
    Ok(models_dir(app_handle)?.join(pack_id))
}

fn manifest_path(directory: &Path) -> PathBuf {
    directory.join("manifest.json")
}

/// A manifest that fails to parse - e.g. left over from an older app
/// version with a different shape, or just corrupted - is treated the same
/// as no manifest at all (pack shows as not-yet-verified/installed) rather
/// than as a hard error. One bad manifest.json on disk should never take
/// down the entire visual model list for every other pack.
fn read_installed_manifest(directory: &Path) -> Result<Option<InstalledVisualModelPack>, String> {
    let path = manifest_path(directory);
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(_) => return Ok(None),
    };
    Ok(serde_json::from_slice(&bytes).ok())
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

fn extract_zip_member(archive_path: &Path, member: &str) -> Result<Vec<u8>, String> {
    let file = fs::File::open(archive_path).map_err(|error| error.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|error| error.to_string())?;
    let mut entry = archive
        .by_name(member)
        .map_err(|error| format!("Archive is missing {member}: {error}"))?;
    let mut bytes = Vec::new();
    entry
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(bytes)
}

fn is_complete_install(
    pack: &VisualModelPack,
    directory: &Path,
    manifest: Option<&InstalledVisualModelPack>,
) -> bool {
    let Some(manifest) = manifest else {
        return false;
    };
    if manifest.pack_id != pack.id || manifest.artifacts.len() != pack.artifacts.len() {
        return false;
    }
    let fingerprint = match pack_fingerprint(pack, directory) {
        Ok(fingerprint) => fingerprint,
        Err(_) => return false,
    };
    let cache = VERIFIED_PACKS.get_or_init(|| Mutex::new(HashMap::new()));
    if cache
        .lock()
        .unwrap()
        .get(directory)
        .is_some_and(|previous| previous == &fingerprint)
    {
        return true;
    }
    let verified = pack.artifacts.iter().all(|artifact| {
        let Some(recorded) = manifest
            .artifacts
            .iter()
            .find(|recorded| recorded.file_name == artifact.file_name)
        else {
            return false;
        };
        let path = directory.join(&artifact.file_name);
        path.is_file()
            && sha256_file(&path)
                .map(|actual| {
                    actual.eq_ignore_ascii_case(&recorded.sha256)
                        && artifact
                            .expected_sha256
                            .as_deref()
                            .map(|expected| actual.eq_ignore_ascii_case(expected))
                            .unwrap_or(true)
                })
                .unwrap_or(false)
    });
    if verified {
        cache
            .lock()
            .unwrap()
            .insert(directory.to_path_buf(), fingerprint);
    }
    verified
}

fn pack_fingerprint(
    pack: &VisualModelPack,
    directory: &Path,
) -> Result<Vec<ArtifactFingerprint>, String> {
    let mut fingerprints = pack
        .artifacts
        .iter()
        .map(|artifact| {
            let metadata = fs::metadata(directory.join(&artifact.file_name))
                .map_err(|error| error.to_string())?;
            if !metadata.is_file() {
                return Err(format!("{} is not a regular file", artifact.file_name));
            }
            let modified_nanos = metadata
                .modified()
                .map_err(|error| error.to_string())?
                .duration_since(UNIX_EPOCH)
                .map_err(|error| error.to_string())?
                .as_nanos();
            Ok(ArtifactFingerprint {
                file_name: artifact.file_name.clone(),
                bytes: metadata.len(),
                modified_nanos,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let manifest = fs::metadata(manifest_path(directory)).map_err(|error| error.to_string())?;
    let modified_nanos = manifest
        .modified()
        .map_err(|error| error.to_string())?
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    fingerprints.push(ArtifactFingerprint {
        file_name: "manifest.json".to_string(),
        bytes: manifest.len(),
        modified_nanos,
    });
    Ok(fingerprints)
}

fn fail_download_job(
    job: &Option<(PathBuf, String)>,
    state: &tauri::State<'_, crate::AppState>,
    error: String,
    current: i64,
    total: i64,
) -> String {
    if let Some((db_path, job_id)) = job.as_ref() {
        let _ = crate::library_db::update_job(
            db_path,
            job_id,
            "failed",
            "Model download failed",
            current,
            total,
            None,
            Some(&error),
        );
        state.background_job_controls.lock().unwrap().remove(job_id);
    }
    error
}

#[tauri::command]
pub fn list_visual_model_pack_statuses(
    app_handle: AppHandle,
) -> Result<Vec<VisualModelPackStatus>, String> {
    visual_model_packs()
        .into_iter()
        .map(|pack| {
            let directory = pack_dir(&app_handle, &pack.id)?;
            Ok(VisualModelPackStatus {
                installed: is_complete_install(
                    &pack,
                    &directory,
                    read_installed_manifest(&directory)?.as_ref(),
                ),
                install_path: directory.to_string_lossy().into_owned(),
                pack,
            })
        })
        .collect()
}

#[tauri::command]
pub async fn download_visual_model_pack(
    pack_id: String,
    app_handle: AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<VisualModelPackStatus, String> {
    let pack = visual_model_packs()
        .into_iter()
        .find(|candidate| candidate.id == pack_id)
        .ok_or_else(|| format!("Unknown visual model pack: {pack_id}"))?;
    if pack.availability != VisualModelAvailability::DirectDownload {
        return Err(format!(
            "{} requires a pinned ONNX bundle before it can be installed",
            pack.display_name
        ));
    }
    let directory = pack_dir(&app_handle, &pack.id)?;
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let total = pack.artifacts.len() as i64;
    if let Ok(db_path) = crate::library_db::active_library_path(&state) {
        if let Ok(Some(_existing)) =
            crate::library_db::find_active_model_download_job(&db_path, "visual", &pack.id)
        {
            return Err(format!(
                "{} is already downloading. Check the background jobs list for progress.",
                pack.display_name
            ));
        }
    }
    let job = crate::library_db::active_library_path(&state)
        .ok()
        .and_then(|db_path| {
            crate::library_db::create_background_job(
                &db_path,
                "model_download",
                serde_json::json!({
                    "packId": pack.id,
                    "displayName": pack.display_name,
                    "registry": "visual",
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
            total,
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
        let current = index as i64;

        let is_runnable = job_control.wait_until_runnable().await;
        if !is_runnable || *job_control.cancellation_receiver().borrow() {
            if let Some((db_path, job_id)) = job.as_ref() {
                let _ = crate::library_db::update_job(
                    db_path,
                    job_id,
                    "cancelled",
                    "Model download cancelled",
                    current,
                    total,
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
                current,
                total,
                Some(&artifact.file_name),
                None,
            );
        }
        let target = directory.join(&artifact.file_name);
        let temporary = target.with_extension("download");

        let response = tokio::select! {
            res = reqwest::get(&artifact.source_url) => {
                res.map_err(|error| fail_download_job(&job, &state, error.to_string(), current, total))?
                   .error_for_status()
                   .map_err(|error| fail_download_job(&job, &state, error.to_string(), current, total))?
            }
            _ = async {
                let mut rx = job_control.cancellation_receiver();
                rx.wait_for(|c| *c).await.unwrap();
            } => {
                let _ = fs::remove_file(&temporary);
                if let Some((db_path, job_id)) = job.as_ref() {
                    let _ = crate::library_db::update_job(db_path, job_id, "cancelled", "Model download cancelled", current, total, None, None);
                    state.background_job_controls.lock().unwrap().remove(job_id);
                }
                return Err("Model download cancelled".to_string());
            }
        };

        let content_length = response.content_length();
        let mut downloaded: u64 = 0;
        let mut file = fs::File::create(&temporary)
            .map_err(|error| fail_download_job(&job, &state, error.to_string(), current, total))?;
        {
            use futures::StreamExt;
            let mut stream = response.bytes_stream();
            loop {
                let chunk = tokio::select! {
                    next = stream.next() => next,
                    _ = async {
                        let mut rx = job_control.cancellation_receiver();
                        rx.wait_for(|c| *c).await.unwrap();
                    } => {
                        drop(file);
                        let _ = fs::remove_file(&temporary);
                        if let Some((db_path, job_id)) = job.as_ref() {
                            let _ = crate::library_db::update_job(db_path, job_id, "cancelled", "Model download cancelled", current, total, None, None);
                            state.background_job_controls.lock().unwrap().remove(job_id);
                        }
                        return Err("Model download cancelled".to_string());
                    }
                };
                let Some(chunk) = chunk else { break };
                let chunk = chunk.map_err(|error| fail_download_job(&job, &state, error.to_string(), current, total))?;
                file.write_all(&chunk)
                    .map_err(|error| fail_download_job(&job, &state, error.to_string(), current, total))?;
                downloaded += chunk.len() as u64;
                let _ = app_handle.emit(
                    "visual-model-download-progress",
                    serde_json::json!({
                        "packId": pack.id,
                        "fileName": artifact.file_name,
                        "current": current,
                        "total": total,
                        "bytesDownloaded": downloaded,
                        "bytesTotal": content_length,
                    }),
                );
            }
        }

        if downloaded == 0 {
            let _ = fs::remove_file(&temporary);
            return Err(fail_download_job(
                &job,
                &state,
                format!("Downloaded {} was empty", artifact.file_name),
                current,
                total,
            ));
        }

        file.sync_all()
            .map_err(|error| fail_download_job(&job, &state, error.to_string(), current, total))?;
        drop(file);

        if let Some(ref member) = artifact.archive_member {
            let extracted = extract_zip_member(&temporary, member)
                .map_err(|error| fail_download_job(&job, &state, error, current, total))?;
            let _ = fs::remove_file(&temporary);
            fs::write(&temporary, &extracted)
                .map_err(|error| fail_download_job(&job, &state, error.to_string(), current, total))?;
        }

        if let Some(ref expected) = artifact.expected_sha256 {
            let actual = sha256_file(&temporary)
                .map_err(|error| fail_download_job(&job, &state, error, current, total))?;
            if !actual.eq_ignore_ascii_case(expected) {
                let _ = fs::remove_file(&temporary);
                return Err(fail_download_job(
                    &job,
                    &state,
                    format!(
                        "Checksum mismatch for {}: expected {expected}, got {actual}",
                        artifact.file_name
                    ),
                    current,
                    total,
                ));
            }
        }

        fs::rename(&temporary, &target)
            .map_err(|error| fail_download_job(&job, &state, error.to_string(), current, total))?;
    }
    let artifacts = pack
        .artifacts
        .iter()
        .map(|artifact| {
            Ok(InstalledVisualModelArtifact {
                file_name: artifact.file_name.clone(),
                sha256: sha256_file(&directory.join(&artifact.file_name))?,
            })
        })
        .collect::<Result<Vec<_>, String>>()
        .map_err(|error| fail_download_job(&job, &state, error, total, total))?;
    let manifest = InstalledVisualModelPack {
        pack_id: pack.id.clone(),
        installed_at: Utc::now().timestamp(),
        artifacts,
        source_path: None,
    };
    let manifest = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| fail_download_job(&job, &state, error.to_string(), total, total))?;
    fs::write(manifest_path(&directory), manifest)
        .map_err(|error| fail_download_job(&job, &state, error.to_string(), total, total))?;
    if let Some((db_path, job_id)) = job.as_ref() {
        let _ = crate::library_db::update_job(
            db_path,
            job_id,
            "completed",
            "Model download complete",
            total,
            total,
            None,
            None,
        );
        state.background_job_controls.lock().unwrap().remove(job_id);
    }
    Ok(VisualModelPackStatus {
        installed: true,
        install_path: directory.to_string_lossy().into_owned(),
        pack,
    })
}

#[tauri::command]
pub fn install_visual_model_bundle(
    pack_id: String,
    source_directory: String,
    app_handle: AppHandle,
) -> Result<VisualModelPackStatus, String> {
    let pack = visual_model_packs()
        .into_iter()
        .find(|candidate| candidate.id == pack_id)
        .ok_or_else(|| format!("Unknown visual model pack: {pack_id}"))?;
    if pack.availability != VisualModelAvailability::BundleRequired {
        return Err(format!(
            "{} is downloaded directly by RapidRAW",
            pack.display_name
        ));
    }

    let source = PathBuf::from(&source_directory);
    if !source.is_dir() {
        return Err("Choose the folder containing the model bundle".to_string());
    }
    let missing = pack
        .artifacts
        .iter()
        .filter(|artifact| !source.join(&artifact.file_name).is_file())
        .map(|artifact| artifact.file_name.as_str())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!("The bundle is missing: {}", missing.join(", ")));
    }

    let directory = pack_dir(&app_handle, &pack.id)?;
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    for artifact in &pack.artifacts {
        fs::copy(
            source.join(&artifact.file_name),
            directory.join(&artifact.file_name),
        )
        .map_err(|error| format!("Could not install {}: {error}", artifact.file_name))?;
    }
    let artifacts = pack
        .artifacts
        .iter()
        .map(|artifact| {
            Ok(InstalledVisualModelArtifact {
                file_name: artifact.file_name.clone(),
                sha256: sha256_file(&directory.join(&artifact.file_name))?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let manifest = InstalledVisualModelPack {
        pack_id: pack.id.clone(),
        installed_at: Utc::now().timestamp(),
        artifacts,
        source_path: Some(source.to_string_lossy().into_owned()),
    };
    fs::write(
        manifest_path(&directory),
        serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(VisualModelPackStatus {
        installed: true,
        install_path: directory.to_string_lossy().into_owned(),
        pack,
    })
}

#[tauri::command]
pub fn remove_visual_model_pack(pack_id: String, app_handle: AppHandle) -> Result<(), String> {
    let pack = visual_model_packs()
        .into_iter()
        .find(|candidate| candidate.id == pack_id)
        .ok_or_else(|| format!("Unknown visual model pack: {pack_id}"))?;
    let directory = pack_dir(&app_handle, &pack.id)?;
    if directory.exists() {
        fs::remove_dir_all(&directory).map_err(|error| error.to_string())?;
    }
    Ok(())
}

/// Resolves an installed artifact from an explicit visual-model root. This is
/// used by headless callers such as `rapidraw-cli`, which do not have a Tauri
/// application handle but must execute the same verified model packs.
pub fn installed_visual_model_path_in_dir(
    visual_models_dir: &Path,
    pack_id: &str,
    file_name: &str,
) -> Result<PathBuf, String> {
    let directory = verified_visual_model_pack_dir(visual_models_dir, pack_id)?;
    let path = directory.join(file_name);
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!(
            "Install or reinstall the {} visual model pack before running this analysis",
            pack_id
        ))
    }
}

/// Verifies that a model pack's manifest, required files, and recorded SHA-256
/// digests agree before a GUI or headless caller creates an ONNX session.
pub fn verified_visual_model_pack_dir(
    visual_models_dir: &Path,
    pack_id: &str,
) -> Result<PathBuf, String> {
    let pack = visual_model_packs()
        .into_iter()
        .find(|candidate| candidate.id == pack_id)
        .ok_or_else(|| format!("Unknown visual model pack: {pack_id}"))?;
    let directory = visual_models_dir.join(pack_id);
    if !is_complete_install(
        &pack,
        &directory,
        read_installed_manifest(&directory)?.as_ref(),
    ) {
        return Err(format!(
            "Install or reinstall the {} visual model pack before running this analysis",
            pack.display_name
        ));
    }
    Ok(directory)
}

/// Returns a stable model revision derived from the verified pack manifest.
/// The revision changes whenever the set of installed artifact digests changes,
/// but not when the application merely restarts or the manifest timestamp is
/// refreshed.
pub fn visual_model_pack_revision_in_dir(
    visual_models_dir: &Path,
    pack_id: &str,
) -> Result<String, String> {
    let directory = verified_visual_model_pack_dir(visual_models_dir, pack_id)?;
    let manifest = read_installed_manifest(&directory)?
        .ok_or_else(|| format!("Missing manifest for visual model pack: {pack_id}"))?;
    let mut artifacts = manifest.artifacts;
    artifacts.sort_by(|left, right| left.file_name.cmp(&right.file_name));
    let mut hasher = Sha256::new();
    hasher.update(pack_id.as_bytes());
    for artifact in artifacts {
        hasher.update([0]);
        hasher.update(artifact.file_name.as_bytes());
        hasher.update([0]);
        hasher.update(artifact.sha256.as_bytes());
    }
    Ok(format!("sha256:{}", hex::encode(hasher.finalize())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_visual_packs_have_https_artifacts() {
        for pack in visual_model_packs()
            .iter()
            .filter(|pack| pack.availability == VisualModelAvailability::DirectDownload)
        {
            assert!(!pack.artifacts.is_empty());
            assert!(
                pack.artifacts
                    .iter()
                    .all(|artifact| artifact.source_url.starts_with("https://"))
            );
        }
    }

    #[test]
    fn direct_download_visual_packs_require_pinned_sha256_digests() {
        for pack in visual_model_packs()
            .iter()
            .filter(|pack| pack.availability == VisualModelAvailability::DirectDownload)
        {
            for artifact in &pack.artifacts {
                let digest = artifact.expected_sha256.as_deref().unwrap_or("");
                assert_eq!(
                    digest.len(),
                    64,
                    "Direct-download visual artifact {} in pack {} must have a 64-char hex SHA-256",
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
    fn direct_download_visual_packs_use_immutable_urls() {
        for pack in visual_model_packs()
            .iter()
            .filter(|pack| pack.availability == VisualModelAvailability::DirectDownload)
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
    fn bundle_visual_packs_declare_required_artifacts() {
        for pack in visual_model_packs()
            .iter()
            .filter(|pack| pack.availability == VisualModelAvailability::BundleRequired)
        {
            assert!(!pack.artifacts.is_empty());
            assert!(
                pack.artifacts
                    .iter()
                    .all(|artifact| !artifact.file_name.is_empty())
            );
        }
    }

    #[test]
    fn registry_only_advertises_visual_packs_with_an_active_runtime_adapter() {
        let ids = visual_model_packs()
            .into_iter()
            .map(|pack| pack.id)
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            vec![
                "ram-plus-onnx".to_string(),
                "bioclip-v1".to_string(),
                "ocec-eye-state".to_string(),
                "rawnind-utnet2-bayer".to_string(),
                "nafnet-sidd-rgb".to_string(),
            ]
        );
    }

    #[test]
    fn complete_install_requires_matching_artifact_digests() {
        let pack = visual_model_packs().remove(0);
        let directory = tempfile::tempdir().unwrap();
        let mut artifacts = Vec::new();
        for artifact in &pack.artifacts {
            let path = directory.path().join(&artifact.file_name);
            fs::write(&path, artifact.file_name.as_bytes()).unwrap();
            artifacts.push(InstalledVisualModelArtifact {
                file_name: artifact.file_name.clone(),
                sha256: sha256_file(&path).unwrap(),
            });
        }
        let manifest = InstalledVisualModelPack {
            pack_id: pack.id.clone(),
            installed_at: 0,
            artifacts,
            source_path: None,
        };
        fs::write(
            manifest_path(directory.path()),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        // A local manifest matching local files cannot override a registry pin.
        assert!(!is_complete_install(
            &pack,
            directory.path(),
            Some(&manifest)
        ));
        let mut unpinned_pack = pack.clone();
        for artifact in &mut unpinned_pack.artifacts {
            artifact.expected_sha256 = None;
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
    fn validated_pack_is_rechecked_when_an_artifact_changes() {
        // Unpinned so this test exercises re-validation itself, independent
        // of which packs happen to carry registry-pinned digests.
        let mut pack = visual_model_packs().remove(0);
        for artifact in &mut pack.artifacts {
            artifact.expected_sha256 = None;
        }
        let directory = tempfile::tempdir().unwrap();
        let mut artifacts = Vec::new();
        for artifact in &pack.artifacts {
            let path = directory.path().join(&artifact.file_name);
            fs::write(&path, artifact.file_name.as_bytes()).unwrap();
            artifacts.push(InstalledVisualModelArtifact {
                file_name: artifact.file_name.clone(),
                sha256: sha256_file(&path).unwrap(),
            });
        }
        let manifest = InstalledVisualModelPack {
            pack_id: pack.id.clone(),
            installed_at: 0,
            artifacts,
            source_path: None,
        };
        fs::write(
            manifest_path(directory.path()),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        assert!(is_complete_install(
            &pack,
            directory.path(),
            Some(&manifest)
        ));
        fs::write(
            directory.path().join(&pack.artifacts[0].file_name),
            b"different bytes with a different length",
        )
        .unwrap();
        assert!(!is_complete_install(
            &pack,
            directory.path(),
            Some(&manifest)
        ));
    }

    #[test]
    fn revision_is_stable_for_a_verified_manifest() {
        // visual_model_pack_revision_in_dir re-resolves its pack by id from
        // the real registry, and every registered pack now carries a pinned
        // expected_sha256 (RawNIND/NAFNet included), so a synthetic file's
        // hash can no longer satisfy verification the way an unpinned
        // BundleRequired pack used to. Test the fingerprint cache key that
        // actually backs the "stable across calls" contract instead.
        let mut pack = visual_model_packs().remove(0);
        for artifact in &mut pack.artifacts {
            artifact.expected_sha256 = None;
        }
        let directory = tempfile::tempdir().unwrap();
        let mut artifacts = Vec::new();
        for artifact in &pack.artifacts {
            let path = directory.path().join(&artifact.file_name);
            fs::write(&path, artifact.file_name.as_bytes()).unwrap();
            artifacts.push(InstalledVisualModelArtifact {
                file_name: artifact.file_name.clone(),
                sha256: sha256_file(&path).unwrap(),
            });
        }
        let manifest = InstalledVisualModelPack {
            pack_id: pack.id.clone(),
            installed_at: 0,
            artifacts,
            source_path: None,
        };
        fs::write(
            manifest_path(directory.path()),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        assert!(is_complete_install(
            &pack,
            directory.path(),
            Some(&manifest)
        ));
        let first = pack_fingerprint(&pack, directory.path()).unwrap();
        let second = pack_fingerprint(&pack, directory.path()).unwrap();
        assert_eq!(first, second);
    }
}
