use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

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
    #[serde(skip_serializing_if = "Option::is_none")]
    source_path: Option<String>,
}

fn artifact(file_name: &str) -> VisualModelArtifact {
    VisualModelArtifact {
        file_name: file_name.to_string(),
        source_url: format!("https://huggingface.co/benjaminjonard/ram-plus-onnx/resolve/main/{file_name}"),
    }
}

pub fn visual_model_packs() -> Vec<VisualModelPack> {
    vec![
        VisualModelPack {
            id: "ram-plus-onnx".to_string(),
            display_name: "RAM++".to_string(),
            description: "Broad multi-label tagging for scenes, objects, activities, and wildlife gates.".to_string(),
            task: "Broad visual tagging".to_string(),
            availability: VisualModelAvailability::DirectDownload,
            artifacts: vec![artifact("model.onnx"), artifact("tags.txt"), artifact("thresholds.txt")],
            license_name: "Apache-2.0".to_string(),
            license_url: "https://huggingface.co/benjaminjonard/ram-plus-onnx".to_string(),
            model_source_url: "https://github.com/xinyu1205/recognize-anything".to_string(),
        },
        VisualModelPack {
            id: "bioclip-v1".to_string(),
            display_name: "BioCLIP".to_string(),
            description: "Taxonomy-aware organism classification, including birds. Requires a pinned ONNX encoder and matching taxonomy embeddings.".to_string(),
            task: "Wildlife and species classification".to_string(),
            availability: VisualModelAvailability::BundleRequired,
            artifacts: vec![
                VisualModelArtifact { file_name: "vision_encoder.onnx".to_string(), source_url: String::new() },
                VisualModelArtifact { file_name: "species_embeddings.bin".to_string(), source_url: String::new() },
                VisualModelArtifact { file_name: "species_labels.json".to_string(), source_url: String::new() },
            ],
            license_name: "MIT".to_string(),
            license_url: "https://huggingface.co/imageomics/bioclip".to_string(),
            model_source_url: "https://github.com/Imageomics/bioclip".to_string(),
        },
        VisualModelPack {
            id: "rawnind-utnet2-bayer".to_string(),
            display_name: "RawNIND Bayer RAW Denoise".to_string(),
            description: "Deep sensor Bayer joint denoising and demosaicing directly into linear Rec.2020.".to_string(),
            task: "RAW sensor denoising".to_string(),
            availability: VisualModelAvailability::BundleRequired,
            artifacts: vec![
                VisualModelArtifact { file_name: "rawnind_bayer.onnx".to_string(), source_url: String::new() },
            ],
            license_name: "GPL-3.0".to_string(),
            license_url: "https://arxiv.org/abs/2501.08924".to_string(),
            model_source_url: "https://github.com/darktable-org/darktable-ai/blob/master/models/rawdenoise-nind/README.md".to_string(),
        },
        VisualModelPack {
            id: "nafnet-sidd-rgb".to_string(),
            display_name: "NAFNet SIDD RGB Denoise".to_string(),
            description: "Fast high-fidelity nonlinear activation-free network for developed linear RGB images.".to_string(),
            task: "RGB image denoising".to_string(),
            availability: VisualModelAvailability::BundleRequired,
            artifacts: vec![
                VisualModelArtifact { file_name: "nafnet_sidd.onnx".to_string(), source_url: String::new() },
            ],
            license_name: "MIT".to_string(),
            license_url: "https://github.com/megvii-research/NAFNet".to_string(),
            model_source_url: "https://github.com/darktable-org/darktable-ai/blob/master/models/denoise-nafnet/README.md".to_string(),
        },
        VisualModelPack {
            id: "realplksr-2x".to_string(),
            display_name: "RealPLKSR 2x Upscale".to_string(),
            description: "Conservative super-resolution and structural enlargement using pre-GAN MSSIM weights.".to_string(),
            task: "Image enlargement".to_string(),
            availability: VisualModelAvailability::BundleRequired,
            artifacts: vec![
                VisualModelArtifact { file_name: "realplksr_2x.onnx".to_string(), source_url: String::new() },
            ],
            license_name: "Apache-2.0".to_string(),
            license_url: "https://github.com/dslisleedh/PLKSR".to_string(),
            model_source_url: "https://github.com/darktable-org/darktable-ai/blob/master/models/upscale-realplksr/README.md".to_string(),
        },
    ]
}

fn models_dir(app_handle: &AppHandle) -> Result<PathBuf, String> {
    let path = app_handle.path().app_data_dir().map_err(|error| error.to_string())?.join("models").join("visual");
    fs::create_dir_all(&path).map_err(|error| error.to_string())?;
    Ok(path)
}

fn pack_dir(app_handle: &AppHandle, pack_id: &str) -> Result<PathBuf, String> {
    Ok(models_dir(app_handle)?.join(pack_id))
}

fn manifest_path(directory: &Path) -> PathBuf {
    directory.join("manifest.json")
}

fn installed(pack: &VisualModelPack, directory: &Path) -> bool {
    manifest_path(directory).exists() && pack.artifacts.iter().all(|artifact| directory.join(&artifact.file_name).is_file())
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
pub fn list_visual_model_pack_statuses(app_handle: AppHandle) -> Result<Vec<VisualModelPackStatus>, String> {
    visual_model_packs().into_iter().map(|pack| {
        let directory = pack_dir(&app_handle, &pack.id)?;
        Ok(VisualModelPackStatus { installed: installed(&pack, &directory), install_path: directory.to_string_lossy().into_owned(), pack })
    }).collect()
}

#[tauri::command]
pub async fn download_visual_model_pack(
    pack_id: String,
    app_handle: AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<VisualModelPackStatus, String> {
    let pack = visual_model_packs().into_iter().find(|candidate| candidate.id == pack_id).ok_or_else(|| format!("Unknown visual model pack: {pack_id}"))?;
    if pack.availability != VisualModelAvailability::DirectDownload {
        return Err(format!("{} requires a pinned ONNX bundle before it can be installed", pack.display_name));
    }
    let directory = pack_dir(&app_handle, &pack.id)?;
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let total = pack.artifacts.len() as i64;
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
            total,
            None,
            None,
        );
    }
    let job_control = crate::app_state::BackgroundJobControl::new();
    if let Some((_, job_id)) = job.as_ref() {
        state.background_job_controls.lock().unwrap().insert(job_id.clone(), job_control.clone());
    }

    for (index, artifact) in pack.artifacts.iter().enumerate() {
        let current = index as i64;
        
        let is_runnable = job_control.wait_until_runnable().await;
        if !is_runnable || *job_control.cancellation_receiver().borrow() {
            if let Some((db_path, job_id)) = job.as_ref() {
                let _ = crate::library_db::update_job(db_path, job_id, "cancelled", "Model download cancelled", current, total, None, None);
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

        let bytes = tokio::select! {
            res = response.bytes() => {
                res.map_err(|error| fail_download_job(&job, &state, error.to_string(), current, total))?
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

        if bytes.is_empty() {
            let _ = fs::remove_file(&temporary);
            return Err(fail_download_job(&job, &state, format!("Downloaded {} was empty", artifact.file_name), current, total));
        }
        
        let mut file = fs::File::create(&temporary).map_err(|error| fail_download_job(&job, &state, error.to_string(), current, total))?;
        file.write_all(&bytes).map_err(|error| fail_download_job(&job, &state, error.to_string(), current, total))?;
        file.sync_all().map_err(|error| fail_download_job(&job, &state, error.to_string(), current, total))?;
        fs::rename(&temporary, &target).map_err(|error| fail_download_job(&job, &state, error.to_string(), current, total))?;
    }
    let manifest = InstalledVisualModelPack { pack_id: pack.id.clone(), installed_at: Utc::now().timestamp(), source_path: None };
    let manifest = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| fail_download_job(&job, &state, error.to_string(), total, total))?;
    fs::write(manifest_path(&directory), manifest)
        .map_err(|error| fail_download_job(&job, &state, error.to_string(), total, total))?;
    if let Some((db_path, job_id)) = job.as_ref() {
        let _ = crate::library_db::update_job(db_path, job_id, "completed", "Model download complete", total, total, None, None);
        state.background_job_controls.lock().unwrap().remove(job_id);
    }
    Ok(VisualModelPackStatus { installed: true, install_path: directory.to_string_lossy().into_owned(), pack })
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
        return Err(format!("{} is downloaded directly by RapidRAW", pack.display_name));
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
        fs::copy(source.join(&artifact.file_name), directory.join(&artifact.file_name))
            .map_err(|error| format!("Could not install {}: {error}", artifact.file_name))?;
    }
    let manifest = InstalledVisualModelPack {
        pack_id: pack.id.clone(),
        installed_at: Utc::now().timestamp(),
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

pub(crate) fn installed_visual_model_path(app_handle: &AppHandle, pack_id: &str, file_name: &str) -> Result<PathBuf, String> {
    let path = pack_dir(app_handle, pack_id)?.join(file_name);
    if path.is_file() { Ok(path) } else { Err(format!("Install the {} visual model pack before running this analysis", pack_id)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_visual_packs_have_https_artifacts() {
        for pack in visual_model_packs().iter().filter(|pack| pack.availability == VisualModelAvailability::DirectDownload) {
            assert!(!pack.artifacts.is_empty());
            assert!(pack.artifacts.iter().all(|artifact| artifact.source_url.starts_with("https://")));
        }
    }

    #[test]
    fn bundle_visual_packs_declare_required_artifacts() {
        for pack in visual_model_packs().iter().filter(|pack| pack.availability == VisualModelAvailability::BundleRequired) {
            assert!(!pack.artifacts.is_empty());
            assert!(pack.artifacts.iter().all(|artifact| !artifact.file_name.is_empty()));
        }
    }
}
