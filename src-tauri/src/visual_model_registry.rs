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
            artifacts: Vec::new(),
            license_name: "MIT".to_string(),
            license_url: "https://huggingface.co/imageomics/bioclip".to_string(),
            model_source_url: "https://github.com/Imageomics/bioclip".to_string(),
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

#[tauri::command]
pub fn list_visual_model_pack_statuses(app_handle: AppHandle) -> Result<Vec<VisualModelPackStatus>, String> {
    visual_model_packs().into_iter().map(|pack| {
        let directory = pack_dir(&app_handle, &pack.id)?;
        Ok(VisualModelPackStatus { installed: installed(&pack, &directory), install_path: directory.to_string_lossy().into_owned(), pack })
    }).collect()
}

#[tauri::command]
pub async fn download_visual_model_pack(pack_id: String, app_handle: AppHandle) -> Result<VisualModelPackStatus, String> {
    let pack = visual_model_packs().into_iter().find(|candidate| candidate.id == pack_id).ok_or_else(|| format!("Unknown visual model pack: {pack_id}"))?;
    if pack.availability != VisualModelAvailability::DirectDownload {
        return Err(format!("{} requires a pinned ONNX bundle before it can be installed", pack.display_name));
    }
    let directory = pack_dir(&app_handle, &pack.id)?;
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    for artifact in &pack.artifacts {
        let response = reqwest::get(&artifact.source_url).await.map_err(|error| error.to_string())?.error_for_status().map_err(|error| error.to_string())?;
        let bytes = response.bytes().await.map_err(|error| error.to_string())?;
        if bytes.is_empty() { return Err(format!("Downloaded {} was empty", artifact.file_name)); }
        let target = directory.join(&artifact.file_name);
        let temporary = target.with_extension("download");
        let mut file = fs::File::create(&temporary).map_err(|error| error.to_string())?;
        file.write_all(&bytes).map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        fs::rename(&temporary, &target).map_err(|error| error.to_string())?;
    }
    let manifest = InstalledVisualModelPack { pack_id: pack.id.clone(), installed_at: Utc::now().timestamp() };
    fs::write(manifest_path(&directory), serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?;
    Ok(VisualModelPackStatus { installed: true, install_path: directory.to_string_lossy().into_owned(), pack })
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
}
