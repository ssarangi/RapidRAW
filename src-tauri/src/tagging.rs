use anyhow::Result;
use futures::stream::{self, StreamExt};
use image::{DynamicImage, imageops::FilterType};
use ndarray::{Array, Axis};
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};
use tokenizers::Tokenizer;
use tokio::task::JoinHandle;
use walkdir::WalkDir;

use crate::file_management::{self, parse_virtual_path};
use crate::formats::is_supported_image_file;
use crate::hierarchy::TAG_HIERARCHY;
use crate::image_processing::ImageMetadata;
use crate::{AppState, candidates::TAG_CANDIDATES};

#[derive(Clone, Debug)]
pub struct ScoredTag {
    pub name: String,
    pub confidence: f32,
}

const RAM_PLUS_MODEL_ID: &str = "ram-plus";
const RAM_PLUS_INPUT_SIZE: u32 = 384;

pub(crate) struct RamPlusModels {
    model: Mutex<Session>,
    tags: Vec<String>,
    thresholds: Vec<f32>,
}

pub(crate) fn load_ram_plus_models(
    app_handle: &AppHandle,
) -> std::result::Result<Arc<RamPlusModels>, String> {
    let models_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("models")
        .join("visual");
    load_ram_plus_models_in_dir(&models_dir)
}

fn load_ram_plus_models_in_dir(
    visual_models_dir: &Path,
) -> std::result::Result<Arc<RamPlusModels>, String> {
    let model_path = crate::visual_model_registry::installed_visual_model_path_in_dir(
        visual_models_dir,
        "ram-plus-onnx",
        "model.onnx",
    )?;
    let tags_path = crate::visual_model_registry::installed_visual_model_path_in_dir(
        visual_models_dir,
        "ram-plus-onnx",
        "tags.txt",
    )?;
    let thresholds_path = crate::visual_model_registry::installed_visual_model_path_in_dir(
        visual_models_dir,
        "ram-plus-onnx",
        "thresholds.txt",
    )?;
    let tags = fs::read_to_string(tags_path)
        .map_err(|error| error.to_string())?
        .lines()
        .map(str::trim)
        .filter(|tag| !tag.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    let thresholds = fs::read_to_string(thresholds_path)
        .map_err(|error| error.to_string())?
        .lines()
        .map(str::trim)
        .filter_map(|value| value.parse::<f32>().ok())
        .collect::<Vec<_>>();
    if tags.is_empty() || tags.len() != thresholds.len() {
        return Err("RAM++ tag metadata is invalid or incomplete".to_string());
    }
    let model = Session::builder()
        .map_err(|error| error.to_string())?
        .commit_from_file(model_path)
        .map_err(|error| error.to_string())?;
    Ok(Arc::new(RamPlusModels {
        model: Mutex::new(model),
        tags,
        thresholds,
    }))
}

fn ram_plus_input(image: &DynamicImage) -> Array<f32, ndarray::Dim<[usize; 4]>> {
    let image = image
        .resize_exact(
            RAM_PLUS_INPUT_SIZE,
            RAM_PLUS_INPUT_SIZE,
            FilterType::Triangle,
        )
        .to_rgb8();
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    let mut input = Array::zeros((
        1,
        3,
        RAM_PLUS_INPUT_SIZE as usize,
        RAM_PLUS_INPUT_SIZE as usize,
    ));
    for (x, y, pixel) in image.enumerate_pixels() {
        for channel in 0..3 {
            input[[0, channel, y as usize, x as usize]] =
                (pixel[channel] as f32 / 255.0 - mean[channel]) / std[channel];
        }
    }
    input
}

pub(crate) fn generate_tags_with_ram_plus(
    image: &DynamicImage,
    models: &RamPlusModels,
    max_tags: usize,
) -> std::result::Result<Vec<ScoredTag>, String> {
    let input = Tensor::from_array(ram_plus_input(image)).map_err(|error| error.to_string())?;
    let mut session = models.model.lock().unwrap();
    let output = session
        .run(ort::inputs![input])
        .map_err(|error| error.to_string())?;
    let logits = output[0]
        .try_extract_array::<f32>()
        .map_err(|error| error.to_string())?
        .iter()
        .copied()
        .collect::<Vec<_>>();
    if logits.len() != models.tags.len() {
        return Err(format!(
            "RAM++ output has {} logits but {} tags",
            logits.len(),
            models.tags.len()
        ));
    }
    let mut results = logits
        .into_iter()
        .zip(models.tags.iter().zip(models.thresholds.iter()))
        .filter_map(|(logit, (tag, threshold))| {
            let confidence = 1.0 / (1.0 + (-logit.clamp(-30.0, 30.0)).exp());
            (confidence > *threshold).then(|| ScoredTag {
                name: tag.clone(),
                confidence,
            })
        })
        .collect::<Vec<_>>();
    results.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    results.truncate(max_tags);
    Ok(results)
}

pub(crate) struct BioClipModels {
    session: std::sync::Mutex<ort::session::Session>,
    model_id: &'static str,
    /// One contiguous, normalized taxonomy matrix. This avoids a separate heap
    /// allocation per species when loading the full BioCLIP 2 taxonomy.
    embeddings: Vec<f32>,
    embedding_dimension: usize,
    labels: Vec<BioClipTaxon>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BioClipTaxonomyManifest {
    embedding_dimension: usize,
    embedding_parts: Vec<String>,
}

/// Number of taxonomy neighbours retained while deciding whether a species
/// name is distinguishable from visually similar organisms.
const BIOCLIP_CANDIDATE_COUNT: usize = 5;
/// Raw cosine similarity is not a probability. These are deliberately
/// conservative admission gates for a *species suggestion*, not an automatic
/// acceptance threshold.
const BIOCLIP_MIN_SPECIES_SIMILARITY: f32 = 0.35;
const BIOCLIP_MIN_SPECIES_MARGIN: f32 = 0.04;

#[derive(Clone, Debug)]
pub(crate) struct BioClipCandidate {
    pub taxon: BioClipTaxon,
    pub similarity: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct BioClipInference {
    candidates: Vec<BioClipCandidate>,
}

impl BioClipInference {
    fn best(&self) -> Option<&BioClipCandidate> {
        self.candidates.first()
    }

    fn runner_up_similarity(&self) -> Option<f32> {
        let best = self.best()?;
        self.candidates
            .iter()
            .skip(1)
            .find(|candidate| candidate.taxon.scientific_name != best.taxon.scientific_name)
            .map(|candidate| candidate.similarity)
    }

    pub(crate) fn is_confident_species(&self) -> bool {
        let Some(best) = self.best() else {
            return false;
        };
        let margin = self
            .runner_up_similarity()
            .map(|score| best.similarity - score)
            // A taxonomy with a single label has no competing species, so its
            // score still has to clear the base threshold.
            .unwrap_or(BIOCLIP_MIN_SPECIES_MARGIN);
        best.similarity >= BIOCLIP_MIN_SPECIES_SIMILARITY && margin >= BIOCLIP_MIN_SPECIES_MARGIN
    }

    pub(crate) fn best_taxon_and_similarity(&self) -> Option<(&BioClipTaxon, f32)> {
        self.best()
            .map(|candidate| (&candidate.taxon, candidate.similarity))
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BioClipTaxon {
    pub scientific_name: String,
    #[serde(default)]
    pub common_name: Option<String>,
    #[serde(default = "default_species_rank")]
    taxon_rank: String,
}

fn default_species_rank() -> String {
    "species".to_string()
}

pub(crate) fn load_bioclip_models(app_handle: &tauri::AppHandle) -> Result<BioClipModels, String> {
    let models_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("models")
        .join("visual");
    load_bioclip_models_in_dir(&models_dir)
}

fn load_bioclip_models_in_dir(visual_models_dir: &Path) -> Result<BioClipModels, String> {
    let mut errors = Vec::new();
    for model_id in crate::visual_model_registry::BIOCLIP_MODEL_IDS_BY_ACCURACY {
        match load_bioclip_model_in_dir(visual_models_dir, model_id) {
            Ok(models) => return Ok(models),
            Err(error) => errors.push(format!("{model_id}: {error}")),
        }
    }
    Err(format!(
        "No compatible BioCLIP model is installed ({})",
        errors.join("; ")
    ))
}

fn load_bioclip_model_in_dir(
    visual_models_dir: &Path,
    model_id: &'static str,
) -> Result<BioClipModels, String> {
    let model_path = crate::visual_model_registry::installed_visual_model_path_in_dir(
        visual_models_dir,
        model_id,
        "vision_encoder.onnx",
    )?;
    let labels_path = crate::visual_model_registry::installed_visual_model_path_in_dir(
        visual_models_dir,
        model_id,
        "species_labels.json",
    )?;

    if !model_path.exists() || !labels_path.exists() {
        return Err("BioCLIP artifacts missing".into());
    }

    let session = ort::session::Session::builder()
        .and_then(|b| b.commit_from_file(&model_path))
        .map_err(|e| e.to_string())?;

    let labels_json = std::fs::read_to_string(&labels_path).map_err(|e| e.to_string())?;
    let labels: Vec<BioClipTaxon> = serde_json::from_str(&labels_json).map_err(|_| {
        "BioCLIP species_labels.json must contain taxonomy records with scientificName".to_string()
    })?;

    let pack_dir = model_path
        .parent()
        .ok_or("BioCLIP model path has no parent directory")?;
    let manifest_path = pack_dir.join("taxonomy_manifest.json");
    let (embedding_dimension, embedding_paths) = if manifest_path.exists() {
        let manifest: BioClipTaxonomyManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path).map_err(|e| e.to_string())?)
                .map_err(|_| "BioCLIP taxonomy_manifest.json is invalid".to_string())?;
        if manifest.embedding_dimension == 0 || manifest.embedding_parts.is_empty() {
            return Err("BioCLIP taxonomy manifest has no embeddings".to_string());
        }
        (
            manifest.embedding_dimension,
            manifest
                .embedding_parts
                .into_iter()
                .map(|part| pack_dir.join(part))
                .collect::<Vec<_>>(),
        )
    } else {
        (
            0,
            vec![
                crate::visual_model_registry::installed_visual_model_path_in_dir(
                    visual_models_dir,
                    model_id,
                    "species_embeddings.bin",
                )?,
            ],
        )
    };

    let total_bytes = embedding_paths.iter().try_fold(0usize, |total, path| {
        let bytes = std::fs::metadata(path)
            .map_err(|_| format!("BioCLIP taxonomy part is missing: {}", path.display()))?
            .len() as usize;
        total
            .checked_add(bytes)
            .ok_or_else(|| "BioCLIP taxonomy is too large".to_string())
    })?;
    if total_bytes % 4 != 0 {
        return Err("BioCLIP embeddings are not a packed f32 array".to_string());
    }
    let total_f32 = total_bytes / 4;
    let dim = if embedding_dimension == 0 {
        if labels.is_empty() || total_f32 % labels.len() != 0 {
            return Err("Invalid BioCLIP embeddings/labels shape".into());
        }
        total_f32 / labels.len()
    } else {
        embedding_dimension
    };
    if labels.is_empty() || total_f32 != labels.len() * dim {
        return Err("Invalid BioCLIP embeddings/labels shape".into());
    }

    let mut embeddings = Vec::with_capacity(total_f32);
    for path in embedding_paths {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        for chunk in bytes.chunks_exact(4) {
            let val = f32::from_le_bytes(chunk.try_into().unwrap());
            if !val.is_finite() {
                return Err("BioCLIP embeddings contain a non-finite value".to_string());
            }
            embeddings.push(val);
        }
    }
    for embedding in embeddings.chunks_exact_mut(dim) {
        let norm = embedding
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if !norm.is_finite() || norm <= 1e-8 {
            return Err("BioCLIP embeddings contain a zero-length vector".to_string());
        }
        for value in embedding {
            *value /= norm;
        }
    }

    Ok(BioClipModels {
        session: std::sync::Mutex::new(session),
        model_id,
        embeddings,
        embedding_dimension: dim,
        labels,
    })
}

fn bioclip_input(image: &image::DynamicImage) -> ndarray::Array<f32, ndarray::Dim<[usize; 4]>> {
    let image = image
        .resize_exact(224, 224, image::imageops::FilterType::Triangle)
        .to_rgb8();
    let mean = [0.48145466, 0.4578275, 0.40821073];
    let std = [0.26862954, 0.26130258, 0.27577711];
    let mut input = ndarray::Array::zeros((1, 3, 224, 224));
    for (x, y, pixel) in image.enumerate_pixels() {
        for channel in 0..3 {
            input[[0, channel, y as usize, x as usize]] =
                (pixel[channel] as f32 / 255.0 - mean[channel]) / std[channel];
        }
    }
    input
}

/// Runs BioCLIP's vision encoder and returns the raw image embedding, before
/// any taxonomy lookup. Exposed separately from taxonomy classification so
/// callers that just want "how visually/taxonomically similar are these two
/// images" (e.g. vetoing a perceptual-hash duplicate match) do not need to
/// evaluate the candidate taxonomy.
pub(crate) fn bioclip_embedding(
    image: &image::DynamicImage,
    models: &BioClipModels,
) -> Result<Vec<f32>, String> {
    let input =
        ort::value::Tensor::from_array(bioclip_input(image)).map_err(|error| error.to_string())?;
    let mut session = models.session.lock().unwrap();
    let output = session
        .run(ort::inputs![input])
        .map_err(|error| error.to_string())?;
    let img_emb = output[0]
        .try_extract_array::<f32>()
        .map_err(|error| error.to_string())?
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let expected_dimension = models.embedding_dimension;
    if img_emb.len() != expected_dimension || img_emb.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "BioCLIP encoder emitted {} values; taxonomy expects {expected_dimension}",
            img_emb.len()
        ));
    }
    Ok(img_emb)
}

/// Returns the closest taxonomy candidates in descending similarity.  Callers
/// must use `is_confident_species` before presenting the top candidate as a
/// species name; close neighbours are deliberately treated as ambiguous.
pub(crate) fn run_bioclip_ranked_inference(
    image: &image::DynamicImage,
    models: &BioClipModels,
) -> Result<BioClipInference, String> {
    let img_emb = bioclip_embedding(image, models)?;
    let norm_img: f32 = img_emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);

    let mut candidates = Vec::with_capacity(BIOCLIP_CANDIDATE_COUNT);

    for (i, tax_emb) in models
        .embeddings
        .chunks_exact(models.embedding_dimension)
        .enumerate()
    {
        let dot: f32 = img_emb.iter().zip(tax_emb).map(|(a, b)| a * b).sum();
        let score = dot / norm_img;
        // Keep a fixed-size leaderboard. Tree-of-Life taxonomies can contain
        // hundreds of thousands of labels, so cloning every taxon for every
        // image would make the ambiguity check itself needlessly expensive.
        let beats_current_cutoff = candidates
            .last()
            .map(|candidate: &BioClipCandidate| score > candidate.similarity)
            .unwrap_or(true);
        if candidates.len() < BIOCLIP_CANDIDATE_COUNT || beats_current_cutoff {
            candidates.push(BioClipCandidate {
                taxon: models.labels[i].clone(),
                similarity: score,
            });
            candidates.sort_by(|left, right| right.similarity.total_cmp(&left.similarity));
            candidates.truncate(BIOCLIP_CANDIDATE_COUNT);
        }
    }
    Ok(BioClipInference { candidates })
}

fn store_bioclip_suggestion(
    db_path: &Path,
    image_id: i64,
    image: &DynamicImage,
    bioclip_models: &Result<BioClipModels, String>,
    model_revision: Option<&str>,
    ai_semaphore: &Arc<tokio::sync::Semaphore>,
) {
    let Some(model_revision) = model_revision else {
        return;
    };
    let Ok(bioclip) = bioclip_models else {
        return;
    };
    let permit = tauri::async_runtime::block_on(ai_semaphore.clone().acquire_owned()).ok();
    let inference_result = run_bioclip_ranked_inference(image, bioclip);
    drop(permit);
    let Ok(inference) = inference_result else {
        return;
    };
    store_bioclip_inference(
        db_path,
        image_id,
        &inference,
        bioclip.model_id,
        model_revision,
    );
}

/// Persists only a clearly separated species candidate.  A near tie is more
/// useful as "needs another look" than as an incorrect name in the catalog.
fn store_bioclip_inference(
    db_path: &Path,
    image_id: i64,
    inference: &BioClipInference,
    model_id: &str,
    model_revision: &str,
) {
    let Ok(conn) = rusqlite::Connection::open(db_path) else {
        return;
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let _ = conn.execute(
        "DELETE FROM species_classifications WHERE image_id = ?1 AND model_id = ?2 AND review_state = 'suggested'",
        rusqlite::params![image_id, model_id],
    );
    // Remove a prior unreviewed guess even when this pass is ambiguous.  We do
    // not leave a stale species name visible merely because the new evidence is
    // insufficient to replace it. Accepted/rejected reviewer decisions stay.
    if !inference.is_confident_species() {
        return;
    }
    let Some((taxon, confidence)) = inference.best_taxon_and_similarity() else {
        return;
    };
    let _ = conn.execute(
        "INSERT INTO species_classifications(image_id, model_id, model_revision, scientific_name, common_name, taxon_rank, confidence, review_state, created_at, updated_at)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, 'suggested', ?8, ?8)",
        rusqlite::params![
            image_id,
            model_id,
            model_revision,
            &taxon.scientific_name,
            taxon.common_name.as_deref(),
            &taxon.taxon_rank,
            confidence as f64,
            now,
        ],
    );
}

/// Executes the RAM++ catalog batch without a Tauri window. It is shared by
/// headless callers and follows the same durable job/state contract as the UI.
pub fn run_catalog_ram_plus_tagging_headless(
    db_path: &Path,
    visual_models_dir: &Path,
    tag_count: usize,
    include_bioclip: bool,
    job_id: &str,
    job_control: &Arc<crate::app_state::BackgroundJobControl>,
    ai_semaphore: Arc<tokio::sync::Semaphore>,
) -> Result<(), String> {
    crate::library_db::update_job(
        db_path,
        job_id,
        "running",
        "Preparing RAM++ tagging",
        0,
        0,
        None,
        None,
    )?;
    let model_revision = crate::visual_model_registry::visual_model_pack_revision_in_dir(
        visual_models_dir,
        "ram-plus-onnx",
    )?;
    let candidates = crate::library_db::list_ai_tag_candidates_for_model(
        db_path,
        RAM_PLUS_MODEL_ID,
        &model_revision,
    )?;
    let total = candidates.len() as i64;
    if candidates.is_empty() {
        return crate::library_db::update_job(
            db_path,
            job_id,
            "completed",
            "All catalog images already have RAM++ tags",
            0,
            0,
            None,
            None,
        );
    }
    crate::library_db::update_job(
        db_path,
        job_id,
        "running",
        "Loading RAM++ model",
        0,
        total,
        None,
        None,
    )?;
    let models = load_ram_plus_models_in_dir(visual_models_dir)?;
    let bioclip_models = include_bioclip
        .then(|| load_bioclip_models_in_dir(visual_models_dir))
        .unwrap_or_else(|| Err("BioCLIP disabled for this job".to_string()));
    let bioclip_revision = bioclip_models.as_ref().ok().and_then(|models| {
        crate::visual_model_registry::visual_model_pack_revision_in_dir(
            visual_models_dir,
            models.model_id,
        )
        .ok()
    });

    for (index, (image_id, path, modified)) in candidates.into_iter().enumerate() {
        if !tauri::async_runtime::block_on(job_control.wait_until_runnable()) {
            return Err("RAM++ tagging cancelled".to_string());
        }
        let current = index as i64 + 1;
        crate::library_db::update_job(
            db_path,
            job_id,
            "running",
            "RAM++ tagging image",
            current,
            total,
            Some(&path),
            None,
        )?;
        crate::library_db::mark_ai_tag_analysis_state_for_model(
            db_path,
            image_id,
            modified,
            RAM_PLUS_MODEL_ID,
            &model_revision,
            "processing",
            None,
        )?;
        match crate::face_detection::load_image_for_local_ai(Path::new(&path)) {
            Ok(image) => {
                let permit =
                    tauri::async_runtime::block_on(ai_semaphore.clone().acquire_owned()).ok();
                let result = generate_tags_with_ram_plus(&image, &models, tag_count.clamp(1, 100));
                drop(permit);
                match result {
                    Ok(tags) => {
                        crate::library_db::replace_ai_tags_for_model(
                            db_path,
                            image_id,
                            RAM_PLUS_MODEL_ID,
                            &model_revision,
                            &tags,
                        )?;
                        crate::library_db::mark_ai_tag_analysis_state_for_model(
                            db_path,
                            image_id,
                            modified,
                            RAM_PLUS_MODEL_ID,
                            &model_revision,
                            "completed",
                            None,
                        )?;
                        if tags.iter().any(|tag| {
                            let name = tag.name.to_ascii_lowercase();
                            name.contains("bird")
                                || name.contains("wildlife")
                                || name.contains("animal")
                        }) {
                            store_bioclip_suggestion(
                                db_path,
                                image_id,
                                &image,
                                &bioclip_models,
                                bioclip_revision.as_deref(),
                                &ai_semaphore,
                            );
                        }
                    }
                    Err(error) => {
                        crate::library_db::mark_ai_tag_analysis_state_for_model(
                            db_path,
                            image_id,
                            modified,
                            RAM_PLUS_MODEL_ID,
                            &model_revision,
                            "failed",
                            Some(&error),
                        )?;
                    }
                }
            }
            Err(error) => crate::library_db::mark_ai_tag_analysis_state_for_model(
                db_path,
                image_id,
                modified,
                RAM_PLUS_MODEL_ID,
                &model_revision,
                "failed",
                Some(&error),
            )?,
        }
    }
    crate::library_db::update_job(
        db_path,
        job_id,
        "completed",
        "RAM++ catalog tagging complete",
        total,
        total,
        None,
        None,
    )
}

#[tauri::command]
pub fn start_catalog_ram_plus_tagging(
    root_id: Option<i64>,
    relative_path: Option<String>,
    force: Option<bool>,
    app_handle: AppHandle,
    state: State<'_, AppState>,
) -> Result<String, String> {
    let db_path = crate::library_db::active_library_path(&state)?;
    let job_id = crate::library_db::create_background_job(
        &db_path,
        "ram_plus_tagging",
        serde_json::json!({ "modelId": RAM_PLUS_MODEL_ID, "rootId": root_id, "relativePath": relative_path, "force": force.unwrap_or(false) }),
    )?;
    crate::library_db::set_background_job_root_id(&db_path, &job_id, root_id)?;
    crate::library_db::update_job(
        &db_path,
        &job_id,
        "queued",
        "RAM++ tagging queued",
        0,
        0,
        None,
        None,
    )?;
    let job_control = crate::app_state::BackgroundJobControl::new();
    state
        .background_job_controls
        .lock()
        .unwrap()
        .insert(job_id.clone(), job_control.clone());
    let worker_state = app_handle.clone();
    let species_app_handle = app_handle.clone();
    let worker_db_path = db_path.clone();
    let worker_job_id = job_id.clone();

    tauri::async_runtime::spawn_blocking(move || {
        // Candidate enumeration and ONNX session construction can both take long enough to
        // visibly stall the UI when done in the command handler. Keep all startup work here.
        let _ = crate::library_db::update_job(
            &worker_db_path,
            &worker_job_id,
            "running",
            "Preparing RAM++ tagging",
            0,
            0,
            None,
            None,
        );
        let visual_models_dir = match worker_state.path().app_data_dir() {
            Ok(path) => path.join("models").join("visual"),
            Err(error) => {
                let error = error.to_string();
                let _ = crate::library_db::update_job(
                    &worker_db_path,
                    &worker_job_id,
                    "failed",
                    "Unable to locate RAM++ model directory",
                    0,
                    0,
                    None,
                    Some(&error),
                );
                cleanup_tag_job(&worker_state, &worker_job_id);
                return;
            }
        };
        let model_revision = match crate::visual_model_registry::visual_model_pack_revision_in_dir(
            &visual_models_dir,
            "ram-plus-onnx",
        ) {
            Ok(revision) => revision,
            Err(error) => {
                let _ = crate::library_db::update_job(
                    &worker_db_path,
                    &worker_job_id,
                    "failed",
                    "Unable to verify RAM++ model",
                    0,
                    0,
                    None,
                    Some(&error),
                );
                cleanup_tag_job(&worker_state, &worker_job_id);
                return;
            }
        };
        let candidates = match crate::library_db::list_ai_tag_candidates_for_model_in_scope(
            &worker_db_path,
            RAM_PLUS_MODEL_ID,
            &model_revision,
            root_id,
            relative_path.as_deref(),
            force.unwrap_or(false),
        ) {
            Ok(candidates) => candidates,
            Err(error) => {
                let _ = crate::library_db::update_job(
                    &worker_db_path,
                    &worker_job_id,
                    "failed",
                    "Unable to prepare RAM++ tagging",
                    0,
                    0,
                    None,
                    Some(&error),
                );
                cleanup_tag_job(&worker_state, &worker_job_id);
                return;
            }
        };
        let total = candidates.len() as i64;
        if candidates.is_empty() {
            let _ = crate::library_db::update_job(
                &worker_db_path,
                &worker_job_id,
                "completed",
                "All catalog images already have RAM++ tags",
                0,
                0,
                None,
                None,
            );
            cleanup_tag_job(&worker_state, &worker_job_id);
            return;
        }
        let _ = crate::library_db::update_job(
            &worker_db_path,
            &worker_job_id,
            "running",
            "Loading RAM++ model",
            0,
            total,
            None,
            None,
        );
        let models = match load_ram_plus_models(&worker_state) {
            Ok(models) => models,
            Err(error) => {
                let _ = crate::library_db::update_job(
                    &worker_db_path,
                    &worker_job_id,
                    "failed",
                    "Unable to load RAM++",
                    0,
                    total,
                    None,
                    Some(&error),
                );
                cleanup_tag_job(&worker_state, &worker_job_id);
                return;
            }
        };
        let tag_count = crate::load_settings(worker_state.clone())
            .ok()
            .and_then(|settings| settings.ai_tag_count)
            .unwrap_or(20) as usize;
        let ai_semaphore = worker_state.state::<AppState>().ai_job_semaphore.clone();
        // BioCLIP is optional, but loading it must not block the Tauri command/UI thread.
        let bioclip_models = load_bioclip_models(&species_app_handle);
        let bioclip_revision = bioclip_models.as_ref().ok().and_then(|models| {
            crate::visual_model_registry::visual_model_pack_revision_in_dir(
                &visual_models_dir,
                models.model_id,
            )
            .ok()
        });
        for (index, (image_id, path, modified)) in candidates.into_iter().enumerate() {
            if !tauri::async_runtime::block_on(job_control.wait_until_runnable()) {
                let _ = crate::library_db::update_job(
                    &worker_db_path,
                    &worker_job_id,
                    "cancelled",
                    "RAM++ tagging cancelled",
                    index as i64,
                    total,
                    Some(&path),
                    None,
                );
                cleanup_tag_job(&worker_state, &worker_job_id);
                return;
            }
            if *job_control.cancellation_receiver().borrow() {
                let _ = crate::library_db::update_job(
                    &worker_db_path,
                    &worker_job_id,
                    "cancelled",
                    "RAM++ tagging cancelled",
                    index as i64,
                    total,
                    Some(&path),
                    None,
                );
                cleanup_tag_job(&worker_state, &worker_job_id);
                return;
            }
            let current = index as i64 + 1;
            let _ = crate::library_db::update_job(
                &worker_db_path,
                &worker_job_id,
                "running",
                "RAM++ tagging image",
                current,
                total,
                Some(&path),
                None,
            );
            let _ = crate::library_db::mark_ai_tag_analysis_state_for_model(
                &worker_db_path,
                image_id,
                modified,
                RAM_PLUS_MODEL_ID,
                &model_revision,
                "processing",
                None,
            );

            let image_res = crate::file_management::get_cached_or_generate_thumbnail_image(
                &path,
                &worker_state,
                None,
            )
            .map_err(|error| error.to_string());
            match image_res {
                Ok(image) => {
                    let _permit =
                        tauri::async_runtime::block_on(ai_semaphore.clone().acquire_owned()).ok();
                    let result = generate_tags_with_ram_plus(&image, &models, tag_count);
                    drop(_permit);
                    match result {
                        Ok(tags) => {
                            let _ = crate::library_db::replace_ai_tags_for_model(
                                &worker_db_path,
                                image_id,
                                RAM_PLUS_MODEL_ID,
                                &model_revision,
                                &tags,
                            );
                            let _ = crate::library_db::mark_ai_tag_analysis_state_for_model(
                                &worker_db_path,
                                image_id,
                                modified,
                                RAM_PLUS_MODEL_ID,
                                &model_revision,
                                "completed",
                                None,
                            );

                            let has_bird_or_wildlife = tags.iter().any(|t| {
                                let lower = t.name.to_ascii_lowercase();
                                lower.contains("bird")
                                    || lower.contains("wildlife")
                                    || lower.contains("animal")
                            });
                            if has_bird_or_wildlife {
                                if let (Ok(bioclip), Some(bioclip_revision)) =
                                    (&bioclip_models, bioclip_revision.as_deref())
                                {
                                    let _permit = tauri::async_runtime::block_on(
                                        ai_semaphore.clone().acquire_owned(),
                                    )
                                    .ok();
                                    let inference_result =
                                        run_bioclip_ranked_inference(&image, bioclip);
                                    drop(_permit);
                                    if let Ok(inference) = inference_result {
                                        store_bioclip_inference(
                                            &worker_db_path,
                                            image_id,
                                            &inference,
                                            bioclip.model_id,
                                            bioclip_revision,
                                        );
                                    }
                                }
                            }
                        }
                        Err(error) => {
                            let _ = crate::library_db::mark_ai_tag_analysis_state_for_model(
                                &worker_db_path,
                                image_id,
                                modified,
                                RAM_PLUS_MODEL_ID,
                                &model_revision,
                                "failed",
                                Some(&error),
                            );
                        }
                    }
                }
                Err(error) => {
                    let _ = crate::library_db::mark_ai_tag_analysis_state_for_model(
                        &worker_db_path,
                        image_id,
                        modified,
                        RAM_PLUS_MODEL_ID,
                        &model_revision,
                        "failed",
                        Some(&error),
                    );
                }
            }
        }
        let _ = crate::library_db::update_job(
            &worker_db_path,
            &worker_job_id,
            "completed",
            "RAM++ catalog tagging complete",
            total,
            total,
            None,
            None,
        );
        cleanup_tag_job(&worker_state, &worker_job_id);
    });
    Ok(job_id)
}

fn cleanup_tag_job(app_handle: &AppHandle, job_id: &str) {
    app_handle
        .state::<AppState>()
        .background_job_controls
        .lock()
        .unwrap()
        .remove(job_id);
}

#[tauri::command]
pub async fn start_catalog_ai_tagging(
    app_handle: AppHandle,
    state: State<'_, AppState>,
) -> Result<String, String> {
    let db_path = crate::library_db::active_library_path(&state)?;
    let job_id = crate::library_db::create_background_job(
        &db_path,
        "ai_tagging",
        serde_json::json!({ "modelId": "clip", "modelRevision": "rapidraw-clip-v1" }),
    )?;
    let candidates = crate::library_db::list_ai_tag_candidates(&db_path)?;
    crate::library_db::update_job(
        &db_path,
        &job_id,
        "queued",
        "Catalog AI tagging queued",
        0,
        candidates.len() as i64,
        None,
        None,
    )?;
    if candidates.is_empty() {
        crate::library_db::update_job(
            &db_path,
            &job_id,
            "completed",
            "All catalog images already have AI tags",
            0,
            0,
            None,
            None,
        )?;
        return Ok(job_id);
    }
    let settings = crate::load_settings(app_handle.clone())?;
    let tag_count = settings.ai_tag_count.unwrap_or(10) as usize;
    let custom_tags = settings.custom_ai_tags.clone();
    let clip_models = match crate::ai_processing::get_or_init_clip_models(
        &app_handle,
        &state.ai_state,
        &state.ai_init_lock,
    )
    .await
    {
        Ok(models) => models,
        Err(error) => {
            let message = error.to_string();
            let _ = crate::library_db::update_job(
                &db_path,
                &job_id,
                "failed",
                "Unable to initialize the AI tagging model",
                0,
                candidates.len() as i64,
                None,
                Some(&message),
            );
            return Err(message);
        }
    };
    let worker_db_path = db_path.clone();
    let worker_job_id = job_id.clone();
    let worker_app_handle = app_handle.clone();
    let job_control = crate::app_state::BackgroundJobControl::new();
    state
        .background_job_controls
        .lock()
        .unwrap()
        .insert(job_id.clone(), job_control.clone());
    tauri::async_runtime::spawn_blocking(move || {
        let ai_semaphore = worker_app_handle
            .state::<AppState>()
            .ai_job_semaphore
            .clone();
        let total = candidates.len() as i64;
        for (index, (image_id, path, modified)) in candidates.into_iter().enumerate() {
            if !tauri::async_runtime::block_on(job_control.wait_until_runnable()) {
                let _ = crate::library_db::update_job(
                    &worker_db_path,
                    &worker_job_id,
                    "cancelled",
                    "Catalog AI tagging cancelled",
                    index as i64,
                    total,
                    None,
                    None,
                );
                return;
            }
            if *job_control.cancellation_receiver().borrow() {
                let _ = crate::library_db::update_job(
                    &worker_db_path,
                    &worker_job_id,
                    "cancelled",
                    "Catalog AI tagging cancelled",
                    index as i64,
                    total,
                    None,
                    None,
                );
                cleanup_tag_job(&worker_app_handle, &worker_job_id);
                return;
            }
            let current = index as i64 + 1;
            let _ = crate::library_db::update_job(
                &worker_db_path,
                &worker_job_id,
                "running",
                "Tagging image",
                current,
                total,
                Some(&path),
                None,
            );
            crate::library_db::mark_ai_tag_analysis_state(
                &worker_db_path,
                image_id,
                modified,
                "processing",
                None,
            )
            .ok();
            let result = file_management::get_cached_or_generate_thumbnail_image(
                &path,
                &worker_app_handle,
                None,
            )
            .map_err(|error| error.to_string())
            .and_then(|image| {
                let _permit =
                    tauri::async_runtime::block_on(ai_semaphore.clone().acquire_owned()).ok();
                let res = generate_tags_with_clip(
                    &image,
                    &clip_models.model,
                    &clip_models.tokenizer,
                    custom_tags.clone(),
                    tag_count,
                )
                .map_err(|error| error.to_string());
                drop(_permit);
                res
            });
            match result {
                Ok(tags) => {
                    let _ =
                        crate::library_db::replace_clip_ai_tags(&worker_db_path, image_id, &tags);
                    let _ = crate::library_db::mark_ai_tag_analysis_state(
                        &worker_db_path,
                        image_id,
                        modified,
                        "completed",
                        None,
                    );
                }
                Err(error) => {
                    let _ = crate::library_db::mark_ai_tag_analysis_state(
                        &worker_db_path,
                        image_id,
                        modified,
                        "failed",
                        Some(&error),
                    );
                }
            }
        }
        let _ = crate::library_db::update_job(
            &worker_db_path,
            &worker_job_id,
            "completed",
            "Catalog AI tagging complete",
            total,
            total,
            None,
            None,
        );
        cleanup_tag_job(&worker_app_handle, &worker_job_id);
    });
    let _ = app_handle.emit(
        "catalog-ai-tagging-started",
        serde_json::json!({ "jobId": job_id }),
    );
    Ok(job_id)
}

pub const COLOR_TAG_PREFIX: &str = "color:";
pub const USER_TAG_PREFIX: &str = "user:";

fn preprocess_clip_image(image: &DynamicImage) -> Array<f32, ndarray::Dim<[usize; 4]>> {
    let input_size = 224;
    let resized = image.resize_to_fill(input_size, input_size, FilterType::Triangle);
    let rgb_image = resized.to_rgb8();

    let mean = [0.48145466, 0.4578275, 0.40821073];
    let std = [0.26862954, 0.261_302_6, 0.275_777_1];

    let mut array = Array::zeros((1, 3, input_size as usize, input_size as usize));
    for (x, y, pixel) in rgb_image.enumerate_pixels() {
        array[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];
        array[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];
        array[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];
    }
    array
}

fn softmax(array: &Array<f32, ndarray::Dim<[usize; 2]>>) -> Array<f32, ndarray::Dim<[usize; 2]>> {
    let mut new_array = array.clone();
    for mut row in new_array.axis_iter_mut(Axis(0)) {
        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|x| (x - max_val).exp());
        let sum = row.sum();
        if sum > 0.0 {
            row.mapv_inplace(|x| x / sum);
        }
    }
    new_array
}

fn rgb_to_hsv((r, g, b): (u8, u8, u8)) -> (f32, f32, f32) {
    let r = r as f32 / 255.0;
    let g = g as f32 / 255.0;
    let b = b as f32 / 255.0;

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let h = if delta.abs() < f32::EPSILON {
        0.0
    } else if (max - r).abs() < f32::EPSILON {
        60.0 * (((g - b) / delta) % 6.0)
    } else if (max - g).abs() < f32::EPSILON {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };

    let s = if max.abs() < f32::EPSILON {
        0.0
    } else {
        delta / max
    };
    let v = max;

    (h, s, v)
}

pub fn extract_color_tags(image: &DynamicImage) -> Vec<String> {
    let resized = image.resize(100, 100, FilterType::Triangle);
    let rgb_image = resized.to_rgb8();
    let mut color_counts: HashMap<String, u32> = HashMap::new();

    for pixel in rgb_image.pixels() {
        let rgb = (pixel[0], pixel[1], pixel[2]);
        let (h, s, v) = rgb_to_hsv(rgb);

        let color_name = if v < 0.2 {
            "black".to_string()
        } else if s < 0.1 {
            if v > 0.8 {
                "white".to_string()
            } else {
                "gray".to_string()
            }
        } else {
            match h {
                _ if !(20.0..340.0).contains(&h) => "red".to_string(),
                _ if (20.0..45.0).contains(&h) => "orange".to_string(),
                _ if (45.0..70.0).contains(&h) => "yellow".to_string(),
                _ if (70.0..160.0).contains(&h) => "green".to_string(),
                _ if (160.0..260.0).contains(&h) => "blue".to_string(),
                _ if (260.0..340.0).contains(&h) => "purple".to_string(),
                _ => "unknown".to_string(),
            }
        };

        if (color_name == "orange" || color_name == "red") && v < 0.6 && s < 0.7 {
            *color_counts.entry("brown".to_string()).or_insert(0) += 1;
        } else {
            *color_counts.entry(color_name).or_insert(0) += 1;
        }
    }

    let mut colorful_tags: Vec<(String, u32)> = color_counts
        .iter()
        .filter(|(name, _)| !matches!(name.as_str(), "black" | "white" | "gray"))
        .map(|(name, &count)| (name.clone(), count))
        .collect();

    colorful_tags.sort_by_key(|b| std::cmp::Reverse(b.1));

    if !colorful_tags.is_empty() {
        colorful_tags
            .into_iter()
            .take(2)
            .map(|(name, _)| name)
            .collect()
    } else {
        color_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(name, _)| vec![name])
            .unwrap_or_default()
    }
}

pub fn generate_tags_with_clip(
    image: &DynamicImage,
    clip_session_mutex: &Mutex<Session>,
    tokenizer: &Tokenizer,
    custom_tags: Option<Vec<String>>,
    max_tags: usize,
) -> Result<Vec<ScoredTag>> {
    let image_input = preprocess_clip_image(image);

    let is_custom = custom_tags.as_ref().map(|t| !t.is_empty()).unwrap_or(false);
    let text_inputs: Vec<String> = if is_custom {
        custom_tags.as_ref().unwrap().clone()
    } else {
        TAG_CANDIDATES.iter().map(|&s| s.to_string()).collect()
    };

    let encodings = tokenizer
        .encode_batch(text_inputs.clone(), true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let max_len = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(0);

    let mut ids_data = Vec::new();
    let mut mask_data = Vec::new();
    for encoding in encodings {
        let mut ids = encoding
            .get_ids()
            .iter()
            .map(|&i| i as i64)
            .collect::<Vec<_>>();
        let mut mask = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect::<Vec<_>>();
        ids.resize(max_len, 0);
        mask.resize(max_len, 0);
        ids_data.extend_from_slice(&ids);
        mask_data.extend_from_slice(&mask);
    }

    let ids_array = Array::from_shape_vec((text_inputs.len(), max_len), ids_data)?;
    let mask_array = Array::from_shape_vec((text_inputs.len(), max_len), mask_data)?;

    let image_input_dyn = image_input.into_dyn();
    let ids_array_dyn = ids_array.into_dyn();
    let mask_array_dyn = mask_array.into_dyn();

    let image_layout = image_input_dyn.as_standard_layout();
    let ids_layout = ids_array_dyn.as_standard_layout();
    let mask_layout = mask_array_dyn.as_standard_layout();

    let image_val = Tensor::from_array(image_layout.into_owned())?;
    let ids_val = Tensor::from_array(ids_layout.into_owned())?;
    let mask_val = Tensor::from_array(mask_layout.into_owned())?;

    let mut clip_session = clip_session_mutex.lock().unwrap();
    let outputs = clip_session.run(ort::inputs![ids_val, image_val, mask_val])?;

    let logits_dyn = outputs[0].try_extract_array::<f32>()?.to_owned();
    let logits = logits_dyn.into_dimensionality::<ndarray::Dim<[usize; 2]>>()?;
    let probs = softmax(&logits);

    let confidence_threshold = 0.005;
    let mut scored_tags: Vec<(String, f32)> = Vec::new();

    let prob_row = probs.row(0);
    for (i, &prob) in prob_row.iter().enumerate() {
        if prob > confidence_threshold {
            scored_tags.push((text_inputs[i].clone(), prob));
        }
    }

    scored_tags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let selected_tags: Vec<(String, f32)> = scored_tags.into_iter().take(max_tags).collect();
    let initial_tags: Vec<String> = selected_tags.iter().map(|(tag, _)| tag.clone()).collect();
    let mut final_tags: HashMap<String, f32> = selected_tags.into_iter().collect();

    if !is_custom {
        let color_tags = extract_color_tags(image);
        for color_tag in color_tags {
            final_tags.entry(color_tag).or_insert(0.0);
        }

        for tag in &initial_tags {
            if let Some(parents) = TAG_HIERARCHY.get(tag.as_str()) {
                for &parent in parents {
                    final_tags.entry(parent.to_string()).or_insert(0.0);
                }
            }
        }
    }

    Ok(final_tags
        .into_iter()
        .map(|(name, confidence)| ScoredTag { name, confidence })
        .collect())
}

#[tauri::command]
pub async fn start_background_indexing(
    folder_path: String,
    app_handle: AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    if let Some(handle) = state.indexing_task_handle.lock().unwrap().take() {
        println!("Cancelling previous indexing task.");
        handle.abort();
    }

    let settings = crate::load_settings(app_handle.clone())?;
    if !settings.enable_ai_tagging.unwrap_or(false) {
        return Ok(());
    }

    let max_concurrent_tasks = settings.tagging_thread_count.unwrap_or(3).max(1) as usize;
    let custom_ai_tags = settings.custom_ai_tags.clone();
    let ai_tag_count = settings.ai_tag_count.unwrap_or(10) as usize;

    let clip_models = crate::ai_processing::get_or_init_clip_models(
        &app_handle,
        &state.ai_state,
        &state.ai_init_lock,
    )
    .await
    .map_err(|e| e.to_string())?;

    let app_handle_clone = app_handle.clone();

    let task: JoinHandle<()> = tokio::spawn(async move {
        let _ = app_handle_clone.emit("indexing-started", ());
        println!("Starting background indexing for: {}", folder_path);
        println!(
            "Using {} concurrent threads for AI tagging.",
            max_concurrent_tasks
        );

        let state_clone = app_handle_clone.state::<AppState>();
        let gpu_context =
            crate::gpu_processing::get_or_init_gpu_context(&state_clone, &app_handle).ok();

        let image_paths: Vec<PathBuf> = match fs::read_dir(&folder_path) {
            Ok(entries) => entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| {
                    path.is_file() && is_supported_image_file(path.to_string_lossy().as_ref())
                })
                .collect(),
            Err(e) => {
                eprintln!("Failed to read directory '{}': {}", folder_path, e);
                let _ = app_handle_clone
                    .emit("indexing-error", format!("Failed to read directory: {}", e));
                *app_handle_clone
                    .state::<AppState>()
                    .indexing_task_handle
                    .lock()
                    .unwrap() = None;
                return;
            }
        };

        println!(
            "Found {} images to process in {}",
            image_paths.len(),
            folder_path
        );
        let total_images = image_paths.len();
        let processed_count = Arc::new(Mutex::new(0));
        let custom_ai_tags_shared = Arc::new(custom_ai_tags);

        stream::iter(image_paths)
            .for_each_concurrent(max_concurrent_tasks, |path| {
                let app_handle_inner = app_handle_clone.clone();
                let clip_models_inner = clip_models.clone();
                let gpu_context_inner = gpu_context.clone();
                let processed_count_inner = Arc::clone(&processed_count);
                let tags_inner = Arc::clone(&custom_ai_tags_shared);

                async move {
                    let path_str = path.to_string_lossy().to_string();
                    let (_, sidecar_path) = parse_virtual_path(&path_str);

                    let mut metadata = crate::exif_processing::load_sidecar(&sidecar_path);

                    let should_generate_tags = match &metadata.tags {
                        None => true,
                        Some(tags) => !tags.iter().any(|tag| {
                            !tag.starts_with(COLOR_TAG_PREFIX) && !tag.starts_with(USER_TAG_PREFIX)
                        }),
                    };

                    if should_generate_tags {
                        match file_management::get_cached_or_generate_thumbnail_image(
                            &path_str,
                            &app_handle_inner,
                            gpu_context_inner.as_ref(),
                        ) {
                            Ok(image) => {
                                if let Ok(ai_tags) = generate_tags_with_clip(
                                    &image,
                                    &clip_models_inner.model,
                                    &clip_models_inner.tokenizer,
                                    (*tags_inner).clone(),
                                    ai_tag_count,
                                ) {
                                    println!("Found AI tags for {}: {:?}", path_str, ai_tags);

                                    let mut existing_tags: HashSet<String> =
                                        metadata.tags.unwrap_or_default().into_iter().collect();

                                    for tag in ai_tags {
                                        existing_tags.insert(tag.name);
                                    }

                                    let mut final_tags: Vec<String> =
                                        existing_tags.into_iter().collect();
                                    final_tags.sort_unstable();

                                    metadata.tags = Some(final_tags);

                                    if let Ok(json_string) = serde_json::to_string_pretty(&metadata)
                                    {
                                        let _ = fs::write(sidecar_path, json_string);
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!(
                                    "Could not get or generate image for tagging {}: {}",
                                    path_str, e
                                );
                            }
                        }
                    }

                    let mut count = processed_count_inner.lock().unwrap();
                    *count += 1;
                    let _ = app_handle_inner.emit(
                        "indexing-progress",
                        serde_json::json!({
                            "current": *count,
                            "total": total_images
                        }),
                    );
                }
            })
            .await;

        println!("Background indexing finished for: {}", folder_path);
        let _ = app_handle_clone.emit("indexing-finished", ());

        *app_handle_clone
            .state::<AppState>()
            .indexing_task_handle
            .lock()
            .unwrap() = None;
    });

    *state.indexing_task_handle.lock().unwrap() = Some(task);

    Ok(())
}

fn modify_tags_for_path(
    path_str: &str,
    app_handle: &AppHandle,
    modify_fn: impl Fn(&mut Vec<String>),
) -> Result<(), String> {
    let (source_path, sidecar_path) = parse_virtual_path(path_str);

    let mut metadata = crate::exif_processing::load_sidecar(&sidecar_path);

    let mut tags = metadata.tags.unwrap_or_default();
    modify_fn(&mut tags);

    tags.sort_unstable();
    tags.dedup();

    if tags.is_empty() {
        metadata.tags = None;
    } else {
        metadata.tags = Some(tags);
    }

    let json_string = serde_json::to_string_pretty(&metadata).map_err(|e| e.to_string())?;
    fs::write(&sidecar_path, json_string).map_err(|e| e.to_string())?;

    if let Ok(settings) = crate::load_settings(app_handle.clone())
        && settings.enable_xmp_sync.unwrap_or(false)
    {
        let create_if_missing = settings.create_xmp_if_missing.unwrap_or(false);
        file_management::sync_metadata_to_xmp(&source_path, &metadata, create_if_missing);
    }

    Ok(())
}

#[tauri::command]
pub fn add_tag_for_paths(
    paths: Vec<String>,
    tag: String,
    app_handle: AppHandle,
) -> Result<(), String> {
    paths.par_iter().for_each(|path| {
        let tag_clone = tag.clone();
        if let Err(e) = modify_tags_for_path(path, &app_handle, |tags| {
            if !tags.contains(&tag_clone) {
                tags.push(tag_clone.clone());
            }
        }) {
            eprintln!("Failed to add tag to {}: {}", path, e);
        }
    });
    Ok(())
}

#[tauri::command]
pub fn remove_tag_for_paths(
    paths: Vec<String>,
    tag: String,
    app_handle: AppHandle,
) -> Result<(), String> {
    paths.par_iter().for_each(|path| {
        let tag_clone = tag.clone();
        if let Err(e) = modify_tags_for_path(path, &app_handle, |tags| {
            tags.retain(|t| t != &tag_clone);
        }) {
            eprintln!("Failed to remove tag from {}: {}", path, e);
        }
    });
    Ok(())
}

fn rrdata_source_path(rrdata: &Path) -> Option<PathBuf> {
    let name = rrdata.file_name()?.to_str()?;
    let base = name.strip_suffix(".rrdata")?;

    let source_filename = if base.len() >= 7 && base.as_bytes()[base.len() - 7] == b'.' {
        let id = &base[base.len() - 6..];
        if id.chars().all(|c| c.is_ascii_hexdigit()) {
            &base[..base.len() - 7]
        } else {
            base
        }
    } else {
        base
    };

    Some(rrdata.with_file_name(source_filename))
}

fn sync_xmp_for_rrdata(
    rrdata_path: &Path,
    metadata: &ImageMetadata,
    enable_xmp_sync: bool,
    create_xmp_if_missing: bool,
) {
    if !enable_xmp_sync {
        return;
    }
    if let Some(source_path) = rrdata_source_path(rrdata_path) {
        file_management::sync_metadata_to_xmp(&source_path, metadata, create_xmp_if_missing);
    }
}

#[tauri::command]
pub fn clear_ai_tags(root_path: String, app_handle: AppHandle) -> Result<usize, String> {
    if !Path::new(&root_path).exists() {
        return Err(format!("Root path does not exist: {}", root_path));
    }

    let settings = crate::load_settings(app_handle.clone()).unwrap_or_default();
    let enable_xmp_sync = settings.enable_xmp_sync.unwrap_or(false);
    let create_xmp_if_missing = settings.create_xmp_if_missing.unwrap_or(false);

    let mut updated_count = 0;
    let walker = WalkDir::new(root_path).into_iter();

    for entry in walker.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file()
            && path.extension().and_then(|s| s.to_str()) == Some("rrdata")
            && let Ok(content) = fs::read_to_string(path)
            && let Ok(mut metadata) = serde_json::from_str::<ImageMetadata>(&content)
            && let Some(tags) = &mut metadata.tags
        {
            let original_len = tags.len();
            // Keep color tags and user tags, remove others (AI tags)
            tags.retain(|tag| {
                tag.starts_with(COLOR_TAG_PREFIX) || tag.starts_with(USER_TAG_PREFIX)
            });

            if tags.len() < original_len {
                if tags.is_empty() {
                    metadata.tags = None;
                }
                if let Ok(json_string) = serde_json::to_string_pretty(&metadata)
                    && fs::write(path, json_string).is_ok()
                {
                    updated_count += 1;
                    sync_xmp_for_rrdata(path, &metadata, enable_xmp_sync, create_xmp_if_missing);
                }
            }
        }
    }
    Ok(updated_count)
}

#[tauri::command]
pub fn clear_all_tags(root_path: String, app_handle: AppHandle) -> Result<usize, String> {
    if !Path::new(&root_path).exists() {
        return Err(format!("Root path does not exist: {}", root_path));
    }

    let settings = crate::load_settings(app_handle.clone()).unwrap_or_default();
    let enable_xmp_sync = settings.enable_xmp_sync.unwrap_or(false);
    let create_xmp_if_missing = settings.create_xmp_if_missing.unwrap_or(false);

    let mut updated_count = 0;
    let walker = WalkDir::new(root_path).into_iter();

    for entry in walker.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file()
            && path.extension().and_then(|s| s.to_str()) == Some("rrdata")
            && let Ok(content) = fs::read_to_string(path)
            && let Ok(mut metadata) = serde_json::from_str::<ImageMetadata>(&content)
            && let Some(tags) = &mut metadata.tags
        {
            let original_len = tags.len();
            // Keep only color tags, remove AI and user tags
            tags.retain(|tag| tag.starts_with(COLOR_TAG_PREFIX));

            if tags.len() < original_len {
                if tags.is_empty() {
                    metadata.tags = None;
                }
                if let Ok(json_string) = serde_json::to_string_pretty(&metadata)
                    && fs::write(path, json_string).is_ok()
                {
                    updated_count += 1;
                    sync_xmp_for_rrdata(path, &metadata, enable_xmp_sync, create_xmp_if_missing);
                }
            }
        }
    }
    Ok(updated_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ram_plus_preprocessing_uses_expected_nchw_shape() {
        let image = DynamicImage::new_rgb8(16, 8);
        assert_eq!(ram_plus_input(&image).dim(), (1, 3, 384, 384));
    }

    #[test]
    fn bioclip_preprocessing_uses_expected_nchw_shape() {
        let image = DynamicImage::new_rgb8(16, 8);
        assert_eq!(bioclip_input(&image).dim(), (1, 3, 224, 224));
    }

    #[test]
    fn bioclip_taxonomy_requires_structured_records() {
        let valid: Vec<BioClipTaxon> = serde_json::from_str(
            r#"[{"scientificName":"Corvus corax","commonName":"Common raven","taxonRank":"species"}]"#,
        )
        .unwrap();
        assert_eq!(valid[0].scientific_name, "Corvus corax");
        assert!(serde_json::from_str::<Vec<BioClipTaxon>>(r#"["Common raven"]"#).is_err());
    }

    #[test]
    fn bioclip_rejects_species_when_nearest_taxa_are_ambiguous() {
        let taxon = |scientific_name: &str| BioClipTaxon {
            scientific_name: scientific_name.to_string(),
            common_name: None,
            taxon_rank: "species".to_string(),
        };
        let ambiguous = BioClipInference {
            candidates: vec![
                BioClipCandidate {
                    taxon: taxon("Canis latrans"),
                    similarity: 0.57,
                },
                BioClipCandidate {
                    taxon: taxon("Vulpes vulpes"),
                    similarity: 0.55,
                },
            ],
        };
        let separated = BioClipInference {
            candidates: vec![
                BioClipCandidate {
                    taxon: taxon("Haliaeetus leucocephalus"),
                    similarity: 0.62,
                },
                BioClipCandidate {
                    taxon: taxon("Buteo jamaicensis"),
                    similarity: 0.49,
                },
            ],
        };

        assert!(!ambiguous.is_confident_species());
        assert!(separated.is_confident_species());
    }
}
