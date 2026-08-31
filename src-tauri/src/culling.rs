use crate::ai_processing::{AiModels, get_or_init_ai_models, run_u2netp_model};
use crate::app_settings::load_settings;
use crate::app_state::AppState;
use image::{GenericImageView, GrayImage, imageops};
use image_hasher::{HashAlg, HasherConfig};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use tauri::{AppHandle, Emitter};

use crate::image_loader;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CullingSettings {
    pub similarity_threshold: u32,
    pub blur_threshold: f64,
    pub group_similar: bool,
    pub filter_blurry: bool,
    pub use_subject_detection: bool,
    #[serde(default = "default_subject_mode")]
    pub subject_mode: String,
}

fn default_subject_mode() -> String {
    "general".to_string()
}

impl Default for CullingSettings {
    fn default() -> Self {
        Self {
            similarity_threshold: 28,
            blur_threshold: 100.0,
            group_similar: true,
            filter_blurry: true,
            use_subject_detection: false,
            subject_mode: default_subject_mode(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub struct BlurRegion {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ImageAnalysisResult {
    pub path: String,
    pub quality_score: f64,
    pub sharpness_metric: f64,
    pub center_focus_metric: f64,
    pub subject_focus_metric: Option<f64>,
    pub subject_composition_metric: Option<f64>,
    pub subject_edge_contact_ratio: Option<f64>,
    pub exposure_metric: f64,
    pub width: u32,
    pub height: u32,
    /// Normalized (0-1) bounding box of the softest-focus region with enough
    /// texture to judge sharpness on, for pointing a "here's the blurry
    /// part" overlay at something real - see `find_blurriest_region`.
    pub blurry_region: Option<BlurRegion>,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CullGroup {
    pub representative: ImageAnalysisResult,
    pub duplicates: Vec<ImageAnalysisResult>,
}

#[derive(Serialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct CullingSuggestions {
    pub similar_groups: Vec<CullGroup>,
    pub blurry_images: Vec<ImageAnalysisResult>,
    pub failed_paths: Vec<String>,
}

/// Result from the window-free technical culling pass. Subject detection is
/// intentionally excluded here: callers must use a model-aware batch worker
/// rather than silently presenting U-2-Net-dependent scores as available.
#[derive(Serialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct HeadlessCullingReport {
    pub analyses: Vec<ImageAnalysisResult>,
    pub suggestions: CullingSuggestions,
}

#[derive(Serialize, Clone)]
struct CullingProgress {
    current: usize,
    total: usize,
    stage: String,
}

#[derive(Clone)]
pub(crate) struct ImageAnalysisData {
    pub(crate) hash: image_hasher::ImageHash,
    pub(crate) result: ImageAnalysisResult,
}

const WEIGHT_SHARPNESS: f64 = 0.38;
const WEIGHT_CENTER_FOCUS: f64 = 0.32;
const WEIGHT_EXPOSURE: f64 = 0.22;
// This intentionally remains a minor cue. Rule-of-thirds placement is not a
// universal aesthetic rule, so it may only break ties between technically
// comparable frames with a confident foreground mask.
const WEIGHT_SUBJECT_COMPOSITION: f64 = 0.08;

// U-2-Netp outputs a min-max normalized 0-255 saliency mask; treat anything
// above this as "subject" when computing subject-aware sharpness.
const SUBJECT_MASK_THRESHOLD: u8 = 128;
// Below this many qualifying pixels, the mask is too small/ambiguous to trust
// (e.g. no clear foreground, or the subject is tiny) - fall back to the
// center-crop metric instead of scoring noise.
const MIN_SUBJECT_PIXELS: usize = 64;

fn calculate_laplacian_variance(image: &GrayImage) -> f64 {
    let (width, height) = image.dimensions();
    if width < 3 || height < 3 {
        return 0.0;
    }

    let mut laplacian_values = Vec::with_capacity(((width - 2) * (height - 2)) as usize);
    let mut sum = 0.0;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let p_center = image.get_pixel(x, y)[0] as i32;
            let p_north = image.get_pixel(x, y - 1)[0] as i32;
            let p_south = image.get_pixel(x, y + 1)[0] as i32;
            let p_west = image.get_pixel(x - 1, y)[0] as i32;
            let p_east = image.get_pixel(x + 1, y)[0] as i32;
            let conv_val = (p_north + p_south + p_west + p_east - 4 * p_center) as f64;
            laplacian_values.push(conv_val);
            sum += conv_val;
        }
    }

    if laplacian_values.is_empty() {
        return 0.0;
    }
    let mean = sum / laplacian_values.len() as f64;

    laplacian_values
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / laplacian_values.len() as f64
}

/// Finds the region most likely responsible for a "blurry" verdict: divides
/// the image into a coarse grid, scores each cell's local sharpness
/// (Laplacian variance) and texture (intensity std-dev), and returns the
/// softest cell among those with enough texture to judge focus on at all.
/// A flat sky or wall always has near-zero Laplacian variance without being
/// an out-of-focus subject, so uniformly flat cells are excluded rather than
/// reported as "the blurry part." Returns `None` for images too small to
/// grid meaningfully, or where every cell is too flat to judge.
fn find_blurriest_region(image: &GrayImage) -> Option<BlurRegion> {
    const GRID: u32 = 4;
    const MIN_TEXTURE_STDDEV: f64 = 4.0;

    let (width, height) = image.dimensions();
    if width < GRID * 4 || height < GRID * 4 {
        return None;
    }

    let cell_w = width / GRID;
    let cell_h = height / GRID;
    let mut softest: Option<(f64, u32, u32)> = None;

    for row in 0..GRID {
        for col in 0..GRID {
            let x = col * cell_w;
            let y = row * cell_h;
            let w = if col == GRID - 1 { width - x } else { cell_w };
            let h = if row == GRID - 1 { height - y } else { cell_h };
            let cell = imageops::crop_imm(image, x, y, w, h).to_image();

            let pixels: Vec<f64> = cell.pixels().map(|p| p[0] as f64).collect();
            if pixels.is_empty() {
                continue;
            }
            let mean = pixels.iter().sum::<f64>() / pixels.len() as f64;
            let variance =
                pixels.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / pixels.len() as f64;
            if variance.sqrt() < MIN_TEXTURE_STDDEV {
                continue;
            }

            let sharpness = calculate_laplacian_variance(&cell);
            let is_softer = softest.map(|(best, _, _)| sharpness < best).unwrap_or(true);
            if is_softer {
                softest = Some((sharpness, col, row));
            }
        }
    }

    softest.map(|(_, col, row)| BlurRegion {
        x: (col * cell_w) as f64 / width as f64,
        y: (row * cell_h) as f64 / height as f64,
        width: cell_w as f64 / width as f64,
        height: cell_h as f64 / height as f64,
    })
}

/// Laplacian variance restricted to pixels the subject mask marks as
/// foreground, rather than a fixed center crop. `mask` must have the same
/// dimensions as `image` (caller runs the mask model on the same thumbnail
/// used to build `image`). Returns `None` when too few pixels qualify, so
/// callers can fall back to the center-crop metric instead of scoring an
/// empty/ambiguous mask.
fn calculate_masked_laplacian_variance(image: &GrayImage, mask: &GrayImage) -> Option<f64> {
    let (width, height) = image.dimensions();
    if width < 3 || height < 3 || mask.dimensions() != (width, height) {
        return None;
    }

    let mut laplacian_values = Vec::new();
    let mut sum = 0.0;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            if mask.get_pixel(x, y)[0] < SUBJECT_MASK_THRESHOLD {
                continue;
            }

            let p_center = image.get_pixel(x, y)[0] as i32;
            let p_north = image.get_pixel(x, y - 1)[0] as i32;
            let p_south = image.get_pixel(x, y + 1)[0] as i32;
            let p_west = image.get_pixel(x - 1, y)[0] as i32;
            let p_east = image.get_pixel(x + 1, y)[0] as i32;
            let conv_val = (p_north + p_south + p_west + p_east - 4 * p_center) as f64;
            laplacian_values.push(conv_val);
            sum += conv_val;
        }
    }

    if laplacian_values.len() < MIN_SUBJECT_PIXELS {
        return None;
    }
    let mean = sum / laplacian_values.len() as f64;

    Some(
        laplacian_values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / laplacian_values.len() as f64,
    )
}

/// Subject-aware exposure: measures clipping only within the detected
/// foreground, so a legitimately dark background (very common in wildlife/bird
/// photography - shadowed foliage, night perches) doesn't tank the score for a
/// well-exposed subject. Falls back to None (whole-frame metric) when the mask
/// is too small/ambiguous to trust, mirroring calculate_masked_laplacian_variance.
fn calculate_masked_exposure_metric(image: &GrayImage, mask: &GrayImage) -> Option<f64> {
    let (width, height) = image.dimensions();
    if mask.dimensions() != (width, height) {
        return None;
    }

    let clip_threshold_dark = 5u8;
    let clip_threshold_bright = 250u8;
    let mut subject_pixels = 0usize;
    let mut dark_pixels = 0usize;
    let mut bright_pixels = 0usize;
    for (pixel, mask_pixel) in image.pixels().zip(mask.pixels()) {
        if mask_pixel[0] < SUBJECT_MASK_THRESHOLD {
            continue;
        }
        subject_pixels += 1;
        let value = pixel[0];
        if value < clip_threshold_dark {
            dark_pixels += 1;
        } else if value >= clip_threshold_bright {
            bright_pixels += 1;
        }
    }

    if subject_pixels < MIN_SUBJECT_PIXELS {
        return None;
    }

    let dark_clip_ratio = dark_pixels as f64 / subject_pixels as f64;
    let bright_clip_ratio = bright_pixels as f64 / subject_pixels as f64;
    let penalty = (dark_clip_ratio * 5.0) + (bright_clip_ratio * 5.0);
    Some((1.0f64 - penalty).max(0.0))
}

/// Returns a conservative composition cue derived from the existing saliency
/// mask. It rewards a compact subject near a thirds intersection and reports
/// how much of the detected subject touches the image edge. It is not an
/// aesthetic classifier and callers should retain the raw measurements.
fn calculate_subject_composition_metrics(mask: &GrayImage) -> Option<(f64, f64)> {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return None;
    }
    let edge_margin_x = ((width as f32 * 0.04).round() as u32).max(1);
    let edge_margin_y = ((height as f32 * 0.04).round() as u32).max(1);
    let mut pixels = 0usize;
    let mut edge_pixels = 0usize;
    let mut sum_x = 0f64;
    let mut sum_y = 0f64;
    for y in 0..height {
        for x in 0..width {
            if mask.get_pixel(x, y)[0] < SUBJECT_MASK_THRESHOLD {
                continue;
            }
            pixels += 1;
            sum_x += x as f64;
            sum_y += y as f64;
            if x < edge_margin_x
                || x >= width.saturating_sub(edge_margin_x)
                || y < edge_margin_y
                || y >= height.saturating_sub(edge_margin_y)
            {
                edge_pixels += 1;
            }
        }
    }
    if pixels < MIN_SUBJECT_PIXELS {
        return None;
    }
    let center_x = sum_x / pixels as f64 / width as f64;
    let center_y = sum_y / pixels as f64 / height as f64;
    let nearest_thirds_distance = [1.0 / 3.0, 2.0 / 3.0]
        .into_iter()
        .flat_map(|third_x| {
            [1.0 / 3.0, 2.0 / 3.0].into_iter().map(move |third_y| {
                ((center_x - third_x).powi(2) + (center_y - third_y).powi(2)).sqrt()
            })
        })
        .fold(f64::INFINITY, f64::min);
    let thirds_score = (1.0 - nearest_thirds_distance / 0.5).clamp(0.0, 1.0);
    let edge_contact_ratio = edge_pixels as f64 / pixels as f64;
    let score = (thirds_score * 0.7 + (1.0 - edge_contact_ratio) * 0.3).clamp(0.0, 1.0);
    Some((score, edge_contact_ratio))
}

fn calculate_exposure_metric(image: &GrayImage) -> f64 {
    let histogram = imageproc::stats::histogram(image);
    let total_pixels = (image.width() * image.height()) as f64;
    if total_pixels == 0.0 {
        return 0.0;
    }

    let clip_threshold_dark = 5;
    let clip_threshold_bright = 250;

    let dark_pixels = histogram.channels[0][0..clip_threshold_dark]
        .iter()
        .sum::<u32>() as f64;
    let bright_pixels = histogram.channels[0][clip_threshold_bright..256]
        .iter()
        .sum::<u32>() as f64;

    let dark_clip_ratio = dark_pixels / total_pixels;
    let bright_clip_ratio = bright_pixels / total_pixels;

    let penalty = (dark_clip_ratio * 5.0) + (bright_clip_ratio * 5.0);

    (1.0f64 - penalty).max(0.0)
}

pub(crate) fn analyze_image(
    path: &str,
    hasher: &image_hasher::Hasher,
    settings: &crate::app_settings::AppSettings,
    ai_models: Option<&Arc<AiModels>>,
    // Round-robins into ai_models.u2netp_pool so concurrent calls (this runs
    // inside a rayon par_iter) don't all contend for a single shared model
    // session. Caller just needs to pass a distinct-ish value per call, e.g.
    // its item index - exact fairness doesn't matter, just avoiding one lock.
    pool_slot: usize,
) -> Result<ImageAnalysisData, String> {
    const ANALYSIS_DIM: u32 = 720; // FIXME: How should we calculate good focus if it's downscaled?!?

    if crate::file_management::is_cloud_placeholder(Path::new(path)) {
        return Err(format!("'{}' is stored in iCloud and not downloaded", path));
    }

    let file_bytes = std::fs::read(path).map_err(|e| e.to_string())?;

    let img = image_loader::load_base_image_from_bytes(&file_bytes, path, true, settings, None)
        .map_err(|e| e.to_string())?;

    let (width, height) = img.dimensions();
    let thumbnail = img.thumbnail(ANALYSIS_DIM, ANALYSIS_DIM);
    let gray_thumbnail = thumbnail.to_luma8();

    let sharpness_metric = calculate_laplacian_variance(&gray_thumbnail);
    let whole_frame_exposure_metric = calculate_exposure_metric(&gray_thumbnail);
    let blurry_region = find_blurriest_region(&gray_thumbnail);

    let (thumb_w, thumb_h) = gray_thumbnail.dimensions();
    let center_crop = imageops::crop_imm(
        &gray_thumbnail,
        thumb_w / 4,
        thumb_h / 4,
        thumb_w / 2,
        thumb_h / 2,
    )
    .to_image();
    let center_focus_metric = calculate_laplacian_variance(&center_crop);

    // Subject-aware focus: run the existing foreground-mask model on this
    // same thumbnail and measure sharpness only where the subject actually
    // is, instead of assuming it's centered. Falls back to the center-crop
    // metric when detection is off, fails, or the mask is too small/empty.
    let subject_metrics = ai_models.and_then(|models| {
        let session = models
            .u2netp_pool
            .get(pool_slot % models.u2netp_pool.len().max(1))?;
        let mask = run_u2netp_model(&thumbnail, session).ok()?;
        let focus = calculate_masked_laplacian_variance(&gray_thumbnail, &mask)?;
        let (composition, edge_contact) = calculate_subject_composition_metrics(&mask)?;
        let exposure = calculate_masked_exposure_metric(&gray_thumbnail, &mask);
        Some((focus, composition, edge_contact, exposure))
    });
    let subject_focus_metric = subject_metrics.map(|metrics| metrics.0);
    let subject_composition_metric = subject_metrics.map(|metrics| metrics.1);
    let subject_edge_contact_ratio = subject_metrics.map(|metrics| metrics.2);
    // Prefer exposure measured within the detected subject over the
    // whole-frame reading, so a legitimately dark/bright background doesn't
    // tank the score (or the "why this decision" text) for a well-exposed
    // subject. Falls back to the whole-frame metric when there's no usable mask.
    let exposure_metric = subject_metrics
        .and_then(|metrics| metrics.3)
        .unwrap_or(whole_frame_exposure_metric);

    let normalized_sharpness = ((sharpness_metric + 1.0).log10() / 3.5).min(1.0);
    let focus_metric_for_score = subject_focus_metric.unwrap_or(center_focus_metric);
    let normalized_focus = ((focus_metric_for_score + 1.0).log10() / 3.5).min(1.0);

    let baseline_quality_score = (normalized_sharpness * WEIGHT_SHARPNESS)
        + (normalized_focus * WEIGHT_CENTER_FOCUS)
        + (exposure_metric * WEIGHT_EXPOSURE);
    let quality_score = match subject_composition_metric {
        Some(composition) => baseline_quality_score + composition * WEIGHT_SUBJECT_COMPOSITION,
        // Do not penalize a legitimate image merely because saliency could
        // not identify a compact foreground subject (common for landscapes).
        None => baseline_quality_score / (1.0 - WEIGHT_SUBJECT_COMPOSITION),
    };

    let hash = hasher.hash_image(&thumbnail);

    Ok(ImageAnalysisData {
        hash,
        result: ImageAnalysisResult {
            path: path.to_string(),
            quality_score,
            sharpness_metric,
            center_focus_metric,
            subject_focus_metric,
            subject_composition_metric,
            subject_edge_contact_ratio,
            exposure_metric,
            width,
            height,
            blurry_region,
        },
    })
}

/// Turns raw per-image analysis into grouped duplicate suggestions + a
/// blurry-images list. Shared by the manual `cull_images` command and the
/// unattended auto-cull planner so both use the exact same grouping logic.
pub(crate) fn build_culling_suggestions(
    successful_analyses: Vec<ImageAnalysisData>,
    failed_paths: Vec<String>,
    settings: &CullingSettings,
) -> CullingSuggestions {
    let mut suggestions = CullingSuggestions {
        failed_paths,
        ..Default::default()
    };
    let mut processed_indices = vec![false; successful_analyses.len()];

    if settings.group_similar {
        for i in 0..successful_analyses.len() {
            if processed_indices[i] {
                continue;
            }

            let mut current_group_indices = vec![];
            let mut queue = VecDeque::new();

            processed_indices[i] = true;
            current_group_indices.push(i);
            queue.push_back(i);

            while let Some(current_idx) = queue.pop_front() {
                for j in (current_idx + 1)..successful_analyses.len() {
                    if processed_indices[j] {
                        continue;
                    }

                    let dist = successful_analyses[current_idx]
                        .hash
                        .dist(&successful_analyses[j].hash);
                    if dist <= settings.similarity_threshold {
                        processed_indices[j] = true;
                        current_group_indices.push(j);
                        queue.push_back(j);
                    }
                }
            }

            if current_group_indices.len() > 1 {
                current_group_indices.sort_by(|&a, &b| {
                    successful_analyses[b]
                        .result
                        .quality_score
                        .partial_cmp(&successful_analyses[a].result.quality_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let representative_idx = current_group_indices[0];
                let duplicate_indices = &current_group_indices[1..];

                suggestions.similar_groups.push(CullGroup {
                    representative: successful_analyses[representative_idx].result.clone(),
                    duplicates: duplicate_indices
                        .iter()
                        .map(|&idx| successful_analyses[idx].result.clone())
                        .collect(),
                });
            }
        }
    }

    if settings.filter_blurry {
        for i in 0..successful_analyses.len() {
            if !processed_indices[i] {
                let item = &successful_analyses[i];
                if item.result.sharpness_metric < settings.blur_threshold {
                    suggestions.blurry_images.push(item.result.clone());
                }
            }
        }
        suggestions.blurry_images.sort_by(|a, b| {
            a.sharpness_metric
                .partial_cmp(&b.sharpness_metric)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    suggestions
}

/// Runs the deterministic duplicate, sharpness, focus, and exposure pass
/// without a Tauri application handle. This is used by `rapidraw-cli` and
/// deliberately rejects subject analysis because loading U-2-Net remains a
/// model-managed operation with its own explicit runtime contract.
pub fn cull_images_headless<F>(
    paths: Vec<String>,
    settings: CullingSettings,
    app_settings: crate::app_settings::AppSettings,
    on_progress: F,
) -> Result<HeadlessCullingReport, String>
where
    F: FnMut(usize, usize, Option<&str>) + Send,
{
    if settings.use_subject_detection {
        return Err(
            "Headless culling currently supports technical analysis only; run subject-aware culling from the desktop workflow"
                .to_string(),
        );
    }
    if paths.is_empty() {
        return Ok(HeadlessCullingReport::default());
    }

    let total = paths.len();
    let hasher = HasherConfig::new()
        .hash_alg(HashAlg::DoubleGradient)
        .hash_size(16, 16)
        .to_hasher();
    let completed = AtomicUsize::new(0);
    let progress = Mutex::new(on_progress);
    let results = paths
        .par_iter()
        .map(|path| {
            let current = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if let Ok(mut callback) = progress.lock() {
                callback(current, total, Some(path));
            }
            analyze_image(path, &hasher, &app_settings, None, current)
                .map_err(|error| (path.clone(), error))
        })
        .collect::<Vec<_>>();

    let mut analyses = Vec::new();
    let mut failed_paths = Vec::new();
    let mut successful = Vec::new();
    for result in results {
        match result {
            Ok(analysis) => {
                analyses.push(analysis.result.clone());
                successful.push(analysis);
            }
            Err((path, error)) => {
                eprintln!("Failed to analyze image {path}: {error}");
                failed_paths.push(path);
            }
        }
    }
    analyses.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(HeadlessCullingReport {
        suggestions: build_culling_suggestions(successful, failed_paths, &settings),
        analyses,
    })
}

#[tauri::command]
pub async fn cull_images(
    paths: Vec<String>,
    settings: CullingSettings,
    app_handle: AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<CullingSuggestions, String> {
    if paths.is_empty() {
        return Ok(CullingSuggestions::default());
    }

    let app_settings = load_settings(app_handle.clone()).unwrap_or_default();

    let total_count = paths.len();
    let completed_count = Arc::new(AtomicUsize::new(0));
    let _ = app_handle.emit("culling-start", total_count);

    let ai_models = if settings.use_subject_detection && settings.subject_mode != "landscape" {
        let _ = app_handle.emit(
            "culling-progress",
            CullingProgress {
                current: 0,
                total: total_count,
                stage: "Loading subject detection model...".to_string(),
            },
        );
        Some(
            get_or_init_ai_models(&app_handle, &state.ai_state, &state.ai_init_lock)
                .await
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };

    let hasher = HasherConfig::new()
        .hash_alg(HashAlg::DoubleGradient)
        .hash_size(16, 16)
        .to_hasher();

    let analysis_results: Vec<Result<ImageAnalysisData, (String, String)>> = paths
        .par_iter()
        .map(|path| {
            let completed = completed_count.fetch_add(1, Ordering::Relaxed) + 1;
            let _ = app_handle.emit(
                "culling-progress",
                CullingProgress {
                    current: completed,
                    total: total_count,
                    stage: "Analyzing images...".to_string(),
                },
            );

            analyze_image(path, &hasher, &app_settings, ai_models.as_ref(), completed)
                .map_err(|e| (path.to_string(), e))
        })
        .collect();

    let mut successful_analyses = Vec::new();
    let mut failed_paths = Vec::new();
    for res in analysis_results {
        match res {
            Ok(data) => successful_analyses.push(data),
            Err((path, error)) => {
                eprintln!("Failed to analyze image {}: {}", path, error);
                failed_paths.push(path);
            }
        }
    }

    let _ = app_handle.emit(
        "culling-progress",
        CullingProgress {
            current: total_count,
            total: total_count,
            stage: "Grouping similar images...".to_string(),
        },
    );

    let suggestions = build_culling_suggestions(successful_analyses, failed_paths, &settings);

    let _ = app_handle.emit("culling-complete", &suggestions);
    Ok(suggestions)
}

#[cfg(test)]
mod tests {
    use super::{CullingSettings, calculate_subject_composition_metrics, cull_images_headless};
    use image::{GrayImage, Luma};

    #[test]
    fn subject_mode_defaults_for_older_culling_requests() {
        let settings: CullingSettings = serde_json::from_str(
            r#"{"similarityThreshold":28,"blurThreshold":100.0,"groupSimilar":true,"filterBlurry":true,"useSubjectDetection":false}"#,
        )
        .expect("older settings remain compatible");

        assert_eq!(settings.subject_mode, "general");
    }

    #[test]
    fn subject_geometry_penalizes_edge_contact() {
        let mut centered = GrayImage::new(90, 90);
        let mut edge = GrayImage::new(90, 90);
        for y in 25..45 {
            for x in 25..45 {
                centered.put_pixel(x, y, Luma([255]));
            }
        }
        for y in 25..45 {
            for x in 0..20 {
                edge.put_pixel(x, y, Luma([255]));
            }
        }
        let (centered_score, centered_edge_contact) =
            calculate_subject_composition_metrics(&centered).unwrap();
        let (edge_score, edge_contact) = calculate_subject_composition_metrics(&edge).unwrap();
        assert!(edge_contact > centered_edge_contact);
        assert!(centered_score > edge_score);
    }

    #[test]
    fn headless_culling_analyzes_regular_images_without_a_tauri_handle() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("test.png");
        image::RgbImage::from_fn(24, 24, |x, y| {
            image::Rgb([((x + y) * 4) as u8, (x * 5) as u8, (y * 5) as u8])
        })
        .save(&path)
        .unwrap();

        let report = cull_images_headless(
            vec![path.to_string_lossy().into_owned()],
            CullingSettings::default(),
            crate::app_settings::AppSettings::default(),
            |_, _, _| {},
        )
        .unwrap();

        assert_eq!(report.analyses.len(), 1);
        assert!(report.suggestions.failed_paths.is_empty());
    }
}
