use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ort::execution_providers::{CUDAExecutionProvider, ExecutionProvider};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use tauri::{Emitter, Manager};

/// RawNIND's exported ONNX graph has a fixed 4×512×512 Bayer input.
const RAWNIND_BAYER_TILE_SIZE: u32 = 512;
const RAWNIND_SOURCE_TILE_SIZE: u32 = RAWNIND_BAYER_TILE_SIZE * 2;

/// RawNIND is an interactive-editor task, not a batch worker. Keep exactly
/// one session alive and serialize access to it: `Session::run` needs `&mut
/// self` anyway, and creating several sessions lets ONNX Runtime create a
/// full CPU pool for every pending slider update. The resulting contention
/// starves the WebView even though the develop request itself runs off the UI
/// thread.
///
/// A GPU-capable ONNX Runtime can replace the CPU provider in this one place
/// without changing the RAW pipeline. Until such a runtime is packaged, the
/// CPU fallback is intentionally capped at one inference thread to preserve
/// editor responsiveness.
static RAWNIND_EDITOR_SESSION: OnceLock<Mutex<Option<(PathBuf, ort::session::Session)>>> =
    OnceLock::new();

fn build_rawnind_cpu_session(model_path: &Path) -> Result<ort::session::Session, String> {
    ort::session::Session::builder()
        .and_then(|builder| builder.with_intra_threads(1))
        .and_then(|builder| builder.with_inter_threads(1))
        .and_then(|builder| builder.with_parallel_execution(false))
        .and_then(|builder| builder.commit_from_file(model_path))
        .map_err(|error| format!("Failed to load RawNIND model: {error}"))
}

fn build_rawnind_editor_session(model_path: &Path) -> Result<ort::session::Session, String> {
    // CUDA is deliberately probed at runtime. A CPU-only package, a machine
    // without an NVIDIA driver, or a GPU package whose CUDA/cuDNN dependencies
    // are unavailable must still open the image through the CPU fallback.
    let cuda = CUDAExecutionProvider::default();
    if cuda.is_available().unwrap_or(false) {
        let provider = cuda.build().error_on_failure();
        match ort::session::Session::builder()
            .and_then(|builder| builder.with_execution_providers([provider]))
            .and_then(|builder| builder.with_intra_threads(1))
            .and_then(|builder| builder.with_inter_threads(1))
            .and_then(|builder| builder.with_parallel_execution(false))
            .and_then(|builder| builder.commit_from_file(model_path))
        {
            Ok(session) => {
                log::info!("RawNIND editor session is using CUDAExecutionProvider");
                return Ok(session);
            }
            Err(error) => {
                log::warn!(
                    "RawNIND CUDA provider could not start ({error}); falling back to CPU inference"
                );
            }
        }
    }

    build_rawnind_cpu_session(model_path)
}

fn rawnind_editor_session(
    model_path: &Path,
) -> Result<std::sync::MutexGuard<'static, Option<(PathBuf, ort::session::Session)>>, String> {
    let session_cache = RAWNIND_EDITOR_SESSION.get_or_init(|| Mutex::new(None));
    let mut session = session_cache
        .lock()
        .map_err(|_| "RawNIND session lock was poisoned".to_string())?;

    let needs_reload = match session.as_ref() {
        Some((loaded_path, _)) => loaded_path != model_path,
        None => true,
    };
    if needs_reload {
        // The runtime defaults to a pool sized to all logical CPUs. RawNIND
        // operates on many 512px tiles, so that default can otherwise pin the
        // machine for the whole render. Sequential, one-thread inference is
        // slower in isolation but keeps the application interactive; reuse
        // of the session removes the expensive model-load cost on later edits.
        let loaded = build_rawnind_editor_session(model_path)?;
        log::info!("RawNIND editor session ready; subsequent renders reuse it");
        *session = Some((model_path.to_path_buf(), loaded));
    }

    Ok(session)
}

/// Recipe and configuration parameters for a restoration operation.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RestorationRecipe {
    pub operation_kind: String, // "raw_denoise" or "rgb_denoise"
    pub model_id: String,
    pub model_revision: String,
    pub denoise_strength: f32,
    pub microcontrast_strength: f32,
    pub detail_recovery: f32,
    pub tile_size: u32,
    pub tile_overlap: u32,
}

impl Default for RestorationRecipe {
    fn default() -> Self {
        Self {
            operation_kind: "raw_denoise".to_string(),
            model_id: crate::visual_model_registry::RAWNIND_MODEL_ID.to_string(),
            model_revision: "v1".to_string(),
            denoise_strength: 0.8,
            // These legacy fields remain in stored recipes for compatibility.
            // Finish-stage adjustments are deliberately never applied here.
            microcontrast_strength: 0.0,
            detail_recovery: 0.0,
            // Bayer tiling halves tile_size before feeding RawNIND's static
            // 512×512 graph, so the source tile must be 1024 pixels.
            tile_size: RAWNIND_SOURCE_TILE_SIZE,
            tile_overlap: 64,
        }
    }
}

pub fn validate_restoration_recipe(recipe: &RestorationRecipe) -> Result<(), String> {
    if !matches!(
        recipe.operation_kind.as_str(),
        "raw_denoise" | "rgb_denoise"
    ) {
        return Err(format!(
            "Unsupported restoration operation: {}",
            recipe.operation_kind
        ));
    }
    if !(0.0..=1.0).contains(&recipe.denoise_strength)
        || !(0.0..=1.0).contains(&recipe.microcontrast_strength)
        || !(0.0..=1.0).contains(&recipe.detail_recovery)
    {
        return Err("Restoration strengths must be between 0 and 1".to_string());
    }
    if recipe.tile_size < 64 || recipe.tile_overlap >= recipe.tile_size {
        return Err(
            "Tile size must be at least 64 and overlap must be smaller than the tile size"
                .to_string(),
        );
    }
    match recipe.operation_kind.as_str() {
        "raw_denoise" if recipe.model_id != crate::visual_model_registry::RAWNIND_MODEL_ID => {
            Err("RAW denoise requires the RawNIND Bayer model".to_string())
        }
        "raw_denoise" if recipe.tile_size != RAWNIND_SOURCE_TILE_SIZE => Err(format!(
            "RawNIND requires a {}px source tile for its fixed {}×{} Bayer input",
            RAWNIND_SOURCE_TILE_SIZE, RAWNIND_BAYER_TILE_SIZE, RAWNIND_BAYER_TILE_SIZE
        )),
        "rgb_denoise" if recipe.model_id != crate::visual_model_registry::NAFNET_MODEL_ID => {
            Err("RGB denoise requires the NAFNet SIDD model".to_string())
        }
        "raw_denoise" | "rgb_denoise" => Ok(()),
        _ => unreachable!("operation kinds are checked above"),
    }
}

/// Result metadata for a completed or generated derivative.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RestorationResult {
    pub derivative_id: i64,
    pub source_image_id: i64,
    pub output_path: String,
    pub output_format: String,
    pub width: u32,
    pub height: u32,
}

/// Applies edge-preserving multi-frequency microcontrast enhancement.
/// Decomposes the luminance channel into coarse structure, medium texture,
/// and fine microcontrast, boosting micro-contrast without increasing noise.
pub fn apply_microcontrast(
    img: &DynamicImage,
    microcontrast_amount: f32,
    detail_amount: f32,
) -> DynamicImage {
    if microcontrast_amount <= 0.0 && detail_amount <= 0.0 {
        return img.clone();
    }

    let (width, height) = img.dimensions();
    let rgb_img = img.to_rgb8();

    // Guided/box approximation for high-pass frequency decomposition
    let radius = 2usize;

    // Extract luminance channel
    let raw_pixels: Vec<[u8; 3]> = rgb_img.pixels().map(|p| p.0).collect();
    let lum: Vec<f32> = raw_pixels
        .iter()
        .map(|p| 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32)
        .collect();

    // Compute blurred base luminance (coarse structure) in parallel rows
    let mut blurred_lum = vec![0.0f32; (width * height) as usize];
    blurred_lum
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y_min = y.saturating_sub(radius);
            let y_max = (y + radius).min(height as usize - 1);
            for (x, out) in row.iter_mut().enumerate() {
                let x_min = x.saturating_sub(radius);
                let x_max = (x + radius).min(width as usize - 1);
                let mut sum = 0.0f32;
                let mut count = 0.0f32;
                for ny in y_min..=y_max {
                    for nx in x_min..=x_max {
                        sum += lum[ny * width as usize + nx];
                        count += 1.0;
                    }
                }
                *out = sum / count;
            }
        });

    // Enhance microcontrast: difference between original luminance and blurred base
    let boost = 1.0 + microcontrast_amount * 0.8;
    let detail_boost = detail_amount * 0.5;

    let enhanced_pixels: Vec<u8> = raw_pixels
        .par_iter()
        .enumerate()
        .flat_map(|(idx, p)| {
            let l_orig = lum[idx];
            let l_base = blurred_lum[idx];
            let high_pass = l_orig - l_base;
            let l_new =
                (l_base + high_pass * boost + high_pass.signum() * detail_boost).clamp(0.0, 255.0);

            let scale = if l_orig > 1e-4 { l_new / l_orig } else { 1.0 };
            let r = (p[0] as f32 * scale).clamp(0.0, 255.0) as u8;
            let g = (p[1] as f32 * scale).clamp(0.0, 255.0) as u8;
            let b = (p[2] as f32 * scale).clamp(0.0, 255.0) as u8;
            [r, g, b]
        })
        .collect();

    if let Some(buf) = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, enhanced_pixels) {
        DynamicImage::ImageRgb8(buf)
    } else {
        img.clone()
    }
}

/// Slices an image into overlapping tiles with a Hanning / cosine window
/// to prevent edge artifacts when performing neural inference.
pub fn calculate_tiles(
    width: u32,
    height: u32,
    tile_size: u32,
    overlap: u32,
) -> Vec<(u32, u32, u32, u32)> {
    let mut tiles = Vec::new();
    let step = tile_size.saturating_sub(overlap).max(1);

    let mut y = 0;
    while y < height {
        let h = tile_size.min(height - y);
        let mut x = 0;
        while x < width {
            let w = tile_size.min(width - x);
            tiles.push((x, y, w, h));
            if x + w >= width {
                break;
            }
            x += step;
        }
        if y + h >= height {
            break;
        }
        y += step;
    }
    tiles
}

/// Executes tiled neural image restoration using an ONNX model session if present,
/// blending tiles with Hanning weight windows to prevent edge artifacts.
pub fn run_neural_restoration_tiled(
    img: &DynamicImage,
    model_session: &mut ort::session::Session,
    tile_size: u32,
    tile_overlap: u32,
    denoise_strength: f32,
) -> Result<DynamicImage, String> {
    let (width, height) = img.dimensions();
    let tiles = calculate_tiles(width, height, tile_size, tile_overlap);
    let rgb = img.to_rgb8();

    let mut output_accum = vec![0.0f32; (width * height * 3) as usize];
    let mut weight_accum = vec![0.0f32; (width * height) as usize];

    for (tx, ty, tw, th) in tiles {
        let mut input_tile = ndarray::Array4::<f32>::zeros((1, 3, th as usize, tw as usize));
        for y in 0..th {
            for x in 0..tw {
                let px = rgb.get_pixel(tx + x, ty + y);
                input_tile[[0, 0, y as usize, x as usize]] = px[0] as f32 / 255.0;
                input_tile[[0, 1, y as usize, x as usize]] = px[1] as f32 / 255.0;
                input_tile[[0, 2, y as usize, x as usize]] = px[2] as f32 / 255.0;
            }
        }

        let tensor_input = ort::value::Tensor::from_array(input_tile).map_err(|e| e.to_string())?;
        let outputs = model_session
            .run(ort::inputs![tensor_input])
            .map_err(|e| format!("Neural inference failed: {e}"))?;

        let out_array = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| e.to_string())?;

        let shape = out_array.shape();
        if shape.len() != 4
            || shape[0] != 1
            || shape[1] != 3
            || shape[2] != th as usize
            || shape[3] != tw as usize
        {
            return Err(format!(
                "Model output shape mismatch. Expected [1, 3, {}, {}], got {:?}",
                th, tw, shape
            ));
        }

        for y in 0..th {
            let wy = (std::f32::consts::PI * (y as f32 + 0.5) / th as f32).sin();
            let wy = wy * wy; // Hann window
            for x in 0..tw {
                let wx = (std::f32::consts::PI * (x as f32 + 0.5) / tw as f32).sin();
                let wx = wx * wx; // Hann window
                let w = (wx * wy).max(0.01);

                let global_x = tx + x;
                let global_y = ty + y;
                let global_idx = (global_y * width + global_x) as usize;

                let r_pred = out_array[[0, 0, y as usize, x as usize]];
                let g_pred = out_array[[0, 1, y as usize, x as usize]];
                let b_pred = out_array[[0, 2, y as usize, x as usize]];

                // Blend prediction with original based on denoise_strength
                let px_orig = rgb.get_pixel(global_x, global_y);
                let r_orig = px_orig[0] as f32 / 255.0;
                let g_orig = px_orig[1] as f32 / 255.0;
                let b_orig = px_orig[2] as f32 / 255.0;

                let r_blended = r_orig * (1.0 - denoise_strength) + r_pred * denoise_strength;
                let g_blended = g_orig * (1.0 - denoise_strength) + g_pred * denoise_strength;
                let b_blended = b_orig * (1.0 - denoise_strength) + b_pred * denoise_strength;

                output_accum[global_idx * 3] += r_blended * w;
                output_accum[global_idx * 3 + 1] += g_blended * w;
                output_accum[global_idx * 3 + 2] += b_blended * w;
                weight_accum[global_idx] += w;
            }
        }
    }

    let mut out_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let w = weight_accum[idx].max(f32::EPSILON);
            let r = (output_accum[idx * 3] / w * 255.0).clamp(0.0, 255.0) as u8;
            let g = (output_accum[idx * 3 + 1] / w * 255.0).clamp(0.0, 255.0) as u8;
            let b = (output_accum[idx * 3 + 2] / w * 255.0).clamp(0.0, 255.0) as u8;
            out_buffer.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    Ok(DynamicImage::ImageRgb8(out_buffer))
}

/// Executes tiled RawNIND inference on an undecoded CFA mosaic
pub fn run_rawnind_restoration_tiled(
    raw_mosaic: &[u16],
    width: usize,
    height: usize,
    cfa_pattern: &str,
    black_levels: [u16; 4],
    white_levels: [u16; 4],
    model_session: &mut ort::session::Session,
    tile_size: u32,
    tile_overlap: u32,
    _denoise_strength: f32,
    camera_sensor: Option<&crate::custom_raw_pipeline::RawSensorData>,
    highlight_compression: f32,
) -> Result<DynamicImage, String> {
    let packed_bayer = pack_bayer_cfa_with_pattern(
        raw_mosaic,
        width,
        height,
        cfa_pattern,
        black_levels,
        white_levels,
    )?;
    let bayer_w = width / 2;
    let bayer_h = height / 2;
    let channel_size = bayer_w * bayer_h;
    let input_mean = packed_bayer.iter().copied().sum::<f32>() / packed_bayer.len().max(1) as f32;

    // We tile on the Bayer domain, so the source tile is halved. Edge tiles
    // are padded below to this fixed graph size before inference.
    let bayer_tile_size = tile_size / 2;
    let bayer_tile_overlap = tile_overlap / 2;
    if bayer_tile_size != RAWNIND_BAYER_TILE_SIZE {
        return Err(format!(
            "RawNIND requires {}×{} Bayer tiles, got {}×{}",
            RAWNIND_BAYER_TILE_SIZE, RAWNIND_BAYER_TILE_SIZE, bayer_tile_size, bayer_tile_size
        ));
    }

    let tiles = calculate_tiles(
        bayer_w as u32,
        bayer_h as u32,
        bayer_tile_size,
        bayer_tile_overlap,
    );

    let mut output_accum = vec![0.0f32; (width * height * 3) as usize];
    let mut weight_accum = vec![0.0f32; (width * height) as usize];

    for (tx, ty, tw, th) in tiles {
        // The model outputs full-resolution RGB. It is static, so every input
        // — including partial right/bottom edge tiles — must remain 512×512.
        // Replicate the nearest valid Bayer sample into padding, then crop the
        // predicted 1024×1024 RGB output to the actual image extent.
        let valid_out_w = tw * 2;
        let valid_out_h = th * 2;
        let model_out_size = RAWNIND_BAYER_TILE_SIZE * 2;
        let mut input_tile = ndarray::Array4::<f32>::zeros((
            1,
            4,
            RAWNIND_BAYER_TILE_SIZE as usize,
            RAWNIND_BAYER_TILE_SIZE as usize,
        ));
        for y in 0..RAWNIND_BAYER_TILE_SIZE as usize {
            for x in 0..RAWNIND_BAYER_TILE_SIZE as usize {
                let source_y = (ty as usize + y).min(bayer_h - 1);
                let source_x = (tx as usize + x).min(bayer_w - 1);
                let src_idx = source_y * bayer_w + source_x;
                input_tile[[0, 0, y, x]] = packed_bayer[src_idx];
                input_tile[[0, 1, y, x]] = packed_bayer[channel_size + src_idx];
                input_tile[[0, 2, y, x]] = packed_bayer[channel_size * 2 + src_idx];
                input_tile[[0, 3, y, x]] = packed_bayer[channel_size * 3 + src_idx];
            }
        }

        let tensor_input = ort::value::Tensor::from_array(input_tile).map_err(|e| e.to_string())?;
        let outputs = model_session
            .run(ort::inputs![tensor_input])
            .map_err(|e| format!("Neural inference failed: {e}"))?;

        let out_array = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| e.to_string())?;

        let shape = out_array.shape();
        if shape.len() != 4
            || shape[0] != 1
            || shape[1] != 3
            || shape[2] != model_out_size as usize
            || shape[3] != model_out_size as usize
        {
            return Err(format!(
                "Model output shape mismatch. Expected [1, 3, {}, {}], got {:?}",
                model_out_size, model_out_size, shape
            ));
        }

        for y in 0..valid_out_h {
            let wy = (std::f32::consts::PI * (y as f32 + 0.5) / valid_out_h as f32).sin();
            let wy = wy * wy; // Hann window
            for x in 0..valid_out_w {
                let wx = (std::f32::consts::PI * (x as f32 + 0.5) / valid_out_w as f32).sin();
                let wx = wx * wx; // Hann window
                let w = (wx * wy).max(0.01);

                let global_x = (tx * 2) + x;
                let global_y = (ty * 2) + y;
                let global_idx = (global_y * width as u32 + global_x) as usize;

                let r_pred = out_array[[0, 0, y as usize, x as usize]];
                let g_pred = out_array[[0, 1, y as usize, x as usize]];
                let b_pred = out_array[[0, 2, y as usize, x as usize]];

                if !r_pred.is_finite() || !g_pred.is_finite() || !b_pred.is_finite() {
                    return Err("RawNIND emitted non-finite pixel data".to_string());
                }
                output_accum[global_idx * 3] += r_pred * w;
                output_accum[global_idx * 3 + 1] += g_pred * w;
                output_accum[global_idx * 3 + 2] += b_pred * w;
                weight_accum[global_idx] += w;
            }
        }
    }

    // The network is trained with arbitrary output gain. Match its global
    // output mean to the normalized Bayer input mean before color handling.
    let mut output_sum = 0.0f64;
    for y in 0..height as u32 {
        for x in 0..width as u32 {
            let idx = (y * width as u32 + x) as usize;
            let weight = weight_accum[idx].max(f32::EPSILON);
            output_sum += (output_accum[idx * 3]
                + output_accum[idx * 3 + 1]
                + output_accum[idx * 3 + 2]) as f64
                / weight as f64;
        }
    }
    let output_mean = (output_sum / (width * height * 3) as f64) as f32;
    let gain_match = if output_mean.is_finite() && output_mean.abs() > f32::EPSILON {
        input_mean / output_mean
    } else {
        return Err("RawNIND produced an invalid output gain".to_string());
    };
    let camera_transform = camera_sensor
        .map(crate::custom_raw_pipeline::normalized_camera_rgb_to_linear_srgb_transform);

    // The editor's RAW base stays linear float data. Returning an already
    // gamma-encoded 8-bit RawNIND render here makes the rest of the editor
    // apply its RAW/display stages a second time, inflating exposure.
    if let Some((camera_to_srgb, wb)) = camera_transform {
        let sensor = camera_sensor.expect("camera transform requires a sensor");
        let exposure_gain = crate::custom_raw_pipeline::estimate_exposure_gain(sensor);
        let clamp_limit = highlight_compression.max(1.01);
        let buffer = ImageBuffer::<image::Rgba<f32>, Vec<f32>>::from_fn(
            width as u32,
            height as u32,
            |x, y| {
                let idx = (y * width as u32 + x) as usize;
                let weight = weight_accum[idx].max(f32::EPSILON);
                let input = nalgebra::Vector3::new(
                    output_accum[idx * 3] / weight * gain_match * wb[0] * exposure_gain,
                    output_accum[idx * 3 + 1] / weight * gain_match * wb[1] * exposure_gain,
                    output_accum[idx * 3 + 2] / weight * gain_match * wb[2] * exposure_gain,
                );
                let output = camera_to_srgb * input;
                let (mut r, mut g, mut b) =
                    (output[0].max(0.0), output[1].max(0.0), output[2].max(0.0));
                let max_c = r.max(g).max(b);
                if max_c > 1.0 {
                    let min_c = r.min(g).min(b);
                    let factor = (1.0 - (max_c - 1.0) / (clamp_limit - 1.0)).clamp(0.0, 1.0);
                    let compressed_r = min_c + (r - min_c) * factor;
                    let compressed_g = min_c + (g - min_c) * factor;
                    let compressed_b = min_c + (b - min_c) * factor;
                    let compressed_max = compressed_r.max(compressed_g).max(compressed_b);
                    if compressed_max > 1e-6 {
                        let rescale = max_c / compressed_max;
                        r = compressed_r * rescale;
                        g = compressed_g * rescale;
                        b = compressed_b * rescale;
                    } else {
                        r = max_c;
                        g = max_c;
                        b = max_c;
                    }
                }
                image::Rgba([
                    r.clamp(0.0, clamp_limit),
                    g.clamp(0.0, clamp_limit),
                    b.clamp(0.0, clamp_limit),
                    1.0,
                ])
            },
        );
        return Ok(DynamicImage::ImageRgba32F(buffer));
    }

    let mut out_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);
    for y in 0..height as u32 {
        for x in 0..width as u32 {
            let idx = (y * width as u32 + x) as usize;
            let w = weight_accum[idx].max(f32::EPSILON);
            let camera_rgb = [
                output_accum[idx * 3] / w * gain_match,
                output_accum[idx * 3 + 1] / w * gain_match,
                output_accum[idx * 3 + 2] / w * gain_match,
            ];
            let (r_lin, g_lin, b_lin) = (
                camera_rgb[0].clamp(0.0, 1.0),
                camera_rgb[1].clamp(0.0, 1.0),
                camera_rgb[2].clamp(0.0, 1.0),
            );
            let to_srgb = |value: f32| {
                if value <= 0.0031308 {
                    value * 12.92
                } else {
                    1.055 * value.powf(1.0 / 2.4) - 0.055
                }
            };

            let r = (to_srgb(r_lin) * 255.0).round().clamp(0.0, 255.0) as u8;
            let g = (to_srgb(g_lin) * 255.0).round().clamp(0.0, 255.0) as u8;
            let b = (to_srgb(b_lin) * 255.0).round().clamp(0.0, 255.0) as u8;

            out_buffer.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    Ok(DynamicImage::ImageRgb8(out_buffer))
}

/// Runs RawNIND as a RAW Develop stage. The returned image is an in-memory
/// render for the editor/export pipeline; it never creates a derivative file.
pub fn develop_rawnind_in_memory(
    file_bytes: &[u8],
    model_path: &Path,
    camera_sensor: &crate::custom_raw_pipeline::RawSensorData,
    highlight_compression: f32,
) -> Result<DynamicImage, String> {
    let source = rawler::rawsource::RawSource::new_from_slice(file_bytes);
    let raw_image = rawler::get_decoder(&source)
        .and_then(|decoder| {
            decoder.raw_image(
                &source,
                &rawler::decoders::RawDecodeParams::default(),
                false,
            )
        })
        .map_err(|error| format!("Failed to decode RAW mosaic for RawNIND: {error}"))?;
    let raw_data = match &raw_image.data {
        rawler::rawimage::RawImageData::Integer(data) => data,
        _ => return Err("RawNIND does not support floating-point RAW data".to_string()),
    };
    let black_levels = raw_image
        .blacklevel
        .as_bayer_array()
        .map(|level| level.clamp(0.0, u16::MAX as f32) as u16);
    let white_levels = raw_image
        .whitelevel
        .as_bayer_array()
        .map(|level| level.clamp(0.0, u16::MAX as f32) as u16);
    // Hold the shared session for the complete tiled run. This both reuses
    // model initialization and makes overlapping RAW Develop requests wait
    // rather than multiplying their CPU thread pools.
    let mut session = rawnind_editor_session(model_path)?;
    let (_, session) = session
        .as_mut()
        .expect("RawNIND editor session is initialized before use");

    run_rawnind_restoration_tiled(
        raw_data,
        raw_image.width,
        raw_image.height,
        raw_image.camera.cfa.name.as_str(),
        black_levels,
        white_levels,
        session,
        RAWNIND_SOURCE_TILE_SIZE,
        64,
        1.0,
        Some(camera_sensor),
        highlight_compression,
    )
}

/// Packs single-channel Bayer mosaic CFA raw values into a 4-channel NCHW tensor
/// format [R, G1, G2, B] normalized by camera black/white level.
pub fn pack_bayer_cfa(
    raw_mosaic: &[u16],
    width: usize,
    height: usize,
    black_level: u16,
    white_level: u16,
) -> Result<Vec<f32>, String> {
    pack_bayer_cfa_with_pattern(
        raw_mosaic,
        width,
        height,
        "RGGB",
        [black_level; 4],
        [white_level; 4],
    )
}

/// Packs a two-by-two Bayer mosaic into canonical [R, G1, G2, B] channels.
/// RawNIND is deliberately limited to standard Bayer patterns; X-Trans and other
/// mosaics need a model trained for their CFA and must not be passed to this runner.
pub fn pack_bayer_cfa_with_pattern(
    raw_mosaic: &[u16],
    width: usize,
    height: usize,
    cfa_pattern: &str,
    black_levels: [u16; 4],
    white_levels: [u16; 4],
) -> Result<Vec<f32>, String> {
    if width % 2 != 0 || height % 2 != 0 {
        return Err("Bayer mosaic dimensions must be even".to_string());
    }
    if raw_mosaic.len() < width.saturating_mul(height) {
        return Err("Bayer mosaic buffer is shorter than its declared dimensions".to_string());
    }
    let positions = match cfa_pattern {
        "RGGB" => [0, 1, 2, 3],
        "BGGR" => [3, 2, 1, 0],
        "GRBG" => [1, 0, 3, 2],
        "GBRG" => [2, 3, 0, 1],
        _ => {
            return Err(format!(
                "RawNIND supports RGGB/BGGR/GRBG/GBRG CFA patterns, got {cfa_pattern}"
            ));
        }
    };
    let half_w = width / 2;
    let half_h = height / 2;
    let channel_size = half_w * half_h;
    let mut tensor = vec![0.0f32; channel_size * 4];

    for y in 0..half_h {
        for x in 0..half_w {
            let src_y = y * 2;
            let src_x = x * 2;
            let dest_idx = y * half_w + x;
            let samples = [
                raw_mosaic[src_y * width + src_x],
                raw_mosaic[src_y * width + src_x + 1],
                raw_mosaic[(src_y + 1) * width + src_x],
                raw_mosaic[(src_y + 1) * width + src_x + 1],
            ];
            for (channel, &position) in positions.iter().enumerate() {
                let range = white_levels[position]
                    .saturating_sub(black_levels[position])
                    .max(1) as f32;
                tensor[channel * channel_size + dest_idx] =
                    (samples[position].saturating_sub(black_levels[position]) as f32 / range)
                        .clamp(0.0, 1.0);
            }
        }
    }

    Ok(tensor)
}

/// Creates a safe derivative output path in the catalog's derivative directory
pub fn derivative_output_path(
    catalog_dir: &Path,
    source_image_id: i64,
    operation: &str,
    format_ext: &str,
) -> Result<PathBuf, String> {
    let dir = catalog_dir.join("derivatives").join(operation);
    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create derivative directory: {e}"))?;
    let timestamp_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let uuid_suffix = &uuid::Uuid::new_v4().to_string()[..8];
    let filename =
        format!("img_{source_image_id}_{operation}_{timestamp_nanos}_{uuid_suffix}.{format_ext}");
    Ok(dir.join(filename))
}

/// Initiates a background restoration job for a specific image in the catalog
#[tauri::command]
pub fn start_image_restoration(
    image_id: i64,
    mut recipe: RestorationRecipe,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<String, String> {
    validate_restoration_recipe(&recipe)?;
    let visual_models_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("models")
        .join("visual");
    recipe.model_revision = crate::visual_model_registry::visual_model_pack_revision_in_dir(
        &visual_models_dir,
        &recipe.model_id,
    )?;
    let db_path = crate::library_db::active_library_path(&state)?;
    let job_id = crate::library_db::create_background_job(
        &db_path,
        &recipe.operation_kind,
        serde_json::json!({
            "imageId": image_id,
            "recipe": recipe,
        }),
    )?;

    let job_control = crate::app_state::BackgroundJobControl::new();
    state
        .background_job_controls
        .lock()
        .unwrap()
        .insert(job_id.clone(), job_control.clone());

    let app = app_handle.clone();
    let worker_job_id = job_id.clone();
    let worker_recipe = recipe.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let ai_semaphore = app.state::<crate::AppState>().ai_job_semaphore.clone();
        let visual_models_dir = app
            .path()
            .app_data_dir()
            .map(|path| path.join("models").join("visual"))
            .map_err(|error| error.to_string());
        let result = match visual_models_dir {
            Ok(visual_models_dir) => run_restoration_worker(
                &db_path,
                image_id,
                &worker_recipe,
                &worker_job_id,
                &job_control,
                &visual_models_dir,
                ai_semaphore,
            ),
            Err(error) => Err(error),
        };

        if let Err(error) = result {
            let job_state = if error == "Restoration cancelled" {
                "cancelled"
            } else {
                "failed"
            };
            let _ = crate::library_db::update_job(
                &db_path,
                &worker_job_id,
                job_state,
                &error,
                0,
                100,
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
            "image-restoration-complete",
            serde_json::json!({ "jobId": worker_job_id, "imageId": image_id }),
        );
    });

    Ok(job_id)
}

pub fn run_restoration_worker(
    db_path: &Path,
    image_id: i64,
    recipe: &RestorationRecipe,
    job_id: &str,
    job_control: &std::sync::Arc<crate::app_state::BackgroundJobControl>,
    visual_models_dir: &Path,
    ai_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
) -> Result<(), String> {
    let conn = rusqlite::Connection::open(db_path).map_err(|e| e.to_string())?;

    let (root_path, relative_path): (String, String) = conn
        .query_row(
            "SELECT r.absolute_path, i.relative_path FROM images i JOIN collection_roots r ON r.id = i.root_id WHERE i.id = ?1",
            [image_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|e| format!("Image record not found: {e}"))?;

    let source_path = Path::new(&root_path).join(&relative_path);
    if !source_path.exists() {
        return Err(format!(
            "Source image file not found: {}",
            source_path.display()
        ));
    }

    let _ = crate::library_db::update_job(
        db_path,
        job_id,
        "running",
        "Loading image for restoration",
        10,
        100,
        Some(&source_path.to_string_lossy()),
        None,
    );

    if *job_control.cancellation_receiver().borrow() {
        return Err("Restoration cancelled".to_string());
    }

    let catalog_dir = db_path.parent().unwrap_or_else(|| Path::new("."));
    let final_output =
        derivative_output_path(catalog_dir, image_id, &recipe.operation_kind, "tiff")?;
    let temp_output = final_output.with_extension("tmp.tiff");

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let recipe_json = serde_json::to_string(recipe).unwrap_or_default();

    // Calculate input hash for provenance verification
    let input_hash = match fs::read(&source_path) {
        Ok(bytes) => {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            Some(hex::encode(hasher.finalize()))
        }
        Err(_) => None,
    };

    // Create queued derivative record
    conn.execute(
        "INSERT INTO image_derivatives(source_image_id, operation_kind, model_id, model_revision, recipe_json, input_hash, output_path, output_format, state, created_at, updated_at)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, 'tiff', 'running', ?8, ?8)",
        rusqlite::params![
            image_id,
            recipe.operation_kind,
            recipe.model_id,
            recipe.model_revision,
            recipe_json,
            input_hash,
            final_output.to_string_lossy().to_string(),
            now,
        ],
    )
    .map_err(|e| format!("Failed to create initial derivative record: {e}"))?;
    let derivative_id = conn.last_insert_rowid();

    let is_raw = crate::formats::is_raw_file(&source_path);
    let file_bytes = if is_raw {
        match fs::read(&source_path) {
            Ok(bytes) => Some(bytes),
            Err(e) => {
                let error_msg = format!("Failed to read raw file: {e}");
                let _ = conn.execute("UPDATE image_derivatives SET state = 'failed', error_message = ?1, updated_at = ?2 WHERE id = ?3", rusqlite::params![error_msg, now, derivative_id]);
                return Err(error_msg);
            }
        }
    } else {
        None
    };

    // Check pause state with cancellation awareness
    if !tauri::async_runtime::block_on(job_control.wait_until_runnable()) {
        let _ = conn.execute(
            "UPDATE image_derivatives SET state = 'cancelled', updated_at = ?1 WHERE id = ?2",
            rusqlite::params![now, derivative_id],
        );
        return Err("Restoration cancelled".to_string());
    }

    let _ = crate::library_db::update_job(
        db_path,
        job_id,
        "running",
        "Executing neural restoration",
        50,
        100,
        None,
        None,
    );

    // Check if neural model is installed and load session
    let restored = if recipe.operation_kind.contains("denoise") {
        match crate::visual_model_registry::installed_visual_model_path_in_dir(
            visual_models_dir,
            &recipe.model_id,
            if recipe.model_id == crate::visual_model_registry::RAWNIND_MODEL_ID {
                crate::visual_model_registry::RAWNIND_MODEL_FILE_NAME
            } else if recipe.model_id == crate::visual_model_registry::NAFNET_MODEL_ID {
                crate::visual_model_registry::NAFNET_MODEL_FILE_NAME
            } else {
                "model.onnx"
            },
        ) {
            Ok(model_file) if model_file.exists() => {
                match ort::session::Session::builder().and_then(|b| b.commit_from_file(&model_file))
                {
                    Ok(mut session) => {
                        if recipe.model_id.contains("rawnind") && is_raw {
                            // Extract Bayer mosaic
                            let bytes = file_bytes.as_ref().unwrap();
                            let source = rawler::rawsource::RawSource::new_from_slice(bytes);
                            let raw_image = match rawler::get_decoder(&source).and_then(|d| {
                                d.raw_image(
                                    &source,
                                    &rawler::decoders::RawDecodeParams::default(),
                                    false,
                                )
                            }) {
                                Ok(ri) => ri,
                                Err(e) => {
                                    let error_msg =
                                        format!("Failed to decode RAW mosaic for RawNIND: {}", e);
                                    let _ = conn.execute("UPDATE image_derivatives SET state = 'failed', error_message = ?1, updated_at = ?2 WHERE id = ?3", rusqlite::params![error_msg, now, derivative_id]);
                                    return Err(error_msg);
                                }
                            };
                            let raw_data = match &raw_image.data {
                                rawler::rawimage::RawImageData::Integer(data) => data,
                                _ => {
                                    let error_msg =
                                        "Unsupported float RAW data for RawNIND".to_string();
                                    let _ = conn.execute("UPDATE image_derivatives SET state = 'failed', error_message = ?1, updated_at = ?2 WHERE id = ?3", rusqlite::params![error_msg, now, derivative_id]);
                                    return Err(error_msg);
                                }
                            };
                            let cfa_pattern = raw_image.camera.cfa.name.as_str();
                            let black_levels = raw_image
                                .blacklevel
                                .as_bayer_array()
                                .map(|level| level.clamp(0.0, u16::MAX as f32) as u16);
                            let white_levels = raw_image
                                .whitelevel
                                .as_bayer_array()
                                .map(|level| level.clamp(0.0, u16::MAX as f32) as u16);

                            let _permit = tauri::async_runtime::block_on(
                                ai_semaphore.clone().acquire_owned(),
                            )
                            .ok();
                            let result = run_rawnind_restoration_tiled(
                                raw_data,
                                raw_image.width,
                                raw_image.height,
                                cfa_pattern,
                                black_levels,
                                white_levels,
                                &mut session,
                                recipe.tile_size,
                                recipe.tile_overlap,
                                recipe.denoise_strength,
                                None,
                                2.5,
                            );
                            drop(_permit);
                            match result {
                                Ok(res) => res,
                                Err(e) => {
                                    let error_msg = format!("RawNIND inference failed: {}", e);
                                    let _ = conn.execute("UPDATE image_derivatives SET state = 'failed', error_message = ?1, updated_at = ?2 WHERE id = ?3", rusqlite::params![error_msg, now, derivative_id]);
                                    return Err(error_msg);
                                }
                            }
                        } else {
                            // Standard RGB fallback/neural run
                            let img = if is_raw {
                                let (_, sidecar_path) = crate::file_management::parse_virtual_path(
                                    &source_path.to_string_lossy(),
                                );
                                let raw_develop_metadata =
                                    crate::exif_processing::load_sidecar(&sidecar_path);
                                match crate::image_loader::load_base_image_from_bytes(
                                    file_bytes.as_ref().unwrap(),
                                    &source_path.to_string_lossy(),
                                    false,
                                    &crate::app_settings::AppSettings::default(),
                                    Some(&raw_develop_metadata.adjustments),
                                    true,
                                    None,
                                ) {
                                    Ok(loaded) => loaded,
                                    Err(e) => {
                                        let error_msg = format!(
                                            "Failed to develop RAW file for restoration: {}",
                                            e
                                        );
                                        let _ = conn.execute("UPDATE image_derivatives SET state = 'failed', error_message = ?1, updated_at = ?2 WHERE id = ?3", rusqlite::params![error_msg, now, derivative_id]);
                                        return Err(error_msg);
                                    }
                                }
                            } else {
                                match image::open(&source_path) {
                                    Ok(loaded) => loaded,
                                    Err(e) => {
                                        let error_msg = format!("Failed to open image: {}", e);
                                        let _ = conn.execute("UPDATE image_derivatives SET state = 'failed', error_message = ?1, updated_at = ?2 WHERE id = ?3", rusqlite::params![error_msg, now, derivative_id]);
                                        return Err(error_msg);
                                    }
                                }
                            };

                            let _permit = tauri::async_runtime::block_on(
                                ai_semaphore.clone().acquire_owned(),
                            )
                            .ok();
                            let result = run_neural_restoration_tiled(
                                &img,
                                &mut session,
                                recipe.tile_size,
                                recipe.tile_overlap,
                                recipe.denoise_strength,
                            );
                            drop(_permit);
                            match result {
                                Ok(res) => res,
                                Err(e) => {
                                    let error_msg = format!("Neural inference failed: {}", e);
                                    let _ = conn.execute("UPDATE image_derivatives SET state = 'failed', error_message = ?1, updated_at = ?2 WHERE id = ?3", rusqlite::params![error_msg, now, derivative_id]);
                                    return Err(error_msg);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let error_msg = format!("Failed to load ONNX model: {}", e);
                        let _ = conn.execute("UPDATE image_derivatives SET state = 'failed', error_message = ?1, updated_at = ?2 WHERE id = ?3", rusqlite::params![error_msg, now, derivative_id]);
                        return Err(error_msg);
                    }
                }
            }
            _ => {
                let error_msg = "Required model artifact is missing or not installed.".to_string();
                let _ = conn.execute("UPDATE image_derivatives SET state = 'failed', error_message = ?1, updated_at = ?2 WHERE id = ?3", rusqlite::params![error_msg, now, derivative_id]);
                return Err(error_msg);
            }
        }
    } else {
        // Non-neural fallback operation
        let img = if is_raw {
            let (_, sidecar_path) =
                crate::file_management::parse_virtual_path(&source_path.to_string_lossy());
            let raw_develop_metadata = crate::exif_processing::load_sidecar(&sidecar_path);
            match crate::image_loader::load_base_image_from_bytes(
                file_bytes.as_ref().unwrap(),
                &source_path.to_string_lossy(),
                false,
                &crate::app_settings::AppSettings::default(),
                Some(&raw_develop_metadata.adjustments),
                true,
                None,
            ) {
                Ok(loaded) => loaded,
                Err(e) => {
                    let error_msg = format!("Failed to develop RAW file for restoration: {}", e);
                    let _ = conn.execute("UPDATE image_derivatives SET state = 'failed', error_message = ?1, updated_at = ?2 WHERE id = ?3", rusqlite::params![error_msg, now, derivative_id]);
                    return Err(error_msg);
                }
            }
        } else {
            match image::open(&source_path) {
                Ok(loaded) => loaded,
                Err(e) => {
                    let error_msg = format!("Failed to open image: {}", e);
                    let _ = conn.execute("UPDATE image_derivatives SET state = 'failed', error_message = ?1, updated_at = ?2 WHERE id = ?3", rusqlite::params![error_msg, now, derivative_id]);
                    return Err(error_msg);
                }
            }
        };
        img
    };

    if *job_control.cancellation_receiver().borrow() {
        let _ = fs::remove_file(&temp_output);
        let _ = conn.execute(
            "UPDATE image_derivatives SET state = 'cancelled', updated_at = ?1 WHERE id = ?2",
            rusqlite::params![now, derivative_id],
        );
        return Err("Restoration cancelled".to_string());
    }

    // Save temporary derivative file
    if let Err(e) = restored.save(&temp_output) {
        let _ = fs::remove_file(&temp_output);
        let error_msg = format!("Failed to write derivative: {e}");
        let _ = conn.execute(
            "UPDATE image_derivatives SET state = 'failed', error_message = ?1, updated_at = ?2 WHERE id = ?3",
            rusqlite::params![error_msg, now, derivative_id],
        );
        return Err(error_msg);
    }

    // Atomic publish
    if let Err(e) = fs::rename(&temp_output, &final_output) {
        let _ = fs::remove_file(&temp_output);
        let error_msg = format!("Failed to publish derivative: {e}");
        let _ = conn.execute(
            "UPDATE image_derivatives SET state = 'failed', error_message = ?1, updated_at = ?2 WHERE id = ?3",
            rusqlite::params![error_msg, now, derivative_id],
        );
        return Err(error_msg);
    }

    let (width, height) = restored.dimensions();

    // Calculate output hash for provenance verification
    let output_hash = match fs::read(&final_output) {
        Ok(bytes) => {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            Some(hex::encode(hasher.finalize()))
        }
        Err(_) => None,
    };

    let completed_now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    // Update derivative record to completed with provenance hashes
    conn.execute(
        "UPDATE image_derivatives SET width = ?1, height = ?2, output_hash = ?3, state = 'completed', completed_at = ?4, updated_at = ?4 WHERE id = ?5",
        rusqlite::params![
            width as i64,
            height as i64,
            output_hash,
            completed_now,
            derivative_id,
        ],
    )
    .map_err(|e| format!("Failed to update derivative in catalog: {e}"))?;

    let _ = crate::library_db::update_job(
        db_path,
        job_id,
        "completed",
        "Restoration complete",
        100,
        100,
        Some(&final_output.to_string_lossy()),
        None,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculate_tiles_covers_entire_image_area() {
        let width = 1920;
        let height = 1080;
        let tile_size = 512;
        let overlap = 64;

        let tiles = calculate_tiles(width, height, tile_size, overlap);
        assert!(!tiles.is_empty());

        // Verify bounds
        for (x, y, w, h) in &tiles {
            assert!(*x + *w <= width);
            assert!(*y + *h <= height);
            assert!(*w <= tile_size);
            assert!(*h <= tile_size);
        }

        // Verify last tile touches edges
        let max_x = tiles.iter().map(|(x, _, w, _)| x + w).max().unwrap();
        let max_y = tiles.iter().map(|(_, y, _, h)| y + h).max().unwrap();
        assert_eq!(max_x, width);
        assert_eq!(max_y, height);
    }

    #[test]
    fn microcontrast_finishing_filter_preserves_dimensions() {
        let img = DynamicImage::new_rgb8(100, 100);
        let enhanced = apply_microcontrast(&img, 0.5, 0.3);
        assert_eq!(enhanced.width(), 100);
        assert_eq!(enhanced.height(), 100);
    }

    #[test]
    fn pack_bayer_cfa_normalizes_to_unit_range_and_nchw_layout() {
        let mosaic: Vec<u16> = vec![512, 1024, 1024, 2048];
        let tensor = pack_bayer_cfa(&mosaic, 2, 2, 0, 4096).unwrap();
        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor[0], 512.0 / 4096.0); // R
        assert_eq!(tensor[1], 1024.0 / 4096.0); // G1
        assert_eq!(tensor[2], 1024.0 / 4096.0); // G2
        assert_eq!(tensor[3], 2048.0 / 4096.0); // B
    }

    #[test]
    fn pack_bayer_cfa_rejects_odd_dimensions() {
        let mosaic: Vec<u16> = vec![1, 2, 3];
        assert!(pack_bayer_cfa(&mosaic, 3, 1, 0, 100).is_err());
    }

    #[test]
    fn restoration_recipe_defaults_are_valid() {
        let recipe = RestorationRecipe::default();
        assert_eq!(recipe.operation_kind, "raw_denoise");
        assert_eq!(
            recipe.model_id,
            crate::visual_model_registry::RAWNIND_MODEL_ID
        );
        assert!(recipe.denoise_strength > 0.0 && recipe.denoise_strength <= 1.0);
        assert_eq!(recipe.microcontrast_strength, 0.0);
        assert_eq!(recipe.detail_recovery, 0.0);
        assert!(recipe.tile_size >= 256);
        assert!(recipe.tile_overlap < recipe.tile_size);
    }
    #[test]
    fn pack_bayer_cfa_reorders_bggr_to_canonical_channels() {
        let mosaic = vec![40, 30, 20, 10]; // BGGR: B, G, G, R
        let packed = pack_bayer_cfa_with_pattern(&mosaic, 2, 2, "BGGR", [0; 4], [100; 4]).unwrap();
        assert_eq!(packed, vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn test_pack_bayer_cfa_dimensions() {
        // 4x4 image
        let mut raw_mosaic = vec![0u16; 16];
        raw_mosaic[0] = 10;
        raw_mosaic[1] = 20;
        raw_mosaic[4] = 30;
        raw_mosaic[5] = 40;

        let packed = pack_bayer_cfa(&raw_mosaic, 4, 4, 0, 100).unwrap();
        assert_eq!(packed.len(), 4 * (2 * 2)); // 4 channels * 4 pixels
        assert_eq!(packed[0], 0.1);
        assert_eq!(packed[4], 0.2); // G1 starts at offset 4
        assert_eq!(packed[8], 0.3); // G2 starts at offset 8
        assert_eq!(packed[12], 0.4); // B starts at offset 12
    }

    #[test]
    fn test_pack_bayer_cfa_odd_dimensions_fail() {
        let raw_mosaic = vec![0u16; 9];
        let packed = pack_bayer_cfa(&raw_mosaic, 3, 3, 0, 100);
        assert!(packed.is_err());
    }

    #[test]
    fn test_pack_bayer_cfa_clamping() {
        let raw_mosaic = vec![0, 150, 0, 0];
        let packed = pack_bayer_cfa(&raw_mosaic, 2, 2, 50, 100).unwrap();
        // 0 -> clamped to 0
        // 150 - 50 = 100 / 50 = 2 -> clamped to 1.0
        assert_eq!(packed[0], 0.0);
        assert_eq!(packed[1], 1.0);
    }

    #[test]
    fn pack_bayer_cfa_rejects_unknown_cfa_and_short_buffer() {
        assert!(pack_bayer_cfa_with_pattern(&[0; 4], 2, 2, "XTRANS", [0; 4], [1; 4]).is_err());
        assert!(pack_bayer_cfa_with_pattern(&[0; 3], 2, 2, "RGGB", [0; 4], [1; 4]).is_err());
    }

    #[test]
    fn restoration_recipe_rejects_invalid_model_or_tile_settings() {
        let mut recipe = RestorationRecipe::default();
        assert!(validate_restoration_recipe(&recipe).is_ok());
        recipe.model_id = crate::visual_model_registry::NAFNET_MODEL_ID.to_string();
        assert!(validate_restoration_recipe(&recipe).is_err());
        recipe.model_id = crate::visual_model_registry::RAWNIND_MODEL_ID.to_string();
        recipe.tile_overlap = recipe.tile_size;
        assert!(validate_restoration_recipe(&recipe).is_err());
        recipe.tile_overlap = 64;
        recipe.operation_kind = "upscale".to_string();
        assert!(validate_restoration_recipe(&recipe).is_err());
    }
}
