use std::fs;
use std::path::{Path, PathBuf};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager};


/// Recipe and configuration parameters for a restoration operation.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RestorationRecipe {
    pub operation_kind: String, // "raw_denoise", "rgb_denoise", "deblur", "upscale"
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
            model_id: "rawnind-utnet2-bayer".to_string(),
            model_revision: "v1".to_string(),
            denoise_strength: 0.8,
            // These legacy fields remain in stored recipes for compatibility.
            // Finish-stage adjustments are deliberately never applied here.
            microcontrast_strength: 0.0,
            detail_recovery: 0.0,
            tile_size: 768,
            tile_overlap: 64,
        }
    }
}

fn validate_restoration_recipe(recipe: &RestorationRecipe) -> Result<(), String> {
    if !matches!(recipe.operation_kind.as_str(), "raw_denoise" | "rgb_denoise" | "deblur" | "upscale") {
        return Err(format!("Unsupported restoration operation: {}", recipe.operation_kind));
    }
    if !(0.0..=1.0).contains(&recipe.denoise_strength)
        || !(0.0..=1.0).contains(&recipe.microcontrast_strength)
        || !(0.0..=1.0).contains(&recipe.detail_recovery)
    {
        return Err("Restoration strengths must be between 0 and 1".to_string());
    }
    if recipe.tile_size < 64 || recipe.tile_overlap >= recipe.tile_size {
        return Err("Tile size must be at least 64 and overlap must be smaller than the tile size".to_string());
    }
    match recipe.operation_kind.as_str() {
        "raw_denoise" if recipe.model_id != "rawnind-utnet2-bayer" => Err("RAW denoise requires the RawNIND Bayer model".to_string()),
        "rgb_denoise" if recipe.model_id != "nafnet-sidd-rgb" => Err("RGB denoise requires the NAFNet SIDD model".to_string()),
        _ => Ok(()),
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
            let l_new = (l_base + high_pass * boost + high_pass.signum() * detail_boost)
                .clamp(0.0, 255.0);

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
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] != th as usize || shape[3] != tw as usize {
            return Err(format!("Model output shape mismatch. Expected [1, 3, {}, {}], got {:?}", th, tw, shape));
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

    // We tile on the Bayer domain, so tile size is halved
    let bayer_tile_size = tile_size / 2;
    let bayer_tile_overlap = tile_overlap / 2;

    let tiles = calculate_tiles(bayer_w as u32, bayer_h as u32, bayer_tile_size, bayer_tile_overlap);

    let mut output_accum = vec![0.0f32; (width * height * 3) as usize];
    let mut weight_accum = vec![0.0f32; (width * height) as usize];

    for (tx, ty, tw, th) in tiles {
        // Output from model is full resolution, so we reconstruct it
        let out_w = tw * 2;
        let out_h = th * 2;

        let mut input_tile = ndarray::Array4::<f32>::zeros((1, 4, th as usize, tw as usize));
        for y in 0..th as usize {
            for x in 0..tw as usize {
                let src_idx = (ty as usize + y) * bayer_w + (tx as usize + x);
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
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] != out_h as usize || shape[3] != out_w as usize {
            return Err(format!("Model output shape mismatch. Expected [1, 3, {}, {}], got {:?}", out_h, out_w, shape));
        }

        for y in 0..out_h {
            let wy = (std::f32::consts::PI * (y as f32 + 0.5) / out_h as f32).sin();
            let wy = wy * wy; // Hann window
            for x in 0..out_w {
                let wx = (std::f32::consts::PI * (x as f32 + 0.5) / out_w as f32).sin();
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

    let mut out_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);
    for y in 0..height as u32 {
        for x in 0..width as u32 {
            let idx = (y * width as u32 + x) as usize;
            let w = weight_accum[idx].max(f32::EPSILON);
            // Model outputs linear Rec.2020. We apply a simple gamma for sRGB preview here
            let r_lin = (output_accum[idx * 3] / w).clamp(0.0, 1.0);
            let g_lin = (output_accum[idx * 3 + 1] / w).clamp(0.0, 1.0);
            let b_lin = (output_accum[idx * 3 + 2] / w).clamp(0.0, 1.0);

            let r = (r_lin.powf(1.0 / 2.2) * 255.0) as u8;
            let g = (g_lin.powf(1.0 / 2.2) * 255.0) as u8;
            let b = (b_lin.powf(1.0 / 2.2) * 255.0) as u8;

            out_buffer.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    Ok(DynamicImage::ImageRgb8(out_buffer))
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
        _ => return Err(format!("RawNIND supports RGGB/BGGR/GRBG/GBRG CFA patterns, got {cfa_pattern}")),
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
                let range = white_levels[position].saturating_sub(black_levels[position]).max(1) as f32;
                tensor[channel * channel_size + dest_idx] =
                    (samples[position].saturating_sub(black_levels[position]) as f32 / range).clamp(0.0, 1.0);
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
    let filename = format!("img_{source_image_id}_{operation}_{timestamp_nanos}_{uuid_suffix}.{format_ext}");
    Ok(dir.join(filename))
}

/// Initiates a background restoration job for a specific image in the catalog
#[tauri::command]
pub fn start_image_restoration(
    image_id: i64,
    recipe: RestorationRecipe,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, crate::AppState>,
) -> Result<String, String> {
    validate_restoration_recipe(&recipe)?;
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
        let result = run_restoration_worker(
            &db_path,
            image_id,
            &worker_recipe,
            &worker_job_id,
            &job_control,
            &app,
            ai_semaphore,
        );

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

fn run_restoration_worker(
    db_path: &Path,
    image_id: i64,
    recipe: &RestorationRecipe,
    job_id: &str,
    job_control: &std::sync::Arc<crate::app_state::BackgroundJobControl>,
    _app_handle: &tauri::AppHandle,
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
        return Err(format!("Source image file not found: {}", source_path.display()));
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
    let final_output = derivative_output_path(catalog_dir, image_id, &recipe.operation_kind, "tiff")?;
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
        let _ = conn.execute("UPDATE image_derivatives SET state = 'cancelled', updated_at = ?1 WHERE id = ?2", rusqlite::params![now, derivative_id]);
        return Err("Restoration cancelled".to_string());
    }

    let _ = crate::library_db::update_job(db_path, job_id, "running", "Executing neural restoration", 50, 100, None, None);

    // Check if neural model is installed and load session
    let restored = if recipe.operation_kind.contains("denoise") {
        match crate::visual_model_registry::installed_visual_model_path(
            _app_handle,
            &recipe.model_id,
            if recipe.model_id.contains("rawnind") {
                "rawnind_bayer.onnx"
            } else if recipe.model_id.contains("nafnet") {
                "nafnet_sidd.onnx"
            } else {
                "model.onnx"
            },
        ) {
            Ok(model_file) if model_file.exists() => {
                match ort::session::Session::builder().and_then(|b| b.commit_from_file(&model_file)) {
                    Ok(mut session) => {
                        if recipe.model_id.contains("rawnind") && is_raw {
                            // Extract Bayer mosaic
                            let bytes = file_bytes.as_ref().unwrap();
                            let source = rawler::rawsource::RawSource::new_from_slice(bytes);
                            let raw_image = match rawler::get_decoder(&source).and_then(|d| d.raw_image(&source, &rawler::decoders::RawDecodeParams::default(), false)) {
                                Ok(ri) => ri,
                                Err(e) => {
                                    let error_msg = format!("Failed to decode RAW mosaic for RawNIND: {}", e);
                                    let _ = conn.execute("UPDATE image_derivatives SET state = 'failed', error_message = ?1, updated_at = ?2 WHERE id = ?3", rusqlite::params![error_msg, now, derivative_id]);
                                    return Err(error_msg);
                                }
                            };
                            let raw_data = match &raw_image.data {
                                rawler::rawimage::RawImageData::Integer(data) => data,
                                _ => {
                                    let error_msg = "Unsupported float RAW data for RawNIND".to_string();
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

                            let _permit = tauri::async_runtime::block_on(ai_semaphore.clone().acquire_owned()).ok();
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
                                match crate::image_loader::load_base_image_from_bytes(file_bytes.as_ref().unwrap(), &source_path.to_string_lossy(), false, &crate::app_settings::AppSettings::default(), None) {
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

                            let _permit = tauri::async_runtime::block_on(ai_semaphore.clone().acquire_owned()).ok();
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
            match crate::image_loader::load_base_image_from_bytes(file_bytes.as_ref().unwrap(), &source_path.to_string_lossy(), false, &crate::app_settings::AppSettings::default(), None) {
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
        let mosaic: Vec<u16> = vec![
            512, 1024,
            1024, 2048,
        ];
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
        assert_eq!(recipe.model_id, "rawnind-utnet2-bayer");
        assert!(recipe.denoise_strength > 0.0 && recipe.denoise_strength <= 1.0);
        assert_eq!(recipe.microcontrast_strength, 0.0);
        assert_eq!(recipe.detail_recovery, 0.0);
        assert!(recipe.tile_size >= 256);
        assert!(recipe.tile_overlap < recipe.tile_size);
    }
    #[test]
    fn pack_bayer_cfa_reorders_bggr_to_canonical_channels() {
        let mosaic = vec![40, 30, 20, 10]; // BGGR: B, G, G, R
        let packed = pack_bayer_cfa_with_pattern(
            &mosaic,
            2,
            2,
            "BGGR",
            [0; 4],
            [100; 4],
        )
        .unwrap();
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
        recipe.model_id = "nafnet-sidd-rgb".to_string();
        assert!(validate_restoration_recipe(&recipe).is_err());
        recipe.model_id = "rawnind-utnet2-bayer".to_string();
        recipe.tile_overlap = recipe.tile_size;
        assert!(validate_restoration_recipe(&recipe).is_err());
    }
}
