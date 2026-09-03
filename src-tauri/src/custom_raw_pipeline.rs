//! Our own RAW development pipeline: rawler is used ONLY to decode the file
//! format into raw (pre-demosaic) sensor data and calibration metadata -
//! demosaic and the rest of the pipeline (white balance, camera-to-sRGB
//! color conversion, cropping) are implemented entirely in this module.
//!
//! This exists so custom demosaic algorithms (AMaZE, IGV, LMMSE) can be
//! implemented without forking or patching the vendored `rawler` crate -
//! see CLAUDE.md's "Open TODO: pluggable demosaic algorithms" for why that
//! constraint exists. Phase 1 (this file, initially) proves the pipeline
//! end-to-end with a simple bilinear demosaic before the real algorithms are
//! implemented.

use anyhow::{Result, anyhow};
use image::{DynamicImage, ImageBuffer, Rgba};
use nalgebra::Matrix3;
use rayon::prelude::*;
use rawler::{
    cfa::CFAColor,
    decoders::{Orientation, RawDecodeParams},
    imgop::xyz::Illuminant,
    rawimage::{RawImage, RawImageData, RawPhotometricInterpretation},
    rawsource::RawSource,
};

/// Toggles for the optional pipeline stages that sit around demosaic:
/// raw-domain preprocessing (before), denoise and sharpening (after).
/// Passing all-default/off values reproduces the original demosaic-only
/// pipeline exactly.
#[derive(Clone, Copy, Debug)]
pub struct DevelopOptions {
    /// Hot/dead pixel correction + CFA row-banding denoise, applied to the
    /// raw mosaic before demosaic. See `raw_preprocess.rs`.
    pub preprocess: bool,
    /// Post-demosaic wavelet luminance/chrominance denoise strength, 0..1
    /// (0 = off). See `raw_denoise.rs`.
    pub denoise_strength: f32,
    /// Post-demosaic sharpening amount, 0..1 (0 = off). See `raw_sharpen.rs`.
    pub sharpen_amount: f32,
    /// Which sharpening algorithm `sharpen_amount` drives. See
    /// `raw_sharpen::SharpenMethod`.
    pub sharpen_method: crate::raw_sharpen::SharpenMethod,
}

impl Default for DevelopOptions {
    fn default() -> Self {
        Self {
            preprocess: true,
            denoise_strength: 0.0,
            sharpen_amount: 0.0,
            sharpen_method: crate::raw_sharpen::SharpenMethod::UnsharpMask,
        }
    }
}

impl DevelopOptions {
    /// Auto-selected options for a given ISO, matching the same
    /// "auto by ISO" pattern as `demosaic_algorithms::select_by_iso`.
    pub fn auto_for_iso(iso: u32) -> Self {
        Self {
            preprocess: true,
            denoise_strength: crate::raw_denoise::suggest_strength_for_iso(iso),
            sharpen_amount: crate::raw_sharpen::suggest_amount_for_iso(iso),
            sharpen_method: crate::raw_sharpen::SharpenMethod::UnsharpMask,
        }
    }
}

/// Raw (pre-demosaic) sensor data plus everything needed to develop it,
/// extracted from rawler's `RawImage` - all fields here come from rawler's
/// own public API, nothing requires modifying rawler itself.
pub struct RawSensorData {
    pub width: usize,
    pub height: usize,
    /// Mosaiced sensor values, one f32 per pixel, row-major, length width*height.
    pub data: Vec<f32>,
    pub cfa_at: Box<dyn Fn(usize, usize) -> CFAColor + Send + Sync>,
    pub wb_coeffs: [f32; 4],
    pub black_level: [f32; 4],
    pub white_level: f32,
    /// XYZ (D65-ish, camera-native illuminant) -> camera RGB, top 3 rows of
    /// rawler's `xyz_to_cam` (the 4th row is for RGBE sensors, unused here).
    pub xyz_to_cam: Matrix3<f32>,
    /// (x, y, width, height) of the active (non-black-masked) sensor area.
    pub active_area: Option<[usize; 4]>,
    /// (x, y, width, height) of the default (post-active-area) crop, in
    /// sensor-absolute coordinates - mirrors rawler's `crop_area`.
    pub crop_area: Option<[usize; 4]>,
    pub iso: u32,
    pub orientation: Orientation,
    /// True only for a standard 2x2 RGB Bayer CFA - the only sensor layout
    /// our own demosaic algorithms understand. X-Trans, 4-channel (CMY/RGBE),
    /// monochrome and DNG `LinearRaw` sources are all ineligible and must
    /// fall back to rawler's own pipeline.
    pub is_standard_bayer: bool,
    /// "<clean_make> <clean_model>" (e.g. "Sony ILCE-7RM2"), used to look up
    /// per-camera PDAF sensor-row data in `raw_pdaf_data.rs`.
    pub camera_name: String,
}

pub fn decode_raw_sensor_data(file_bytes: &[u8]) -> Result<RawSensorData> {
    let source = RawSource::new_from_slice(file_bytes);
    let decoder = rawler::get_decoder(&source).map_err(|e| anyhow!("get_decoder: {e}"))?;
    let raw_image: RawImage = decoder
        .raw_image(&source, &RawDecodeParams::default(), false)
        .map_err(|e| anyhow!("raw_image: {e}"))?;
    let metadata = decoder
        .raw_metadata(&source, &RawDecodeParams::default())
        .map_err(|e| anyhow!("raw_metadata: {e}"))?;

    let orientation = metadata
        .exif
        .orientation
        .map(Orientation::from_u16)
        .unwrap_or(Orientation::Normal);
    let iso = metadata.exif.iso_speed_ratings.unwrap_or(0) as u32;

    let is_standard_bayer = matches!(
        &raw_image.photometric,
        RawPhotometricInterpretation::Cfa(config)
            if config.cfa.is_rgb() && config.cfa.width == 2 && config.cfa.height == 2
    );

    let width = raw_image.width;
    let height = raw_image.height;

    let data: Vec<f32> = match &raw_image.data {
        RawImageData::Integer(v) => v.iter().map(|&p| p as f32).collect(),
        RawImageData::Float(v) => v.clone(),
    };

    if data.len() != width * height {
        return Err(anyhow!(
            "unexpected raw data length: {} != {}x{}",
            data.len(),
            width,
            height
        ));
    }

    let black_level: [f32; 4] = {
        let mut levels = [0.0f32; 4];
        for (i, l) in raw_image.blacklevel.levels.iter().take(4).enumerate() {
            levels[i] = l.as_f32();
        }
        if raw_image.blacklevel.levels.len() == 1 {
            levels = [levels[0]; 4];
        }
        levels
    };

    let white_level = raw_image
        .whitelevel
        .0
        .first()
        .copied()
        .unwrap_or(u16::MAX as u32) as f32;

    // `raw_image.xyz_to_cam` is explicitly marked deprecated by rawler
    // itself ("TODO: deprecated, use color_matrix") and was the actual
    // cause of a strong magenta/purple color cast on every image - it does
    // not reliably hold the file's real calibrated matrix. rawler's own
    // production Calibrate step (`imgop::develop::RawDevelop`) reads
    // `color_matrix` instead, preferring the D65 illuminant entry and
    // falling back to whatever's first available; mirror that exactly.
    let mut xyz_to_cam = Matrix3::identity();
    let found_matrix = raw_image
        .color_matrix
        .iter()
        .find(|(illuminant, _m)| **illuminant == Illuminant::D65)
        .or_else(|| raw_image.color_matrix.iter().next());
    if let Some((_illuminant, color_matrix)) = found_matrix
        && color_matrix.len() >= 9
    {
        for r in 0..3 {
            for c in 0..3 {
                xyz_to_cam[(r, c)] = color_matrix[r * 3 + c];
            }
        }
        if xyz_to_cam == Matrix3::zeros() {
            xyz_to_cam = Matrix3::identity();
        }
    }

    let cfa = raw_image.camera.cfa.clone();
    let cfa_at = Box::new(move |row: usize, col: usize| cfa.cfa_color_at(row, col));

    Ok(RawSensorData {
        width,
        height,
        data,
        cfa_at,
        wb_coeffs: raw_image.wb_coeffs,
        black_level,
        white_level,
        xyz_to_cam,
        active_area: raw_image.active_area.map(|r| [r.p.x, r.p.y, r.d.w, r.d.h]),
        crop_area: raw_image.crop_area.map(|r| [r.p.x, r.p.y, r.d.w, r.d.h]),
        iso,
        orientation,
        is_standard_bayer,
        camera_name: format!(
            "{} {}",
            raw_image.clean_make.trim(),
            raw_image.clean_model.trim()
        )
        .trim()
        .to_string(),
    })
}

#[inline]
fn sample(sensor: &RawSensorData, row: isize, col: isize) -> f32 {
    let row = row.clamp(0, sensor.height as isize - 1) as usize;
    let col = col.clamp(0, sensor.width as isize - 1) as usize;
    sensor.data[row * sensor.width + col]
}

/// Simple bilinear Bayer demosaic - Phase 1 placeholder to validate the rest
/// of the pipeline (extraction, white balance, color conversion) before the
/// real algorithms (AMaZE/IGV/LMMSE) replace this function.
pub fn bilinear_demosaic(sensor: &RawSensorData) -> Vec<[f32; 3]> {
    let (w, h) = (sensor.width, sensor.height);
    let mut out = vec![[0.0f32; 3]; w * h];

    for row in 0..h {
        for col in 0..w {
            let center_color = (sensor.cfa_at)(row, col);
            let mut rgb = [0.0f32; 3];
            let center_val = sensor.data[row * w + col];

            match center_color {
                CFAColor::RED => {
                    rgb[0] = center_val;
                    rgb[1] = average_neighbors(sensor, row as isize, col as isize, true);
                    rgb[2] = average_neighbors(sensor, row as isize, col as isize, false);
                }
                CFAColor::BLUE => {
                    rgb[2] = center_val;
                    rgb[1] = average_neighbors(sensor, row as isize, col as isize, true);
                    rgb[0] = average_neighbors(sensor, row as isize, col as isize, false);
                }
                _ => {
                    // GREEN (or anything else, fails open to green): red/blue
                    // come from the four diagonal neighbors, whichever color
                    // they actually are determined per-neighbor.
                    rgb[1] = center_val;
                    let mut r_sum = 0.0;
                    let mut r_n = 0.0;
                    let mut b_sum = 0.0;
                    let mut b_n = 0.0;
                    for (dr, dc) in [(-1_isize, -1_isize), (-1, 1), (1, -1), (1, 1)] {
                        let nr = row as isize + dr;
                        let nc = col as isize + dc;
                        let v = sample(sensor, nr, nc);
                        let color = (sensor.cfa_at)(
                            nr.clamp(0, h as isize - 1) as usize,
                            nc.clamp(0, w as isize - 1) as usize,
                        );
                        match color {
                            CFAColor::RED => {
                                r_sum += v;
                                r_n += 1.0;
                            }
                            CFAColor::BLUE => {
                                b_sum += v;
                                b_n += 1.0;
                            }
                            _ => {}
                        }
                    }
                    rgb[0] = if r_n > 0.0 { r_sum / r_n } else { center_val };
                    rgb[2] = if b_n > 0.0 { b_sum / b_n } else { center_val };
                }
            }

            out[row * w + col] = rgb;
        }
    }

    out
}

/// Averages the 4 orthogonal (cross) or 4 diagonal neighbors of a pixel.
/// Used to interpolate green (cross) or the opposite red/blue channel
/// (diagonal) at a red/blue sensor site.
fn average_neighbors(sensor: &RawSensorData, row: isize, col: isize, cross: bool) -> f32 {
    let offsets: [(isize, isize); 4] = if cross {
        [(-1, 0), (1, 0), (0, -1), (0, 1)]
    } else {
        [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    };
    let sum: f32 = offsets
        .iter()
        .map(|&(dr, dc)| sample(sensor, row + dr, col + dc))
        .sum();
    sum / 4.0
}

/// Applies black-level subtraction + white-balance scaling to mosaiced RGB
/// triples (each triple only has one "real" sensor-measured channel per
/// pixel before demosaic, but callers pass already-demosaiced RGB here).
fn apply_white_balance(rgb: &mut [[f32; 3]], sensor: &RawSensorData) {
    let bl = sensor.black_level;
    let wb = sensor.wb_coeffs;
    let wb_max = wb[0]
        .max(wb[1])
        .max(wb[2])
        .max(if wb[3] > 0.0 { wb[3] } else { 0.0 });
    let denom = (sensor.white_level - bl[0]).max(1.0);

    rgb.par_iter_mut().for_each(|px| {
        for c in 0..3 {
            let black = bl[c];
            let gain = if wb_max > 0.0 { wb[c] / wb_max } else { 1.0 };
            px[c] = ((px[c] - black) * gain / denom).max(0.0);
        }
    });
}

// Standard sRGB (D65) <- XYZ matrix (IEC 61966-2-1).
const XYZ_TO_SRGB: [[f32; 3]; 3] = [
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
];

#[inline]
fn srgb_gamma(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Converts white-balanced camera-RGB triples to (still-linear) sRGB
/// primaries, via the camera's own XYZ<->camera-RGB calibration matrix
/// (rawler's `xyz_to_cam`). No gamma is applied here - this mirrors rawler's
/// own `Calibrate` step, which also leaves the result linear.
fn apply_color_matrix(rgb: &mut [[f32; 3]], xyz_to_cam: &Matrix3<f32>) {
    let cam_to_xyz = match xyz_to_cam.try_inverse() {
        Some(m) => m,
        None => Matrix3::identity(),
    };
    let xyz_to_srgb = Matrix3::from_row_slice(&[
        XYZ_TO_SRGB[0][0],
        XYZ_TO_SRGB[0][1],
        XYZ_TO_SRGB[0][2],
        XYZ_TO_SRGB[1][0],
        XYZ_TO_SRGB[1][1],
        XYZ_TO_SRGB[1][2],
        XYZ_TO_SRGB[2][0],
        XYZ_TO_SRGB[2][1],
        XYZ_TO_SRGB[2][2],
    ]);
    let cam_to_srgb = xyz_to_srgb * cam_to_xyz;

    rgb.par_iter_mut().for_each(|px| {
        let v = nalgebra::Vector3::new(px[0], px[1], px[2]);
        let out = cam_to_srgb * v;
        px[0] = out[0];
        px[1] = out[1];
        px[2] = out[2];
    });
}

/// Same as `apply_color_matrix`, but also applies sRGB gamma - used only by
/// the standalone test-dump path below, which produces a display-ready
/// image rather than the linear intermediate the live app pipeline expects.
fn apply_color_matrix_and_gamma(rgb: &mut [[f32; 3]], xyz_to_cam: &Matrix3<f32>) {
    apply_color_matrix(rgb, xyz_to_cam);
    rgb.par_iter_mut().for_each(|px| {
        px[0] = srgb_gamma(px[0]);
        px[1] = srgb_gamma(px[1]);
        px[2] = srgb_gamma(px[2]);
    });
}

/// Production entry point: develops a RAW file into the same LINEAR,
/// pre-tonemap `Rgba32F` intermediate that `raw_processing::develop_raw_image`
/// produces (values scaled so ~1.0 is the white point, up to
/// `highlight_compression` for softly-clipped highlights), but using our own
/// demosaic algorithms (AMaZE/IGV/LMMSE, auto-selected by ISO) instead of
/// rawler's PPG. Orientation is applied, matching `develop_raw_image`.
///
/// Returns `Err` for anything our demosaic doesn't support (X-Trans,
/// 4-channel, monochrome, DNG `LinearRaw`) - callers must fall back to
/// `raw_processing::develop_raw_image` in that case.
pub fn develop_raw_image_custom(
    file_bytes: &[u8],
    highlight_compression: f32,
) -> Result<DynamicImage> {
    let sensor = decode_raw_sensor_data(file_bytes)?;
    if !sensor.is_standard_bayer {
        return Err(anyhow!(
            "not a standard Bayer CFA; custom pipeline not applicable"
        ));
    }
    let algo = crate::demosaic_algorithms::select_by_iso(sensor.iso);
    let options = DevelopOptions::auto_for_iso(sensor.iso);
    develop_raw_image_custom_with_algorithm(file_bytes, algo, highlight_compression, &options)
}

/// Same as `develop_raw_image_custom`, but with per-image overrides (the
/// "Raw Develop" adjustments panel section) layered on top of the ISO-based
/// auto defaults - `None`/`Ok(None)` for any override falls back to auto.
/// This is what `raw_processing::develop_raw_image_for_editor` calls; used
/// only by the one real "open an image for editing" load path, not by every
/// `raw_processing::develop_raw_image` caller (thumbnails, export, culling,
/// etc. all keep using plain ISO-auto behavior unchanged).
#[allow(clippy::too_many_arguments)]
pub fn develop_raw_image_custom_resolved(
    file_bytes: &[u8],
    highlight_compression: f32,
    demosaic_override: Option<&str>,
    denoise_override: Option<f32>,
    sharpen_override: Option<f32>,
    sharpen_method_override: Option<&str>,
    preprocess: bool,
) -> Result<DynamicImage> {
    let sensor = decode_raw_sensor_data(file_bytes)?;
    if !sensor.is_standard_bayer {
        return Err(anyhow!(
            "not a standard Bayer CFA; custom pipeline not applicable"
        ));
    }

    let algo = match demosaic_override {
        None | Some("auto") | Some("") => crate::demosaic_algorithms::select_by_iso(sensor.iso),
        Some(name) => crate::demosaic_algorithms::parse_algorithm_name(name)
            .ok_or_else(|| anyhow!("unknown demosaic override '{name}'"))?,
    };
    let sharpen_method = match sharpen_method_override {
        None | Some("") => crate::raw_sharpen::SharpenMethod::UnsharpMask,
        Some(name) => crate::raw_sharpen::parse_method_name(name)
            .ok_or_else(|| anyhow!("unknown sharpen method override '{name}'"))?,
    };
    let options = DevelopOptions {
        preprocess,
        denoise_strength: denoise_override
            .unwrap_or_else(|| crate::raw_denoise::suggest_strength_for_iso(sensor.iso)),
        sharpen_amount: sharpen_override
            .unwrap_or_else(|| crate::raw_sharpen::suggest_amount_for_iso(sensor.iso)),
        sharpen_method,
    };
    develop_raw_image_custom_with_algorithm(file_bytes, algo, highlight_compression, &options)
}

/// Same as `develop_raw_image_custom`, but with an explicit algorithm choice
/// and pipeline options instead of ISO auto-selection - used by the CLI
/// (`raw develop --demosaic amaze|igv|lmmse|bilinear --denoise ... --sharpen
/// ...`) to debug/compare stages directly against the exact linear
/// intermediate the live app produces.
pub fn develop_raw_image_custom_with_algorithm(
    file_bytes: &[u8],
    algo: crate::demosaic_algorithms::DemosaicAlgorithm,
    highlight_compression: f32,
    options: &DevelopOptions,
) -> Result<DynamicImage> {
    let mut sensor = decode_raw_sensor_data(file_bytes)?;
    if !sensor.is_standard_bayer {
        return Err(anyhow!(
            "not a standard Bayer CFA; custom pipeline not applicable"
        ));
    }

    let _t_preprocess = std::time::Instant::now();
    if options.preprocess {
        crate::raw_preprocess::correct_pdaf_pixels(&mut sensor);
        crate::raw_preprocess::correct_hot_dead_pixels(&mut sensor, 0.15);
        crate::raw_preprocess::correct_cfa_line_banding(&mut sensor, 0.5);
        crate::raw_preprocess::equalize_green_channels(&mut sensor);
    }
    log::debug!("[raw develop timing] preprocess: {:?}", _t_preprocess.elapsed());

    let _t_demosaic = std::time::Instant::now();
    let mut rgb = crate::demosaic_algorithms::demosaic(&sensor, algo);
    log::debug!("[raw develop timing] demosaic ({:?}): {:?}", algo, _t_demosaic.elapsed());
    let _t_wbcm = std::time::Instant::now();
    apply_white_balance(&mut rgb, &sensor);
    apply_color_matrix(&mut rgb, &sensor.xyz_to_cam);
    log::debug!("[raw develop timing] wb+colormatrix: {:?}", _t_wbcm.elapsed());

    // Mirrors raw_processing::develop_internal's rescale + highlight
    // compression: apply_white_balance already normalized by
    // (white_level - black_level), so here rescale is a no-op scale of 1.0
    // and we only need the highlight-compression clamp/desaturation.
    let safe_highlight_compression = highlight_compression.max(1.01);
    let clamp_limit = safe_highlight_compression;
    rgb.par_iter_mut().for_each(|px| {
        let (r, g, b) = (px[0].max(0.0), px[1].max(0.0), px[2].max(0.0));
        let max_c = r.max(g).max(b);
        let (final_r, final_g, final_b) = if max_c > 1.0 {
            let min_c = r.min(g).min(b);
            let compression_factor =
                (1.0 - (max_c - 1.0) / (safe_highlight_compression - 1.0)).clamp(0.0, 1.0);
            let compressed_r = min_c + (r - min_c) * compression_factor;
            let compressed_g = min_c + (g - min_c) * compression_factor;
            let compressed_b = min_c + (b - min_c) * compression_factor;
            let compressed_max = compressed_r.max(compressed_g).max(compressed_b);
            if compressed_max > 1e-6 {
                let rescale = max_c / compressed_max;
                (
                    compressed_r * rescale,
                    compressed_g * rescale,
                    compressed_b * rescale,
                )
            } else {
                (max_c, max_c, max_c)
            }
        } else {
            (r, g, b)
        };
        px[0] = final_r.clamp(0.0, clamp_limit);
        px[1] = final_g.clamp(0.0, clamp_limit);
        px[2] = final_b.clamp(0.0, clamp_limit);
    });

    let _t_denoise = std::time::Instant::now();
    if options.denoise_strength > 0.0 {
        crate::raw_denoise::wavelet_denoise(
            &mut rgb,
            sensor.width,
            sensor.height,
            options.denoise_strength,
        );
    }
    log::debug!("[raw develop timing] denoise: {:?}", _t_denoise.elapsed());
    let _t_sharpen = std::time::Instant::now();
    if options.sharpen_amount > 0.0 {
        crate::raw_sharpen::sharpen(
            &mut rgb,
            sensor.width,
            sensor.height,
            options.sharpen_method,
            options.sharpen_amount,
            1.0,
        );
    }
    log::debug!("[raw develop timing] sharpen: {:?}", _t_sharpen.elapsed());

    // Crop to active area, then to the default crop (crop_area), matching
    // rawler's CropActiveArea + CropDefault steps in that order.
    let (aa_x, aa_y, aa_w, aa_h) = match sensor.active_area {
        Some([x, y, w, h]) => (x, y, w, h),
        None => (0, 0, sensor.width, sensor.height),
    };
    let (crop_x, crop_y, crop_w, crop_h) = match sensor.crop_area {
        Some([x, y, w, h]) => {
            // crop_area is active-area-relative in rawler's own semantics
            // for Bayer sensors post-demosaic; intersect defensively so an
            // out-of-range value can never panic on the buffer below.
            let x = (aa_x + x).min(sensor.width.saturating_sub(1));
            let y = (aa_y + y).min(sensor.height.saturating_sub(1));
            let w = w.min(sensor.width.saturating_sub(x));
            let h = h.min(sensor.height.saturating_sub(y));
            (x, y, w, h)
        }
        None => (aa_x, aa_y, aa_w, aa_h),
    };

    let buffer = ImageBuffer::<Rgba<f32>, _>::from_fn(crop_w as u32, crop_h as u32, |x, y| {
        let src_row = crop_y + y as usize;
        let src_col = crop_x + x as usize;
        let px = rgb[src_row * sensor.width + src_col];
        Rgba([px[0], px[1], px[2], 1.0])
    });

    Ok(crate::image_processing::apply_orientation(
        DynamicImage::ImageRgba32F(buffer),
        sensor.orientation,
    ))
}

/// Full pipeline: decode -> demosaic -> white balance -> color matrix ->
/// crop to active area -> DynamicImage. Mirrors
/// `raw_processing::develop_raw_image`'s shape so output can be compared
/// directly against the existing rawler-pipeline result.
pub fn develop_raw_custom(file_bytes: &[u8]) -> Result<DynamicImage> {
    develop_raw_custom_with_algorithm(
        file_bytes,
        crate::demosaic_algorithms::DemosaicAlgorithm::Bilinear,
    )
}

pub fn develop_raw_custom_with_algorithm(
    file_bytes: &[u8],
    algo: crate::demosaic_algorithms::DemosaicAlgorithm,
) -> Result<DynamicImage> {
    develop_raw_custom_with_options(file_bytes, algo, &DevelopOptions::default())
}

/// Same as `develop_raw_custom_with_algorithm`, with explicit preprocess/
/// denoise/sharpen options - used by the CLI's non-`--linear` (display-ready
/// gamma-encoded) preview path. Note denoise/sharpen run in gamma-encoded
/// space here (after `apply_color_matrix_and_gamma`), unlike
/// `develop_raw_image_custom_with_algorithm`'s linear space - this path is a
/// display convenience, not meant for exact numeric parity between the two.
pub fn develop_raw_custom_with_options(
    file_bytes: &[u8],
    algo: crate::demosaic_algorithms::DemosaicAlgorithm,
    options: &DevelopOptions,
) -> Result<DynamicImage> {
    let mut sensor = decode_raw_sensor_data(file_bytes)?;

    if options.preprocess {
        crate::raw_preprocess::correct_pdaf_pixels(&mut sensor);
        crate::raw_preprocess::correct_hot_dead_pixels(&mut sensor, 0.15);
        crate::raw_preprocess::correct_cfa_line_banding(&mut sensor, 0.5);
        crate::raw_preprocess::equalize_green_channels(&mut sensor);
    }

    let mut rgb = crate::demosaic_algorithms::demosaic(&sensor, algo);
    apply_white_balance(&mut rgb, &sensor);
    apply_color_matrix_and_gamma(&mut rgb, &sensor.xyz_to_cam);

    if options.denoise_strength > 0.0 {
        crate::raw_denoise::wavelet_denoise(
            &mut rgb,
            sensor.width,
            sensor.height,
            options.denoise_strength,
        );
    }
    if options.sharpen_amount > 0.0 {
        crate::raw_sharpen::sharpen(
            &mut rgb,
            sensor.width,
            sensor.height,
            options.sharpen_method,
            options.sharpen_amount,
            1.0,
        );
    }

    // Crop to active area, then to the default crop (crop_area), matching
    // the production-parity `develop_raw_image_custom_with_algorithm` path
    // (this used to only crop to active_area - a Phase-1 gap that only
    // affected this CLI/test preview path, not the live app).
    let (aa_x, aa_y, aa_w, aa_h) = match sensor.active_area {
        Some([x, y, w, h]) => (x, y, w, h),
        None => (0, 0, sensor.width, sensor.height),
    };
    let (crop_x, crop_y, crop_w, crop_h) = match sensor.crop_area {
        Some([x, y, w, h]) => {
            let x = (aa_x + x).min(sensor.width.saturating_sub(1));
            let y = (aa_y + y).min(sensor.height.saturating_sub(1));
            let w = w.min(sensor.width.saturating_sub(x));
            let h = h.min(sensor.height.saturating_sub(y));
            (x, y, w, h)
        }
        None => (aa_x, aa_y, aa_w, aa_h),
    };

    let buffer = ImageBuffer::<Rgba<u8>, _>::from_fn(crop_w as u32, crop_h as u32, |x, y| {
        let src_row = crop_y + y as usize;
        let src_col = crop_x + x as usize;
        let px = rgb[src_row * sensor.width + src_col];
        Rgba([
            (px[0] * 255.0).round().clamp(0.0, 255.0) as u8,
            (px[1] * 255.0).round().clamp(0.0, 255.0) as u8,
            (px[2] * 255.0).round().clamp(0.0, 255.0) as u8,
            255,
        ])
    });

    Ok(DynamicImage::ImageRgba8(buffer))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Temporary validation test: dumps develop_raw_custom's output next to
    // rawler's own PPG-based develop_raw_image output for the same file, so
    // they can be compared visually. Run with:
    //   RAPIDRAW_TEST_RAW_PATH=/path/to/file.arw RAPIDRAW_TEST_DUMP_DIR=/tmp/dump \
    //     cargo test --lib custom_raw_pipeline::tests::dump_comparison -- --nocapture
    #[test]
    fn dump_comparison() {
        let path = match std::env::var("RAPIDRAW_TEST_RAW_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("RAPIDRAW_TEST_RAW_PATH not set, skipping");
                return;
            }
        };
        let dump_dir =
            std::env::var("RAPIDRAW_TEST_DUMP_DIR").unwrap_or_else(|_| "/tmp".to_string());
        let bytes = std::fs::read(&path).expect("read raw file");

        use crate::demosaic_algorithms::DemosaicAlgorithm;
        for (name, algo) in [
            ("bilinear", DemosaicAlgorithm::Bilinear),
            ("amaze", DemosaicAlgorithm::AMaZE),
            ("igv", DemosaicAlgorithm::IGV),
            ("lmmse", DemosaicAlgorithm::LMMSE),
        ] {
            let img = develop_raw_custom_with_algorithm(&bytes, algo)
                .expect("develop_raw_custom_with_algorithm");
            let out_path = format!("{dump_dir}/custom_{name}.png");
            img.save(&out_path).expect("save custom variant");
            println!("saved {} ({}x{})", out_path, img.width(), img.height());
        }

        // develop_raw_image returns a LINEAR (pre-tonemap, pre-gamma) Rgba32F
        // intermediate meant for further GPU-side processing, not a display-
        // ready image - so this is only a rough sanity-check conversion
        // (naive gamma + clamp), not a true apples-to-apples comparison
        // against develop_raw_custom's already-gamma-encoded sRGB output.
        let reference =
            crate::raw_processing::develop_raw_image(&bytes, false, 2.0, "auto".to_string(), None)
                .expect("develop_raw_image");
        let reference_f32 = reference.to_rgba32f();
        let reference_rgba8 = image::ImageBuffer::<image::Rgba<u8>, _>::from_fn(
            reference.width(),
            reference.height(),
            |x, y| {
                let p = reference_f32.get_pixel(x, y).0;
                image::Rgba([
                    (srgb_gamma(p[0]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    (srgb_gamma(p[1]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    (srgb_gamma(p[2]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    255,
                ])
            },
        );
        let reference = DynamicImage::ImageRgba8(reference_rgba8);
        let reference_path = format!("{dump_dir}/reference_pipeline.png");
        reference.save(&reference_path).expect("save reference");
        println!(
            "saved {} ({}x{})",
            reference_path,
            reference.width(),
            reference.height()
        );
    }

    // Validates the live-pipeline wiring in raw_processing::develop_raw_image:
    // with RAPIDRAW_CUSTOM_DEMOSAIC=1 it should route through
    // develop_raw_image_custom (ISO-selected algorithm, oriented, linear
    // Rgba32F) instead of rawler's PPG. Run with:
    //   RAPIDRAW_TEST_RAW_PATH=/path/to/file.arw RAPIDRAW_TEST_DUMP_DIR=/tmp/dump \
    //     RAPIDRAW_CUSTOM_DEMOSAIC=1 \
    //     cargo test --lib custom_raw_pipeline::tests::wired_pipeline_dump -- --nocapture
    #[test]
    fn wired_pipeline_dump() {
        let path = match std::env::var("RAPIDRAW_TEST_RAW_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("RAPIDRAW_TEST_RAW_PATH not set, skipping");
                return;
            }
        };
        let dump_dir =
            std::env::var("RAPIDRAW_TEST_DUMP_DIR").unwrap_or_else(|_| "/tmp".to_string());
        let bytes = std::fs::read(&path).expect("read raw file");

        let sensor = decode_raw_sensor_data(&bytes).expect("decode_raw_sensor_data");
        println!(
            "sensor: {}x{}, iso={}, orientation={:?}, is_standard_bayer={}, active_area={:?}, crop_area={:?}",
            sensor.width,
            sensor.height,
            sensor.iso,
            sensor.orientation,
            sensor.is_standard_bayer,
            sensor.active_area,
            sensor.crop_area
        );

        let selected = crate::demosaic_algorithms::select_by_iso(sensor.iso);
        println!("select_by_iso({}) -> {:?}", sensor.iso, selected);

        unsafe {
            std::env::set_var("RAPIDRAW_CUSTOM_DEMOSAIC", "1");
        }
        let wired =
            crate::raw_processing::develop_raw_image(&bytes, false, 2.5, "auto".to_string(), None)
                .expect("develop_raw_image with RAPIDRAW_CUSTOM_DEMOSAIC=1");
        unsafe {
            std::env::remove_var("RAPIDRAW_CUSTOM_DEMOSAIC");
        }

        println!(
            "wired develop_raw_image (custom path) -> {}x{}, color={:?}",
            wired.width(),
            wired.height(),
            wired.color()
        );

        // wired is a LINEAR Rgba32F intermediate (same contract as the
        // rawler-PPG path) - apply the same naive gamma+clamp used for the
        // reference dump above so it's visually inspectable as a PNG.
        let wired_f32 = wired.to_rgba32f();
        let wired_rgba8 = image::ImageBuffer::<image::Rgba<u8>, _>::from_fn(
            wired.width(),
            wired.height(),
            |x, y| {
                let p = wired_f32.get_pixel(x, y).0;
                image::Rgba([
                    (srgb_gamma(p[0]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    (srgb_gamma(p[1]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    (srgb_gamma(p[2]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    255,
                ])
            },
        );
        let out_path = format!("{dump_dir}/wired_custom_pipeline.png");
        image::DynamicImage::ImageRgba8(wired_rgba8)
            .save(&out_path)
            .expect("save wired dump");
        println!("saved {} ({}x{})", out_path, wired.width(), wired.height());
    }

    // Clean timing of develop_raw_image_custom (the actual production
    // path the live app calls, minus PNG encoding - the app never encodes
    // PNG mid-pipeline, it hands the linear Rgba32F straight to the GPU
    // shader, so PNG-encode time in the other dump tests above is not
    // representative of real in-app latency). Run with:
    //   RAPIDRAW_TEST_RAW_PATH=/path/to/file.arw \
    //     cargo test --release --lib custom_raw_pipeline::tests::time_develop_only -- --nocapture
    #[test]
    fn time_develop_only() {
        let path = match std::env::var("RAPIDRAW_TEST_RAW_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("RAPIDRAW_TEST_RAW_PATH not set, skipping");
                return;
            }
        };
        let bytes = std::fs::read(&path).expect("read raw file");

        let t0 = std::time::Instant::now();
        let sensor = decode_raw_sensor_data(&bytes).expect("decode");
        let t_decode = t0.elapsed();

        let t1 = std::time::Instant::now();
        let img = develop_raw_image_custom(&bytes, 2.5).expect("develop_raw_image_custom");
        let t_develop = t1.elapsed();

        println!(
            "decode={:?} develop_total={:?} iso={} dims={}x{}",
            t_decode,
            t_develop,
            sensor.iso,
            img.width(),
            img.height()
        );
    }
}
