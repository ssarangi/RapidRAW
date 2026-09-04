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
use rawler::{
    cfa::CFAColor,
    decoders::{Orientation, RawDecodeParams},
    imgop::xyz::Illuminant,
    rawimage::{RawImage, RawImageData, RawPhotometricInterpretation},
    rawsource::RawSource,
};
use rayon::prelude::*;

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
    /// Flat linear gain applied to the white-balanced, color-matrixed RGB
    /// before highlight compression - 1.0 is a no-op. See
    /// `estimate_exposure_gain`: this exists only to keep a severely
    /// underexposed capture from rendering essentially black by default: a
    /// normally- or deliberately-low-key-exposed shot always gets 1.0.
    pub exposure_gain: f32,
}

impl Default for DevelopOptions {
    fn default() -> Self {
        Self {
            preprocess: true,
            denoise_strength: 0.0,
            sharpen_amount: 0.0,
            sharpen_method: crate::raw_sharpen::SharpenMethod::UnsharpMask,
            exposure_gain: 1.0,
        }
    }
}

impl DevelopOptions {
    /// Auto-selected options for a given (possibly exposure-gain-adjusted)
    /// ISO, matching the same "auto by ISO" pattern as
    /// `demosaic_algorithms::select_by_iso`. `effective_iso` should already
    /// have `estimate_exposure_gain`'s boost folded in by the caller (see
    /// `develop_raw_image_custom`/`develop_raw_image_custom_resolved`) -
    /// denoise/sharpen strength need to react to the *post-boost* noise
    /// level, not the camera's nominal ISO.
    pub fn auto_for_iso(effective_iso: u32, exposure_gain: f32) -> Self {
        Self {
            preprocess: true,
            denoise_strength: crate::raw_denoise::suggest_strength_for_iso(effective_iso),
            sharpen_amount: crate::raw_sharpen::suggest_amount_for_iso(effective_iso),
            sharpen_method: crate::raw_sharpen::SharpenMethod::UnsharpMask,
            exposure_gain,
        }
    }
}

/// Estimates a flat linear exposure gain for a severely underexposed
/// capture, so it doesn't render essentially black by default - a
/// deliberate, conservative "expose to the right"-style correction, not a
/// general auto-exposure algorithm. A normally-exposed frame, or one that's
/// genuinely dark by intent (a deliberate low-key/night shot with real
/// content only in the shadows), is left at gain 1.0: the trigger is
/// specifically "even the near-brightest real signal in the whole frame
/// never got close to using the sensor's headroom," which a deliberately
/// dark scene with a bright subject wouldn't hit.
pub fn estimate_exposure_gain(sensor: &RawSensorData) -> f32 {
    // Prime-ish stride so the sample doesn't alias with the 2x2 CFA tile -
    // this only needs to be a fast, rough scene-brightness estimate (it
    // informs a single scalar gain), not a precise per-channel histogram.
    const STRIDE: usize = 7;
    const TRIGGER_BELOW: f32 = 0.20;
    const TARGET_PEAK: f32 = 0.7;
    const MAX_GAIN: f32 = 8.0; // +3 stops

    let black = sensor.black_level[1];
    let denom = (sensor.white_level - black).max(1.0);

    let mut samples: Vec<f32> = sensor
        .data
        .iter()
        .step_by(STRIDE)
        .map(|&v| ((v - black) / denom).max(0.0))
        .collect();
    if samples.len() < 100 {
        return 1.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let peak = samples[((samples.len() - 1) as f32 * 0.999) as usize];

    // Below the noise floor entirely (lens cap, corrupt frame) - a huge
    // gain here would just amplify noise into nothing meaningful. Also
    // skip if already reasonably exposed.
    if !(1e-4..TRIGGER_BELOW).contains(&peak) {
        return 1.0;
    }

    (TARGET_PEAK / peak).clamp(1.0, MAX_GAIN)
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

/// Applies the same active-area/default-crop geometry used by the custom RAW
/// development pipeline to an RGB image produced directly from the sensor.
/// RawNIND emits the full mosaic extent, so it must pass through this before
/// the editor applies EXIF orientation.
pub fn crop_sensor_develop_image(image: DynamicImage, sensor: &RawSensorData) -> DynamicImage {
    let (aa_x, aa_y, aa_w, aa_h) = match sensor.active_area {
        Some([x, y, width, height]) => (x, y, width, height),
        None => (0, 0, sensor.width, sensor.height),
    };
    let (crop_x, crop_y, crop_w, crop_h) = match sensor.crop_area {
        Some([x, y, width, height]) => {
            let x = (aa_x + x).min(sensor.width.saturating_sub(1));
            let y = (aa_y + y).min(sensor.height.saturating_sub(1));
            let width = width.min(sensor.width.saturating_sub(x));
            let height = height.min(sensor.height.saturating_sub(y));
            (x, y, width, height)
        }
        None => (aa_x, aa_y, aa_w, aa_h),
    };
    image.crop_imm(crop_x as u32, crop_y as u32, crop_w as u32, crop_h as u32)
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
    let denom = (sensor.white_level - bl[0]).max(1.0);

    // wb_coeffs are already the direct per-channel multipliers a neutral
    // subject needs (green is conventionally the reference at ~1.0, e.g.
    // [1.47, 1.0, 2.93] for a warm/tungsten-lit scene needing more blue) -
    // they must be applied AS-IS, not re-normalized by dividing by
    // whichever channel happens to be numerically largest. Doing that
    // (the previous bug here) inverts the correction: it suppresses the
    // channels that actually needed boosting instead of boosting them,
    // producing a strong, systematic color cast on every image regardless
    // of the color-matrix step downstream. A channel legitimately going
    // above the original white point after this (e.g. blue at 2.93x) is
    // expected and handled by the highlight-compression step that follows.
    rgb.par_iter_mut().for_each(|px| {
        for c in 0..3 {
            let black = bl[c];
            px[c] = ((px[c] - black) * wb[c] / denom).max(0.0);
        }
    });
}

#[inline]
fn srgb_gamma(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

// Standard sRGB (D65) -> XYZ matrix, matching rawler's own
// `imgop::xyz::SRGB_TO_XYZ_D65` constant exactly (IEC 61966-2-1).
const SRGB_TO_XYZ: [[f32; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];

/// Converts white-balanced camera-RGB triples to (still-linear) sRGB
/// primaries, via the camera's own XYZ<->camera-RGB calibration matrix
/// (rawler's `color_matrix`). No gamma is applied here - this mirrors
/// rawler's own `Calibrate` step, which also leaves the result linear.
///
/// This is a direct port of rawler's actual (but `pub(crate)`-only, hence
/// unreachable) `imgop::raw::map_3ch_to_rgb` - read from the vendored
/// source rather than reverse-engineered, after an earlier attempt (invert
/// xyz_to_cam, multiply by XYZ_TO_SRGB, *then* row-normalize the combined
/// camera->sRGB matrix) turned out to only fix neutral gray while leaving
/// other colors wrong - measurably: it still produced a strong cast on a
/// real photo. rawler's actual order is different in a way that matters:
/// it builds the *forward* sRGB->camera matrix first
/// (`xyz_to_cam * srgb_to_xyz`), row-normalizes *that* (so sRGB-white
/// truly maps to equal-RGB camera values), and only *then* inverts to get
/// camera->sRGB - normalizing before inverting is not the same operation
/// as inverting then normalizing the result.
fn camera_to_linear_srgb_matrix(xyz_to_cam: &Matrix3<f32>) -> Matrix3<f32> {
    let srgb_to_xyz = Matrix3::from_row_slice(&[
        SRGB_TO_XYZ[0][0],
        SRGB_TO_XYZ[0][1],
        SRGB_TO_XYZ[0][2],
        SRGB_TO_XYZ[1][0],
        SRGB_TO_XYZ[1][1],
        SRGB_TO_XYZ[1][2],
        SRGB_TO_XYZ[2][0],
        SRGB_TO_XYZ[2][1],
        SRGB_TO_XYZ[2][2],
    ]);
    // sRGB -> camera, fused (rawler's `rgb2cam`).
    let mut rgb_to_cam = xyz_to_cam * srgb_to_xyz;
    // rawler's `normalize`: each row scaled to sum to 1.0, so sRGB-white
    // [1,1,1] maps to equal-valued camera RGB.
    for r in 0..3 {
        let row_sum = rgb_to_cam[(r, 0)] + rgb_to_cam[(r, 1)] + rgb_to_cam[(r, 2)];
        if row_sum.abs() > 1e-8 {
            rgb_to_cam[(r, 0)] /= row_sum;
            rgb_to_cam[(r, 1)] /= row_sum;
            rgb_to_cam[(r, 2)] /= row_sum;
        }
    }
    // camera -> sRGB, fused (rawler's `cam2rgb`, via pseudo-inverse for
    // the general 4-channel case - a plain inverse is equivalent here
    // since we only ever build a full-rank 3x3 matrix).
    rgb_to_cam.try_inverse().unwrap_or_else(Matrix3::identity)
}

/// Returns the transform required after a normalized camera-RGB model output.
/// RawNIND has already had black level removed and white level normalized, so
/// only camera white balance and camera-RGB → linear sRGB remain.
pub fn normalized_camera_rgb_to_linear_srgb_transform(
    sensor: &RawSensorData,
) -> (Matrix3<f32>, [f32; 3]) {
    (
        camera_to_linear_srgb_matrix(&sensor.xyz_to_cam),
        [
            sensor.wb_coeffs[0],
            sensor.wb_coeffs[1],
            sensor.wb_coeffs[2],
        ],
    )
}

fn apply_color_matrix(rgb: &mut [[f32; 3]], xyz_to_cam: &Matrix3<f32>) {
    let cam_to_srgb = camera_to_linear_srgb_matrix(xyz_to_cam);

    rgb.par_iter_mut().for_each(|px| {
        let v = nalgebra::Vector3::new(px[0], px[1], px[2]);
        let out = cam_to_srgb * v;
        px[0] = out[0];
        px[1] = out[1];
        px[2] = out[2];
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
    let exposure_gain = estimate_exposure_gain(&sensor);
    let effective_iso = ((sensor.iso as f32) * exposure_gain).round() as u32;
    let algo = crate::demosaic_algorithms::select_by_iso(effective_iso);
    let options = DevelopOptions::auto_for_iso(effective_iso, exposure_gain);
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

    let exposure_gain = estimate_exposure_gain(&sensor);
    let effective_iso = ((sensor.iso as f32) * exposure_gain).round() as u32;

    let algo = match demosaic_override {
        None | Some("auto") | Some("") => crate::demosaic_algorithms::select_by_iso(effective_iso),
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
            .unwrap_or_else(|| crate::raw_denoise::suggest_strength_for_iso(effective_iso)),
        sharpen_amount: sharpen_override
            .unwrap_or_else(|| crate::raw_sharpen::suggest_amount_for_iso(effective_iso)),
        sharpen_method,
        exposure_gain,
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
    log::debug!(
        "[raw develop timing] preprocess: {:?}",
        _t_preprocess.elapsed()
    );

    let _t_demosaic = std::time::Instant::now();
    let mut rgb = crate::demosaic_algorithms::demosaic(&sensor, algo);
    log::debug!(
        "[raw develop timing] demosaic ({:?}): {:?}",
        algo,
        _t_demosaic.elapsed()
    );
    let _t_wbcm = std::time::Instant::now();
    apply_white_balance(&mut rgb, &sensor);
    apply_color_matrix(&mut rgb, &sensor.xyz_to_cam);
    log::debug!(
        "[raw develop timing] wb+colormatrix: {:?}",
        _t_wbcm.elapsed()
    );

    // Auto-exposure boost for severely underexposed captures (see
    // `estimate_exposure_gain`) - applied here, before highlight
    // compression, as a flat linear gain so the compression step still
    // gets a chance to soften any highlights the boost pushes over 1.0.
    // A no-op (1.0) for anything normally- or deliberately-exposed.
    if options.exposure_gain != 1.0 {
        let gain = options.exposure_gain;
        rgb.par_iter_mut().for_each(|px| {
            px[0] *= gain;
            px[1] *= gain;
            px[2] *= gain;
        });
    }

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
    apply_color_matrix(&mut rgb, &sensor.xyz_to_cam);
    // Exposure gain must land here - after the (linear) color matrix, before
    // gamma - not via `apply_color_matrix_and_gamma`, which would apply
    // gamma first and make a linear multiply afterward meaningless.
    if options.exposure_gain != 1.0 {
        let gain = options.exposure_gain;
        rgb.par_iter_mut().for_each(|px| {
            px[0] *= gain;
            px[1] *= gain;
            px[2] *= gain;
        });
    }
    rgb.par_iter_mut().for_each(|px| {
        px[0] = srgb_gamma(px[0]);
        px[1] = srgb_gamma(px[1]);
        px[2] = srgb_gamma(px[2]);
    });

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

    #[test]
    fn debug_color_matrix() {
        let path = match std::env::var("RAPIDRAW_TEST_RAW_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("RAPIDRAW_TEST_RAW_PATH not set, skipping");
                return;
            }
        };
        let bytes = std::fs::read(&path).expect("read raw file");
        let source = RawSource::new_from_slice(&bytes);
        let decoder = rawler::get_decoder(&source).expect("get_decoder");
        let raw_image: RawImage = decoder
            .raw_image(&source, &RawDecodeParams::default(), false)
            .expect("raw_image");
        println!("color_matrix entries: {}", raw_image.color_matrix.len());
        for (illuminant, matrix) in raw_image.color_matrix.iter() {
            println!(
                "  {:?}: len={} values={:?}",
                illuminant,
                matrix.len(),
                matrix
            );
        }
        println!("deprecated xyz_to_cam: {:?}", raw_image.xyz_to_cam);
        println!("wb_coeffs: {:?}", raw_image.wb_coeffs);
        let sensor = decode_raw_sensor_data(&bytes).expect("decode_raw_sensor_data");
        println!("resolved sensor.xyz_to_cam: {:?}", sensor.xyz_to_cam);
        println!(
            "black_level: {:?} white_level: {} wb_coeffs: {:?}",
            sensor.black_level, sensor.white_level, sensor.wb_coeffs
        );
        let mut sorted = sensor.data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let pct = |p: f64| sorted[((n - 1) as f64 * p) as usize];
        println!(
            "raw ADU stats: min={} p50={} p90={} p99={} p99.9={} max={}",
            sorted[0],
            pct(0.5),
            pct(0.9),
            pct(0.99),
            pct(0.999),
            sorted[n - 1]
        );
    }
}
