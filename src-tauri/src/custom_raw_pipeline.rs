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
    rawimage::{RawImage, RawImageData, RawPhotometricInterpretation},
    rawsource::RawSource,
};

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

    let mut xyz_to_cam = Matrix3::identity();
    for r in 0..3 {
        for c in 0..3 {
            xyz_to_cam[(r, c)] = raw_image.xyz_to_cam[r][c];
        }
    }
    // rawler leaves unset rows/matrices as zero; fall back to identity so we
    // don't divide by a singular matrix downstream.
    if xyz_to_cam == Matrix3::zeros() {
        xyz_to_cam = Matrix3::identity();
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
    let sum: f32 = offsets.iter().map(|&(dr, dc)| sample(sensor, row + dr, col + dc)).sum();
    sum / 4.0
}

/// Applies black-level subtraction + white-balance scaling to mosaiced RGB
/// triples (each triple only has one "real" sensor-measured channel per
/// pixel before demosaic, but callers pass already-demosaiced RGB here).
fn apply_white_balance(rgb: &mut [[f32; 3]], sensor: &RawSensorData) {
    let bl = sensor.black_level;
    let wb = sensor.wb_coeffs;
    let wb_max = wb[0].max(wb[1]).max(wb[2]).max(if wb[3] > 0.0 { wb[3] } else { 0.0 });
    let denom = (sensor.white_level - bl[0]).max(1.0);

    for px in rgb.iter_mut() {
        for c in 0..3 {
            let black = bl[c];
            let gain = if wb_max > 0.0 { wb[c] / wb_max } else { 1.0 };
            px[c] = ((px[c] - black) * gain / denom).max(0.0);
        }
    }
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

    for px in rgb.iter_mut() {
        let v = nalgebra::Vector3::new(px[0], px[1], px[2]);
        let out = cam_to_srgb * v;
        px[0] = out[0];
        px[1] = out[1];
        px[2] = out[2];
    }
}

/// Same as `apply_color_matrix`, but also applies sRGB gamma - used only by
/// the standalone test-dump path below, which produces a display-ready
/// image rather than the linear intermediate the live app pipeline expects.
fn apply_color_matrix_and_gamma(rgb: &mut [[f32; 3]], xyz_to_cam: &Matrix3<f32>) {
    apply_color_matrix(rgb, xyz_to_cam);
    for px in rgb.iter_mut() {
        px[0] = srgb_gamma(px[0]);
        px[1] = srgb_gamma(px[1]);
        px[2] = srgb_gamma(px[2]);
    }
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
pub fn develop_raw_image_custom(file_bytes: &[u8], highlight_compression: f32) -> Result<DynamicImage> {
    let sensor = decode_raw_sensor_data(file_bytes)?;
    if !sensor.is_standard_bayer {
        return Err(anyhow!("not a standard Bayer CFA; custom pipeline not applicable"));
    }
    let algo = crate::demosaic_algorithms::select_by_iso(sensor.iso);
    develop_raw_image_custom_with_algorithm(file_bytes, algo, highlight_compression)
}

/// Same as `develop_raw_image_custom`, but with an explicit algorithm choice
/// instead of ISO auto-selection - used by the CLI (`raw develop
/// --demosaic amaze|igv|lmmse|bilinear`) to debug/compare algorithms
/// directly against the exact linear intermediate the live app produces.
pub fn develop_raw_image_custom_with_algorithm(
    file_bytes: &[u8],
    algo: crate::demosaic_algorithms::DemosaicAlgorithm,
    highlight_compression: f32,
) -> Result<DynamicImage> {
    let sensor = decode_raw_sensor_data(file_bytes)?;
    if !sensor.is_standard_bayer {
        return Err(anyhow!("not a standard Bayer CFA; custom pipeline not applicable"));
    }

    let mut rgb = crate::demosaic_algorithms::demosaic(&sensor, algo);
    apply_white_balance(&mut rgb, &sensor);
    apply_color_matrix(&mut rgb, &sensor.xyz_to_cam);

    // Mirrors raw_processing::develop_internal's rescale + highlight
    // compression: apply_white_balance already normalized by
    // (white_level - black_level), so here rescale is a no-op scale of 1.0
    // and we only need the highlight-compression clamp/desaturation.
    let safe_highlight_compression = highlight_compression.max(1.01);
    let clamp_limit = safe_highlight_compression;
    for px in rgb.iter_mut() {
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
                (compressed_r * rescale, compressed_g * rescale, compressed_b * rescale)
            } else {
                (max_c, max_c, max_c)
            }
        } else {
            (r, g, b)
        };
        px[0] = final_r.clamp(0.0, clamp_limit);
        px[1] = final_g.clamp(0.0, clamp_limit);
        px[2] = final_b.clamp(0.0, clamp_limit);
    }

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
    develop_raw_custom_with_algorithm(file_bytes, crate::demosaic_algorithms::DemosaicAlgorithm::Bilinear)
}

pub fn develop_raw_custom_with_algorithm(
    file_bytes: &[u8],
    algo: crate::demosaic_algorithms::DemosaicAlgorithm,
) -> Result<DynamicImage> {
    let sensor = decode_raw_sensor_data(file_bytes)?;
    let mut rgb = crate::demosaic_algorithms::demosaic(&sensor, algo);
    apply_white_balance(&mut rgb, &sensor);
    apply_color_matrix_and_gamma(&mut rgb, &sensor.xyz_to_cam);

    let (crop_x, crop_y, crop_w, crop_h) = match sensor.active_area {
        Some([x, y, w, h]) => (x, y, w, h),
        None => (0, 0, sensor.width, sensor.height),
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
        let dump_dir = std::env::var("RAPIDRAW_TEST_DUMP_DIR").unwrap_or_else(|_| "/tmp".to_string());
        let bytes = std::fs::read(&path).expect("read raw file");

        use crate::demosaic_algorithms::DemosaicAlgorithm;
        for (name, algo) in [
            ("bilinear", DemosaicAlgorithm::Bilinear),
            ("amaze", DemosaicAlgorithm::AMaZE),
            ("igv", DemosaicAlgorithm::IGV),
            ("lmmse", DemosaicAlgorithm::LMMSE),
        ] {
            let img = develop_raw_custom_with_algorithm(&bytes, algo).expect("develop_raw_custom_with_algorithm");
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
        let reference_rgba8 =
            image::ImageBuffer::<image::Rgba<u8>, _>::from_fn(reference.width(), reference.height(), |x, y| {
                let p = reference_f32.get_pixel(x, y).0;
                image::Rgba([
                    (srgb_gamma(p[0]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    (srgb_gamma(p[1]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    (srgb_gamma(p[2]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    255,
                ])
            });
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
        let dump_dir = std::env::var("RAPIDRAW_TEST_DUMP_DIR").unwrap_or_else(|_| "/tmp".to_string());
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
        let wired_rgba8 =
            image::ImageBuffer::<image::Rgba<u8>, _>::from_fn(wired.width(), wired.height(), |x, y| {
                let p = wired_f32.get_pixel(x, y).0;
                image::Rgba([
                    (srgb_gamma(p[0]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    (srgb_gamma(p[1]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    (srgb_gamma(p[2]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    255,
                ])
            });
        let out_path = format!("{dump_dir}/wired_custom_pipeline.png");
        image::DynamicImage::ImageRgba8(wired_rgba8)
            .save(&out_path)
            .expect("save wired dump");
        println!("saved {} ({}x{})", out_path, wired.width(), wired.height());
    }
}
