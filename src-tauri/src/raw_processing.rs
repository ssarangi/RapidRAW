use crate::image_processing::apply_orientation;
use anyhow::{Result, anyhow};
use image::{DynamicImage, ImageBuffer, Rgba};
use rawler::{
    decoders::{Orientation, RawDecodeParams},
    imgop::develop::{DemosaicAlgorithm, Intermediate, ProcessingStep, RawDevelop},
    rawimage::{RawImage, RawPhotometricInterpretation},
    rawsource::RawSource,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

pub fn develop_raw_image(
    file_bytes: &[u8],
    fast_demosaic: bool,
    highlight_compression: f32,
    linear_mode: String,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<DynamicImage> {
    // Opt-in (env-var gated, default off) experimental path: our own
    // AMaZE/IGV/LMMSE demosaic pipeline (see custom_raw_pipeline.rs and
    // CLAUDE.md's "Open TODO: pluggable demosaic algorithms"), used only for
    // full-quality (non-fast) decodes of standard Bayer sensors. Falls back
    // to the normal rawler-PPG pipeline below on any error or ineligible
    // sensor, so this can never break existing behavior when unset.
    if !fast_demosaic && std::env::var("RAPIDRAW_CUSTOM_DEMOSAIC").as_deref() == Ok("1") {
        match crate::custom_raw_pipeline::develop_raw_image_custom(
            file_bytes,
            highlight_compression,
        ) {
            Ok(image) => return Ok(image),
            Err(e) => {
                log::info!(
                    "custom demosaic pipeline unavailable ({e}), falling back to rawler PPG"
                );
            }
        }
    }

    let (developed_image, orientation) = develop_internal(
        file_bytes,
        fast_demosaic,
        highlight_compression,
        linear_mode,
        cancel_token,
    )?;
    Ok(apply_orientation(developed_image, orientation))
}

/// Real "open a RAW file for processing" entry point for every call site
/// that has (or can cheaply load) that image's persisted "Raw Develop"
/// adjustments - the editor's own load path, preview regeneration, export,
/// and now every other `develop_raw_image` caller (thumbnails, culling,
/// HDR, focus stacking, panorama, restoration, negative conversion) that
/// reads a sidecar for the image anyway. Tries our own demosaic/preprocess/
/// denoise/sharpen pipeline first for full-quality decodes of standard
/// Bayer sensors, falling back to the normal rawler-PPG `develop_raw_image`
/// above on any error or ineligible sensor (X-Trans, 4-channel, monochrome,
/// linear RAW) - so an image this pipeline can't handle degrades to the
/// exact prior behavior, never fails.
///
/// `raw_develop_adjustments` is that image's persisted adjustments blob
/// (or `None` for a caller with no adjustments in scope, e.g. because the
/// image isn't part of a catalog) - `rawDemosaicAlgorithm`/
/// `rawDenoiseAmount`/`rawSharpenAmount`/`rawSharpenMethod`/
/// `rawPreprocessEnabled` are read from it here, with `None`/missing/"auto"
/// falling back to ISO-auto behavior, exactly as if no override were set.
///
/// `allow_custom_pipeline` is a separate gate from `fast_demosaic`: it lets
/// a caller force the fast rawler-PPG path even for a full-quality decode -
/// used for the editor's "phase 1" fast-first-paint load (see
/// `image_loader::load_image`), which needs a fast, full-resolution
/// decode (unlike `fast_demosaic`, which is rawler's quarter-resolution
/// thumbnail mode), with the slower custom pipeline applied moments later
/// as a background "phase 2" upgrade once it's ready.
pub fn develop_raw_image_for_editor(
    file_bytes: &[u8],
    fast_demosaic: bool,
    highlight_compression: f32,
    linear_mode: String,
    raw_develop_adjustments: Option<&serde_json::Value>,
    allow_custom_pipeline: bool,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<DynamicImage> {
    let raw_nind_enabled = raw_develop_adjustments
        .and_then(|adjustments| adjustments.get("rawAiDenoiseEnabled"))
        .and_then(|value| value.as_bool())
        .unwrap_or(false);

    // RawNIND replaces demosaic for this development pass: it denoises the
    // Bayer mosaic and emits RGB directly. This remains entirely in memory,
    // so toggling it is a normal non-destructive RAW Develop adjustment.
    if raw_nind_enabled && !fast_demosaic && allow_custom_pipeline {
        let sensor = crate::custom_raw_pipeline::decode_raw_sensor_data(file_bytes)?;
        if !sensor.is_standard_bayer {
            return Err(anyhow!(
                "AI sensor denoise requires a standard Bayer RAW sensor"
            ));
        }
        let model_path = crate::image_loader::raw_nind_model_path()?;
        let image = crate::image_restoration::develop_rawnind_in_memory(file_bytes, &model_path)
            .map_err(|error| anyhow!(error))?;
        let image = crate::custom_raw_pipeline::crop_sensor_develop_image(image, &sensor);
        return Ok(apply_orientation(image, sensor.orientation));
    }

    let demosaic_override = raw_develop_adjustments
        .and_then(|a| a.get("rawDemosaicAlgorithm"))
        .and_then(|v| v.as_str());
    // "ppg" is rawler's own demosaic (not one of our AMaZE/IGV/LMMSE/
    // Bilinear algorithms) - an explicit user choice to skip this pipeline
    // entirely and fall straight through to the rawler-PPG path below,
    // rather than an override `develop_raw_image_custom_resolved` could
    // ever satisfy itself.
    let wants_ppg = demosaic_override == Some("ppg");

    if !fast_demosaic && allow_custom_pipeline && !wants_ppg {
        // Sliders store 0..100 with a negative sentinel meaning "auto"
        // (ISO-based suggestion) - matches how other percentage-style
        // adjustments in this app are already stored.
        let denoise_override = raw_develop_adjustments
            .and_then(|a| a.get("rawDenoiseAmount"))
            .and_then(|v| v.as_f64())
            .filter(|v| *v >= 0.0)
            .map(|v| (v as f32 / 100.0).clamp(0.0, 1.0));
        let sharpen_override = raw_develop_adjustments
            .and_then(|a| a.get("rawSharpenAmount"))
            .and_then(|v| v.as_f64())
            .filter(|v| *v >= 0.0)
            .map(|v| (v as f32 / 100.0).clamp(0.0, 1.0));
        let sharpen_method_override = raw_develop_adjustments
            .and_then(|a| a.get("rawSharpenMethod"))
            .and_then(|v| v.as_str());
        let preprocess = raw_develop_adjustments
            .and_then(|a| a.get("rawPreprocessEnabled"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        match crate::custom_raw_pipeline::develop_raw_image_custom_resolved(
            file_bytes,
            highlight_compression,
            demosaic_override,
            denoise_override,
            sharpen_override,
            sharpen_method_override,
            preprocess,
        ) {
            Ok(image) => return Ok(image),
            Err(e) => {
                log::info!(
                    "custom raw develop pipeline unavailable ({e}), falling back to rawler PPG"
                );
            }
        }
    }

    develop_raw_image(
        file_bytes,
        fast_demosaic,
        highlight_compression,
        linear_mode,
        cancel_token,
    )
}

fn is_linear_raw_format(raw_image: &RawImage) -> bool {
    matches!(
        raw_image.photometric,
        RawPhotometricInterpretation::LinearRaw
    )
}

#[inline]
fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(3.0)
    }
}

fn develop_internal(
    file_bytes: &[u8],
    fast_demosaic: bool,
    highlight_compression: f32,
    linear_mode: String,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<(DynamicImage, Orientation)> {
    let check_cancel = || -> Result<()> {
        if let Some((tracker, generation)) = &cancel_token
            && tracker.load(Ordering::SeqCst) != *generation
        {
            return Err(anyhow!("Load cancelled"));
        }
        Ok(())
    };

    check_cancel()?;

    let source = RawSource::new_from_slice(file_bytes);
    let decoder = rawler::get_decoder(&source)?;

    check_cancel()?;
    let mut raw_image: RawImage = decoder.raw_image(&source, &RawDecodeParams::default(), false)?;

    let metadata = decoder.raw_metadata(&source, &RawDecodeParams::default())?;
    let orientation = metadata
        .exif
        .orientation
        .map(Orientation::from_u16)
        .unwrap_or(Orientation::Normal);

    let is_linear_format = is_linear_raw_format(&raw_image);

    let (apply_ungamma, apply_calibration) = match linear_mode.as_str() {
        "gamma" => (true, true),
        "skip_calib" => (false, false),
        "gamma_skip_calib" => (true, false),
        _ => (false, true),
    };

    let original_white_level = raw_image
        .whitelevel
        .0
        .first()
        .cloned()
        .unwrap_or(u16::MAX as u32) as f32;
    let original_black_level = raw_image
        .blacklevel
        .levels
        .first()
        .map(|r| r.as_f32())
        .unwrap_or(0.0);

    for level in raw_image.whitelevel.0.iter_mut() {
        *level = u32::MAX;
    }

    let mut developer = RawDevelop::default();

    if is_linear_format {
        developer.steps.retain(|&step| {
            step != ProcessingStep::SRgb
                && step != ProcessingStep::Demosaic
                && (apply_calibration || step != ProcessingStep::Calibrate)
        });
    } else if fast_demosaic {
        developer.demosaic_algorithm = DemosaicAlgorithm::Speed;
        developer.steps.retain(|&step| step != ProcessingStep::SRgb);
    } else {
        developer.steps.retain(|&step| step != ProcessingStep::SRgb);
    }

    raw_image.wb_coeffs =
        crate::multi_exposure::neutralize_wb_if_multiexposure(raw_image.wb_coeffs, file_bytes);

    check_cancel()?;
    let mut developed_intermediate = developer.develop_intermediate(&raw_image)?;

    drop(raw_image);

    let denominator = (original_white_level - original_black_level).max(1.0);
    let rescale_factor = (u32::MAX as f32 - original_black_level) / denominator;

    let safe_highlight_compression = highlight_compression.max(1.01);
    let clamp_limit = safe_highlight_compression;

    check_cancel()?;

    match &mut developed_intermediate {
        Intermediate::Monochrome(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                let mut linear_val = *p * rescale_factor;
                if is_linear_format && apply_ungamma {
                    linear_val = srgb_to_linear(linear_val.clamp(0.0, 1.0));
                }
                *p = linear_val.clamp(0.0, clamp_limit);
            });
        }
        Intermediate::ThreeColor(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                let mut r = (p[0] * rescale_factor).max(0.0);
                let mut g = (p[1] * rescale_factor).max(0.0);
                let mut b = (p[2] * rescale_factor).max(0.0);

                if is_linear_format && apply_ungamma {
                    r = srgb_to_linear(r.clamp(0.0, 1.0));
                    g = srgb_to_linear(g.clamp(0.0, 1.0));
                    b = srgb_to_linear(b.clamp(0.0, 1.0));
                }

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

                p[0] = final_r.clamp(0.0, clamp_limit);
                p[1] = final_g.clamp(0.0, clamp_limit);
                p[2] = final_b.clamp(0.0, clamp_limit);
            });
        }
        Intermediate::FourColor(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                p.iter_mut().for_each(|c| {
                    let mut linear_val = *c * rescale_factor;
                    if is_linear_format && apply_ungamma {
                        linear_val = srgb_to_linear(linear_val.clamp(0.0, 1.0));
                    }
                    *c = linear_val.clamp(0.0, clamp_limit);
                });
            });
        }
    }

    let (width, height) = {
        let dim = developed_intermediate.dim();
        (dim.w as u32, dim.h as u32)
    };

    check_cancel()?;

    let dynamic_image = match developed_intermediate {
        Intermediate::ThreeColor(pixels) => {
            let buffer = ImageBuffer::<Rgba<f32>, _>::from_fn(width, height, |x, y| {
                let p = pixels.data[(y * width + x) as usize];
                Rgba([p[0], p[1], p[2], 1.0])
            });
            DynamicImage::ImageRgba32F(buffer)
        }
        Intermediate::Monochrome(pixels) => {
            let buffer = ImageBuffer::<Rgba<f32>, _>::from_fn(width, height, |x, y| {
                let p = pixels.data[(y * width + x) as usize];
                Rgba([p, p, p, 1.0])
            });
            DynamicImage::ImageRgba32F(buffer)
        }
        _ => {
            return Err(anyhow!("Unsupported intermediate format for conversion"));
        }
    };

    Ok((dynamic_image, orientation))
}

pub fn get_fast_demosaic_scale_factor(
    file_bytes: &[u8],
    decoded_width: u32,
    decoded_height: u32,
) -> f32 {
    let source = RawSource::new_from_slice(file_bytes);
    if let Ok(decoder) = rawler::get_decoder(&source)
        && let Ok(raw_img) = decoder.raw_image(&source, &RawDecodeParams::default(), true)
    {
        let max_orig = (raw_img.width as f32).max(raw_img.height as f32);
        let max_comp = (decoded_width as f32).max(decoded_height as f32);
        if max_orig > 0.0 {
            let ratio = max_comp / max_orig;
            if ratio > 0.1 && ratio < 0.35 {
                return 0.25;
            } else if (0.35..0.75).contains(&ratio) {
                return 0.5;
            }
        }
    }
    1.0
}

/// Reads a RAW file's pixel dimensions via rawler's fast (non-demosaic) decode
/// path - the same one used by `get_fast_demosaic_scale_factor` - so callers can
/// learn the image's aspect ratio near-instantly, long before a full decode
/// (which can take several seconds) completes. Swaps width/height for a 90/270
/// EXIF orientation so the result matches what the real decode eventually
/// produces (the real decode always ends with `apply_orientation`) - this used
/// to be documented as a known sensor-native-only limitation, but once a
/// caller actually renders a box sized from this result (rather than just
/// using it for an early aspect-ratio guess that gets silently replaced), a
/// portrait file reported as landscape stretches the on-screen image visibly,
/// so it's worth getting right here too.
pub fn get_fast_raw_dimensions(file_bytes: &[u8]) -> Option<(u32, u32)> {
    let source = RawSource::new_from_slice(file_bytes);
    let decoder = rawler::get_decoder(&source).ok()?;
    let raw_img = decoder
        .raw_image(&source, &RawDecodeParams::default(), true)
        .ok()?;
    let (width, height) = (raw_img.width as u32, raw_img.height as u32);

    let orientation = decoder
        .raw_metadata(&source, &RawDecodeParams::default())
        .ok()
        .and_then(|metadata| metadata.exif.orientation)
        .map(Orientation::from_u16)
        .unwrap_or(Orientation::Normal);

    match orientation {
        Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Transpose
        | Orientation::Transverse => Some((height, width)),
        _ => Some((width, height)),
    }
}
