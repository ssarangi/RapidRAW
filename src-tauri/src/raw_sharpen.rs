//! Post-demosaic capture sharpening on the luminance channel only (never
//! chroma, to avoid color fringing), with an edge-aware blend mask that
//! suppresses sharpening in flat or noisy regions. Two methods, matching
//! ART's own choices (`rtengine/ipsharpen.cc`): classic unsharp mask
//! (ART's default) and Richardson-Lucy deconvolution ("rld" in ART; ART's
//! third method, "psf", uses a real measured per-lens PSF and is not
//! implemented here).

use rayon::prelude::*;

/// Which sharpening algorithm to use - mirrors `demosaic_algorithms`'s
/// naming/parsing pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharpenMethod {
    UnsharpMask,
    RlDeconvolution,
}

pub fn parse_method_name(name: &str) -> Option<SharpenMethod> {
    match name.to_ascii_lowercase().as_str() {
        "unsharp" | "usm" => Some(SharpenMethod::UnsharpMask),
        "rld" | "deconvolution" | "rl" => Some(SharpenMethod::RlDeconvolution),
        _ => None,
    }
}

pub fn method_name(method: SharpenMethod) -> &'static str {
    match method {
        SharpenMethod::UnsharpMask => "unsharp",
        SharpenMethod::RlDeconvolution => "rld",
    }
}

/// Suggests a default sharpen amount (0..1) from ISO - lower at high ISO so
/// sharpening doesn't re-amplify noise the denoise stage didn't fully
/// remove. A starting point, not tuned against a real test set.
pub fn suggest_amount_for_iso(iso: u32) -> f32 {
    let base = 0.5f32;
    let falloff = ((iso as f32 - 800.0) / 6400.0).clamp(0.0, 0.35);
    (base - falloff).max(0.15)
}

/// Splits a demosaiced RGB buffer into independent Y/Cb/Cr planes.
fn split_yc(rgb: &[[f32; 3]]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let planes: Vec<(f32, f32, f32)> = rgb
        .par_iter()
        .map(|px| crate::image_processing::rgb_to_yc_only(px[0], px[1], px[2]))
        .collect();
    let y_plane = planes.iter().map(|p| p.0).collect();
    let cb_plane = planes.iter().map(|p| p.1).collect();
    let cr_plane = planes.iter().map(|p| p.2).collect();
    (y_plane, cb_plane, cr_plane)
}

/// Recombines Y/Cb/Cr planes back into `rgb` in place.
fn merge_yc(rgb: &mut [[f32; 3]], y_plane: &[f32], cb_plane: &[f32], cr_plane: &[f32]) {
    rgb.par_iter_mut().enumerate().for_each(|(i, px)| {
        let (r, g, b) = crate::image_processing::yc_to_rgb(y_plane[i], cb_plane[i], cr_plane[i]);
        px[0] = r.max(0.0);
        px[1] = g.max(0.0);
        px[2] = b.max(0.0);
    });
}

/// Dispatches to the selected sharpening method - the entry point
/// `custom_raw_pipeline.rs` should call.
pub fn sharpen(
    rgb: &mut [[f32; 3]],
    width: usize,
    height: usize,
    method: SharpenMethod,
    amount: f32,
    radius: f32,
) {
    match method {
        SharpenMethod::UnsharpMask => unsharp_mask(rgb, width, height, amount, radius),
        SharpenMethod::RlDeconvolution => rl_deconv_sharpen(rgb, width, height, amount, radius),
    }
}

/// Separable Gaussian blur, sigma in pixels. Kernel radius follows the
/// common 3*sigma rule of thumb.
fn gaussian_blur(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let radius = (sigma * 3.0).ceil().max(1.0) as isize;
    let mut kernel = vec![0.0f32; (2 * radius + 1) as usize];
    let mut sum = 0.0;
    for (i, k) in kernel.iter_mut().enumerate() {
        let x = i as isize - radius;
        *k = (-((x * x) as f32) / (2.0 * sigma * sigma)).exp();
        sum += *k;
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    let mut horizontal = vec![0.0f32; src.len()];
    horizontal
        .par_chunks_mut(w)
        .zip(src.par_chunks(w))
        .for_each(|(out_row, row)| {
            for (x, out) in out_row.iter_mut().enumerate() {
                let mut acc = 0.0;
                for (k, &weight) in kernel.iter().enumerate() {
                    let dx = k as isize - radius;
                    let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                    acc += weight * row[sx];
                }
                *out = acc;
            }
        });

    let mut vertical = vec![0.0f32; src.len()];
    vertical
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, out_row)| {
            for (x, out) in out_row.iter_mut().enumerate() {
                let mut acc = 0.0;
                for (k, &weight) in kernel.iter().enumerate() {
                    let dy = k as isize - radius;
                    let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                    acc += weight * horizontal[sy * w + x];
                }
                *out = acc;
            }
        });

    vertical
}

/// Local-contrast-based blend mask: 0 in flat/noisy regions (where
/// sharpening would only amplify noise), approaching 1 near real edges.
/// `contrast_threshold` is the local-gradient magnitude (in the same units
/// as `y`, roughly 0..1 for our linear-ish pipeline) at which the mask
/// reaches ~0.5.
fn build_blend_mask(y: &[f32], w: usize, h: usize, contrast_threshold: f32) -> Vec<f32> {
    let mut mask = vec![1.0f32; y.len()];
    let threshold = contrast_threshold.max(1e-4);

    mask.par_chunks_mut(w).enumerate().for_each(|(row, out_row)| {
        for (col, out) in out_row.iter_mut().enumerate() {
            let left = y[row * w + col.saturating_sub(1)];
            let right = y[row * w + (col + 1).min(w - 1)];
            let up = y[row.saturating_sub(1) * w + col];
            let down = y[(row + 1).min(h - 1) * w + col];
            let gx = right - left;
            let gy = down - up;
            let gradient = (gx * gx + gy * gy).sqrt();
            // Smoothstep-like ramp from 0 to 1 around the threshold.
            let t = (gradient / (threshold * 2.0)).clamp(0.0, 1.0);
            *out = t * t * (3.0 - 2.0 * t);
        }
    });

    mask
}

/// Sharpens a demosaiced RGB buffer (row-major, one `[f32; 3]` per pixel) in
/// place via luminance-only unsharp mask. `amount` is 0..1 (0 = no-op);
/// `radius` is the Gaussian blur sigma in pixels (ART's default is small,
/// around 0.5-1.0 for typical preview resolutions - larger values sharpen
/// coarser detail).
pub fn unsharp_mask(rgb: &mut [[f32; 3]], width: usize, height: usize, amount: f32, radius: f32) {
    if amount <= 0.0 || rgb.is_empty() {
        return;
    }

    let (mut y_plane, cb_plane, cr_plane) = split_yc(rgb);

    let blurred = gaussian_blur(&y_plane, width, height, radius.max(0.1));
    let blend = build_blend_mask(&y_plane, width, height, 0.02);

    for i in 0..y_plane.len() {
        let detail = y_plane[i] - blurred[i];
        y_plane[i] += detail * amount * 2.0 * blend[i];
    }

    merge_yc(rgb, &y_plane, &cb_plane, &cr_plane);
}

/// Sharpens via Richardson-Lucy deconvolution against a symmetric Gaussian
/// PSF - a real iterative deblur (distinct from unsharp mask's single-pass
/// blur-and-subtract), simplified from ART's `deconvsharpening`
/// (`rtengine/ipsharpen.cc`, ART's "rld" method): same classic single-
/// channel RL update (`estimate *= blur(observed / blur(estimate))`,
/// repeated `MAX_ITER` times), mixed back with the original luminance via
/// the same edge-aware blend mask `unsharp_mask` uses, scaled by `amount`.
///
/// Two things ART's version does that this doesn't: a per-pixel divergence
/// check that stops iterating early once a pixel's estimate has moved more
/// than 20% from its original value (ART's `check_stop`/`delta_factor`,
/// which mainly guards against ringing on precisely the same lines the
/// shared blend mask already suppresses), and a dedicated impulse-noise
/// exclusion mask (ART's `markImpulse`) - skipped here because
/// `raw_preprocess::correct_hot_dead_pixels` already corrects impulse noise
/// before demosaic even runs, so there should be little left for it to
/// protect against by this stage.
pub fn rl_deconv_sharpen(
    rgb: &mut [[f32; 3]],
    width: usize,
    height: usize,
    amount: f32,
    radius: f32,
) {
    if amount <= 0.0 || rgb.is_empty() || radius < 0.2 {
        return;
    }
    const MAX_ITER: usize = 20;
    // Keeps the multiplicative RL update well away from zero/negative
    // territory - our working values are roughly 0..~few (post white-
    // balance, pre-crop), unlike ART's 16-bit-ish scale, so a small
    // additive offset is enough (ART uses 1000.0 against its own scale).
    const OFFSET: f32 = 1.0;
    let sigma = radius.max(0.2);

    let (mut y_plane, cb_plane, cr_plane) = split_yc(rgb);

    let observed: Vec<f32> = y_plane.iter().map(|&v| v + OFFSET).collect();
    let mut estimate = observed.clone();

    for _ in 0..MAX_ITER {
        let blurred_estimate = gaussian_blur(&estimate, width, height, sigma);
        let ratio: Vec<f32> = observed
            .iter()
            .zip(blurred_estimate.iter())
            .map(|(&o, &b)| o / b.max(1e-6))
            .collect();
        let correction = gaussian_blur(&ratio, width, height, sigma);
        for (e, c) in estimate.iter_mut().zip(correction.iter()) {
            *e = (*e * c).max(0.0);
        }
    }

    let blend = build_blend_mask(&y_plane, width, height, 0.02);
    for i in 0..y_plane.len() {
        let deconvolved = estimate[i] - OFFSET;
        let w = (blend[i] * amount).clamp(0.0, 1.0);
        y_plane[i] += (deconvolved - y_plane[i]) * w;
    }

    merge_yc(rgb, &y_plane, &cb_plane, &cr_plane);
}
