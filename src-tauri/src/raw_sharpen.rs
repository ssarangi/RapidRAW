//! Post-demosaic capture sharpening: classic unsharp mask on the luminance
//! channel only (never chroma, to avoid color fringing), with an edge-aware
//! blend mask that suppresses sharpening in flat or noisy regions - matching
//! ART's default sharpening method (`rtengine/ipsharpen.cc`'s `unsharp_mask`
//! + `buildBlendMask`; ART also offers Richardson-Lucy deconvolution modes
//! ("rld"/"psf") which are not implemented here - unsharp mask is ART's own
//! default and the simplest to get right first).

/// Suggests a default sharpen amount (0..1) from ISO - lower at high ISO so
/// sharpening doesn't re-amplify noise the denoise stage didn't fully
/// remove. A starting point, not tuned against a real test set.
pub fn suggest_amount_for_iso(iso: u32) -> f32 {
    let base = 0.5f32;
    let falloff = ((iso as f32 - 800.0) / 6400.0).clamp(0.0, 0.35);
    (base - falloff).max(0.15)
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
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let out_row = &mut horizontal[y * w..(y + 1) * w];
        for (x, out) in out_row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (k, &weight) in kernel.iter().enumerate() {
                let dx = k as isize - radius;
                let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                acc += weight * row[sx];
            }
            *out = acc;
        }
    }

    let mut vertical = vec![0.0f32; src.len()];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, &weight) in kernel.iter().enumerate() {
                let dy = k as isize - radius;
                let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                acc += weight * horizontal[sy * w + x];
            }
            vertical[y * w + x] = acc;
        }
    }

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

    for row in 0..h {
        for col in 0..w {
            let idx = row * w + col;
            let left = y[row * w + col.saturating_sub(1)];
            let right = y[row * w + (col + 1).min(w - 1)];
            let up = y[row.saturating_sub(1) * w + col];
            let down = y[(row + 1).min(h - 1) * w + col];
            let gx = right - left;
            let gy = down - up;
            let gradient = (gx * gx + gy * gy).sqrt();
            // Smoothstep-like ramp from 0 to 1 around the threshold.
            let t = (gradient / (threshold * 2.0)).clamp(0.0, 1.0);
            mask[idx] = t * t * (3.0 - 2.0 * t);
        }
    }

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

    let mut y_plane = vec![0.0f32; rgb.len()];
    let mut cb_plane = vec![0.0f32; rgb.len()];
    let mut cr_plane = vec![0.0f32; rgb.len()];
    for (i, px) in rgb.iter().enumerate() {
        let (y, cb, cr) = crate::image_processing::rgb_to_yc_only(px[0], px[1], px[2]);
        y_plane[i] = y;
        cb_plane[i] = cb;
        cr_plane[i] = cr;
    }

    let blurred = gaussian_blur(&y_plane, width, height, radius.max(0.1));
    let blend = build_blend_mask(&y_plane, width, height, 0.02);

    for i in 0..y_plane.len() {
        let detail = y_plane[i] - blurred[i];
        y_plane[i] += detail * amount * 2.0 * blend[i];
    }

    for (i, px) in rgb.iter_mut().enumerate() {
        let (r, g, b) = crate::image_processing::yc_to_rgb(y_plane[i], cb_plane[i], cr_plane[i]);
        px[0] = r.max(0.0);
        px[1] = g.max(0.0);
        px[2] = b.max(0.0);
    }
}
