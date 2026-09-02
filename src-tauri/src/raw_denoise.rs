//! Post-demosaic wavelet-domain denoise: luminance and chrominance are
//! decomposed and denoised independently, matching the shape of ART's
//! `RGB_denoise` (`rtengine/ipdenoise.cc`) - separate luminance/chrominance
//! strength, coarse structure left alone so edges/texture aren't smeared.
//!
//! This is a deliberately simplified stand-in for ART's actual algorithm,
//! not a port: ART uses a directional complex wavelet transform
//! (`cplx_wavelet_dec.cc`) with per-subband statistical noise estimation.
//! Here we use the classic "à trous" (undecimated, hole-punched) wavelet
//! transform - a real, standard multiscale decomposition (the same family
//! used in astronomical image denoising), with a fixed per-level
//! attenuation curve rather than statistically-estimated thresholds. Finer
//! levels (which carry most sensor noise) are attenuated more than coarse
//! levels (which carry structure), and chrominance is attenuated more
//! aggressively than luminance at the same `strength`, since chroma noise
//! is both more visually objectionable and less texture-bearing than
//! luminance noise.

const NUM_LEVELS: usize = 4;
// Fine-to-coarse per-level attenuation weight at strength = 1.0. Level 0 is
// the finest (highest-frequency, most noise-like) detail band.
const LUMA_LEVEL_WEIGHTS: [f32; NUM_LEVELS] = [0.9, 0.7, 0.4, 0.15];
const CHROMA_LEVEL_WEIGHTS: [f32; NUM_LEVELS] = [1.0, 0.9, 0.6, 0.3];

/// Suggests a default denoise strength (0..1) from ISO - higher ISO gets
/// more denoising. A starting point, not tuned against a real test set;
/// mirrors `demosaic_algorithms::select_by_iso`'s "starting point" framing.
pub fn suggest_strength_for_iso(iso: u32) -> f32 {
    ((iso as f32 - 100.0) / 6300.0).clamp(0.0, 1.0)
}

/// Separable 5-tap [1,4,6,4,1]/16 "à trous" (hole-punched) smoothing pass -
/// `gap` (1, 2, 4, 8...) controls the hole spacing, doubling at each
/// successive decomposition level so each pass captures coarser structure
/// without downsampling (keeping the transform shift-invariant).
fn atrous_smooth(src: &[f32], w: usize, h: usize, gap: usize) -> Vec<f32> {
    const TAPS: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let gap = gap as isize;

    let mut horizontal = vec![0.0f32; src.len()];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let out_row = &mut horizontal[y * w..(y + 1) * w];
        for (x, out) in out_row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (k, &t) in TAPS.iter().enumerate() {
                let dx = (k as isize - 2) * gap;
                let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                acc += t * row[sx];
            }
            *out = acc;
        }
    }

    let mut vertical = vec![0.0f32; src.len()];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, &t) in TAPS.iter().enumerate() {
                let dy = (k as isize - 2) * gap;
                let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                acc += t * horizontal[sy * w + x];
            }
            vertical[y * w + x] = acc;
        }
    }

    vertical
}

/// Decomposes `plane` into NUM_LEVELS à trous detail bands plus a residual,
/// attenuates each detail band by `level_weights[level] * strength`, and
/// reconstructs in place.
fn denoise_plane(
    plane: &mut [f32],
    w: usize,
    h: usize,
    strength: f32,
    level_weights: &[f32; NUM_LEVELS],
) {
    if strength <= 0.0 {
        return;
    }
    let mut current = plane.to_vec();
    let mut result = vec![0.0f32; plane.len()];

    for (level, &weight) in level_weights.iter().enumerate() {
        let gap = 1usize << level;
        let smoothed = atrous_smooth(&current, w, h, gap);
        let attenuation = 1.0 - (strength * weight).clamp(0.0, 1.0);
        for i in 0..plane.len() {
            let detail = current[i] - smoothed[i];
            result[i] += detail * attenuation;
        }
        current = smoothed;
    }
    for i in 0..plane.len() {
        result[i] += current[i];
    }
    plane.copy_from_slice(&result);
}

/// Denoises a demosaiced RGB buffer (row-major, one `[f32; 3]` per pixel) in
/// place. `strength` is 0..1 (0 = no-op). Luminance and chrominance are
/// denoised independently in YCbCr space so edge/texture-carrying luminance
/// detail is preserved more than chroma detail at the same strength.
pub fn wavelet_denoise(rgb: &mut [[f32; 3]], width: usize, height: usize, strength: f32) {
    if strength <= 0.0 || rgb.is_empty() {
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

    denoise_plane(&mut y_plane, width, height, strength, &LUMA_LEVEL_WEIGHTS);
    denoise_plane(
        &mut cb_plane,
        width,
        height,
        strength,
        &CHROMA_LEVEL_WEIGHTS,
    );
    denoise_plane(
        &mut cr_plane,
        width,
        height,
        strength,
        &CHROMA_LEVEL_WEIGHTS,
    );

    for (i, px) in rgb.iter_mut().enumerate() {
        let (r, g, b) = crate::image_processing::yc_to_rgb(y_plane[i], cb_plane[i], cr_plane[i]);
        px[0] = r.max(0.0);
        px[1] = g.max(0.0);
        px[2] = b.max(0.0);
    }
}
