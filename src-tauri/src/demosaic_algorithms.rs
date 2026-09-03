//! Custom Bayer demosaic algorithms - AMaZE, IGV, and LMMSE - implemented
//! entirely in our own code (see custom_raw_pipeline.rs and CLAUDE.md's
//! "Open TODO: pluggable demosaic algorithms" for why: no rawler fork).
//!
//! Verification status (checked directly against reference source, not just
//! literature/memory - see CLAUDE.md for the full writeup):
//! - AMaZE and IGV's core directional green-interpolation formula (the
//!   asymmetric 4-direction 5-tap Hamilton-Adams estimate in
//!   `directional_estimate` below) is transcribed from RawTherapee's actual
//!   amaze_demosaic_RT.cc and demosaic_algos.cc (igv_interpolate).
//! - LMMSE's Gaussian smoothing filter (`lmmse_gaussian_taps`) uses the
//!   exact weights from RawTherapee's lmmse_demosaic.cc, applied to the
//!   color-difference plane as RT does (an earlier version of this file
//!   smoothed raw pixel values with an invented filter - wrong, since fixed).
//! - None of the three is a full line-for-line port: AMaZE (1600+ lines) has
//!   additional Nyquist-aliasing detection and iterative refinement passes
//!   not implemented here; IGV's real refinement stage integrates a
//!   Gaussian-weighted variance over a ±6-pixel neighborhood of the
//!   color-difference planes, replaced here with a simpler direct-neighbor
//!   averaging pass; LMMSE's H/V combination logic past the Gaussian
//!   smoothing step (RT has further passes) was approximated with a local-
//!   variance-weighted average rather than ported in full.
//!
//! All three build on the same two-stage structure, which is standard
//! practice for high-quality Bayer demosaic:
//! 1. Build a full-resolution GREEN plane (green is known at half the
//!    pixels; interpolate it at the rest). This is where the algorithms
//!    differ from each other.
//! 2. Build RED and BLUE planes via color-difference interpolation
//!    (R-G and B-G, which vary much more smoothly than R and B themselves,
//!    interpolated then added back to green) - shared by all three.

use crate::custom_raw_pipeline::RawSensorData;
use rawler::cfa::CFAColor;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemosaicAlgorithm {
    /// Plain bilinear - kept as a fast fallback (e.g. tiny ROIs).
    Bilinear,
    /// Directional, gradient-adaptive green interpolation, hard-selecting
    /// the lower-gradient axis (interpolate along an edge, not across it) -
    /// AMaZE's central aliasing/zipper-avoiding idea, built on the same
    /// verified Hamilton-Adams estimate IGV uses (see module docs above).
    AMaZE,
    /// Same directional estimates as AMaZE, combined via RT's actual
    /// gradient-cross-weighted vdif/hdif formula (soft, not hard-selected),
    /// plus a refinement pass against known green neighbors approximating
    /// IGV's real (and more elaborate) iterative refinement stage.
    IGV,
    /// Zhang & Wu (2005) LMMSE demosaicking: directional color-difference
    /// estimates smoothed with RT's actual 9-tap Gaussian kernel, then
    /// combined by local-variance-weighted averaging. Tends to be the
    /// smoothest/most noise-resistant of the three, which is why it's
    /// selected for high-ISO images.
    LMMSE,
}

/// Chooses a demosaic algorithm from the shot ISO. LMMSE/IGV favor
/// noise-smoothness at high ISO; AMaZE favors detail/sharpness otherwise.
/// Thresholds are a starting point, not tuned against real-world test sets.
pub fn select_by_iso(iso: u32) -> DemosaicAlgorithm {
    if iso >= 1600 {
        DemosaicAlgorithm::LMMSE
    } else if iso >= 800 {
        DemosaicAlgorithm::IGV
    } else {
        DemosaicAlgorithm::AMaZE
    }
}

/// Parses a CLI/debug-facing algorithm name (case-insensitive). Returns
/// `None` for "auto" (meaning: caller should use `select_by_iso`) or an
/// unrecognized name.
pub fn parse_algorithm_name(name: &str) -> Option<DemosaicAlgorithm> {
    match name.to_ascii_lowercase().as_str() {
        "bilinear" => Some(DemosaicAlgorithm::Bilinear),
        "amaze" => Some(DemosaicAlgorithm::AMaZE),
        "igv" => Some(DemosaicAlgorithm::IGV),
        "lmmse" => Some(DemosaicAlgorithm::LMMSE),
        _ => None,
    }
}

pub fn algorithm_name(algo: DemosaicAlgorithm) -> &'static str {
    match algo {
        DemosaicAlgorithm::Bilinear => "bilinear",
        DemosaicAlgorithm::AMaZE => "amaze",
        DemosaicAlgorithm::IGV => "igv",
        DemosaicAlgorithm::LMMSE => "lmmse",
    }
}

pub fn demosaic(sensor: &RawSensorData, algo: DemosaicAlgorithm) -> Vec<[f32; 3]> {
    let green = match algo {
        DemosaicAlgorithm::Bilinear => {
            return crate::custom_raw_pipeline::bilinear_demosaic(sensor);
        }
        DemosaicAlgorithm::AMaZE => amaze_green_plane(sensor),
        DemosaicAlgorithm::IGV => igv_green_plane(sensor),
        DemosaicAlgorithm::LMMSE => lmmse_green_plane(sensor),
    };

    let (red_diff, blue_diff) = rayon::join(
        || interpolate_color_diff(sensor, &green, CFAColor::RED),
        || interpolate_color_diff(sensor, &green, CFAColor::BLUE),
    );

    let (w, h) = (sensor.width, sensor.height);
    let mut out = vec![[0.0f32; 3]; w * h];
    out.par_iter_mut().enumerate().for_each(|(i, px)| {
        *px = [green[i] + red_diff[i], green[i], green[i] + blue_diff[i]];
    });
    out
}

#[inline]
fn sample(sensor: &RawSensorData, row: isize, col: isize) -> f32 {
    let row = row.clamp(0, sensor.height as isize - 1) as usize;
    let col = col.clamp(0, sensor.width as isize - 1) as usize;
    sensor.data[row * sensor.width + col]
}

/// The four (N/E/W/S) directional green estimates + their gradient weights
/// at a red/blue sensor site, each verified against RawTherapee's actual
/// source (amaze_demosaic_RT.cc, which explicitly labels this step "G
/// interpolated in vert/hor directions using Hamilton-Adams method", and
/// demosaic_algos.cc's igv_interpolate, where all four formulas below are
/// taken from directly). Each direction's formula is asymmetric (3 taps
/// biased toward that direction, 1 "closing" tap on the opposite side) -
/// deliberately NOT simplified to a symmetric H/V pair, since that would be
/// a different (and wrong) formula, not an equivalent simplification.
/// Divided by the coefficient sum (48) rather than RT's `48*65535`, since
/// this pipeline works in raw ADU units, not values pre-normalized to
/// 0..65535.
struct DirectionalEstimate {
    n: f32,
    e: f32,
    w: f32,
    s: f32,
    /// Gradient/confidence weight per direction (RT's ng/eg/wg/sg): lower
    /// means a smoother, more trustworthy estimate in that direction.
    grad_n: f32,
    grad_e: f32,
    grad_w: f32,
    grad_s: f32,
}

fn directional_estimate(sensor: &RawSensorData, row: isize, col: isize) -> DirectionalEstimate {
    let center = sample(sensor, row, col);
    let g = |dr: isize, dc: isize| sample(sensor, row + dr, col + dc);
    let eps = 1e-5;

    let n = (23.0 * g(-1, 0) + 23.0 * g(-3, 0) + g(-5, 0) + g(1, 0) + 40.0 * center
        - 32.0 * g(-2, 0)
        - 8.0 * g(-4, 0))
        / 48.0;
    let s = (23.0 * g(1, 0) + 23.0 * g(3, 0) + g(5, 0) + g(-1, 0) + 40.0 * center
        - 32.0 * g(2, 0)
        - 8.0 * g(4, 0))
        / 48.0;
    let e = (23.0 * g(0, 1) + 23.0 * g(0, 3) + g(0, 5) + g(0, -1) + 40.0 * center
        - 32.0 * g(0, 2)
        - 8.0 * g(0, 4))
        / 48.0;
    let w = (23.0 * g(0, -1) + 23.0 * g(0, -3) + g(0, -5) + g(0, 1) + 40.0 * center
        - 32.0 * g(0, -2)
        - 8.0 * g(0, -4))
        / 48.0;

    let grad_n = eps + (g(-1, 0) - g(-3, 0)).abs() + (center - g(-2, 0)).abs();
    let grad_s = eps + (g(1, 0) - g(3, 0)).abs() + (center - g(2, 0)).abs();
    let grad_e = eps + (g(0, 1) - g(0, 3)).abs() + (center - g(0, 2)).abs();
    let grad_w = eps + (g(0, -1) - g(0, -3)).abs() + (center - g(0, -2)).abs();

    DirectionalEstimate {
        n,
        e,
        w,
        s,
        grad_n,
        grad_e,
        grad_w,
        grad_s,
    }
}

fn amaze_green_plane(sensor: &RawSensorData) -> Vec<f32> {
    let (w, h) = (sensor.width, sensor.height);
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(row, out_row)| {
        for (col, out_px) in out_row.iter_mut().enumerate() {
            let idx = row * w + col;
            if (sensor.cfa_at)(row, col) == CFAColor::GREEN {
                *out_px = sensor.data[idx];
                continue;
            }
            let e = directional_estimate(sensor, row as isize, col as isize);
            let grad_v = e.grad_n + e.grad_s;
            let grad_h = e.grad_e + e.grad_w;
            // Hard direction choice: interpolate along the lower-gradient
            // axis (i.e. along the edge, not across it) - this is the
            // aliasing/zipper-avoiding behavior AMaZE is built around.
            *out_px = if grad_h < grad_v {
                (e.grad_w * e.e + e.grad_e * e.w) / (e.grad_e + e.grad_w)
            } else {
                (e.grad_s * e.n + e.grad_n * e.s) / (e.grad_n + e.grad_s)
            };
        }
    });
    out
}

fn igv_green_plane(sensor: &RawSensorData) -> Vec<f32> {
    let (w, h) = (sensor.width, sensor.height);
    let mut out = vec![0.0f32; w * h];
    let mut is_interpolated = vec![false; w * h];

    // Pass 1: RT's actual vdif/hdif combination - each pair (N/S, E/W)
    // cross-weighted by the OTHER direction's gradient (a lower gradient
    // means "trust this direction's neighbor more"), verified against
    // demosaic_algos.cc's igv_interpolate.
    out.par_chunks_mut(w)
        .zip(is_interpolated.par_chunks_mut(w))
        .enumerate()
        .for_each(|(row, (out_row, interp_row))| {
            for (col, out_px) in out_row.iter_mut().enumerate() {
                let idx = row * w + col;
                if (sensor.cfa_at)(row, col) == CFAColor::GREEN {
                    *out_px = sensor.data[idx];
                    continue;
                }
                let d = directional_estimate(sensor, row as isize, col as isize);
                let v = (d.grad_s * d.n + d.grad_n * d.s) / (d.grad_n + d.grad_s);
                let hh = (d.grad_w * d.e + d.grad_e * d.w) / (d.grad_e + d.grad_w);
                let gv = 1.0 / (d.grad_n + d.grad_s);
                let gh = 1.0 / (d.grad_e + d.grad_w);
                *out_px = (gh * v + gv * hh) / (gh + gv);
                interp_row[col] = true;
            }
        });

    // Pass 2: refine each interpolated site against its already-known
    // green neighbors - a cheap stand-in for the "integrated Gaussian
    // vector over variance" refinement pass RT does over a wider (±6)
    // neighborhood of the color-difference planes, which was not ported
    // in full given its complexity.
    let pass1 = out.clone();
    out.par_chunks_mut(w).enumerate().for_each(|(row, out_row)| {
        for (col, out_px) in out_row.iter_mut().enumerate() {
            let idx = row * w + col;
            if !is_interpolated[idx] {
                continue;
            }
            let mut sum = pass1[idx];
            let mut n = 1.0;
            for (dr, dc) in [(-1_isize, 0_isize), (1, 0), (0, -1), (0, 1)] {
                let nr = row as isize + dr;
                let nc = col as isize + dc;
                let nidx_row = nr.clamp(0, h as isize - 1) as usize;
                let nidx_col = nc.clamp(0, w as isize - 1) as usize;
                let nidx = nidx_row * w + nidx_col;
                if (sensor.cfa_at)(nidx_row, nidx_col) == CFAColor::GREEN {
                    sum += sensor.data[nidx];
                    n += 1.0;
                }
            }
            *out_px = sum / n;
        }
    });

    out
}

// Verified against RawTherapee's actual lmmse_demosaic.cc (rtengine): these
// are the real 9-tap Gaussian weights (h0..h4 = exp(-k^2/8) for k=0..4,
// normalized so h0 + 2*(h1+h2+h3+h4) == 1), NOT an invented filter. RT
// applies this to smooth the (green - same-channel) DIFFERENCE plane, which
// is the actual source of LMMSE's noise-resistance at high ISO - smoothing
// raw pixel values directly (what this file did before checking against the
// reference source) is a different, weaker operation.
fn lmmse_gaussian_taps() -> [f32; 5] {
    let h0 = 1.0f32;
    let h1 = (-1.0f32 / 8.0).exp();
    let h2 = (-4.0f32 / 8.0).exp();
    let h3 = (-9.0f32 / 8.0).exp();
    let h4 = (-16.0f32 / 8.0).exp();
    let hs = h0 + 2.0 * (h1 + h2 + h3 + h4);
    [h0 / hs, h1 / hs, h2 / hs, h3 / hs, h4 / hs]
}

/// Estimate of (green - same_channel) at a red/blue site, one direction at a
/// time - the Laplacian-style formula RT uses before smoothing (`-0.25 *
/// (C[-2]+C[2]) + 0.5*(G[-1]+C[0]+G[1]) - C[0]`, algebraically simplified).
fn diff_estimate(sensor: &RawSensorData, row: isize, col: isize, horizontal: bool) -> f32 {
    let center = sample(sensor, row, col);
    let (g_a, g_b, c_a, c_b) = if horizontal {
        (
            sample(sensor, row, col - 1),
            sample(sensor, row, col + 1),
            sample(sensor, row, col - 2),
            sample(sensor, row, col + 2),
        )
    } else {
        (
            sample(sensor, row - 1, col),
            sample(sensor, row + 1, col),
            sample(sensor, row - 2, col),
            sample(sensor, row + 2, col),
        )
    };
    0.5 * (g_a + g_b) - 0.25 * (c_a + c_b) - 0.5 * center
}

fn local_variance(values: &[f32]) -> f32 {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32
}

fn lmmse_green_plane(sensor: &RawSensorData) -> Vec<f32> {
    let (w, h) = (sensor.width, sensor.height);
    let taps = lmmse_gaussian_taps();

    // Stage 1: raw (unsmoothed) H/V difference-domain estimates at every
    // red/blue site.
    let mut diff_h = vec![0.0f32; w * h];
    let mut diff_v = vec![0.0f32; w * h];
    diff_h
        .par_chunks_mut(w)
        .zip(diff_v.par_chunks_mut(w))
        .enumerate()
        .for_each(|(row, (dh_row, dv_row))| {
            for col in 0..w {
                if (sensor.cfa_at)(row, col) == CFAColor::GREEN {
                    continue;
                }
                dh_row[col] = diff_estimate(sensor, row as isize, col as isize, true);
                dv_row[col] = diff_estimate(sensor, row as isize, col as isize, false);
            }
        });

    // Stage 2: smooth each difference plane along its own axis with the
    // real 9-tap Gaussian - this is LMMSE's actual noise-reduction step.
    let smooth = |plane: &[f32], horizontal: bool| -> Vec<f32> {
        let mut out = vec![0.0f32; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(row, out_row)| {
            for (col, out_px) in out_row.iter_mut().enumerate() {
                let mut acc = 0.0;
                for (i, &tap) in taps.iter().enumerate() {
                    if i == 0 {
                        acc += tap * plane[row * w + col];
                        continue;
                    }
                    // Stride 2, not 1: `plane` only has real (non-placeholder)
                    // values at every OTHER pixel along this axis (the same
                    // CFA color repeats with period 2 in a Bayer row/column -
                    // the interleaved slots are the opposite color/green and
                    // were left as 0.0 in stage 1). A stride-1 tap pattern
                    // averaged in those zero placeholders, systematically
                    // pulling every diff estimate toward 0 and producing a
                    // strong, systematic green-channel deficiency (visible as
                    // a magenta/purple cast) - unit-offset taps only make
                    // sense on a plane that's dense at every pixel, which
                    // this one deliberately isn't.
                    let offset = (i as isize) * 2;
                    let (r1, c1, r2, c2) = if horizontal {
                        (
                            row,
                            (col as isize - offset).clamp(0, w as isize - 1) as usize,
                            row,
                            (col as isize + offset).clamp(0, w as isize - 1) as usize,
                        )
                    } else {
                        (
                            (row as isize - offset).clamp(0, h as isize - 1) as usize,
                            col,
                            (row as isize + offset).clamp(0, h as isize - 1) as usize,
                            col,
                        )
                    };
                    acc += tap * (plane[r1 * w + c1] + plane[r2 * w + c2]);
                }
                *out_px = acc;
            }
        });
        out
    };
    let (smoothed_h, smoothed_v) = rayon::join(|| smooth(&diff_h, true), || smooth(&diff_v, false));

    // Stage 3: combine the two smoothed directional estimates by local
    // variance (lower-variance/"quieter" direction trusted more) - a
    // statistically-motivated combination in the same spirit as LMMSE's
    // minimum-mean-square-error weighting, though RT's own combination
    // logic (further in lmmse_demosaic.cc) was not ported line-for-line.
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(row, out_row)| {
        for (col, out_px) in out_row.iter_mut().enumerate() {
            let idx = row * w + col;
            if (sensor.cfa_at)(row, col) == CFAColor::GREEN {
                *out_px = sensor.data[idx];
                continue;
            }
            // Fixed-size arrays, not Vec: this loop runs once per red/blue
            // pixel (~half the image), and a heap allocation per pixel here
            // was a real, measured performance bug (millions of tiny
            // allocations dominating this stage's wall time).
            let mut h_window = [0.0f32; 5];
            let mut v_window = [0.0f32; 5];
            for (i, offset) in (-2isize..=2).enumerate() {
                let hc = (col as isize + offset).clamp(0, w as isize - 1) as usize;
                h_window[i] = smoothed_h[row * w + hc];
                let vr = (row as isize + offset).clamp(0, h as isize - 1) as usize;
                v_window[i] = smoothed_v[vr * w + col];
            }
            let wh = 1.0 / local_variance(&h_window).max(1e-6);
            let wv = 1.0 / local_variance(&v_window).max(1e-6);
            let diff = (smoothed_h[idx] * wh + smoothed_v[idx] * wv) / (wh + wv);
            *out_px = sensor.data[idx] + diff;
        }
    });

    out
}

/// Interpolates a color-difference plane (target_channel - green) to every
/// pixel: exact where `target` is the native CFA color, averaged from
/// same-color pixels in a 3x3 neighborhood elsewhere. Shared by all three
/// algorithms - they differ in the green plane feeding into this, not in
/// how red/blue are reconstructed from it.
fn interpolate_color_diff(sensor: &RawSensorData, green: &[f32], target: CFAColor) -> Vec<f32> {
    let (w, h) = (sensor.width, sensor.height);
    let mut diff = vec![0.0f32; w * h];
    let mut has_diff = vec![false; w * h];

    diff.par_chunks_mut(w)
        .zip(has_diff.par_chunks_mut(w))
        .enumerate()
        .for_each(|(row, (diff_row, has_row))| {
            for (col, diff_px) in diff_row.iter_mut().enumerate() {
                if (sensor.cfa_at)(row, col) == target {
                    let idx = row * w + col;
                    *diff_px = sensor.data[idx] - green[idx];
                    has_row[col] = true;
                }
            }
        });

    let known = diff.clone();
    diff.par_chunks_mut(w).enumerate().for_each(|(row, diff_row)| {
        for (col, diff_px) in diff_row.iter_mut().enumerate() {
            let idx = row * w + col;
            if has_diff[idx] {
                continue;
            }
            let mut sum = 0.0;
            let mut n = 0.0;
            for dr in -1isize..=1 {
                for dc in -1isize..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = (row as isize + dr).clamp(0, h as isize - 1) as usize;
                    let nc = (col as isize + dc).clamp(0, w as isize - 1) as usize;
                    let nidx = nr * w + nc;
                    if has_diff[nidx] {
                        sum += known[nidx];
                        n += 1.0;
                    }
                }
            }
            *diff_px = if n > 0.0 { sum / n } else { 0.0 };
        }
    });

    diff
}
