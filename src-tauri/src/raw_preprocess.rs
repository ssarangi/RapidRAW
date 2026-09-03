//! Raw-domain (pre-demosaic) sensor corrections: hot/dead pixel removal and
//! CFA row-banding denoise. Mirrors two of the narrower fixes ART/RawTherapee
//! apply in `RawImageSource::preprocess` (see CLAUDE.md's "RAW develop
//! pipeline" notes) before demosaic ever runs - deliberately NOT the whole of
//! ART's preprocess step (dark-frame/flat-field subtraction and PDAF-line
//! filtering are out of scope: the former needs a user-managed calibration
//! frame library, the latter needs a per-camera-model pixel-location database
//! we don't have).
//!
//! Both corrections rely on one property of a standard 2x2 Bayer CFA: the
//! color at (row, col) is identical to the color at (row+2, col) and
//! (row, col+2), since the pattern repeats with period 2 in each axis. That
//! makes "same-color neighbor" trivial to find without consulting `cfa_at`
//! at all.

use crate::custom_raw_pipeline::RawSensorData;
use rawler::cfa::CFAColor;

/// Detects and replaces hot/dead (stuck) pixels: sensor sites whose value is
/// a statistical outlier compared to the same-color pixels immediately
/// around them (up/down/left/right at distance 2, per the CFA-periodicity
/// note above). Outliers are replaced with the local same-color median.
///
/// `threshold` is a fraction of the sensor's white level (0..1) - the
/// minimum deviation from the local median before a pixel is considered
/// "hot" or "dead" rather than genuine detail. Not tuned against a real
/// test set; 0.15 (15% of white level) is a starting point matching the
/// rough magnitude of RawTherapee's default `hotdeadpix_thresh`.
pub fn correct_hot_dead_pixels(sensor: &mut RawSensorData, threshold: f32) {
    let (w, h) = (sensor.width, sensor.height);
    if w < 5 || h < 5 {
        return;
    }
    let abs_threshold = threshold * sensor.white_level;
    let original = sensor.data.clone();

    let get = |row: isize, col: isize| -> f32 {
        let row = row.clamp(0, h as isize - 1) as usize;
        let col = col.clamp(0, w as isize - 1) as usize;
        original[row * w + col]
    };

    for row in 0..h {
        for col in 0..w {
            let center = original[row * w + col];
            let mut neighbors = [
                get(row as isize - 2, col as isize),
                get(row as isize + 2, col as isize),
                get(row as isize, col as isize - 2),
                get(row as isize, col as isize + 2),
            ];
            neighbors.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = (neighbors[1] + neighbors[2]) * 0.5;

            if (center - median).abs() > abs_threshold {
                sensor.data[row * w + col] = median;
            }
        }
    }
}

/// Corrects per-row banding (a fixed or slowly-varying horizontal-line
/// pattern noise some sensors exhibit, especially at high ISO - Sony bodies
/// are a commonly cited example) by subtracting each row's deviation from a
/// locally-smoothed reference, computed separately per CFA color and kept
/// separate for even/odd rows (a Bayer row is either all "R/G" or all
/// "G/B", and only rows of the same type are comparable).
///
/// `strength` (0..1) blends between no correction (0.0) and the full
/// per-row offset correction (1.0), so callers can dial it back if it ever
/// over-corrects genuine low-frequency lighting gradients.
pub fn correct_cfa_line_banding(sensor: &mut RawSensorData, strength: f32) {
    if strength <= 0.0 {
        return;
    }
    let (w, h) = (sensor.width, sensor.height);
    if h < 9 {
        return;
    }

    // Two "row types" per the CFA's 2-row period (RGRG.../GBGB... for a
    // standard Bayer pattern); average each row's value, per row-type,
    // across the whole row (mixing both colors in that row is fine here -
    // we only care about detecting a uniform-across-the-row banding offset,
    // not per-color bias, which correct_hot_dead_pixels already handles).
    let mut row_means = vec![0.0f32; h];
    for (row, mean) in row_means.iter_mut().enumerate() {
        let row_data = &sensor.data[row * w..(row + 1) * w];
        *mean = row_data.iter().sum::<f32>() / w as f32;
    }

    // Reference: a wide local average of same-row-type means, excluding the
    // row itself, so a genuinely banded row doesn't smooth into its own
    // reference.
    const RADIUS: isize = 8;
    let mut offsets = vec![0.0f32; h];
    for row in 0..h {
        let row_type = row % 2;
        let mut sum = 0.0f32;
        let mut count = 0.0f32;
        for d in -RADIUS..=RADIUS {
            if d == 0 {
                continue;
            }
            let r = row as isize + d * 2; // step by 2 to stay on the same row-type
            if r < 0 || r >= h as isize {
                continue;
            }
            let r = r as usize;
            if r % 2 != row_type {
                continue;
            }
            sum += row_means[r];
            count += 1.0;
        }
        if count > 0.0 {
            offsets[row] = (row_means[row] - sum / count) * strength;
        }
    }

    for row in 0..h {
        let offset = offsets[row];
        if offset.abs() < 1e-6 {
            continue;
        }
        for col in 0..w {
            sensor.data[row * w + col] = (sensor.data[row * w + col] - offset).max(0.0);
        }
    }
}

/// Corrects on-sensor PDAF (phase-detect autofocus) pixels for cameras
/// whose row pattern is known (see `raw_pdaf_data.rs`) - a no-op for any
/// other camera. PDAF pixels replace ordinary green photosites on specific
/// rows and read back brighter than their neighbors; left alone, demosaic
/// spreads them into small colored specks.
///
/// Mirrors ART's `PDAFLinesFilter::markLine` (`rtengine/pdaflinesfilter.cc`):
/// walks the camera's known candidate rows (and their immediate ±1
/// neighbors), and within each row flags a green pixel as PDAF-affected
/// only if it's brighter than all four same-color diagonal neighbors *and*
/// those neighbors are locally consistent (ruling out a real high-contrast
/// edge) - then corrects it the same way `correct_hot_dead_pixels` does,
/// via the same-color 4-neighbor median.
pub fn correct_pdaf_pixels(sensor: &mut RawSensorData) {
    let Some(pattern) = crate::raw_pdaf_data::lookup(&sensor.camera_name) else {
        return;
    };
    let (w, h) = (sensor.width, sensor.height);
    if h < 5 || w < 5 || pattern.pattern.is_empty() {
        return;
    }

    // Replicates ART's sequential row-pattern walk: advance through the
    // repeating pattern as `y` increases, wrapping and adding the pattern's
    // own span (its last element) back onto the running offset each cycle.
    let period = *pattern.pattern.last().unwrap();
    let mut target_rows = Vec::new();
    let mut idx = 0usize;
    let mut off = pattern.offset as i64;
    for y in 2..(h as i64 - 2) {
        let yy = pattern.pattern[idx] as i64 + off;
        if y >= yy {
            if y == yy {
                target_rows.push(y);
            }
            idx += 1;
            if idx >= pattern.pattern.len() {
                idx = 0;
                off += period as i64;
            }
        }
    }

    let original = sensor.data.clone();
    let get = |row: i64, col: i64| -> f32 {
        let row = row.clamp(0, h as i64 - 1) as usize;
        let col = col.clamp(0, w as i64 - 1) as usize;
        original[row * w + col]
    };

    for &y in &target_rows {
        for dy in -1i64..=1 {
            let row = y + dy;
            if row < 2 || row >= h as i64 - 2 {
                continue;
            }
            let row_u = row as usize;
            for col in 2..(w - 2) {
                if (sensor.cfa_at)(row_u, col) != CFAColor::GREEN {
                    continue;
                }
                let g0 = original[row_u * w + col];
                if g0 <= 1e-6 {
                    continue;
                }
                let g1 = get(row - 1, col as i64 + 1);
                let g2 = get(row + 1, col as i64 + 1);
                let g3 = get(row - 1, col as i64 - 1);
                let g4 = get(row + 1, col as i64 - 1);
                if g0 <= g1.max(g2).max(g3).max(g4) {
                    continue;
                }
                let gu = g2 + g4;
                let gd = g1 + g3;
                let g_max = gu.max(gd);
                let g_min = gu.min(gd);
                if g_max <= 1e-6 {
                    continue;
                }
                let d = (g_max - g_min) / g_max;
                if d < 0.2 && (1.0 - (g_min + g_max) / (4.0 * g0)) > d.min(0.1) {
                    let mut neighbors = [
                        get(row - 2, col as i64),
                        get(row + 2, col as i64),
                        get(row, col as i64 - 2),
                        get(row, col as i64 + 2),
                    ];
                    neighbors.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    sensor.data[row_u * w + col] = (neighbors[1] + neighbors[2]) * 0.5;
                }
            }
        }
    }
}

/// Global green-channel equilibration: a standard Bayer CFA has two
/// distinct green sub-populations (green pixels on "red rows" vs. green
/// pixels on "blue rows"), which can read back with a slight sensitivity
/// mismatch on some sensors, showing up as a faint checkerboard/maze
/// pattern after demosaic. Camera-agnostic - unlike PDAF correction, no
/// per-model data is needed: this computes each sub-population's average
/// directly from the image and rescales one to match the other. Mirrors
/// ART's `green_equilibrate_global` (`rtengine/green_equil_RT.cc`).
pub fn equalize_green_channels(sensor: &mut RawSensorData) {
    let (w, h) = (sensor.width, sensor.height);
    if h < 4 || w < 4 {
        return;
    }

    let mut sum_even = 0.0f64;
    let mut count_even = 0u64;
    let mut sum_odd = 0.0f64;
    let mut count_odd = 0u64;

    for row in 0..h {
        for col in 0..w {
            if (sensor.cfa_at)(row, col) != CFAColor::GREEN {
                continue;
            }
            let value = sensor.data[row * w + col] as f64;
            if row % 2 == 0 {
                sum_even += value;
                count_even += 1;
            } else {
                sum_odd += value;
                count_odd += 1;
            }
        }
    }

    if count_even == 0 || count_odd == 0 {
        return;
    }
    let avg_even = sum_even / count_even as f64;
    let avg_odd = sum_odd / count_odd as f64;
    if avg_even <= 1e-6 || avg_odd <= 1e-6 {
        return;
    }

    // Scale both sub-populations toward their shared mean, rather than
    // picking one as the reference - avoids a directional brightness bias.
    let target = (avg_even + avg_odd) / 2.0;
    let scale_even = (target / avg_even) as f32;
    let scale_odd = (target / avg_odd) as f32;

    for row in 0..h {
        let scale = if row % 2 == 0 { scale_even } else { scale_odd };
        if (scale - 1.0).abs() < 1e-4 {
            continue;
        }
        for col in 0..w {
            if (sensor.cfa_at)(row, col) == CFAColor::GREEN {
                sensor.data[row * w + col] *= scale;
            }
        }
    }
}
