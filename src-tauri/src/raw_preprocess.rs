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
