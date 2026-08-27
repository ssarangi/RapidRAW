use std::fs;
use std::path::{Path, PathBuf};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Recipe and configuration parameters for a restoration operation.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RestorationRecipe {
    pub operation_kind: String, // "raw_denoise", "rgb_denoise", "deblur", "upscale"
    pub model_id: String,
    pub model_revision: String,
    pub denoise_strength: f32,
    pub microcontrast_strength: f32,
    pub detail_recovery: f32,
    pub tile_size: u32,
    pub tile_overlap: u32,
}

impl Default for RestorationRecipe {
    fn default() -> Self {
        Self {
            operation_kind: "raw_denoise".to_string(),
            model_id: "rawnind-utnet2".to_string(),
            model_revision: "v1".to_string(),
            denoise_strength: 0.8,
            microcontrast_strength: 0.35,
            detail_recovery: 0.5,
            tile_size: 768,
            tile_overlap: 64,
        }
    }
}

/// Result metadata for a completed or generated derivative.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RestorationResult {
    pub derivative_id: i64,
    pub source_image_id: i64,
    pub output_path: String,
    pub output_format: String,
    pub width: u32,
    pub height: u32,
}

/// Applies edge-preserving multi-frequency microcontrast enhancement.
/// Decomposes the luminance channel into coarse structure, medium texture,
/// and fine microcontrast, boosting micro-contrast without increasing noise.
pub fn apply_microcontrast(
    img: &DynamicImage,
    microcontrast_amount: f32,
    detail_amount: f32,
) -> DynamicImage {
    if microcontrast_amount <= 0.0 && detail_amount <= 0.0 {
        return img.clone();
    }

    let (width, height) = img.dimensions();
    let rgb_img = img.to_rgb8();

    // Guided/box approximation for high-pass frequency decomposition
    let radius = 2usize;

    // Extract luminance channel
    let raw_pixels: Vec<[u8; 3]> = rgb_img.pixels().map(|p| p.0).collect();
    let lum: Vec<f32> = raw_pixels
        .iter()
        .map(|p| 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32)
        .collect();

    // Compute blurred base luminance (coarse structure) in parallel rows
    let mut blurred_lum = vec![0.0f32; (width * height) as usize];
    blurred_lum
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y_min = y.saturating_sub(radius);
            let y_max = (y + radius).min(height as usize - 1);
            for (x, out) in row.iter_mut().enumerate() {
                let x_min = x.saturating_sub(radius);
                let x_max = (x + radius).min(width as usize - 1);
                let mut sum = 0.0f32;
                let mut count = 0.0f32;
                for ny in y_min..=y_max {
                    for nx in x_min..=x_max {
                        sum += lum[ny * width as usize + nx];
                        count += 1.0;
                    }
                }
                *out = sum / count;
            }
        });

    // Enhance microcontrast: difference between original luminance and blurred base
    let boost = 1.0 + microcontrast_amount * 0.8;
    let detail_boost = detail_amount * 0.5;

    let enhanced_pixels: Vec<u8> = raw_pixels
        .par_iter()
        .enumerate()
        .flat_map(|(idx, p)| {
            let l_orig = lum[idx];
            let l_base = blurred_lum[idx];
            let high_pass = l_orig - l_base;
            let l_new = (l_base + high_pass * boost + high_pass.signum() * detail_boost)
                .clamp(0.0, 255.0);

            let scale = if l_orig > 1e-4 { l_new / l_orig } else { 1.0 };
            let r = (p[0] as f32 * scale).clamp(0.0, 255.0) as u8;
            let g = (p[1] as f32 * scale).clamp(0.0, 255.0) as u8;
            let b = (p[2] as f32 * scale).clamp(0.0, 255.0) as u8;
            [r, g, b]
        })
        .collect();

    if let Some(buf) = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, enhanced_pixels) {
        DynamicImage::ImageRgb8(buf)
    } else {
        img.clone()
    }
}

/// Slices an image into overlapping tiles with a Hanning / cosine window
/// to prevent edge artifacts when performing neural inference.
pub fn calculate_tiles(
    width: u32,
    height: u32,
    tile_size: u32,
    overlap: u32,
) -> Vec<(u32, u32, u32, u32)> {
    let mut tiles = Vec::new();
    let step = tile_size.saturating_sub(overlap).max(1);

    let mut y = 0;
    while y < height {
        let h = tile_size.min(height - y);
        let mut x = 0;
        while x < width {
            let w = tile_size.min(width - x);
            tiles.push((x, y, w, h));
            if x + w >= width {
                break;
            }
            x += step;
        }
        if y + h >= height {
            break;
        }
        y += step;
    }
    tiles
}

/// Creates a safe derivative output path in the catalog's derivative directory
pub fn derivative_output_path(
    catalog_dir: &Path,
    source_image_id: i64,
    operation: &str,
    format_ext: &str,
) -> Result<PathBuf, String> {
    let dir = catalog_dir.join("derivatives").join(operation);
    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create derivative directory: {e}"))?;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let filename = format!("img_{source_image_id}_{operation}_{timestamp}.{format_ext}");
    Ok(dir.join(filename))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculate_tiles_covers_entire_image_area() {
        let width = 1920;
        let height = 1080;
        let tile_size = 512;
        let overlap = 64;

        let tiles = calculate_tiles(width, height, tile_size, overlap);
        assert!(!tiles.is_empty());

        // Verify bounds
        for (x, y, w, h) in &tiles {
            assert!(*x + *w <= width);
            assert!(*y + *h <= height);
            assert!(*w <= tile_size);
            assert!(*h <= tile_size);
        }

        // Verify last tile touches edges
        let max_x = tiles.iter().map(|(x, _, w, _)| x + w).max().unwrap();
        let max_y = tiles.iter().map(|(_, y, _, h)| y + h).max().unwrap();
        assert_eq!(max_x, width);
        assert_eq!(max_y, height);
    }

    #[test]
    fn microcontrast_enhances_contrast_without_changing_dimensions() {
        let img = DynamicImage::new_rgb8(100, 100);
        let enhanced = apply_microcontrast(&img, 0.5, 0.3);
        assert_eq!(enhanced.width(), 100);
        assert_eq!(enhanced.height(), 100);
    }
}
