//! GPU implementations of RAW-domain operations.
//!
//! These kernels share the editor's WGPU device through CubeCL. Every entry
//! point returns `false` when that device is unavailable or execution fails;
//! callers then run their established CPU implementation to preserve both
//! correctness and support for machines without a usable GPU.

use std::sync::OnceLock;

use cubecl::prelude::*;

static SHARED_WGPU_DEVICE: OnceLock<cubecl::wgpu::WgpuDevice> = OnceLock::new();

pub fn register_shared_wgpu_device(device: cubecl::wgpu::WgpuDevice) {
    let _ = SHARED_WGPU_DEVICE.set(device);
}

#[cube(launch_unchecked)]
fn correct_hot_dead_pixels_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: usize,
    height: usize,
    threshold: f32,
) {
    let index = ABSOLUTE_POS;
    if index >= input.len() {
        return;
    }

    let row = index / width;
    let col = index % width;
    let row_up = if row >= 2 { row - 2 } else { 0 };
    let row_down = if row + 2 < height { row + 2 } else { height - 1 };
    let col_left = if col >= 2 { col - 2 } else { 0 };
    let col_right = if col + 2 < width { col + 2 } else { width - 1 };

    let a = input[row_up * width + col];
    let b = input[row_down * width + col];
    let c = input[row * width + col_left];
    let d = input[row * width + col_right];

    // The median of four values is the mean of the two middle values. This
    // small sorting network matches the CPU implementation without needing
    // per-pixel temporary allocation in the GPU kernel.
    let low_ab = if a < b { a } else { b };
    let high_ab = if a < b { b } else { a };
    let low_cd = if c < d { c } else { d };
    let high_cd = if c < d { d } else { c };
    let lower_middle = if low_ab > low_cd { low_ab } else { low_cd };
    let upper_middle = if high_ab < high_cd { high_ab } else { high_cd };
    let median = (lower_middle + upper_middle) * 0.5;
    let center = input[index];

    output[index] = if (center - median).abs() > threshold {
        median
    } else {
        center
    };
}

#[cube(launch_unchecked)]
fn white_balance_and_color_matrix_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    pixels: usize,
    black_r: f32,
    black_g: f32,
    black_b: f32,
    wb_r: f32,
    wb_g: f32,
    wb_b: f32,
    denominator: f32,
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    m20: f32,
    m21: f32,
    m22: f32,
) {
    let pixel = ABSOLUTE_POS;
    if pixel >= pixels {
        return;
    }
    let offset = pixel * 3;
    let r_unclamped = (input[offset] - black_r) * wb_r / denominator;
    let g_unclamped = (input[offset + 1] - black_g) * wb_g / denominator;
    let b_unclamped = (input[offset + 2] - black_b) * wb_b / denominator;
    let r = if r_unclamped > 0.0 { r_unclamped } else { 0.0 };
    let g = if g_unclamped > 0.0 { g_unclamped } else { 0.0 };
    let b = if b_unclamped > 0.0 { b_unclamped } else { 0.0 };

    output[offset] = m00 * r + m01 * g + m02 * b;
    output[offset + 1] = m10 * r + m11 * g + m12 * b;
    output[offset + 2] = m20 * r + m21 * g + m22 * b;
}

/// Performs CFA-periodic hot/dead pixel correction on the shared GPU.
///
/// The transfer overhead outweighs a GPU dispatch on tiny RAW previews, so
/// those deliberately use the CPU fallback. Full sensor frames use the GPU
/// whenever CubeCL has been initialized by the editor's WGPU context.
pub fn correct_hot_dead_pixels(
    data: &mut [f32],
    width: usize,
    height: usize,
    threshold: f32,
) -> bool {
    const MIN_GPU_PIXELS: usize = 1_048_576;

    if width < 5 || height < 5 || data.len() != width.saturating_mul(height) {
        return false;
    }
    if data.len() < MIN_GPU_PIXELS {
        return false;
    }
    let Some(device) = SHARED_WGPU_DEVICE.get() else {
        return false;
    };

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let client = cubecl::wgpu::WgpuRuntime::client(device);
        let input = client.create_from_slice(bytemuck::cast_slice(data));
        let output = client.empty(std::mem::size_of_val(data));
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::Static(
            data.len().div_ceil(cube_dim.num_elems() as usize) as u32,
            1,
            1,
        );

        unsafe {
            correct_hot_dead_pixels_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(input, data.len()),
                ArrayArg::from_raw_parts(output.clone(), data.len()),
                width,
                height,
                threshold,
            );
        }

        let bytes = client.read_one(output).ok()?;
        let corrected = f32::from_bytes(&bytes);
        if corrected.len() != data.len() {
            return None;
        }
        data.copy_from_slice(corrected);
        Some(())
    }))
    .ok()
    .flatten()
    .is_some()
}

/// Fuses the post-demosaic white-balance and camera-to-linear-sRGB stages.
/// The input is packed RGB triplets and is changed only after a complete,
/// correctly-sized GPU result has been received.
pub fn apply_white_balance_and_color_matrix(
    rgb: &mut [[f32; 3]],
    black: [f32; 3],
    white_balance: [f32; 3],
    denominator: f32,
    matrix: [f32; 9],
) -> bool {
    const MIN_GPU_PIXELS: usize = 1_048_576;

    if rgb.len() < MIN_GPU_PIXELS || !denominator.is_finite() || denominator <= 0.0 {
        return false;
    }
    let Some(device) = SHARED_WGPU_DEVICE.get() else {
        return false;
    };

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let client = cubecl::wgpu::WgpuRuntime::client(device);
        let input = client.create_from_slice(bytemuck::cast_slice(rgb));
        let output = client.empty(std::mem::size_of_val(rgb));
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::Static(
            rgb.len().div_ceil(cube_dim.num_elems() as usize) as u32,
            1,
            1,
        );

        unsafe {
            white_balance_and_color_matrix_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(input, rgb.len() * 3),
                ArrayArg::from_raw_parts(output.clone(), rgb.len() * 3),
                rgb.len(),
                black[0], black[1], black[2],
                white_balance[0], white_balance[1], white_balance[2],
                denominator,
                matrix[0], matrix[1], matrix[2],
                matrix[3], matrix[4], matrix[5],
                matrix[6], matrix[7], matrix[8],
            );
        }

        let bytes = client.read_one(output).ok()?;
        let transformed = f32::from_bytes(&bytes);
        if transformed.len() != rgb.len() * 3 {
            return None;
        }
        for (pixel, transformed) in rgb.iter_mut().zip(transformed.chunks_exact(3)) {
            *pixel = [transformed[0], transformed[1], transformed[2]];
        }
        Some(())
    }))
    .ok()
    .flatten()
    .is_some()
}
