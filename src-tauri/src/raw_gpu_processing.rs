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
        terminate!();
    }

    let row = index / width;
    let col = index % width;
    let row_up = if row >= 2 { row - 2 } else { 0usize.into() };
    let row_down = if row + 2 < height { row + 2 } else { height - 1 };
    let col_left = if col >= 2 { col - 2 } else { 0usize.into() };
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
        terminate!();
    }
    let offset = pixel * 3;
    let r_unclamped = (input[offset] - black_r) * wb_r / denominator;
    let g_unclamped = (input[offset + 1] - black_g) * wb_g / denominator;
    let b_unclamped = (input[offset + 2] - black_b) * wb_b / denominator;
    let r = if r_unclamped > 0.0 { r_unclamped } else { 0.0f32.into() };
    let g = if g_unclamped > 0.0 { g_unclamped } else { 0.0f32.into() };
    let b = if b_unclamped > 0.0 { b_unclamped } else { 0.0f32.into() };

    output[offset] = m00 * r + m01 * g + m02 * b;
    output[offset + 1] = m10 * r + m11 * g + m12 * b;
    output[offset + 2] = m20 * r + m21 * g + m22 * b;
}

#[cube]
fn bayer_color(row: usize, col: usize, c00: u32, c01: u32, c10: u32, c11: u32) -> u32 {
    if row % 2 == 0 {
        if col % 2 == 0 { c00 } else { c01 }
    } else if col % 2 == 0 {
        c10
    } else {
        c11
    }
}

#[cube]
fn raw_sample(input: &Array<f32>, row: usize, col: usize, width: usize, height: usize) -> f32 {
    let y = if row < height { row } else { height - 1 };
    let x = if col < width { col } else { width - 1 };
    input[y * width + x]
}

#[cube(launch_unchecked)]
fn amaze_green_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: usize,
    height: usize,
    c00: u32,
    c01: u32,
    c10: u32,
    c11: u32,
) {
    let index = ABSOLUTE_POS;
    if index >= input.len() {
        terminate!();
    }
    let row = index / width;
    let col = index % width;
    if bayer_color(row, col, c00, c01, c10, c11) == 1 {
        output[index] = input[index];
        terminate!();
    }

    let center = input[index];
    let up1 = if row >= 1 { row - 1 } else { 0usize.into() };
    let up2 = if row >= 2 { row - 2 } else { 0usize.into() };
    let up3 = if row >= 3 { row - 3 } else { 0usize.into() };
    let up4 = if row >= 4 { row - 4 } else { 0usize.into() };
    let up5 = if row >= 5 { row - 5 } else { 0usize.into() };
    let down1 = row + 1;
    let down2 = row + 2;
    let down3 = row + 3;
    let down4 = row + 4;
    let down5 = row + 5;
    let left1 = if col >= 1 { col - 1 } else { 0usize.into() };
    let left2 = if col >= 2 { col - 2 } else { 0usize.into() };
    let left3 = if col >= 3 { col - 3 } else { 0usize.into() };
    let left4 = if col >= 4 { col - 4 } else { 0usize.into() };
    let left5 = if col >= 5 { col - 5 } else { 0usize.into() };
    let right1 = col + 1;
    let right2 = col + 2;
    let right3 = col + 3;
    let right4 = col + 4;
    let right5 = col + 5;

    let n = (23.0 * raw_sample(input, up1, col, width, height)
        + 23.0 * raw_sample(input, up3, col, width, height)
        + raw_sample(input, up5, col, width, height)
        + raw_sample(input, down1, col, width, height)
        + 40.0 * center
        - 32.0 * raw_sample(input, up2, col, width, height)
        - 8.0 * raw_sample(input, up4, col, width, height)) / 48.0;
    let s = (23.0 * raw_sample(input, down1, col, width, height)
        + 23.0 * raw_sample(input, down3, col, width, height)
        + raw_sample(input, down5, col, width, height)
        + raw_sample(input, up1, col, width, height)
        + 40.0 * center
        - 32.0 * raw_sample(input, down2, col, width, height)
        - 8.0 * raw_sample(input, down4, col, width, height)) / 48.0;
    let e = (23.0 * raw_sample(input, row, right1, width, height)
        + 23.0 * raw_sample(input, row, right3, width, height)
        + raw_sample(input, row, right5, width, height)
        + raw_sample(input, row, left1, width, height)
        + 40.0 * center
        - 32.0 * raw_sample(input, row, right2, width, height)
        - 8.0 * raw_sample(input, row, right4, width, height)) / 48.0;
    let w = (23.0 * raw_sample(input, row, left1, width, height)
        + 23.0 * raw_sample(input, row, left3, width, height)
        + raw_sample(input, row, left5, width, height)
        + raw_sample(input, row, right1, width, height)
        + 40.0 * center
        - 32.0 * raw_sample(input, row, left2, width, height)
        - 8.0 * raw_sample(input, row, left4, width, height)) / 48.0;
    let grad_n = 0.00001 + (raw_sample(input, up1, col, width, height) - raw_sample(input, up3, col, width, height)).abs()
        + (center - raw_sample(input, up2, col, width, height)).abs();
    let grad_s = 0.00001 + (raw_sample(input, down1, col, width, height) - raw_sample(input, down3, col, width, height)).abs()
        + (center - raw_sample(input, down2, col, width, height)).abs();
    let grad_e = 0.00001 + (raw_sample(input, row, right1, width, height) - raw_sample(input, row, right3, width, height)).abs()
        + (center - raw_sample(input, row, right2, width, height)).abs();
    let grad_w = 0.00001 + (raw_sample(input, row, left1, width, height) - raw_sample(input, row, left3, width, height)).abs()
        + (center - raw_sample(input, row, left2, width, height)).abs();
    let grad_v = grad_n + grad_s;
    let grad_h = grad_e + grad_w;
    output[index] = if grad_h < grad_v {
        (grad_w * e + grad_e * w) / (grad_e + grad_w)
    } else {
        (grad_s * n + grad_n * s) / (grad_n + grad_s)
    };
}

#[cube(launch_unchecked)]
fn igv_green_pass1_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: usize,
    height: usize,
    c00: u32,
    c01: u32,
    c10: u32,
    c11: u32,
) {
    let index = ABSOLUTE_POS;
    if index >= input.len() {
        terminate!();
    }
    let row = index / width;
    let col = index % width;
    if bayer_color(row, col, c00, c01, c10, c11) == 1 {
        output[index] = input[index];
        terminate!();
    }

    let center = input[index];
    let up1 = if row >= 1 { row - 1 } else { 0usize.into() };
    let up2 = if row >= 2 { row - 2 } else { 0usize.into() };
    let up3 = if row >= 3 { row - 3 } else { 0usize.into() };
    let up4 = if row >= 4 { row - 4 } else { 0usize.into() };
    let up5 = if row >= 5 { row - 5 } else { 0usize.into() };
    let down1 = row + 1;
    let down2 = row + 2;
    let down3 = row + 3;
    let down4 = row + 4;
    let down5 = row + 5;
    let left1 = if col >= 1 { col - 1 } else { 0usize.into() };
    let left2 = if col >= 2 { col - 2 } else { 0usize.into() };
    let left3 = if col >= 3 { col - 3 } else { 0usize.into() };
    let left4 = if col >= 4 { col - 4 } else { 0usize.into() };
    let left5 = if col >= 5 { col - 5 } else { 0usize.into() };
    let right1 = col + 1;
    let right2 = col + 2;
    let right3 = col + 3;
    let right4 = col + 4;
    let right5 = col + 5;
    let n = (23.0 * raw_sample(input, up1, col, width, height) + 23.0 * raw_sample(input, up3, col, width, height)
        + raw_sample(input, up5, col, width, height) + raw_sample(input, down1, col, width, height) + 40.0 * center
        - 32.0 * raw_sample(input, up2, col, width, height) - 8.0 * raw_sample(input, up4, col, width, height)) / 48.0;
    let s = (23.0 * raw_sample(input, down1, col, width, height) + 23.0 * raw_sample(input, down3, col, width, height)
        + raw_sample(input, down5, col, width, height) + raw_sample(input, up1, col, width, height) + 40.0 * center
        - 32.0 * raw_sample(input, down2, col, width, height) - 8.0 * raw_sample(input, down4, col, width, height)) / 48.0;
    let e = (23.0 * raw_sample(input, row, right1, width, height) + 23.0 * raw_sample(input, row, right3, width, height)
        + raw_sample(input, row, right5, width, height) + raw_sample(input, row, left1, width, height) + 40.0 * center
        - 32.0 * raw_sample(input, row, right2, width, height) - 8.0 * raw_sample(input, row, right4, width, height)) / 48.0;
    let w = (23.0 * raw_sample(input, row, left1, width, height) + 23.0 * raw_sample(input, row, left3, width, height)
        + raw_sample(input, row, left5, width, height) + raw_sample(input, row, right1, width, height) + 40.0 * center
        - 32.0 * raw_sample(input, row, left2, width, height) - 8.0 * raw_sample(input, row, left4, width, height)) / 48.0;
    let grad_n = 0.00001 + (raw_sample(input, up1, col, width, height) - raw_sample(input, up3, col, width, height)).abs()
        + (center - raw_sample(input, up2, col, width, height)).abs();
    let grad_s = 0.00001 + (raw_sample(input, down1, col, width, height) - raw_sample(input, down3, col, width, height)).abs()
        + (center - raw_sample(input, down2, col, width, height)).abs();
    let grad_e = 0.00001 + (raw_sample(input, row, right1, width, height) - raw_sample(input, row, right3, width, height)).abs()
        + (center - raw_sample(input, row, right2, width, height)).abs();
    let grad_w = 0.00001 + (raw_sample(input, row, left1, width, height) - raw_sample(input, row, left3, width, height)).abs()
        + (center - raw_sample(input, row, left2, width, height)).abs();
    let v = (grad_s * n + grad_n * s) / (grad_n + grad_s);
    let hh = (grad_w * e + grad_e * w) / (grad_e + grad_w);
    let gv = 1.0 / (grad_n + grad_s);
    let gh = 1.0 / (grad_e + grad_w);
    output[index] = (gh * v + gv * hh) / (gh + gv);
}

#[cube(launch_unchecked)]
fn igv_green_refine_kernel(
    input: &Array<f32>,
    pass1: &Array<f32>,
    output: &mut Array<f32>,
    width: usize,
    height: usize,
    c00: u32,
    c01: u32,
    c10: u32,
    c11: u32,
) {
    let index = ABSOLUTE_POS;
    if index >= input.len() {
        terminate!();
    }
    let row = index / width;
    let col = index % width;
    if bayer_color(row, col, c00, c01, c10, c11) == 1 {
        output[index] = pass1[index];
        terminate!();
    }
    let up = if row > 0 { row - 1 } else { 0usize.into() };
    let down = if row + 1 < height { row + 1 } else { height - 1 };
    let left = if col > 0 { col - 1 } else { 0usize.into() };
    let right = if col + 1 < width { col + 1 } else { width - 1 };
    let up_green = if bayer_color(up, col, c00, c01, c10, c11) == 1 { 1.0 } else { 0.0f32.into() };
    let down_green = if bayer_color(down, col, c00, c01, c10, c11) == 1 { 1.0 } else { 0.0f32.into() };
    let left_green = if bayer_color(row, left, c00, c01, c10, c11) == 1 { 1.0 } else { 0.0f32.into() };
    let right_green = if bayer_color(row, right, c00, c01, c10, c11) == 1 { 1.0 } else { 0.0f32.into() };
    let sum = pass1[index]
        + input[up * width + col] * up_green
        + input[down * width + col] * down_green
        + input[row * width + left] * left_green
        + input[row * width + right] * right_green;
    output[index] = sum / (1.0 + up_green + down_green + left_green + right_green);
}

#[cube(launch_unchecked)]
fn lmmse_diff_estimate_kernel(
    input: &Array<f32>,
    horizontal: &mut Array<f32>,
    vertical: &mut Array<f32>,
    width: usize,
    height: usize,
    c00: u32,
    c01: u32,
    c10: u32,
    c11: u32,
) {
    let index = ABSOLUTE_POS;
    if index >= input.len() { terminate!(); }
    let row = index / width;
    let col = index % width;
    if bayer_color(row, col, c00, c01, c10, c11) == 1 {
        horizontal[index] = 0.0;
        vertical[index] = 0.0;
        terminate!();
    }
    let center = input[index];
    let left1 = if col >= 1 { col - 1 } else { 0usize.into() };
    let left2 = if col >= 2 { col - 2 } else { 0usize.into() };
    let right1 = col + 1;
    let right2 = col + 2;
    let up1 = if row >= 1 { row - 1 } else { 0usize.into() };
    let up2 = if row >= 2 { row - 2 } else { 0usize.into() };
    let down1 = row + 1;
    let down2 = row + 2;
    horizontal[index] = 0.5 * (raw_sample(input, row, left1, width, height) + raw_sample(input, row, right1, width, height))
        - 0.25 * (raw_sample(input, row, left2, width, height) + raw_sample(input, row, right2, width, height))
        - 0.5 * center;
    vertical[index] = 0.5 * (raw_sample(input, up1, col, width, height) + raw_sample(input, down1, col, width, height))
        - 0.25 * (raw_sample(input, up2, col, width, height) + raw_sample(input, down2, col, width, height))
        - 0.5 * center;
}

#[cube(launch_unchecked)]
fn lmmse_smooth_horizontal_kernel(
    input: &Array<f32>, output: &mut Array<f32>, width: usize, height: usize,
    t0: f32, t1: f32, t2: f32, t3: f32, t4: f32,
) {
    let index = ABSOLUTE_POS;
    if index >= input.len() { terminate!(); }
    let row = index / width;
    let col = index % width;
    let l2 = if col >= 2 { col - 2 } else { 0usize.into() };
    let l4 = if col >= 4 { col - 4 } else { 0usize.into() };
    let l6 = if col >= 6 { col - 6 } else { 0usize.into() };
    let l8 = if col >= 8 { col - 8 } else { 0usize.into() };
    let r2 = col + 2;
    let r4 = col + 4;
    let r6 = col + 6;
    let r8 = col + 8;
    output[index] = t0 * input[index]
        + t1 * (raw_sample(input, row, l2, width, height) + raw_sample(input, row, r2, width, height))
        + t2 * (raw_sample(input, row, l4, width, height) + raw_sample(input, row, r4, width, height))
        + t3 * (raw_sample(input, row, l6, width, height) + raw_sample(input, row, r6, width, height))
        + t4 * (raw_sample(input, row, l8, width, height) + raw_sample(input, row, r8, width, height));
}

#[cube(launch_unchecked)]
fn lmmse_smooth_vertical_kernel(
    input: &Array<f32>, output: &mut Array<f32>, width: usize, height: usize,
    t0: f32, t1: f32, t2: f32, t3: f32, t4: f32,
) {
    let index = ABSOLUTE_POS;
    if index >= input.len() { terminate!(); }
    let row = index / width;
    let col = index % width;
    let u2 = if row >= 2 { row - 2 } else { 0usize.into() };
    let u4 = if row >= 4 { row - 4 } else { 0usize.into() };
    let u6 = if row >= 6 { row - 6 } else { 0usize.into() };
    let u8 = if row >= 8 { row - 8 } else { 0usize.into() };
    let d2 = row + 2;
    let d4 = row + 4;
    let d6 = row + 6;
    let d8 = row + 8;
    output[index] = t0 * input[index]
        + t1 * (raw_sample(input, u2, col, width, height) + raw_sample(input, d2, col, width, height))
        + t2 * (raw_sample(input, u4, col, width, height) + raw_sample(input, d4, col, width, height))
        + t3 * (raw_sample(input, u6, col, width, height) + raw_sample(input, d6, col, width, height))
        + t4 * (raw_sample(input, u8, col, width, height) + raw_sample(input, d8, col, width, height));
}

#[cube(launch_unchecked)]
fn lmmse_combine_kernel(
    sensor: &Array<f32>, horizontal: &Array<f32>, vertical: &Array<f32>, output: &mut Array<f32>,
    width: usize, height: usize, c00: u32, c01: u32, c10: u32, c11: u32,
) {
    let index = ABSOLUTE_POS;
    if index >= sensor.len() { terminate!(); }
    let row = index / width;
    let col = index % width;
    if bayer_color(row, col, c00, c01, c10, c11) == 1 { output[index] = sensor[index]; terminate!(); }
    let left2 = if col >= 2 { col - 2 } else { 0usize.into() };
    let left1 = if col >= 1 { col - 1 } else { 0usize.into() };
    let right1 = col + 1;
    let right2 = col + 2;
    let up2 = if row >= 2 { row - 2 } else { 0usize.into() };
    let up1 = if row >= 1 { row - 1 } else { 0usize.into() };
    let down1 = row + 1;
    let down2 = row + 2;
    let h0 = raw_sample(horizontal, row, left2, width, height);
    let h1 = raw_sample(horizontal, row, left1, width, height);
    let h2 = horizontal[index];
    let h3 = raw_sample(horizontal, row, right1, width, height);
    let h4 = raw_sample(horizontal, row, right2, width, height);
    let v0 = raw_sample(vertical, up2, col, width, height);
    let v1 = raw_sample(vertical, up1, col, width, height);
    let v2 = vertical[index];
    let v3 = raw_sample(vertical, down1, col, width, height);
    let v4 = raw_sample(vertical, down2, col, width, height);
    let h_mean = (h0 + h1 + h2 + h3 + h4) / 5.0;
    let v_mean = (v0 + v1 + v2 + v3 + v4) / 5.0;
    let h_var = ((h0-h_mean)*(h0-h_mean) + (h1-h_mean)*(h1-h_mean) + (h2-h_mean)*(h2-h_mean) + (h3-h_mean)*(h3-h_mean) + (h4-h_mean)*(h4-h_mean)) / 5.0;
    let v_var = ((v0-v_mean)*(v0-v_mean) + (v1-v_mean)*(v1-v_mean) + (v2-v_mean)*(v2-v_mean) + (v3-v_mean)*(v3-v_mean) + (v4-v_mean)*(v4-v_mean)) / 5.0;
    let wh = 1.0 / if h_var > 0.000001 { h_var } else { 0.000001f32.into() };
    let wv = 1.0 / if v_var > 0.000001 { v_var } else { 0.000001f32.into() };
    output[index] = sensor[index] + (horizontal[index] * wh + vertical[index] * wv) / (wh + wv);
}

#[cube(launch_unchecked)]
fn initialize_color_diff_kernel(
    sensor: &Array<f32>,
    green: &Array<f32>,
    output: &mut Array<f32>,
    width: usize,
    c00: u32,
    c01: u32,
    c10: u32,
    c11: u32,
    target: u32,
) {
    let index = ABSOLUTE_POS;
    if index >= sensor.len() {
        terminate!();
    }
    let row = index / width;
    let col = index % width;
    output[index] = if bayer_color(row, col, c00, c01, c10, c11) == target {
        sensor[index] - green[index]
    } else {
        0.0f32.into()
    };
}

#[cube(launch_unchecked)]
fn interpolate_color_diff_kernel(
    known: &Array<f32>,
    output: &mut Array<f32>,
    width: usize,
    height: usize,
    c00: u32,
    c01: u32,
    c10: u32,
    c11: u32,
    target: u32,
) {
    let index = ABSOLUTE_POS;
    if index >= known.len() {
        terminate!();
    }
    let row = index / width;
    let col = index % width;
    if bayer_color(row, col, c00, c01, c10, c11) == target {
        output[index] = known[index];
        terminate!();
    }
    let up = if row > 0 { row - 1 } else { 0usize.into() };
    let down = if row + 1 < height { row + 1 } else { height - 1 };
    let left = if col > 0 { col - 1 } else { 0usize.into() };
    let right = if col + 1 < width { col + 1 } else { width - 1 };
    let nw_index = up * width + left;
    let n_index = up * width + col;
    let ne_index = up * width + right;
    let w_index = row * width + left;
    let e_index = row * width + right;
    let sw_index = down * width + left;
    let s_index = down * width + col;
    let se_index = down * width + right;
    let nw_known = if bayer_color(up, left, c00, c01, c10, c11) == target { 1.0 } else { 0.0f32.into() };
    let n_known = if bayer_color(up, col, c00, c01, c10, c11) == target { 1.0 } else { 0.0f32.into() };
    let ne_known = if bayer_color(up, right, c00, c01, c10, c11) == target { 1.0 } else { 0.0f32.into() };
    let w_known = if bayer_color(row, left, c00, c01, c10, c11) == target { 1.0 } else { 0.0f32.into() };
    let e_known = if bayer_color(row, right, c00, c01, c10, c11) == target { 1.0 } else { 0.0f32.into() };
    let sw_known = if bayer_color(down, left, c00, c01, c10, c11) == target { 1.0 } else { 0.0f32.into() };
    let s_known = if bayer_color(down, col, c00, c01, c10, c11) == target { 1.0 } else { 0.0f32.into() };
    let se_known = if bayer_color(down, right, c00, c01, c10, c11) == target { 1.0 } else { 0.0f32.into() };
    let sum = known[nw_index] * nw_known + known[n_index] * n_known + known[ne_index] * ne_known
        + known[w_index] * w_known + known[e_index] * e_known
        + known[sw_index] * sw_known + known[s_index] * s_known + known[se_index] * se_known;
    let count = nw_known + n_known + ne_known + w_known + e_known + sw_known + s_known + se_known;
    output[index] = if count > 0.0 { sum / count } else { 0.0f32.into() };
}

#[cube(launch_unchecked)]
fn reconstruct_color_diff_kernel(
    green: &Array<f32>,
    red_diff: &Array<f32>,
    blue_diff: &Array<f32>,
    output: &mut Array<f32>,
) {
    let index = ABSOLUTE_POS;
    if index >= green.len() {
        terminate!();
    }
    let offset = index * 3;
    output[offset] = green[index] + red_diff[index];
    output[offset + 1] = green[index];
    output[offset + 2] = green[index] + blue_diff[index];
}

#[cube(launch_unchecked)]
fn bilinear_demosaic_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: usize,
    height: usize,
    c00: u32,
    c01: u32,
    c10: u32,
    c11: u32,
) {
    let index = ABSOLUTE_POS;
    if index >= input.len() {
        terminate!();
    }

    let row = index / width;
    let col = index % width;
    let up = if row > 0 { row - 1 } else { 0usize.into() };
    let down = if row + 1 < height { row + 1 } else { height - 1 };
    let left = if col > 0 { col - 1 } else { 0usize.into() };
    let right = if col + 1 < width { col + 1 } else { width - 1 };
    let center = input[index];
    let cross = (
        input[up * width + col]
            + input[down * width + col]
            + input[row * width + left]
            + input[row * width + right]
    ) * 0.25;
    let diagonal = (
        input[up * width + left]
            + input[up * width + right]
            + input[down * width + left]
            + input[down * width + right]
    ) * 0.25;
    let color = bayer_color(row, col, c00, c01, c10, c11);
    let offset = index * 3;

    if color == 0 {
        output[offset] = center;
        output[offset + 1] = cross;
        output[offset + 2] = diagonal;
        terminate!();
    }
    if color == 2 {
        output[offset] = diagonal;
        output[offset + 1] = cross;
        output[offset + 2] = center;
        terminate!();
    }

    let nw_color = bayer_color(up, left, c00, c01, c10, c11);
    let ne_color = bayer_color(up, right, c00, c01, c10, c11);
    let sw_color = bayer_color(down, left, c00, c01, c10, c11);
    let se_color = bayer_color(down, right, c00, c01, c10, c11);
    let nw = input[up * width + left];
    let ne = input[up * width + right];
    let sw = input[down * width + left];
    let se = input[down * width + right];
    let red_sum = if nw_color == 0 { nw } else { 0.0f32.into() }
        + if ne_color == 0 { ne } else { 0.0f32.into() }
        + if sw_color == 0 { sw } else { 0.0f32.into() }
        + if se_color == 0 { se } else { 0.0f32.into() };
    let red_count = if nw_color == 0 { 1.0 } else { 0.0f32.into() }
        + if ne_color == 0 { 1.0 } else { 0.0f32.into() }
        + if sw_color == 0 { 1.0 } else { 0.0f32.into() }
        + if se_color == 0 { 1.0 } else { 0.0f32.into() };
    let blue_sum = if nw_color == 2 { nw } else { 0.0f32.into() }
        + if ne_color == 2 { ne } else { 0.0f32.into() }
        + if sw_color == 2 { sw } else { 0.0f32.into() }
        + if se_color == 2 { se } else { 0.0f32.into() };
    let blue_count = if nw_color == 2 { 1.0 } else { 0.0f32.into() }
        + if ne_color == 2 { 1.0 } else { 0.0f32.into() }
        + if sw_color == 2 { 1.0 } else { 0.0f32.into() }
        + if se_color == 2 { 1.0 } else { 0.0f32.into() };
    output[offset] = if red_count > 0.0 { red_sum / red_count } else { center };
    output[offset + 1] = center;
    output[offset + 2] = if blue_count > 0.0 { blue_sum / blue_count } else { center };
}

#[cube(launch_unchecked)]
fn rgb_to_ycbcr_kernel(input: &Array<f32>, y: &mut Array<f32>, cb: &mut Array<f32>, cr: &mut Array<f32>) {
    let index = ABSOLUTE_POS;
    if index >= y.len() {
        terminate!();
    }
    let offset = index * 3;
    let r = input[offset];
    let g = input[offset + 1];
    let b = input[offset + 2];
    y[index] = 0.299 * r + 0.587 * g + 0.114 * b;
    cb[index] = -0.168736 * r - 0.331264 * g + 0.5 * b;
    cr[index] = 0.5 * r - 0.418688 * g - 0.081312 * b;
}

#[cube(launch_unchecked)]
fn atrous_horizontal_kernel(input: &Array<f32>, output: &mut Array<f32>, width: usize, gap: usize) {
    let index = ABSOLUTE_POS;
    if index >= input.len() {
        terminate!();
    }
    let row = index / width;
    let col = index % width;
    let x0 = if col >= gap * 2 { col - gap * 2 } else { 0usize.into() };
    let x1 = if col >= gap { col - gap } else { 0usize.into() };
    let x3 = if col + gap < width { col + gap } else { width - 1 };
    let x4 = if col + gap * 2 < width { col + gap * 2 } else { width - 1 };
    output[index] = (input[row * width + x0]
        + 4.0 * input[row * width + x1]
        + 6.0 * input[index]
        + 4.0 * input[row * width + x3]
        + input[row * width + x4]) / 16.0;
}

#[cube(launch_unchecked)]
fn atrous_vertical_kernel(input: &Array<f32>, output: &mut Array<f32>, width: usize, height: usize, gap: usize) {
    let index = ABSOLUTE_POS;
    if index >= input.len() {
        terminate!();
    }
    let row = index / width;
    let col = index % width;
    let y0 = if row >= gap * 2 { row - gap * 2 } else { 0usize.into() };
    let y1 = if row >= gap { row - gap } else { 0usize.into() };
    let y3 = if row + gap < height { row + gap } else { height - 1 };
    let y4 = if row + gap * 2 < height { row + gap * 2 } else { height - 1 };
    output[index] = (input[y0 * width + col]
        + 4.0 * input[y1 * width + col]
        + 6.0 * input[index]
        + 4.0 * input[y3 * width + col]
        + input[y4 * width + col]) / 16.0;
}

#[cube(launch_unchecked)]
fn clear_kernel(output: &mut Array<f32>) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        output[index] = 0.0;
    }
}

#[cube(launch_unchecked)]
fn accumulate_detail_kernel(
    current: &Array<f32>,
    smoothed: &Array<f32>,
    result: &mut Array<f32>,
    attenuation: f32,
) {
    let index = ABSOLUTE_POS;
    if index < current.len() {
        result[index] = result[index] + (current[index] - smoothed[index]) * attenuation;
    }
}

#[cube(launch_unchecked)]
fn ycbcr_to_rgb_kernel(
    y_current: &Array<f32>,
    y_detail: &Array<f32>,
    cb_current: &Array<f32>,
    cb_detail: &Array<f32>,
    cr_current: &Array<f32>,
    cr_detail: &Array<f32>,
    output: &mut Array<f32>,
) {
    let index = ABSOLUTE_POS;
    if index >= y_current.len() {
        terminate!();
    }
    let y = y_current[index] + y_detail[index];
    let cb = cb_current[index] + cb_detail[index];
    let cr = cr_current[index] + cr_detail[index];
    let r = y + 1.402 * cr;
    let g = y - 0.344136 * cb - 0.714136 * cr;
    let b = y + 1.772 * cb;
    let offset = index * 3;
    output[offset] = if r > 0.0 { r } else { 0.0f32.into() };
    output[offset + 1] = if g > 0.0 { g } else { 0.0f32.into() };
    output[offset + 2] = if b > 0.0 { b } else { 0.0f32.into() };
}

#[cube(launch_unchecked)]
fn gaussian_horizontal_7_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: usize,
    w0: f32,
    w1: f32,
    w2: f32,
    w3: f32,
) {
    let index = ABSOLUTE_POS;
    if index >= input.len() {
        terminate!();
    }
    let row = index / width;
    let col = index % width;
    let x0 = if col >= 3 { col - 3 } else { 0usize.into() };
    let x1 = if col >= 2 { col - 2 } else { 0usize.into() };
    let x2 = if col >= 1 { col - 1 } else { 0usize.into() };
    let x4 = if col + 1 < width { col + 1 } else { width - 1 };
    let x5 = if col + 2 < width { col + 2 } else { width - 1 };
    let x6 = if col + 3 < width { col + 3 } else { width - 1 };
    output[index] = w3 * input[row * width + x0]
        + w2 * input[row * width + x1]
        + w1 * input[row * width + x2]
        + w0 * input[index]
        + w1 * input[row * width + x4]
        + w2 * input[row * width + x5]
        + w3 * input[row * width + x6];
}

#[cube(launch_unchecked)]
fn gaussian_vertical_7_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: usize,
    height: usize,
    w0: f32,
    w1: f32,
    w2: f32,
    w3: f32,
) {
    let index = ABSOLUTE_POS;
    if index >= input.len() {
        terminate!();
    }
    let row = index / width;
    let col = index % width;
    let y0 = if row >= 3 { row - 3 } else { 0usize.into() };
    let y1 = if row >= 2 { row - 2 } else { 0usize.into() };
    let y2 = if row >= 1 { row - 1 } else { 0usize.into() };
    let y4 = if row + 1 < height { row + 1 } else { height - 1 };
    let y5 = if row + 2 < height { row + 2 } else { height - 1 };
    let y6 = if row + 3 < height { row + 3 } else { height - 1 };
    output[index] = w3 * input[y0 * width + col]
        + w2 * input[y1 * width + col]
        + w1 * input[y2 * width + col]
        + w0 * input[index]
        + w1 * input[y4 * width + col]
        + w2 * input[y5 * width + col]
        + w3 * input[y6 * width + col];
}

#[cube(launch_unchecked)]
fn sharpen_blend_mask_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: usize,
    height: usize,
    threshold: f32,
) {
    let index = ABSOLUTE_POS;
    if index >= input.len() {
        terminate!();
    }
    let row = index / width;
    let col = index % width;
    let left = if col > 0 { col - 1 } else { 0usize.into() };
    let right = if col + 1 < width { col + 1 } else { width - 1 };
    let up = if row > 0 { row - 1 } else { 0usize.into() };
    let down = if row + 1 < height { row + 1 } else { height - 1 };
    let gx = input[row * width + right] - input[row * width + left];
    let gy = input[down * width + col] - input[up * width + col];
    let gradient = (gx * gx + gy * gy).sqrt();
    // `gradient` is non-negative by construction. Keep this expression in
    // CubeCL's expanded scalar domain rather than materializing an inferred
    // host-side local, which would make shader compilation reject it.
    let t = if gradient > threshold {
        1.0f32.into()
    } else {
        gradient / threshold
    };
    output[index] = t * t * (3.0 - 2.0 * t);
}

#[cube(launch_unchecked)]
fn unsharp_combine_kernel(
    source: &Array<f32>,
    blurred: &Array<f32>,
    blend: &Array<f32>,
    output: &mut Array<f32>,
    amount: f32,
) {
    let index = ABSOLUTE_POS;
    if index < source.len() {
        output[index] = source[index] + (source[index] - blurred[index]) * amount * 2.0 * blend[index];
    }
}

#[cube(launch_unchecked)]
fn ycbcr_planes_to_rgb_kernel(
    y: &Array<f32>,
    cb: &Array<f32>,
    cr: &Array<f32>,
    output: &mut Array<f32>,
) {
    let index = ABSOLUTE_POS;
    if index >= y.len() {
        terminate!();
    }
    let r = y[index] + 1.402 * cr[index];
    let g = y[index] - 0.344136 * cb[index] - 0.714136 * cr[index];
    let b = y[index] + 1.772 * cb[index];
    let offset = index * 3;
    output[offset] = if r > 0.0 { r } else { 0.0f32.into() };
    output[offset + 1] = if g > 0.0 { g } else { 0.0f32.into() };
    output[offset + 2] = if b > 0.0 { b } else { 0.0f32.into() };
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

fn cfa_color_code(color: rawler::cfa::CFAColor) -> u32 {
    match color {
        rawler::cfa::CFAColor::RED => 0,
        rawler::cfa::CFAColor::BLUE => 2,
        _ => 1,
    }
}

fn sensor_cfa_pattern(sensor: &crate::custom_raw_pipeline::RawSensorData) -> [u32; 4] {
    [
        cfa_color_code((sensor.cfa_at)(0, 0)),
        cfa_color_code((sensor.cfa_at)(0, 1)),
        cfa_color_code((sensor.cfa_at)(1, 0)),
        cfa_color_code((sensor.cfa_at)(1, 1)),
    ]
}

/// GPU implementation of the AMaZE-compatible directional pipeline used by
/// the standard low-ISO RAW Develop mode.
pub fn amaze_demosaic(
    sensor: &crate::custom_raw_pipeline::RawSensorData,
) -> Option<Vec<[f32; 3]>> {
    const MIN_GPU_PIXELS: usize = 1_048_576;

    if sensor.data.len() < MIN_GPU_PIXELS
        || sensor.data.len() != sensor.width.saturating_mul(sensor.height)
    {
        return None;
    }
    let device = SHARED_WGPU_DEVICE.get()?;
    let [c00, c01, c10, c11] = sensor_cfa_pattern(sensor);

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let client = cubecl::wgpu::WgpuRuntime::client(device);
        let len = sensor.data.len();
        let (cube_count, cube_dim) = launch_1d(&client, len);
        let input = client.create_from_slice(bytemuck::cast_slice(&sensor.data));
        let green = client.empty(len * std::mem::size_of::<f32>());
        let red_known = client.empty(len * std::mem::size_of::<f32>());
        let blue_known = client.empty(len * std::mem::size_of::<f32>());
        let red_diff = client.empty(len * std::mem::size_of::<f32>());
        let blue_diff = client.empty(len * std::mem::size_of::<f32>());
        let output = client.empty(len * 3 * std::mem::size_of::<f32>());

        unsafe {
            amaze_green_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim.clone(),
                ArrayArg::from_raw_parts(input.clone(), len),
                ArrayArg::from_raw_parts(green.clone(), len),
                sensor.width,
                sensor.height,
                c00,
                c01,
                c10,
                c11,
            );
            initialize_color_diff_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim.clone(),
                ArrayArg::from_raw_parts(input.clone(), len),
                ArrayArg::from_raw_parts(green.clone(), len),
                ArrayArg::from_raw_parts(red_known.clone(), len),
                sensor.width,
                c00,
                c01,
                c10,
                c11,
                0,
            );
            initialize_color_diff_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim.clone(),
                ArrayArg::from_raw_parts(input, len),
                ArrayArg::from_raw_parts(green.clone(), len),
                ArrayArg::from_raw_parts(blue_known.clone(), len),
                sensor.width,
                c00,
                c01,
                c10,
                c11,
                2,
            );
            interpolate_color_diff_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim.clone(),
                ArrayArg::from_raw_parts(red_known, len),
                ArrayArg::from_raw_parts(red_diff.clone(), len),
                sensor.width,
                sensor.height,
                c00,
                c01,
                c10,
                c11,
                0,
            );
            interpolate_color_diff_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim.clone(),
                ArrayArg::from_raw_parts(blue_known, len),
                ArrayArg::from_raw_parts(blue_diff.clone(), len),
                sensor.width,
                sensor.height,
                c00,
                c01,
                c10,
                c11,
                2,
            );
            reconstruct_color_diff_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(green, len),
                ArrayArg::from_raw_parts(red_diff, len),
                ArrayArg::from_raw_parts(blue_diff, len),
                ArrayArg::from_raw_parts(output.clone(), len * 3),
            );
        }

        let bytes = client.read_one(output).ok()?;
        let demosaiced = f32::from_bytes(&bytes);
        if demosaiced.len() != len * 3 {
            return None;
        }
        Some(
            demosaiced
                .chunks_exact(3)
                .map(|pixel| [pixel[0], pixel[1], pixel[2]])
                .collect(),
        )
    }))
    .ok()
    .flatten()
}

/// GPU implementation of the IGV green interpolation and refinement path.
pub fn igv_demosaic(
    sensor: &crate::custom_raw_pipeline::RawSensorData,
) -> Option<Vec<[f32; 3]>> {
    const MIN_GPU_PIXELS: usize = 1_048_576;
    if sensor.data.len() < MIN_GPU_PIXELS
        || sensor.data.len() != sensor.width.saturating_mul(sensor.height)
    {
        return None;
    }
    let device = SHARED_WGPU_DEVICE.get()?;
    let [c00, c01, c10, c11] = sensor_cfa_pattern(sensor);
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let client = cubecl::wgpu::WgpuRuntime::client(device);
        let len = sensor.data.len();
        let (cube_count, cube_dim) = launch_1d(&client, len);
        let input = client.create_from_slice(bytemuck::cast_slice(&sensor.data));
        let pass1 = client.empty(len * std::mem::size_of::<f32>());
        let green = client.empty(len * std::mem::size_of::<f32>());
        let red_known = client.empty(len * std::mem::size_of::<f32>());
        let blue_known = client.empty(len * std::mem::size_of::<f32>());
        let red_diff = client.empty(len * std::mem::size_of::<f32>());
        let blue_diff = client.empty(len * std::mem::size_of::<f32>());
        let output = client.empty(len * 3 * std::mem::size_of::<f32>());
        unsafe {
            igv_green_pass1_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client, cube_count.clone(), cube_dim.clone(),
                ArrayArg::from_raw_parts(input.clone(), len), ArrayArg::from_raw_parts(pass1.clone(), len),
                sensor.width, sensor.height, c00, c01, c10, c11,
            );
            igv_green_refine_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client, cube_count.clone(), cube_dim.clone(),
                ArrayArg::from_raw_parts(input.clone(), len), ArrayArg::from_raw_parts(pass1, len),
                ArrayArg::from_raw_parts(green.clone(), len), sensor.width, sensor.height, c00, c01, c10, c11,
            );
            initialize_color_diff_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client, cube_count.clone(), cube_dim.clone(),
                ArrayArg::from_raw_parts(input.clone(), len), ArrayArg::from_raw_parts(green.clone(), len),
                ArrayArg::from_raw_parts(red_known.clone(), len), sensor.width, c00, c01, c10, c11, 0,
            );
            initialize_color_diff_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client, cube_count.clone(), cube_dim.clone(),
                ArrayArg::from_raw_parts(input, len), ArrayArg::from_raw_parts(green.clone(), len),
                ArrayArg::from_raw_parts(blue_known.clone(), len), sensor.width, c00, c01, c10, c11, 2,
            );
            interpolate_color_diff_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client, cube_count.clone(), cube_dim.clone(),
                ArrayArg::from_raw_parts(red_known, len), ArrayArg::from_raw_parts(red_diff.clone(), len),
                sensor.width, sensor.height, c00, c01, c10, c11, 0,
            );
            interpolate_color_diff_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client, cube_count.clone(), cube_dim.clone(),
                ArrayArg::from_raw_parts(blue_known, len), ArrayArg::from_raw_parts(blue_diff.clone(), len),
                sensor.width, sensor.height, c00, c01, c10, c11, 2,
            );
            reconstruct_color_diff_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client, cube_count, cube_dim,
                ArrayArg::from_raw_parts(green, len), ArrayArg::from_raw_parts(red_diff, len),
                ArrayArg::from_raw_parts(blue_diff, len), ArrayArg::from_raw_parts(output.clone(), len * 3),
            );
        }
        let bytes = client.read_one(output).ok()?;
        let demosaiced = f32::from_bytes(&bytes);
        if demosaiced.len() != len * 3 { return None; }
        Some(demosaiced.chunks_exact(3).map(|pixel| [pixel[0], pixel[1], pixel[2]]).collect())
    }))
    .ok()
    .flatten()
}

/// GPU implementation of the high-ISO LMMSE demosaic path.
pub fn lmmse_demosaic(
    sensor: &crate::custom_raw_pipeline::RawSensorData,
) -> Option<Vec<[f32; 3]>> {
    const MIN_GPU_PIXELS: usize = 1_048_576;
    if sensor.data.len() < MIN_GPU_PIXELS || sensor.data.len() != sensor.width.saturating_mul(sensor.height) {
        return None;
    }
    let device = SHARED_WGPU_DEVICE.get()?;
    let [c00, c01, c10, c11] = sensor_cfa_pattern(sensor);
    let mut taps = [0.0f32; 5];
    for i in 0..=4usize { taps[i] = (-((i * i) as f32) / 8.0).exp(); }
    let tap_sum = taps[0] + 2.0 * (taps[1] + taps[2] + taps[3] + taps[4]);
    for tap in &mut taps { *tap /= tap_sum; }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let client = cubecl::wgpu::WgpuRuntime::client(device);
        let len = sensor.data.len();
        let (cube_count, cube_dim) = launch_1d(&client, len);
        let input = client.create_from_slice(bytemuck::cast_slice(&sensor.data));
        let diff_h = client.empty(len * std::mem::size_of::<f32>());
        let diff_v = client.empty(len * std::mem::size_of::<f32>());
        let smooth_h = client.empty(len * std::mem::size_of::<f32>());
        let smooth_v = client.empty(len * std::mem::size_of::<f32>());
        let green = client.empty(len * std::mem::size_of::<f32>());
        let red_known = client.empty(len * std::mem::size_of::<f32>());
        let blue_known = client.empty(len * std::mem::size_of::<f32>());
        let red_diff = client.empty(len * std::mem::size_of::<f32>());
        let blue_diff = client.empty(len * std::mem::size_of::<f32>());
        let output = client.empty(len * 3 * std::mem::size_of::<f32>());
        unsafe {
            lmmse_diff_estimate_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client, cube_count.clone(), cube_dim.clone(),
                ArrayArg::from_raw_parts(input.clone(), len), ArrayArg::from_raw_parts(diff_h.clone(), len), ArrayArg::from_raw_parts(diff_v.clone(), len),
                sensor.width, sensor.height, c00, c01, c10, c11,
            );
            lmmse_smooth_horizontal_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client, cube_count.clone(), cube_dim.clone(), ArrayArg::from_raw_parts(diff_h, len), ArrayArg::from_raw_parts(smooth_h.clone(), len),
                sensor.width, sensor.height, taps[0], taps[1], taps[2], taps[3], taps[4],
            );
            lmmse_smooth_vertical_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client, cube_count.clone(), cube_dim.clone(), ArrayArg::from_raw_parts(diff_v, len), ArrayArg::from_raw_parts(smooth_v.clone(), len),
                sensor.width, sensor.height, taps[0], taps[1], taps[2], taps[3], taps[4],
            );
            lmmse_combine_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client, cube_count.clone(), cube_dim.clone(), ArrayArg::from_raw_parts(input.clone(), len),
                ArrayArg::from_raw_parts(smooth_h, len), ArrayArg::from_raw_parts(smooth_v, len), ArrayArg::from_raw_parts(green.clone(), len),
                sensor.width, sensor.height, c00, c01, c10, c11,
            );
            initialize_color_diff_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client, cube_count.clone(), cube_dim.clone(), ArrayArg::from_raw_parts(input.clone(), len), ArrayArg::from_raw_parts(green.clone(), len),
                ArrayArg::from_raw_parts(red_known.clone(), len), sensor.width, c00, c01, c10, c11, 0,
            );
            initialize_color_diff_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client, cube_count.clone(), cube_dim.clone(), ArrayArg::from_raw_parts(input, len), ArrayArg::from_raw_parts(green.clone(), len),
                ArrayArg::from_raw_parts(blue_known.clone(), len), sensor.width, c00, c01, c10, c11, 2,
            );
            interpolate_color_diff_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client, cube_count.clone(), cube_dim.clone(), ArrayArg::from_raw_parts(red_known, len), ArrayArg::from_raw_parts(red_diff.clone(), len),
                sensor.width, sensor.height, c00, c01, c10, c11, 0,
            );
            interpolate_color_diff_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client, cube_count.clone(), cube_dim.clone(), ArrayArg::from_raw_parts(blue_known, len), ArrayArg::from_raw_parts(blue_diff.clone(), len),
                sensor.width, sensor.height, c00, c01, c10, c11, 2,
            );
            reconstruct_color_diff_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client, cube_count, cube_dim, ArrayArg::from_raw_parts(green, len), ArrayArg::from_raw_parts(red_diff, len),
                ArrayArg::from_raw_parts(blue_diff, len), ArrayArg::from_raw_parts(output.clone(), len * 3),
            );
        }
        let bytes = client.read_one(output).ok()?;
        let demosaiced = f32::from_bytes(&bytes);
        if demosaiced.len() != len * 3 { return None; }
        Some(demosaiced.chunks_exact(3).map(|pixel| [pixel[0], pixel[1], pixel[2]]).collect())
    }))
    .ok()
    .flatten()
}

/// GPU implementation of the explicit Bilinear RAW Develop choice.
pub fn bilinear_demosaic(
    sensor: &crate::custom_raw_pipeline::RawSensorData,
) -> Option<Vec<[f32; 3]>> {
    const MIN_GPU_PIXELS: usize = 1_048_576;

    if sensor.data.len() < MIN_GPU_PIXELS
        || sensor.data.len() != sensor.width.saturating_mul(sensor.height)
    {
        return None;
    }
    let device = SHARED_WGPU_DEVICE.get()?;
    let [c00, c01, c10, c11] = sensor_cfa_pattern(sensor);

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let client = cubecl::wgpu::WgpuRuntime::client(device);
        let input = client.create_from_slice(bytemuck::cast_slice(&sensor.data));
        let output = client.empty(sensor.data.len() * 3 * std::mem::size_of::<f32>());
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::Static(
            sensor.data.len().div_ceil(cube_dim.num_elems() as usize) as u32,
            1,
            1,
        );

        unsafe {
            bilinear_demosaic_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(input, sensor.data.len()),
                ArrayArg::from_raw_parts(output.clone(), sensor.data.len() * 3),
                sensor.width,
                sensor.height,
                c00,
                c01,
                c10,
                c11,
            );
        }

        let bytes = client.read_one(output).ok()?;
        let demosaiced = f32::from_bytes(&bytes);
        if demosaiced.len() != sensor.data.len() * 3 {
            return None;
        }
        Some(
            demosaiced
                .chunks_exact(3)
                .map(|pixel| [pixel[0], pixel[1], pixel[2]])
                .collect(),
        )
    }))
    .ok()
    .flatten()
}

fn launch_1d<R: Runtime>(_client: &ComputeClient<R>, len: usize) -> (CubeCount, CubeDim) {
    let cube_dim = CubeDim::new_1d(256);
    let cube_count = CubeCount::Static(
        len.div_ceil(cube_dim.num_elems() as usize) as u32,
        1,
        1,
    );
    (cube_count, cube_dim)
}

fn denoise_plane(
    client: &ComputeClient<cubecl::wgpu::WgpuRuntime>,
    initial: cubecl::server::Handle,
    width: usize,
    height: usize,
    strength: f32,
    level_weights: [f32; 4],
) -> (cubecl::server::Handle, cubecl::server::Handle) {
    let len = width * height;
    let (cube_count, cube_dim) = launch_1d(client, len);
    let horizontal = client.empty(len * std::mem::size_of::<f32>());
    let mut current = initial;
    let mut next = client.empty(len * std::mem::size_of::<f32>());
    let result = client.empty(len * std::mem::size_of::<f32>());

    unsafe {
        clear_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            client,
            cube_count.clone(),
            cube_dim.clone(),
            ArrayArg::from_raw_parts(result.clone(), len),
        );
    }

    for (level, weight) in level_weights.into_iter().enumerate() {
        let gap = 1usize << level;
        let attenuation = 1.0 - (strength * weight).clamp(0.0, 1.0);
        unsafe {
            atrous_horizontal_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                client,
                cube_count.clone(),
                cube_dim.clone(),
                ArrayArg::from_raw_parts(current.clone(), len),
                ArrayArg::from_raw_parts(horizontal.clone(), len),
                width,
                gap,
            );
            atrous_vertical_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                client,
                cube_count.clone(),
                cube_dim.clone(),
                ArrayArg::from_raw_parts(horizontal.clone(), len),
                ArrayArg::from_raw_parts(next.clone(), len),
                width,
                height,
                gap,
            );
            accumulate_detail_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                client,
                cube_count.clone(),
                cube_dim.clone(),
                ArrayArg::from_raw_parts(current.clone(), len),
                ArrayArg::from_raw_parts(next.clone(), len),
                ArrayArg::from_raw_parts(result.clone(), len),
                attenuation,
            );
        }
        std::mem::swap(&mut current, &mut next);
    }

    (current, result)
}

/// Four-level à-trous RAW denoise on the shared GPU. This is structurally
/// identical to `raw_denoise::wavelet_denoise`: identical YCbCr coefficients,
/// taps, boundary clamping, detail attenuation and final non-negative clamp.
pub fn wavelet_denoise(
    rgb: &mut [[f32; 3]],
    width: usize,
    height: usize,
    strength: f32,
) -> bool {
    const MIN_GPU_PIXELS: usize = 1_048_576;
    const LUMA_WEIGHTS: [f32; 4] = [0.25, 0.45, 0.35, 0.15];
    const CHROMA_WEIGHTS: [f32; 4] = [1.0, 0.9, 0.6, 0.3];

    if strength <= 0.0
        || rgb.len() < MIN_GPU_PIXELS
        || rgb.len() != width.saturating_mul(height)
    {
        return false;
    }
    let Some(device) = SHARED_WGPU_DEVICE.get() else {
        return false;
    };

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let client = cubecl::wgpu::WgpuRuntime::client(device);
        let len = rgb.len();
        let (cube_count, cube_dim) = launch_1d(&client, len);
        let input = client.create_from_slice(bytemuck::cast_slice(rgb));
        let y = client.empty(len * std::mem::size_of::<f32>());
        let cb = client.empty(len * std::mem::size_of::<f32>());
        let cr = client.empty(len * std::mem::size_of::<f32>());

        unsafe {
            rgb_to_ycbcr_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim.clone(),
                ArrayArg::from_raw_parts(input, len * 3),
                ArrayArg::from_raw_parts(y.clone(), len),
                ArrayArg::from_raw_parts(cb.clone(), len),
                ArrayArg::from_raw_parts(cr.clone(), len),
            );
        }

        let (y_current, y_detail) = denoise_plane(&client, y, width, height, strength, LUMA_WEIGHTS);
        let (cb_current, cb_detail) = denoise_plane(&client, cb, width, height, strength, CHROMA_WEIGHTS);
        let (cr_current, cr_detail) = denoise_plane(&client, cr, width, height, strength, CHROMA_WEIGHTS);
        let output = client.empty(len * 3 * std::mem::size_of::<f32>());

        unsafe {
            ycbcr_to_rgb_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(y_current, len),
                ArrayArg::from_raw_parts(y_detail, len),
                ArrayArg::from_raw_parts(cb_current, len),
                ArrayArg::from_raw_parts(cb_detail, len),
                ArrayArg::from_raw_parts(cr_current, len),
                ArrayArg::from_raw_parts(cr_detail, len),
                ArrayArg::from_raw_parts(output.clone(), len * 3),
            );
        }

        let bytes = client.read_one(output).ok()?;
        let denoised = f32::from_bytes(&bytes);
        if denoised.len() != len * 3 {
            return None;
        }
        for (pixel, denoised) in rgb.iter_mut().zip(denoised.chunks_exact(3)) {
            *pixel = [denoised[0], denoised[1], denoised[2]];
        }
        Some(())
    }))
    .ok()
    .flatten()
    .is_some()
}

/// GPU implementation of the default radius-1 luminance unsharp mask.
/// Other radii and Richardson-Lucy deconvolution intentionally use their
/// existing CPU paths until their dynamic/iterative kernels are ported.
pub fn unsharp_mask(
    rgb: &mut [[f32; 3]],
    width: usize,
    height: usize,
    amount: f32,
    radius: f32,
) -> bool {
    const MIN_GPU_PIXELS: usize = 1_048_576;

    if amount <= 0.0
        || rgb.len() < MIN_GPU_PIXELS
        || rgb.len() != width.saturating_mul(height)
        || (radius - 1.0).abs() > 1e-6
    {
        return false;
    }
    let Some(device) = SHARED_WGPU_DEVICE.get() else {
        return false;
    };

    // These are precisely the CPU path's normalized Gaussian weights for
    // sigma=1 and radius=ceil(3*sigma)=3, passed as scalar uniforms.
    let mut weights = [0.0f32; 4];
    let mut sum = 0.0f32;
    for offset in -3i32..=3 {
        sum += (-((offset * offset) as f32) / 2.0).exp();
    }
    for offset in 0..=3usize {
        weights[offset] = (-((offset * offset) as f32) / 2.0).exp() / sum;
    }

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let client = cubecl::wgpu::WgpuRuntime::client(device);
        let len = rgb.len();
        let (cube_count, cube_dim) = launch_1d(&client, len);
        let input = client.create_from_slice(bytemuck::cast_slice(rgb));
        let y = client.empty(len * std::mem::size_of::<f32>());
        let cb = client.empty(len * std::mem::size_of::<f32>());
        let cr = client.empty(len * std::mem::size_of::<f32>());
        let horizontal = client.empty(len * std::mem::size_of::<f32>());
        let blurred = client.empty(len * std::mem::size_of::<f32>());
        let blend = client.empty(len * std::mem::size_of::<f32>());
        let sharpened = client.empty(len * std::mem::size_of::<f32>());
        let output = client.empty(len * 3 * std::mem::size_of::<f32>());

        unsafe {
            rgb_to_ycbcr_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim.clone(),
                ArrayArg::from_raw_parts(input, len * 3),
                ArrayArg::from_raw_parts(y.clone(), len),
                ArrayArg::from_raw_parts(cb.clone(), len),
                ArrayArg::from_raw_parts(cr.clone(), len),
            );
            gaussian_horizontal_7_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim.clone(),
                ArrayArg::from_raw_parts(y.clone(), len),
                ArrayArg::from_raw_parts(horizontal.clone(), len),
                width,
                weights[0],
                weights[1],
                weights[2],
                weights[3],
            );
            gaussian_vertical_7_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim.clone(),
                ArrayArg::from_raw_parts(horizontal, len),
                ArrayArg::from_raw_parts(blurred.clone(), len),
                width,
                height,
                weights[0],
                weights[1],
                weights[2],
                weights[3],
            );
            sharpen_blend_mask_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim.clone(),
                ArrayArg::from_raw_parts(y.clone(), len),
                ArrayArg::from_raw_parts(blend.clone(), len),
                width,
                height,
                0.04,
            );
            unsharp_combine_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim.clone(),
                ArrayArg::from_raw_parts(y, len),
                ArrayArg::from_raw_parts(blurred, len),
                ArrayArg::from_raw_parts(blend, len),
                ArrayArg::from_raw_parts(sharpened.clone(), len),
                amount,
            );
            ycbcr_planes_to_rgb_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(sharpened, len),
                ArrayArg::from_raw_parts(cb, len),
                ArrayArg::from_raw_parts(cr, len),
                ArrayArg::from_raw_parts(output.clone(), len * 3),
            );
        }

        let bytes = client.read_one(output).ok()?;
        let sharpened = f32::from_bytes(&bytes);
        if sharpened.len() != len * 3 {
            return None;
        }
        for (pixel, sharpened) in rgb.iter_mut().zip(sharpened.chunks_exact(3)) {
            *pixel = [sharpened[0], sharpened[1], sharpened[2]];
        }
        Some(())
    }))
    .ok()
    .flatten()
    .is_some()
}
