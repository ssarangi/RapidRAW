use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

fn verify_sha256(path: &Path, expected_hash: &str) -> Result<bool, io::Error> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    let hash_bytes = hasher.finalize();
    let calculated_hash = hex::encode(hash_bytes);
    Ok(calculated_hash == expected_hash)
}

fn download_and_verify(
    url: &str,
    dest_path: &Path,
    expected_hash: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let temp_filename = dest_path.file_name().unwrap();
    let temp_path = out_dir.join(temp_filename);

    println!(
        "cargo:warning=Downloading to temporary path: {:?}",
        temp_path
    );
    let mut response = reqwest::blocking::get(url)?;

    if !response.status().is_success() {
        let status = response.status();
        let error_body = response
            .text()
            .unwrap_or_else(|_| "Could not read error body".to_string());
        return Err(format!("Download failed with status {}: {}", status, error_body).into());
    }

    let mut temp_file = fs::File::create(&temp_path)?;
    response.copy_to(&mut temp_file)?;
    println!("cargo:warning=Download complete. Verifying file integrity...");

    match verify_sha256(&temp_path, expected_hash) {
        Ok(true) => {
            fs::copy(&temp_path, dest_path)?;
            fs::remove_file(&temp_path)?;
            println!(
                "cargo:warning=Successfully downloaded and verified {:?}.",
                dest_path
            );
            Ok(())
        }
        Ok(false) => {
            fs::remove_file(&temp_path)?;
            Err("Verification failed! The downloaded file is corrupt.".into())
        }
        Err(e) => {
            fs::remove_file(&temp_path).ok();
            Err(format!("Could not verify file after download: {}", e).into())
        }
    }
}

/// Fetches the official ONNX Runtime CUDA package into a separate resource
/// directory. The CPU runtime remains the default resource; this pack is
/// selected at app startup only when all CUDA provider files are present.
fn download_cuda_runtime_pack(dest_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    const PACKAGE_URL: &str = "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.gpu.linux/1.22.0/microsoft.ml.onnxruntime.gpu.linux.1.22.0.nupkg";
    const PACKAGE_SHA256: &str = "d3dabd8235b7a04a67838a3b93437420f1a709528a8d8611e08bedbb08206d1d";
    const CORE_SHA256: &str = "5ede9537894434c221de73323aa1f644967afeed6569fd570ea356a366e98ecd";
    const CUDA_PROVIDER_SHA256: &str =
        "937e28f8d3fca43c77be9afa6795b6b600fe5bc6c97ba4e513fe05f0ee22a546";
    const NATIVE_ROOT: &str = "runtimes/linux-x64/native/";
    const FILES: &[&str] = &[
        "libonnxruntime.so",
        "libonnxruntime_providers_cuda.so",
        "libonnxruntime_providers_shared.so",
        "LICENSE",
        "ThirdPartyNotices.txt",
    ];

    let core = dest_dir.join("libonnxruntime.so");
    let provider = dest_dir.join("libonnxruntime_providers_cuda.so");
    if core.is_file()
        && provider.is_file()
        && verify_sha256(&core, CORE_SHA256)?
        && verify_sha256(&provider, CUDA_PROVIDER_SHA256)?
    {
        println!("cargo:warning=ONNX Runtime CUDA pack already exists and is valid.");
        return Ok(());
    }

    fs::create_dir_all(dest_dir)?;
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let package_path = out_dir.join("onnxruntime-gpu-linux-x64-1.22.0.nupkg");
    download_and_verify(PACKAGE_URL, &package_path, PACKAGE_SHA256)?;

    let package = fs::File::open(&package_path)?;
    let mut archive = zip::ZipArchive::new(package)?;
    for file_name in FILES {
        let archive_name = if matches!(*file_name, "LICENSE" | "ThirdPartyNotices.txt") {
            (*file_name).to_string()
        } else {
            format!("{NATIVE_ROOT}{file_name}")
        };
        let mut entry = archive.by_name(&archive_name)?;
        let destination = dest_dir.join(file_name);
        let mut output = fs::File::create(destination)?;
        io::copy(&mut entry, &mut output)?;
    }
    fs::remove_file(package_path)?;
    println!("cargo:warning=Installed ONNX Runtime CUDA acceleration pack.");
    Ok(())
}

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let (download_filename, lib_name, expected_hash) =
        match (target_os.as_str(), target_arch.as_str()) {
            ("windows", "x86_64") => (
                "onnxruntime-windows-x86_64.dll",
                "onnxruntime.dll",
                "579b636403983254346a5c1d80bd28f1519cd1e284cd204f8d4ff41f8d711559",
            ),
            ("windows", "aarch64") => (
                "onnxruntime-windows-aarch64.dll",
                "onnxruntime.dll",
                "79281671a386ed1baab9dbdbb09fe55f99577011472e9526cf9d0b468bb6bcc7",
            ),
            ("linux", "x86_64") => (
                "libonnxruntime-linux-x86_64.so",
                "libonnxruntime.so",
                "3da6146e14e7b8aaec625dde11d6114c7457c87a5f93d744897da8781e35c673",
            ),
            ("linux", "aarch64") => (
                "libonnxruntime-linux-aarch64.so",
                "libonnxruntime.so",
                "0afd69a0ae38c5099fd0e8604dda398ac43dee67cd9c6394b5142b19e82528de",
            ),
            ("macos", "x86_64") => (
                "libonnxruntime-macos-x86_64.dylib",
                "libonnxruntime.dylib",
                "283e595e61cf65df7a6b1d59a1616cbd35c8b6399dd90d799d99b71a3ff83160",
            ),
            ("macos", "aarch64") => (
                "libonnxruntime-macos-aarch64.dylib",
                "libonnxruntime.dylib",
                "2b885992d3d6fa4130d39ec84a80d7504ff52750027c547bb22c86165f19406a",
            ),
            ("android", "aarch64") => (
                "libonnxruntime-android-arm64-v8a.so",
                "libonnxruntime.so",
                "999ecfdb5b5a13e4097487773b6d71ce8a075408a237daab072e8f5e817bd78e",
            ),
            _ => panic!("Unsupported target: {}-{}", target_os, target_arch),
        };

    let dest_dir = if target_os == "android" {
        manifest_dir.join("libs").join("arm64-v8a")
    } else {
        manifest_dir.join("resources")
    };

    fs::create_dir_all(&dest_dir).unwrap();
    let dest_path = dest_dir.join(lib_name);

    let mut is_valid = false;
    if dest_path.exists() {
        match verify_sha256(&dest_path, expected_hash) {
            Ok(true) => {
                println!(
                    "cargo:warning=ONNX Runtime library already exists and is valid. Skipping download."
                );
                is_valid = true;
            }
            Ok(false) => {
                println!(
                    "cargo:warning=File {:?} exists but has incorrect hash. Deleting and re-downloading.",
                    dest_path
                );
                fs::remove_file(&dest_path).unwrap();
            }
            Err(e) => {
                println!(
                    "cargo:warning=Could not verify file {:?}: {}. Re-downloading.",
                    dest_path, e
                );
            }
        }
    }

    if !is_valid {
        println!(
            "cargo:warning=Downloading ONNX Runtime library for {}-{}...",
            target_os, target_arch
        );
        let base_url =
            "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/onnxruntimes-v1.22.0/";
        let download_url = format!("{}{}?download=true", base_url, download_filename);
        println!("cargo:warning=URL: {}", download_url);

        if let Err(e) = download_and_verify(&download_url, &dest_path, expected_hash) {
            panic!("Failed to download and verify ONNX Runtime library: {}", e);
        }
    }

    if env::var("RAPIDRAW_ONNX_RUNTIME").as_deref() == Ok("cuda") {
        if target_os != "linux" || target_arch != "x86_64" {
            panic!(
                "The CUDA ONNX Runtime pack is currently supported only for linux-x86_64 builds"
            );
        }
        let cuda_dir = manifest_dir.join("resources").join("onnxruntime-gpu");
        if let Err(error) = download_cuda_runtime_pack(&cuda_dir) {
            panic!("Failed to download ONNX Runtime CUDA pack: {error}");
        }
    }

    if target_os == "android" {
        let jni_libs_dir = manifest_dir.join("gen/android/app/src/main/jniLibs/arm64-v8a");
        fs::create_dir_all(&jni_libs_dir).unwrap();
        fs::copy(&dest_path, jni_libs_dir.join(lib_name)).unwrap();

        println!("cargo:rustc-env=ORT_LIB_LOCATION={}", dest_dir.display());
        println!("cargo:rustc-env=ORT_STRATEGY=manual");
        println!("cargo:rustc-link-search=native={}", dest_dir.display());
    }

    println!("cargo:rerun-if-changed=build.rs");

    tauri_build::build()
}
