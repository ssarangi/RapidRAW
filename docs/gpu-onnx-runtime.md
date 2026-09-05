# NVIDIA GPU inference runtime

RawNIND uses ONNX Runtime's CUDA execution provider when a CUDA runtime pack
is present. The normal application build continues to include the CPU runtime
so it works on every supported machine.

## Building the NVIDIA variant (Linux x86_64)

Set `RAPIDRAW_ONNX_RUNTIME=cuda` for the build. `src-tauri/build.rs` downloads
the pinned, checksum-verified official Microsoft ONNX Runtime 1.22 CUDA package
and places its core, CUDA provider, shared provider, and license notices in
`src-tauri/resources/onnxruntime-gpu/`. Tauri then includes that directory in
the app bundle.

The application detects the complete pack before ONNX Runtime is initialized,
sets `ORT_DYLIB_PATH` to it, and RawNIND creates a CUDA session. If creation
fails, it logs the reason and falls back to the CPU session without losing the
edit.

## CUDA dependencies

The pinned ONNX Runtime 1.22 CUDA provider requires these Linux SONAMEs:

- `libcublasLt.so.12`, `libcublas.so.12`, `libcurand.so.10`,
  `libcufft.so.11`, `libcudart.so.12`, and `libnvrtc.so.12`
- `libcudnn.so.9`
- an NVIDIA driver compatible with CUDA 12

The driver must be installed by the user. CUDA Runtime and cuDNN can either be
installed system-wide from NVIDIA's CUDA 12/cuDNN 9 packages or bundled as a
separate NVIDIA acceleration pack. Do not substitute cuDNN 8: ONNX Runtime's
CUDA 12 package requires cuDNN 9.

The official ONNX Runtime CUDA provider is MIT-licensed and carries its own
third-party notices. NVIDIA permits redistribution of listed CUDA runtime
libraries under its EULA; retain the NVIDIA notices and distribute only the
exact runtime files used by the pack. Confirm cuDNN's current redistribution
terms before publishing a self-contained acceleration pack.

## Validation

On a GPU-enabled build, the application log must contain:

`RawNIND editor session is using CUDAExecutionProvider`

If that line is absent, the logged CUDA provider error identifies the missing
driver/runtime library and RawNIND will use the deliberately throttled CPU
fallback.
