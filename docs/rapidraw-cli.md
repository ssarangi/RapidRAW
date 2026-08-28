# RapidRAW CLI

`rapidraw-cli` is the window-free companion for catalog maintenance and local batch analysis. Successful commands write one JSON value to stdout. Diagnostics and errors go to stderr.

## Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | Command completed. |
| `1` | Operational failure, such as unavailable storage, a database error, or a missing model artifact. |
| `2` | Invalid command or arguments. |
| `3` | A job was cancelled. |

## Catalog Lifecycle

```sh
rapidraw-cli library create --name "Archive" --database /data/archive/rapidraw.db
rapidraw-cli library open --database /data/archive/rapidraw.db
rapidraw-cli library add-root --database /data/archive/rapidraw.db --path /photos/2026 --label "2026"
rapidraw-cli library scan --database /data/archive/rapidraw.db --root 1
rapidraw-cli library roots --database /data/archive/rapidraw.db
rapidraw-cli library metrics --database /data/archive/rapidraw.db
rapidraw-cli library remove-root --database /data/archive/rapidraw.db --root 1
```

`library scan` creates a durable `catalog_scan` job and returns its ID with the scan totals. It does not require a Tauri window. A configured but unavailable NAS root remains a catalog root; the scan reports the failure rather than silently deleting catalog records.

## Batch Analysis

```sh
rapidraw-cli tags run --database /data/archive/rapidraw.db --models-dir /data/models/visual
rapidraw-cli tags run --database /data/archive/rapidraw.db --models-dir /data/models/visual --with-bioclip
rapidraw-cli faces detect --database /data/archive/rapidraw.db --face-models-dir /data/models/face --root 1
rapidraw-cli faces recognize --database /data/archive/rapidraw.db --face-models-dir /data/models/face --root 1
rapidraw-cli people list --database /data/archive/rapidraw.db
rapidraw-cli people images --database /data/archive/rapidraw.db --person 12
rapidraw-cli cull analyze --database /data/archive/rapidraw.db --root 1
```

`cull analyze` only proposes a review session. It does not move, delete, rate, or label source files. RAW/JPEG siblings are represented as one logical capture. The headless path currently performs deterministic duplicate, sharpness, focus, and exposure analysis; subject-aware model analysis remains an explicit desktop workflow.

## Inspection And Derivatives

```sh
rapidraw-cli jobs list --database /data/archive/rapidraw.db
rapidraw-cli jobs show --database /data/archive/rapidraw.db --id <job-id>
rapidraw-cli models list
rapidraw-cli models verify --id ram-plus-onnx --models-dir /data/models/visual
rapidraw-cli restore list --database /data/archive/rapidraw.db --image 42
rapidraw-cli restore run --database /data/archive/rapidraw.db --image 42 --models-dir /data/models/visual --operation raw_denoise
```

Model commands only report an installed pack as runnable when the current application contains a compatible runtime adapter. Face recognition is limited to the YuNet/SFace stack; other visible packs are never claimed to be runnable.
