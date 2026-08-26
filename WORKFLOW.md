# Fork workflow

This is a fork of [CyberTimon/RapidRAW](https://github.com/CyberTimon/RapidRAW) (AGPL-3.0),
extended with a deeper auto-culling pipeline for multi-camera, high-volume shooting
(bursts, wildlife/bird photography). Credit and license terms to the upstream project
and its contributors are preserved; this fork stays AGPL-3.0.

## Remotes & branches

- `origin` = this fork (`ssarangi/RapidRAW`)
- `upstream` = `CyberTimon/RapidRAW`
- `main` — mirrors `upstream/main` exactly. Fast-forward only. **Never commit here.**
- `culling` — the actual work branch. Tracks `origin/culling`.

## Syncing upstream into the fork

```
git checkout main
git merge --ff-only upstream/main
git push origin main

git checkout culling
git merge main
git push
```

Do this roughly weekly — upstream ships close to daily, and conflict cost grows
faster than linearly with drift.

`rerere.enabled=true` is set locally. Trivial recurring conflicts (e.g. a
command-registration line in `lib.rs`) auto-resolve after you fix them once.

## Known upstream hot files

These get touched often by upstream — treat edits here as expected merge-conflict
surface, keep them as small as possible, and never run a formatter across the
whole file:

- `src-tauri/src/culling.rs` — existing scoring/grouping engine (see below)
- `src-tauri/src/image_processing.rs`
- `src/components/panel/library/CullingView.tsx`
- `src/components/panel/MainLibrary.tsx`
- `src/components/modals/CullingModal.tsx`
- `src/store/*` (library/UI state)
- `src/hooks/useKeyboardShortcuts.ts`, `useAppContextMenus.ts`

## Isolation discipline for new work

- New capability → new file/module where at all possible.
- Touch existing files only for minimal wiring (one new command registration
  line, one new render hook/prop).
- Prefer additive optional fields over restructuring existing types/structs.
- Never reformat a file beyond your actual change.

## Existing culling engine baseline (as forked)

`src-tauri/src/culling.rs` already computes, per image:

- **Sharpness**: Laplacian variance over the whole downscaled (720px) thumbnail.
- **Center focus**: Laplacian variance over a center 50% crop — assumes the
  subject is centered, which breaks for off-center compositions.
- **Exposure**: histogram dark/bright clipping penalty.
- **quality_score** = 0.40·sharpness + 0.35·center_focus + 0.25·exposure (normalized).
- **Grouping**: perceptual hash (`image_hasher`, DoubleGradient, 16x16) distance
  threshold, BFS-clustered. No EXIF timestamp/continuous-shooting signal used.

No subject or eye detection, despite `ort` (ONNX Runtime) already being a
dependency used elsewhere in the codebase (masking, denoise, depth). No
personalization from a user's own rating history.

Gaps this fork is targeting:
1. Subject-aware sharpness/focus (via the existing `ort` infra) instead of the
   naive center-crop assumption.
2. EXIF-timestamp/continuous-shooting-based burst grouping, independent of or
   combined with perceptual-hash similarity.
3. Eye-open / closed-eye detection for portrait and wildlife subjects.
4. Personalized ranking learned from the user's own historical star/reject picks.
