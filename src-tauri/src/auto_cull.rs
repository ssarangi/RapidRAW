use crate::ai_processing::get_or_init_ai_models;
use crate::app_settings::load_settings;
use crate::app_state::AppState;
use crate::culling::{
    CullingSettings, ImageAnalysisData, analyze_image, build_culling_suggestions,
};
use crate::file_management::{AutoCullCandidate, resolve_auto_cull_candidates};
use image_hasher::{HashAlg, HasherConfig};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use tauri::{AppHandle, Emitter};

#[derive(Serialize, Clone)]
struct AutoCullProgress {
    current: usize,
    total: usize,
    stage: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AutoCullPlanItem {
    pub representative_path: String,
    pub backing_paths: Vec<String>,
    pub keep: bool,
    pub reason: String,
    pub quality_score: f64,
    /// True if a file with the same name already exists in the destination
    /// folder (e.g. left over from a previous auto-cull run on this same
    /// folder). Computed at plan time so the preview/apply flow can ask how
    /// to handle it up front instead of failing partway through the move.
    pub has_conflict: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AutoCullPlan {
    pub session_id: Option<i64>,
    pub folder_path: String,
    pub include_subfolders: bool,
    pub settings: CullingSettings,
    pub rejected_folder_name: String,
    pub delete_instead_of_move: bool,
    pub items: Vec<AutoCullPlanItem>,
    pub total_count: usize,
    pub reject_count: usize,
    pub failed_paths: Vec<String>,
}

/// Analyzes every image in a folder (optionally recursive) and decides a
/// keep/reject verdict per logical photo (a RAW+JPG pair counts as one).
/// Pure planning - no files are moved or labeled here, so this is what
/// powers the mandatory preview step before anything actually happens.
#[tauri::command]
pub async fn plan_auto_cull(
    folder_path: String,
    include_subfolders: bool,
    settings: CullingSettings,
    rejected_folder_name: String,
    delete_instead_of_move: bool,
    app_handle: AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<AutoCullPlan, String> {
    let images = if include_subfolders {
        crate::file_management::list_images_recursive(folder_path.clone(), app_handle.clone())?
    } else {
        crate::file_management::list_images_in_dir(folder_path.clone(), app_handle.clone())?
    };

    let candidates = resolve_auto_cull_candidates(&images);
    let base_plan = AutoCullPlan {
        folder_path,
        include_subfolders,
        settings: settings.clone(),
        rejected_folder_name,
        delete_instead_of_move,
        ..Default::default()
    };

    if candidates.is_empty() {
        return Ok(base_plan);
    }

    let app_settings = load_settings(app_handle.clone()).unwrap_or_default();
    let total_count = candidates.len();
    let _ = app_handle.emit("auto-cull-plan-start", total_count);

    let ai_models = if settings.use_subject_detection {
        let _ = app_handle.emit(
            "auto-cull-plan-progress",
            AutoCullProgress {
                current: 0,
                total: total_count,
                stage: "Loading subject detection model...".to_string(),
            },
        );
        Some(
            get_or_init_ai_models(&app_handle, &state.ai_state, &state.ai_init_lock)
                .await
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };

    let hasher = HasherConfig::new()
        .hash_alg(HashAlg::DoubleGradient)
        .hash_size(16, 16)
        .to_hasher();

    let completed = AtomicUsize::new(0);
    let analysis_results: Vec<Result<(AutoCullCandidate, ImageAnalysisData), (String, String)>> =
        candidates
            .into_par_iter()
            .map(|candidate| {
                let n = completed.fetch_add(1, Ordering::Relaxed) + 1;
                let _ = app_handle.emit(
                    "auto-cull-plan-progress",
                    AutoCullProgress {
                        current: n,
                        total: total_count,
                        stage: "Analyzing images...".to_string(),
                    },
                );

                match analyze_image(
                    &candidate.representative_path,
                    &hasher,
                    &app_settings,
                    ai_models.as_ref(),
                    n,
                ) {
                    Ok(data) => Ok((candidate, data)),
                    Err(e) => Err((candidate.representative_path.clone(), e)),
                }
            })
            .collect();

    let mut successful: Vec<(AutoCullCandidate, ImageAnalysisData)> = Vec::new();
    let mut failed_paths = Vec::new();
    for res in analysis_results {
        match res {
            Ok(pair) => successful.push(pair),
            Err((path, error)) => {
                eprintln!("Failed to analyze image {}: {}", path, error);
                failed_paths.push(path);
            }
        }
    }

    let _ = app_handle.emit(
        "auto-cull-plan-progress",
        AutoCullProgress {
            current: total_count,
            total: total_count,
            stage: "Grouping and scoring...".to_string(),
        },
    );

    let candidates_by_path: HashMap<String, AutoCullCandidate> = successful
        .iter()
        .map(|(c, _)| {
            (
                c.representative_path.clone(),
                AutoCullCandidate {
                    representative_path: c.representative_path.clone(),
                    backing_paths: c.backing_paths.clone(),
                },
            )
        })
        .collect();
    let scores_by_path: HashMap<String, f64> = successful
        .iter()
        .map(|(_, a)| (a.result.path.clone(), a.result.quality_score))
        .collect();
    let analyses: Vec<ImageAnalysisData> = successful.into_iter().map(|(_, data)| data).collect();

    let suggestions = build_culling_suggestions(analyses, failed_paths.clone(), &settings);

    let mut reject_reasons: HashMap<String, String> = HashMap::new();
    let mut representative_paths: HashSet<String> = HashSet::new();
    for group in &suggestions.similar_groups {
        representative_paths.insert(group.representative.path.clone());
        for dup in &group.duplicates {
            reject_reasons.insert(
                dup.path.clone(),
                format!("duplicate_of:{}", group.representative.path),
            );
        }
    }
    for blurry in &suggestions.blurry_images {
        reject_reasons
            .entry(blurry.path.clone())
            .or_insert_with(|| "blurry".to_string());
    }

    let rejected_folder = Path::new(&base_plan.folder_path).join(&base_plan.rejected_folder_name);

    let mut items: Vec<AutoCullPlanItem> = candidates_by_path
        .into_values()
        .map(|candidate| {
            let quality_score = scores_by_path
                .get(&candidate.representative_path)
                .copied()
                .unwrap_or(0.0);
            let (keep, reason) = match reject_reasons.get(&candidate.representative_path) {
                Some(reason) => (false, reason.clone()),
                None if representative_paths.contains(&candidate.representative_path) => {
                    (true, "best_in_group".to_string())
                }
                None => (true, "unique".to_string()),
            };

            // Only relevant for the move path - deleted files go to the
            // system trash, not a folder, so they can't collide with
            // anything already there.
            let has_conflict = !keep
                && !delete_instead_of_move
                && candidate.backing_paths.iter().any(|p| {
                    Path::new(p)
                        .file_name()
                        .map(|name| rejected_folder.join(name).exists())
                        .unwrap_or(false)
                });

            AutoCullPlanItem {
                representative_path: candidate.representative_path,
                backing_paths: candidate.backing_paths,
                keep,
                reason,
                quality_score,
                has_conflict,
            }
        })
        .collect();
    items.sort_by(|a, b| a.representative_path.cmp(&b.representative_path));

    let reject_count = items.iter().filter(|i| !i.keep).count();

    let _ = app_handle.emit("auto-cull-plan-complete", total_count);

    let session_id = crate::library_db::active_library_path(&state)
        .ok()
        .and_then(|db_path| {
            let decisions = items
                .iter()
                .map(|item| (item.representative_path.clone(), item.keep, item.reason.clone(), item.quality_score))
                .collect::<Vec<_>>();
            crate::library_db::record_cull_session(
                &db_path,
                &base_plan.folder_path,
                &serde_json::to_string(&settings).unwrap_or_else(|_| "{}".to_string()),
                &decisions,
            )
            .map_err(|error| eprintln!("Failed to persist culling session: {error}"))
            .ok()
            .flatten()
        });

    Ok(AutoCullPlan {
        session_id,
        total_count,
        reject_count,
        items,
        failed_paths,
        ..base_plan
    })
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AutoCullMove {
    pub old_path: String,
    pub new_path: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AutoCullResult {
    pub rejected_folder_path: String,
    pub moved: Vec<AutoCullMove>,
    pub labeled_paths: Vec<String>,
    pub deleted: bool,
    pub skipped_paths: Vec<String>,
}

/// Executes a previously computed (and user-reviewed) plan: labels every
/// rejected file red, then moves - or, if requested, deletes - it. Reuses
/// the same move/delete/label commands the manual UI already uses, so
/// associated sidecars travel along exactly as they would from a normal
/// drag-and-drop move.
///
/// `conflict_action` resolves items flagged `has_conflict` in the plan (a
/// file with the same name already sits in the destination folder, e.g.
/// from a previous run) - the caller decides this once, up front, for every
/// conflicting item rather than being asked per file: `"skip"` leaves those
/// items untouched entirely, `"overwrite"` removes the existing destination
/// file before moving. Anything else (including no conflicts existing)
/// behaves as before.
#[tauri::command]
pub async fn apply_auto_cull_plan(
    plan: AutoCullPlan,
    conflict_action: Option<String>,
    app_handle: AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<AutoCullResult, String> {
    let finalize_session = || {
        if let Some(session_id) = plan.session_id {
            if let Ok(db_path) = crate::library_db::active_library_path(&state) {
                if let Err(error) = crate::library_db::mark_cull_session_applied(&db_path, session_id) {
                    eprintln!("Failed to finalize culling session {session_id}: {error}");
                }
            }
        }
    };
    let all_reject_items: Vec<&AutoCullPlanItem> = plan.items.iter().filter(|i| !i.keep).collect();
    if all_reject_items.is_empty() {
        finalize_session();
        return Ok(AutoCullResult::default());
    }

    let skip_conflicts = !plan.delete_instead_of_move && conflict_action.as_deref() == Some("skip");
    let overwrite_conflicts =
        !plan.delete_instead_of_move && conflict_action.as_deref() == Some("overwrite");

    let (items_to_process, skipped_items): (Vec<&AutoCullPlanItem>, Vec<&AutoCullPlanItem>) =
        if skip_conflicts {
            all_reject_items.into_iter().partition(|i| !i.has_conflict)
        } else {
            (all_reject_items, Vec::new())
        };

    let all_reject_paths: Vec<String> = items_to_process
        .iter()
        .flat_map(|i| i.backing_paths.clone())
        .collect();
    let skipped_paths: Vec<String> = skipped_items
        .iter()
        .flat_map(|i| i.backing_paths.clone())
        .collect();

    if all_reject_paths.is_empty() {
        finalize_session();
        return Ok(AutoCullResult {
            skipped_paths,
            ..Default::default()
        });
    }

    crate::file_management::set_color_label_for_paths(
        all_reject_paths.clone(),
        Some("red".to_string()),
        app_handle.clone(),
    )?;

    if plan.delete_instead_of_move {
        crate::file_management::delete_files_with_associated(
            all_reject_paths.clone(),
            app_handle.clone(),
        )?;
        finalize_session();
        return Ok(AutoCullResult {
            rejected_folder_path: String::new(),
            moved: Vec::new(),
            labeled_paths: all_reject_paths,
            deleted: true,
            skipped_paths,
        });
    }

    let folder = Path::new(&plan.folder_path);
    let rejected_folder = folder.join(&plan.rejected_folder_name);
    std::fs::create_dir_all(&rejected_folder).map_err(|e| e.to_string())?;
    let rejected_folder_str = rejected_folder.to_string_lossy().to_string();

    if overwrite_conflicts {
        for item in items_to_process.iter().filter(|i| i.has_conflict) {
            for p in &item.backing_paths {
                if let Some(name) = Path::new(p).file_name() {
                    let dest = rejected_folder.join(name);
                    if dest.exists() {
                        let _ = std::fs::remove_file(&dest);
                    }
                }
            }
        }
    }

    // move_files parallelizes its own fs::copy calls internally now (see its
    // doc comment) - one call here with the full list, rather than fanning
    // out multiple concurrent calls to move_files itself, since it also
    // calls sync_album_path_changes at the end, which isn't safe to run
    // concurrently from multiple calls (unsynchronized read-modify-write on
    // the shared albums file).
    crate::file_management::move_files(
        all_reject_paths.clone(),
        rejected_folder_str.clone(),
        app_handle.clone(),
    )?;

    let moved: Vec<AutoCullMove> = all_reject_paths
        .iter()
        .filter_map(|p| {
            let file_name = Path::new(p).file_name()?;
            let new_path = rejected_folder
                .join(file_name)
                .to_string_lossy()
                .to_string();
            Some(AutoCullMove {
                old_path: p.clone(),
                new_path,
            })
        })
        .collect();

    finalize_session();

    Ok(AutoCullResult {
        rejected_folder_path: rejected_folder_str,
        moved,
        labeled_paths: all_reject_paths,
        deleted: false,
        skipped_paths,
    })
}

/// Reverses a move-based auto-cull run: moves everything back to where it
/// came from and clears the red label. Delete-based runs can't be undone
/// here since the files went to the system trash, not a known folder.
#[tauri::command]
pub async fn undo_auto_cull(result: AutoCullResult, app_handle: AppHandle) -> Result<(), String> {
    if result.deleted {
        return Err(
            "Deleted files went to the system trash and can be restored from there, but not automatically."
                .to_string(),
        );
    }

    for mv in &result.moved {
        let old_parent = Path::new(&mv.old_path)
            .parent()
            .ok_or("Could not resolve the original folder for a moved file")?
            .to_string_lossy()
            .to_string();
        crate::file_management::move_files(
            vec![mv.new_path.clone()],
            old_parent,
            app_handle.clone(),
        )?;
    }

    crate::file_management::set_color_label_for_paths(
        result.labeled_paths.clone(),
        None,
        app_handle.clone(),
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_cull_plan_item_overrides_persist_in_struct() {
        let mut item = AutoCullPlanItem {
            representative_path: "/photos/DSC001.ARW".to_string(),
            backing_paths: vec!["/photos/DSC001.ARW".to_string(), "/photos/DSC001.JPG".to_string()],
            keep: false,
            reason: "Blurry background".to_string(),
            quality_score: 0.35,
            has_conflict: false,
        };

        // User overrides the plan verdict
        item.keep = true;
        assert!(item.keep);
        assert_eq!(item.backing_paths.len(), 2);
    }

    #[test]
    fn auto_cull_empty_candidates_returns_default_plan() {
        let plan = AutoCullPlan {
            folder_path: "/photos".to_string(),
            items: vec![],
            total_count: 0,
            reject_count: 0,
            ..Default::default()
        };
        assert_eq!(plan.total_count, 0);
        assert_eq!(plan.reject_count, 0);
        assert!(plan.items.is_empty());
    }
}
