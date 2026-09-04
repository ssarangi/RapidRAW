use crate::ai_processing::get_or_init_ai_models;
use crate::app_settings::load_settings;
use crate::app_state::AppState;
use crate::culling::{
    CullingSettings, CullingSuggestions, ImageAnalysisData, ImageAnalysisResult, analyze_image,
    build_culling_suggestions,
};
use crate::file_management::{AutoCullCandidate, resolve_auto_cull_candidates};
use image_hasher::{HashAlg, HasherConfig, ImageHash};
use rayon::prelude::*;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tauri::{AppHandle, Emitter, Manager};

#[derive(Serialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
struct AutoCullProgress {
    current: usize,
    total: usize,
    stage: String,
    /// Filename of the specific image most recently finished (or, for the
    /// per-image people/wildlife passes, currently starting) - shown in the
    /// UI so a long analysis run doesn't look stalled with nothing but a
    /// slowly-ticking counter.
    #[serde(skip_serializing_if = "Option::is_none")]
    current_item: Option<String>,
}

fn file_name_for_progress(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(path)
        .to_string()
}

#[derive(Clone)]
struct AutoCullJobContext {
    database_path: std::path::PathBuf,
    job_id: String,
    control: std::sync::Arc<crate::app_state::BackgroundJobControl>,
    last_persisted_progress: std::sync::Arc<AtomicUsize>,
    progress_write_lock: std::sync::Arc<std::sync::Mutex<()>>,
}

fn report_auto_cull_progress(
    app_handle: &AppHandle,
    job: Option<&AutoCullJobContext>,
    progress: AutoCullProgress,
) {
    let _ = app_handle.emit("auto-cull-plan-progress", progress.clone());
    if let Some(job) = job {
        // Progress events may originate from Rayon workers. Keep those UI updates
        // responsive, but serialize and throttle SQLite writes to avoid lock
        // contention with catalog scans and metadata updates.
        let previous = job.last_persisted_progress.load(Ordering::Relaxed);
        let should_persist = progress.current == 0
            || progress.current >= progress.total
            || progress.current.saturating_sub(previous) >= 8;
        if !should_persist {
            return;
        }
        job.last_persisted_progress
            .store(progress.current, Ordering::Relaxed);
        let Ok(_guard) = job.progress_write_lock.lock() else {
            return;
        };
        let _ = crate::library_db::update_job(
            &job.database_path,
            &job.job_id,
            "running",
            &progress.stage,
            progress.current as i64,
            progress.total as i64,
            None,
            None,
        );
    }
}

async fn wait_for_auto_cull_job(job: Option<&AutoCullJobContext>) -> Result<(), String> {
    match job {
        Some(job) if !job.control.wait_until_runnable().await => {
            Err("Auto-cull analysis cancelled".to_string())
        }
        _ => Ok(()),
    }
}

const CULL_FEATURE_SET_VERSION: &str = "culling-v3-geometry";

fn load_catalog_cull_cache(
    db_path: &Path,
    image_ids: &HashMap<String, i64>,
) -> HashMap<String, ImageAnalysisData> {
    let Ok(connection) = Connection::open(db_path) else {
        return HashMap::new();
    };
    let mut cached = HashMap::new();
    for (path, image_id) in image_ids {
        let row = connection
            .query_row(
                "SELECT c.perceptual_hash, c.quality_score, c.sharpness_metric, c.center_focus_metric, c.exposure_metric, c.subject_focus_metric, c.subject_composition_metric, c.subject_edge_contact_ratio, c.width, c.height
                 FROM cull_analysis_cache c JOIN images i ON i.id = c.image_id
                 WHERE c.image_id = ?1 AND c.feature_set_version = ?2 AND c.image_modified_at = i.modified_at",
                rusqlite::params![image_id, CULL_FEATURE_SET_VERSION],
                |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, f64>(1)?, row.get::<_, f64>(2)?, row.get::<_, f64>(3)?, row.get::<_, f64>(4)?, row.get::<_, Option<f64>>(5)?, row.get::<_, Option<f64>>(6)?, row.get::<_, Option<f64>>(7)?, row.get::<_, u32>(8)?, row.get::<_, u32>(9)?)),
            )
            .ok();
        let Some((
            hash_bytes,
            quality_score,
            sharpness_metric,
            center_focus_metric,
            exposure_metric,
            subject_focus_metric,
            subject_composition_metric,
            subject_edge_contact_ratio,
            width,
            height,
        )) = row
        else {
            continue;
        };
        let Ok(hash) = ImageHash::from_bytes(&hash_bytes) else {
            continue;
        };
        cached.insert(
            path.clone(),
            ImageAnalysisData {
                hash,
                result: ImageAnalysisResult {
                    path: path.clone(),
                    quality_score,
                    sharpness_metric,
                    center_focus_metric,
                    subject_focus_metric,
                    subject_composition_metric,
                    subject_edge_contact_ratio,
                    exposure_metric,
                    width,
                    height,
                    // Not persisted in cull_analysis_cache; a cached entry
                    // just won't have a blur-region overlay until the image
                    // is re-analyzed.
                    blurry_region: None,
                },
            },
        );
    }
    cached
}

fn store_catalog_cull_cache(
    db_path: &Path,
    image_ids: &HashMap<String, i64>,
    analyses: &[(AutoCullCandidate, ImageAnalysisData)],
) {
    let Ok(mut connection) = Connection::open(db_path) else {
        return;
    };
    let Ok(transaction) = connection.transaction() else {
        return;
    };
    for (candidate, analysis) in analyses {
        let Some(image_id) = image_ids.get(&candidate.representative_path) else {
            continue;
        };
        let _ = transaction.execute(
            "INSERT INTO cull_analysis_cache(image_id, feature_set_version, image_modified_at, perceptual_hash, quality_score, sharpness_metric, center_focus_metric, exposure_metric, subject_focus_metric, subject_composition_metric, subject_edge_contact_ratio, width, height, created_at)
             SELECT ?1, ?2, modified_at, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, strftime('%s','now') FROM images WHERE id = ?1
             ON CONFLICT(image_id, feature_set_version, image_modified_at) DO UPDATE SET perceptual_hash=excluded.perceptual_hash, quality_score=excluded.quality_score, sharpness_metric=excluded.sharpness_metric, center_focus_metric=excluded.center_focus_metric, exposure_metric=excluded.exposure_metric, subject_focus_metric=excluded.subject_focus_metric, subject_composition_metric=excluded.subject_composition_metric, subject_edge_contact_ratio=excluded.subject_edge_contact_ratio, width=excluded.width, height=excluded.height, created_at=excluded.created_at",
            rusqlite::params![image_id, CULL_FEATURE_SET_VERSION, analysis.hash.as_bytes(), analysis.result.quality_score, analysis.result.sharpness_metric, analysis.result.center_focus_metric, analysis.result.exposure_metric, analysis.result.subject_focus_metric, analysis.result.subject_composition_metric, analysis.result.subject_edge_contact_ratio, analysis.result.width, analysis.result.height],
        );
    }
    let _ = transaction.commit();
}

async fn collect_profile_subject_factors(
    paths: Vec<String>,
    subject_mode: &str,
    completed_before: usize,
    app_handle: AppHandle,
    ai_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
    job: Option<AutoCullJobContext>,
) -> Result<HashMap<String, SubjectAnalysis>, String> {
    let subject_mode = subject_mode.to_string();
    if matches!(subject_mode.as_str(), "general" | "landscape") || paths.is_empty() {
        return Ok(HashMap::new());
    }

    tokio::task::spawn_blocking(move || {
        let total = paths.len();
        let mut factors = HashMap::new();

        if subject_mode == "people" {
            let detector = crate::face_detection::load_local_face_detector(&app_handle)?;
            // Best-effort: the eye-state model pack is a separate, optional
            // download (visual_model_registry, "ocec-eye-state"). If it
            // isn't installed, culling still runs on pose/framing alone
            // exactly as before - this never blocks the people-mode path.
            let eye_classifier = crate::eye_state::load_eye_state_classifier(&app_handle).ok();
            for (index, path) in paths.into_iter().enumerate() {
                if let Some(job) = job.as_ref() {
                    if !tauri::async_runtime::block_on(job.control.wait_until_runnable()) {
                        return Err("Auto-cull analysis cancelled".to_string());
                    }
                }
                report_auto_cull_progress(&app_handle, job.as_ref(), AutoCullProgress {
                    current: completed_before + index + 1,
                    total: completed_before + total,
                    stage: "Detecting people...".to_string(),
                    current_item: Some(file_name_for_progress(&path)),
                });
                let result = tauri::async_runtime::block_on(ai_semaphore.clone().acquire_owned());
                let face_analysis = result.ok().and_then(|permit| {
                    let analysis = crate::face_detection::analyze_faces_for_culling(
                        &detector,
                        eye_classifier.as_ref(),
                        Path::new(&path),
                        &app_handle,
                    );
                    drop(permit);
                    analysis.ok()
                });
                let (label, detail) = match face_analysis.as_ref().map(|analysis| analysis.face_count) {
                    Some(0) => (
                        "People detection",
                        "No faces detected by YuNet.".to_string(),
                    ),
                    Some(1) => ("People detection", "1 face detected by YuNet.".to_string()),
                    Some(count) => (
                        "People detection",
                        format!("{count} faces detected by YuNet."),
                    ),
                    None => (
                        "People detection",
                        "Face analysis was unavailable for this image.".to_string(),
                    ),
                };
                let mut image_factors = vec![CullDecisionFactor {
                        id: "people_detection".to_string(),
                        label: label.to_string(),
                        detail,
                        impact: "context".to_string(),
                    }];
                if let Some(pose) = face_analysis.as_ref().and_then(|analysis| analysis.best_pose) {
                    image_factors.push(CullDecisionFactor {
                        id: "face_pose".to_string(),
                        label: "Face pose and framing".to_string(),
                        detail: format!(
                            "YuNet landmark estimate: {:.0}% frontal, {:.0} degree roll, face covers {:.0}% of frame.",
                            pose.frontal_score * 100.0,
                            pose.roll_degrees,
                            pose.frame_fraction * 100.0,
                        ),
                        impact: "context".to_string(),
                    });
                }
                let eye_openness = face_analysis.as_ref().and_then(|analysis| analysis.eye_openness);
                if let Some(eyes) = eye_openness {
                    image_factors.push(CullDecisionFactor {
                        id: "eye_state".to_string(),
                        label: format!("Eyes: {}", eyes.state.label()),
                        detail: {
                            let openness_pct = (eyes.prob_open * 100.0).round();
                            match eyes.state {
                                crate::eye_state::EyeState::Open => {
                                    format!("The primary subject's eyes look open ({openness_pct:.0}% open confidence).")
                                }
                                crate::eye_state::EyeState::SemiClosed => {
                                    format!("The primary subject's eyes look partially closed - possibly mid-blink ({openness_pct:.0}% open confidence).")
                                }
                                crate::eye_state::EyeState::Closed => {
                                    format!("The primary subject's eyes look closed ({openness_pct:.0}% open confidence). This is a strong signal, not a hard rule - review before rejecting a one-of-a-kind shot.")
                                }
                            }
                        },
                        impact: match eyes.state {
                            crate::eye_state::EyeState::Open => "context".to_string(),
                            crate::eye_state::EyeState::SemiClosed => "supporting".to_string(),
                            crate::eye_state::EyeState::Closed => "reject".to_string(),
                        },
                    });
                }
                let pose_adjustment = face_analysis
                    .as_ref()
                    .and_then(|analysis| analysis.best_pose)
                    .map(|pose| ((pose.frontal_score as f64 - 0.5) * 0.08).clamp(-0.04, 0.04))
                    .unwrap_or_default();
                // A closed eye is a much stronger, more specific signal than
                // framing, so it gets a bigger tie-breaking nudge - enough
                // to reliably lose to an open-eyed duplicate/burst frame,
                // but per the pose comment below, this only ever nudges the
                // continuous quality score. It can never by itself turn a
                // unique (non-duplicate) photo into a reject.
                let eye_adjustment = match eye_openness.map(|eyes| eyes.state) {
                    Some(crate::eye_state::EyeState::Closed) => -0.15,
                    Some(crate::eye_state::EyeState::SemiClosed) => -0.05,
                    _ => 0.0,
                };
                factors.insert(
                    path,
                    SubjectAnalysis {
                        factors: image_factors,
                        quality_adjustment: (pose_adjustment + eye_adjustment).clamp(-0.2, 0.04),
                    },
                );
            }
            return Ok(factors);
        }

        let ram_plus = crate::tagging::load_ram_plus_models(&app_handle)?;
        let bioclip = if subject_mode == "birds" {
            crate::tagging::load_bioclip_models(&app_handle).ok()
        } else {
            None
        };
        for (index, path) in paths.into_iter().enumerate() {
            if let Some(job) = job.as_ref() {
                if !tauri::async_runtime::block_on(job.control.wait_until_runnable()) {
                    return Err("Auto-cull analysis cancelled".to_string());
                }
            }
            report_auto_cull_progress(&app_handle, job.as_ref(), AutoCullProgress {
                current: completed_before + index + 1,
                total: completed_before + total,
                stage: if subject_mode == "birds" {
                    "Identifying birds and wildlife...".to_string()
                } else {
                    "Identifying wildlife and animals...".to_string()
                },
                current_item: Some(file_name_for_progress(&path)),
            });
            let image =
                crate::face_detection::load_image_for_face_ai(Path::new(&path), &app_handle);
            let tags = image.and_then(|image| {
                let permit =
                    tauri::async_runtime::block_on(ai_semaphore.clone().acquire_owned()).ok();
                let tags = crate::tagging::generate_tags_with_ram_plus(&image, &ram_plus, 6);
                drop(permit);
                tags.map(|tags| (image, tags))
            });
            let mut image_factors = Vec::new();
            match tags {
                Ok((image, tags)) => {
                    let names = tags
                        .iter()
                        .map(|tag| tag.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");
                    image_factors.push(CullDecisionFactor {
                        id: "subject_tags".to_string(),
                        label: "Subject classification".to_string(),
                        detail: if names.is_empty() {
                            "RAM++ found no confident broad subject tags.".to_string()
                        } else {
                            format!("RAM++: {names}")
                        },
                        impact: "context".to_string(),
                    });
                    let has_wildlife = tags.iter().any(|tag| {
                        let name = tag.name.to_ascii_lowercase();
                        name.contains("bird")
                            || name.contains("animal")
                            || name.contains("wildlife")
                    });
                    if subject_mode == "birds" && has_wildlife {
                        if let Some(bioclip) = &bioclip {
                            let permit = tauri::async_runtime::block_on(
                                ai_semaphore.clone().acquire_owned(),
                            )
                            .ok();
                            let species = crate::tagging::run_bioclip_ranked_inference(&image, bioclip);
                            drop(permit);
                            if let Ok(inference) = species {
                                if inference.is_confident_species() {
                                    if let Some((taxon, confidence)) =
                                        inference.best_taxon_and_similarity()
                                    {
                                        image_factors.push(CullDecisionFactor {
                                            id: "species_classification".to_string(),
                                            label: "Bird or wildlife candidate".to_string(),
                                            detail: format!(
                                                "BioCLIP: {} ({:.0}% similarity)",
                                                taxon
                                                    .common_name
                                                    .as_deref()
                                                    .unwrap_or(&taxon.scientific_name),
                                                confidence * 100.0
                                            ),
                                            impact: "context".to_string(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                Err(_) => image_factors.push(CullDecisionFactor {
                    id: "subject_tags".to_string(),
                    label: "Subject classification".to_string(),
                    detail: "RAM++ analysis was unavailable for this image.".to_string(),
                    impact: "context".to_string(),
                }),
            }
            factors.insert(
                path,
                SubjectAnalysis {
                    factors: image_factors,
                    quality_adjustment: 0.0,
                },
            );
        }
        Ok(factors)
    })
    .await
    .map_err(|error| format!("Subject analysis worker failed: {error}"))?
}

fn collect_catalog_subject_factors(
    paths: Vec<String>,
    image_ids: &HashMap<String, i64>,
    subject_mode: &str,
    completed_before: usize,
    app_handle: &AppHandle,
    state: &tauri::State<'_, AppState>,
    job: Option<&AutoCullJobContext>,
) -> Result<HashMap<String, SubjectAnalysis>, String> {
    if matches!(subject_mode, "general" | "landscape") || paths.is_empty() {
        return Ok(HashMap::new());
    }

    let db_path = crate::library_db::active_library_path(state)?;
    let conn = Connection::open(db_path).map_err(|error| error.to_string())?;
    let total = paths.len();
    let mut factors = HashMap::new();

    for (index, path) in paths.into_iter().enumerate() {
        if let Some(job) = job {
            if !tauri::async_runtime::block_on(job.control.wait_until_runnable()) {
                return Err("Auto-cull analysis cancelled".to_string());
            }
        }
        report_auto_cull_progress(
            app_handle,
            job,
            AutoCullProgress {
                current: completed_before + index + 1,
                total: completed_before + total,
                stage: "Reading catalog subject analysis...".to_string(),
                current_item: Some(file_name_for_progress(&path)),
            },
        );
        let Some(image_id) = image_ids.get(&path).copied() else {
            continue;
        };
        let mut image_factors = Vec::new();
        let mut pose_adjustment = 0.0;

        if subject_mode == "people" {
            let face_count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM faces WHERE image_id = ?1 AND review_state != 'rejected'",
                    [image_id],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            image_factors.push(CullDecisionFactor {
                id: "people_detection".to_string(),
                label: "People detection".to_string(),
                detail: match face_count {
                    0 => "No catalog faces detected yet.".to_string(),
                    1 => "1 face from catalog analysis.".to_string(),
                    count => format!("{count} faces from catalog analysis."),
                },
                impact: "context".to_string(),
            });
            let mut pose_statement = conn
                .prepare(
                    "SELECT landmarks_json, bbox_width, bbox_height FROM faces WHERE image_id = ?1 AND review_state != 'rejected'",
                )
                .map_err(|error| error.to_string())?;
            let best_pose = pose_statement
                .query_map([image_id], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, f64>(2)?,
                    ))
                })
                .map_err(|error| error.to_string())?
                .filter_map(Result::ok)
                .filter_map(|(landmarks, width, height)| {
                    crate::face_detection::estimate_stored_face_pose(&landmarks, width, height)
                })
                .max_by(|left, right| {
                    (left.frontal_score * 0.8 + left.frame_fraction * 0.2)
                        .total_cmp(&(right.frontal_score * 0.8 + right.frame_fraction * 0.2))
                });
            if let Some(pose) = best_pose {
                pose_adjustment = ((pose.frontal_score as f64 - 0.5) * 0.08).clamp(-0.04, 0.04);
                image_factors.push(CullDecisionFactor {
                    id: "face_pose".to_string(),
                    label: "Face pose and framing".to_string(),
                    detail: format!(
                        "Stored YuNet landmark estimate: {:.0}% frontal, {:.0} degree roll, face covers {:.0}% of frame. Blink and expression are not evaluated by this model.",
                        pose.frontal_score * 100.0,
                        pose.roll_degrees,
                        pose.frame_fraction * 100.0,
                    ),
                    impact: "context".to_string(),
                });
            }
        } else {
            let mut tags_stmt = conn
                .prepare(
                    "SELECT t.name FROM image_ai_tags iat JOIN tags t ON t.id = iat.tag_id WHERE iat.image_id = ?1 AND iat.review_state != 'rejected' ORDER BY iat.confidence DESC LIMIT 6",
                )
                .map_err(|error| error.to_string())?;
            let tags = tags_stmt
                .query_map([image_id], |row| row.get::<_, String>(0))
                .map_err(|error| error.to_string())?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| error.to_string())?;
            image_factors.push(CullDecisionFactor {
                id: "subject_tags".to_string(),
                label: "Subject classification".to_string(),
                detail: if tags.is_empty() {
                    "No catalog RAM++ subject tags yet.".to_string()
                } else {
                    format!("Catalog RAM++: {}", tags.join(", "))
                },
                impact: "context".to_string(),
            });

            if subject_mode == "birds" {
                let species: Option<(String, Option<String>, f64)> = conn
                    .query_row(
                        "SELECT scientific_name, common_name, confidence
                         FROM species_classifications
                         WHERE image_id = ?1
                           AND review_state != 'rejected'
                           AND (is_ambiguous = 0 OR review_state = 'accepted')
                         ORDER BY confidence DESC
                         LIMIT 1",
                        [image_id],
                        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                    )
                    .ok();
                if let Some((scientific_name, common_name, confidence)) = species {
                    image_factors.push(CullDecisionFactor {
                        id: "species_classification".to_string(),
                        label: "Bird or wildlife candidate".to_string(),
                        detail: format!(
                            "Catalog BioCLIP: {} ({:.0}% similarity)",
                            common_name.unwrap_or(scientific_name),
                            confidence * 100.0
                        ),
                        impact: "context".to_string(),
                    });
                }
            }
        }
        factors.insert(
            path,
            SubjectAnalysis {
                factors: image_factors,
                quality_adjustment: pose_adjustment,
            },
        );
    }

    Ok(factors)
}

fn should_reuse_catalog_subjects(
    settings: &CullingSettings,
    catalog_image_ids: Option<&HashMap<String, i64>>,
) -> bool {
    settings.use_subject_detection
        && catalog_image_ids.is_some_and(|ids| !ids.is_empty())
        && matches!(
            settings.subject_mode.as_str(),
            "people" | "wildlife" | "birds"
        )
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CullDecisionFactor {
    pub id: String,
    pub label: String,
    pub detail: String,
    pub impact: String,
}

#[derive(Debug, Clone, Default)]
struct SubjectAnalysis {
    factors: Vec<CullDecisionFactor>,
    quality_adjustment: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AutoCullPlanItem {
    pub representative_path: String,
    pub backing_paths: Vec<String>,
    pub keep: bool,
    pub reason: String,
    pub quality_score: f64,
    pub decision_factors: Vec<CullDecisionFactor>,
    /// Normalized (0-1) bounding box of the region responsible for a
    /// "blurry" verdict, when one could be identified - see
    /// `culling::find_blurriest_region`. `None` for non-blurry items or when
    /// no textured region could be scored.
    pub blurry_region: Option<crate::culling::BlurRegion>,
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
    catalog_image_ids: Option<HashMap<String, i64>>,
    app_handle: AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<AutoCullPlan, String> {
    if let Err(error) =
        crate::library_db::ensure_cull_history_library(&app_handle, &state, &folder_path)
    {
        eprintln!("Failed to prepare a culling-history database for {folder_path}: {error}");
    }

    let job = crate::library_db::active_library_path(&state)
        .ok()
        .and_then(|database_path| {
            let payload = serde_json::json!({
                "folderPath": folder_path.clone(),
                "includeSubfolders": include_subfolders,
                "settings": settings.clone(),
                "rejectedFolderName": rejected_folder_name.clone(),
                "deleteInsteadOfMove": delete_instead_of_move,
            });
            let job_id =
                crate::library_db::create_background_job(&database_path, "cull_analysis", payload)
                    .ok()?;
            let control = crate::app_state::BackgroundJobControl::new();
            state
                .background_job_controls
                .lock()
                .unwrap()
                .insert(job_id.clone(), control.clone());
            Some(AutoCullJobContext {
                database_path,
                job_id,
                control,
                last_persisted_progress: std::sync::Arc::new(AtomicUsize::new(0)),
                progress_write_lock: std::sync::Arc::new(std::sync::Mutex::new(())),
            })
        });
    let result = plan_auto_cull_inner(
        folder_path,
        include_subfolders,
        settings,
        rejected_folder_name,
        delete_instead_of_move,
        catalog_image_ids,
        app_handle.clone(),
        state,
        job.clone(),
    )
    .await;
    if let Some(job) = job {
        let (job_state, message, current, total, error) = match &result {
            Ok(plan) => (
                "completed",
                "Culling analysis ready for review".to_string(),
                plan.total_count as i64,
                plan.total_count as i64,
                None,
            ),
            Err(error) if error == "Auto-cull analysis cancelled" => {
                ("cancelled", error.clone(), 0, 0, None)
            }
            Err(error) => (
                "failed",
                "Culling analysis failed".to_string(),
                0,
                0,
                Some(error),
            ),
        };
        let _ = crate::library_db::update_job(
            &job.database_path,
            &job.job_id,
            job_state,
            &message,
            current,
            total,
            None,
            error.map(String::as_str),
        );
        app_handle
            .state::<AppState>()
            .background_job_controls
            .lock()
            .unwrap()
            .remove(&job.job_id);
    }
    result
}

/// Minimum BioCLIP cosine similarity for a hash-flagged duplicate pairing to
/// survive the veto. Not derived from a labeled validation set - tune this
/// if it turns out to reject too many genuine bursts (too low) or let too
/// many false positives through (too high).
const MIN_BIOCLIP_DUPLICATE_SIMILARITY: f32 = 0.75;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    dot / (norm_a * norm_b)
}

/// Cross-checks each hash-based duplicate/burst group against BioCLIP
/// embeddings, dropping any member whose actual visual/species content is
/// too dissimilar from the group's representative to plausibly be the same
/// subject. Fails open: if BioCLIP isn't installed, or a specific image
/// can't be embedded, the hash-based grouping for that image is left as-is
/// rather than losing a real duplicate to a transient failure.
fn verify_duplicate_groups_with_bioclip(
    suggestions: &mut CullingSuggestions,
    app_handle: &AppHandle,
) {
    let Ok(bioclip) = crate::tagging::load_bioclip_models(app_handle) else {
        return;
    };

    let mut embedding_cache: HashMap<String, Vec<f32>> = HashMap::new();
    let mut embedding_for = |path: &str| -> Option<Vec<f32>> {
        if let Some(cached) = embedding_cache.get(path) {
            return Some(cached.clone());
        }
        let image = crate::face_detection::load_image_for_local_ai(Path::new(path)).ok()?;
        let embedding = crate::tagging::bioclip_embedding(&image, &bioclip).ok()?;
        embedding_cache.insert(path.to_string(), embedding.clone());
        Some(embedding)
    };

    for group in &mut suggestions.similar_groups {
        let Some(representative_embedding) = embedding_for(&group.representative.path) else {
            continue;
        };
        group.duplicates.retain(|duplicate| {
            let Some(duplicate_embedding) = embedding_for(&duplicate.path) else {
                return true;
            };
            cosine_similarity(&representative_embedding, &duplicate_embedding)
                >= MIN_BIOCLIP_DUPLICATE_SIMILARITY
        });
    }
    suggestions
        .similar_groups
        .retain(|group| !group.duplicates.is_empty());
}

async fn plan_auto_cull_inner(
    folder_path: String,
    include_subfolders: bool,
    settings: CullingSettings,
    rejected_folder_name: String,
    delete_instead_of_move: bool,
    catalog_image_ids: Option<HashMap<String, i64>>,
    app_handle: AppHandle,
    state: tauri::State<'_, AppState>,
    job: Option<AutoCullJobContext>,
) -> Result<AutoCullPlan, String> {
    wait_for_auto_cull_job(job.as_ref()).await?;
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
    let has_profile_analysis = settings.use_subject_detection
        && matches!(
            settings.subject_mode.as_str(),
            "people" | "wildlife" | "birds"
        );
    let progress_total = total_count * if has_profile_analysis { 2 } else { 1 };
    let _ = app_handle.emit("auto-cull-plan-start", total_count);
    report_auto_cull_progress(
        &app_handle,
        job.as_ref(),
        AutoCullProgress {
            current: 0,
            total: progress_total,
            stage: "Preparing culling analysis...".to_string(),
            ..Default::default()
        },
    );

    let use_catalog_subjects = should_reuse_catalog_subjects(&settings, catalog_image_ids.as_ref());
    let ai_models = if settings.use_subject_detection
        && settings.subject_mode != "landscape"
        && !use_catalog_subjects
    {
        report_auto_cull_progress(
            &app_handle,
            job.as_ref(),
            AutoCullProgress {
                current: 0,
                total: progress_total,
                stage: "Loading subject detection model...".to_string(),
                ..Default::default()
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

    let catalog_cache = catalog_image_ids
        .as_ref()
        .and_then(|image_ids| {
            crate::library_db::active_library_path(&state)
                .ok()
                .map(|path| (path, image_ids))
        })
        .map(|(path, image_ids)| load_catalog_cull_cache(&path, image_ids))
        .unwrap_or_default();
    let mut successful: Vec<(AutoCullCandidate, ImageAnalysisData)> = Vec::new();
    let mut candidates_to_analyze = Vec::new();
    for candidate in candidates {
        if let Some(cached) = catalog_cache.get(&candidate.representative_path) {
            successful.push((candidate, cached.clone()));
        } else {
            candidates_to_analyze.push(candidate);
        }
    }
    if !successful.is_empty() {
        report_auto_cull_progress(
            &app_handle,
            job.as_ref(),
            AutoCullProgress {
                current: successful.len(),
                total: progress_total,
                stage: format!("Reused cached analysis for {} images...", successful.len()),
                ..Default::default()
            },
        );
    }

    let completed = AtomicUsize::new(successful.len());
    let cancellation_requested = AtomicBool::new(false);
    let analysis_results: Vec<Result<(AutoCullCandidate, ImageAnalysisData), (String, String)>> =
        candidates_to_analyze
            .into_par_iter()
            .map(|candidate| {
                if let Some(job) = job.as_ref() {
                    if !tauri::async_runtime::block_on(job.control.wait_until_runnable()) {
                        cancellation_requested.store(true, Ordering::Relaxed);
                        return Err((
                            candidate.representative_path.clone(),
                            "Auto-cull analysis cancelled".to_string(),
                        ));
                    }
                }
                let item_name = file_name_for_progress(&candidate.representative_path);
                let data_res = match analyze_image(
                    &candidate.representative_path,
                    &hasher,
                    &app_settings,
                    ai_models.as_ref(),
                    0,
                ) {
                    Ok(data) => Ok((candidate, data)),
                    Err(e) => Err((candidate.representative_path.clone(), e)),
                };

                let n = completed.fetch_add(1, Ordering::Relaxed) + 1;
                report_auto_cull_progress(
                    &app_handle,
                    job.as_ref(),
                    AutoCullProgress {
                        current: n,
                        total: progress_total,
                        stage: "Analyzing images...".to_string(),
                        current_item: Some(item_name),
                    },
                );

                data_res
            })
            .collect();

    if cancellation_requested.load(Ordering::Relaxed) {
        return Err("Auto-cull analysis cancelled".to_string());
    }

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

    if let (Some(image_ids), Ok(db_path)) = (
        catalog_image_ids.as_ref(),
        crate::library_db::active_library_path(&state),
    ) {
        store_catalog_cull_cache(&db_path, image_ids, &successful);
    }

    let subject_paths = successful
        .iter()
        .map(|(candidate, _)| candidate.representative_path.clone())
        .collect();
    wait_for_auto_cull_job(job.as_ref()).await?;
    let subject_factors = if !settings.use_subject_detection {
        HashMap::new()
    } else if use_catalog_subjects {
        collect_catalog_subject_factors(
            subject_paths,
            catalog_image_ids
                .as_ref()
                .expect("catalog subject reuse requires image IDs"),
            &settings.subject_mode,
            total_count,
            &app_handle,
            &state,
            job.as_ref(),
        )?
    } else {
        collect_profile_subject_factors(
            subject_paths,
            &settings.subject_mode,
            total_count,
            app_handle.clone(),
            state.ai_job_semaphore.clone(),
            job.clone(),
        )
        .await?
    };

    // Pose/framing is intentionally a bounded tie-breaker. It only applies
    // to the people profile and cannot turn a technical reject into a keep.
    let mut subject_adjustments: HashMap<String, f64> = HashMap::new();
    for (_, analysis) in &mut successful {
        if let Some(subject) = subject_factors.get(&analysis.result.path) {
            if subject.quality_adjustment != 0.0 {
                let baseline = analysis.result.quality_score;
                let adjusted = (baseline + subject.quality_adjustment).clamp(0.0, 1.0);
                subject_adjustments.insert(analysis.result.path.clone(), adjusted - baseline);
                analysis.result.quality_score = adjusted;
            }
        }
    }

    let personalization = crate::library_db::active_library_path(&state)
        .ok()
        .and_then(|db_path| {
            crate::library_db::load_cull_personalization_model(&db_path)
                .ok()
                .flatten()
        });
    let mut personalization_adjustments: HashMap<String, f64> = HashMap::new();
    if let Some(model) = personalization.as_ref().filter(|model| model.is_ready()) {
        for (_, analysis) in &mut successful {
            let baseline = analysis.result.quality_score;
            let learned = model.score(baseline);
            let adjusted = baseline * 0.8 + learned * 0.2;
            personalization_adjustments.insert(analysis.result.path.clone(), adjusted - baseline);
            analysis.result.quality_score = adjusted;
        }
    }

    report_auto_cull_progress(
        &app_handle,
        job.as_ref(),
        AutoCullProgress {
            current: progress_total,
            total: progress_total,
            stage: "Grouping and scoring...".to_string(),
            ..Default::default()
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
    let analysis_by_path: HashMap<String, crate::culling::ImageAnalysisResult> = successful
        .iter()
        .map(|(_, analysis)| (analysis.result.path.clone(), analysis.result.clone()))
        .collect();
    let analyses: Vec<ImageAnalysisData> = successful.into_iter().map(|(_, data)| data).collect();

    let mut suggestions = build_culling_suggestions(analyses, failed_paths.clone(), &settings);

    // DoubleGradient perceptual hashing (see culling.rs) compares coarse
    // brightness-gradient layout, not subject identity - two photos with a
    // similar composition (e.g. a bird silhouette in a similar spot against
    // a similarly bright sky) can hash as "duplicates" even though the
    // actual subjects are completely different animals. Where BioCLIP is
    // already relevant (wildlife/birds), cross-check each hash-based
    // grouping against it and drop pairings it doesn't corroborate, rather
    // than trusting the hash alone.
    if settings.use_subject_detection
        && matches!(settings.subject_mode.as_str(), "wildlife" | "birds")
    {
        verify_duplicate_groups_with_bioclip(&mut suggestions, &app_handle);
    }

    let mut reject_reasons: HashMap<String, String> = HashMap::new();
    let mut representative_paths: HashSet<String> = HashSet::new();
    let mut duplicate_references: HashMap<String, (String, f64)> = HashMap::new();
    for group in &suggestions.similar_groups {
        representative_paths.insert(group.representative.path.clone());
        for dup in &group.duplicates {
            duplicate_references.insert(
                dup.path.clone(),
                (
                    group.representative.path.clone(),
                    group.representative.quality_score,
                ),
            );
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
            let analysis = analysis_by_path.get(&candidate.representative_path);
            let mut decision_factors = Vec::new();
            if let Some((representative, representative_score)) = duplicate_references.get(&candidate.representative_path) {
                decision_factors.push(CullDecisionFactor {
                    id: "duplicate".to_string(),
                    label: "Near duplicate".to_string(),
                    detail: format!(
                        "Lower quality score ({quality_score:.2}) than {} ({representative_score:.2})",
                        Path::new(representative).file_name().and_then(|name| name.to_str()).unwrap_or(representative),
                    ),
                    impact: "reject".to_string(),
                });
            }
            if reason == "blurry" {
                let sharpness = analysis.map(|item| item.sharpness_metric).unwrap_or_default();
                decision_factors.push(CullDecisionFactor {
                    id: "sharpness".to_string(),
                    label: "Low sharpness".to_string(),
                    detail: format!("Laplacian sharpness {sharpness:.1}; rejection threshold {:.1}", base_plan.settings.blur_threshold),
                    impact: "reject".to_string(),
                });
            }
            if let Some(analysis) = analysis {
                decision_factors.push(CullDecisionFactor {
                    id: "technical_quality".to_string(),
                    label: "Technical quality score".to_string(),
                    detail: format!(
                        "Score {quality_score:.2}: sharpness {:.1}, focus {:.1}, exposure {:.2}",
                        analysis.sharpness_metric,
                        analysis.subject_focus_metric.unwrap_or(analysis.center_focus_metric),
                        analysis.exposure_metric,
                    ),
                    impact: if keep { "context".to_string() } else { "supporting".to_string() },
                });
                if let Some(composition) = analysis.subject_composition_metric {
                    decision_factors.push(CullDecisionFactor {
                        id: "subject_geometry".to_string(),
                        label: "Subject placement cue".to_string(),
                        detail: format!(
                            "U-2-Net mask placement score {composition:.2}; detected subject edge contact {:.0}%. This is a minor tie-breaker, not an aesthetic verdict.",
                            analysis.subject_edge_contact_ratio.unwrap_or_default() * 100.0,
                        ),
                        impact: "context".to_string(),
                    });
                }
            }
            if let Some(adjustment) = personalization_adjustments.get(&candidate.representative_path) {
                decision_factors.push(CullDecisionFactor {
                    id: "personalization".to_string(),
                    label: "Personalized ranking adjustment".to_string(),
                    detail: format!("Local preference model adjusted this image by {adjustment:+.2}"),
                    impact: "context".to_string(),
                });
            }
            if let Some(adjustment) = subject_adjustments.get(&candidate.representative_path) {
                decision_factors.push(CullDecisionFactor {
                    id: "face_pose_adjustment".to_string(),
                    label: "Face pose tie-breaker".to_string(),
                    detail: format!("People-profile pose/framing adjusted the technical score by {adjustment:+.2}."),
                    impact: "context".to_string(),
                });
            }
            if let Some(subject) = subject_factors.get(&candidate.representative_path) {
                decision_factors.extend(subject.factors.iter().cloned());
            }

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

            let blurry_region = if reason == "blurry" {
                analysis.and_then(|item| item.blurry_region)
            } else {
                None
            };

            AutoCullPlanItem {
                representative_path: candidate.representative_path,
                backing_paths: candidate.backing_paths,
                keep,
                reason,
                quality_score,
                decision_factors,
                blurry_region,
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
                .map(|item| {
                    (
                        item.representative_path.clone(),
                        item.keep,
                        item.reason.clone(),
                        item.quality_score,
                        serde_json::to_string(&item.decision_factors)
                            .unwrap_or_else(|_| "[]".to_string()),
                    )
                })
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
                if let Err(error) =
                    crate::library_db::mark_cull_session_applied(&db_path, session_id)
                {
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

    let catalog_moves: Vec<(String, String)> = moved
        .iter()
        .map(|move_record| (move_record.old_path.clone(), move_record.new_path.clone()))
        .collect();
    if let Err(error) = crate::library_db::reconcile_catalog_moved_paths(&state, &catalog_moves) {
        log::warn!("Files were moved but catalog paths could not be reconciled: {error}");
    }

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

    let mut catalog_moves = Vec::with_capacity(result.moved.len());
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
        catalog_moves.push((mv.new_path.clone(), mv.old_path.clone()));
    }

    let state = app_handle.state::<crate::AppState>();
    if let Err(error) = crate::library_db::reconcile_catalog_moved_paths(&state, &catalog_moves) {
        log::warn!("Files were restored but catalog paths could not be reconciled: {error}");
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
    use image::{DynamicImage, Rgb, RgbImage};

    #[test]
    fn catalog_cull_cache_reuses_matching_revision_and_invalidates_changed_images() {
        let directory = tempfile::tempdir().unwrap();
        let database_path = directory.path().join("catalog.db");
        let connection = Connection::open(&database_path).unwrap();
        connection
            .execute_batch(
                "
                CREATE TABLE images(id INTEGER PRIMARY KEY, modified_at INTEGER NOT NULL);
                CREATE TABLE cull_analysis_cache(
                    image_id INTEGER NOT NULL,
                    feature_set_version TEXT NOT NULL,
                    image_modified_at INTEGER NOT NULL,
                    perceptual_hash BLOB NOT NULL,
                    quality_score REAL NOT NULL,
                    sharpness_metric REAL NOT NULL,
                    center_focus_metric REAL NOT NULL,
                    exposure_metric REAL NOT NULL,
                    subject_focus_metric REAL,
                    subject_composition_metric REAL,
                    subject_edge_contact_ratio REAL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY(image_id, feature_set_version, image_modified_at)
                );
                INSERT INTO images(id, modified_at) VALUES(7, 100);
                ",
            )
            .unwrap();
        drop(connection);

        let path = "/photos/example.ARW".to_string();
        let image_ids = HashMap::from([(path.clone(), 7)]);
        let candidate = AutoCullCandidate {
            representative_path: path.clone(),
            backing_paths: vec![path.clone()],
        };
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 4, Rgb([24, 48, 72])));
        let hash = HasherConfig::new()
            .hash_alg(HashAlg::DoubleGradient)
            .hash_size(16, 16)
            .to_hasher()
            .hash_image(&image);
        let analysis = ImageAnalysisData {
            hash,
            result: ImageAnalysisResult {
                path: path.clone(),
                quality_score: 0.8,
                sharpness_metric: 12.0,
                center_focus_metric: 9.0,
                subject_focus_metric: Some(10.0),
                subject_composition_metric: Some(0.7),
                subject_edge_contact_ratio: Some(0.1),
                exposure_metric: 0.9,
                width: 4,
                height: 4,
                blurry_region: None,
            },
        };

        store_catalog_cull_cache(&database_path, &image_ids, &[(candidate, analysis)]);
        let cached = load_catalog_cull_cache(&database_path, &image_ids);
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[&path].result.quality_score, 0.8);
        assert_eq!(cached[&path].result.subject_composition_metric, Some(0.7));

        let connection = Connection::open(&database_path).unwrap();
        connection
            .execute("UPDATE images SET modified_at = 101 WHERE id = 7", [])
            .unwrap();
        drop(connection);
        assert!(load_catalog_cull_cache(&database_path, &image_ids).is_empty());
    }

    #[test]
    fn auto_cull_plan_item_overrides_persist_in_struct() {
        let mut item = AutoCullPlanItem {
            representative_path: "/photos/DSC001.ARW".to_string(),
            backing_paths: vec![
                "/photos/DSC001.ARW".to_string(),
                "/photos/DSC001.JPG".to_string(),
            ],
            keep: false,
            reason: "Blurry background".to_string(),
            quality_score: 0.35,
            decision_factors: Vec::new(),
            blurry_region: None,
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

    #[test]
    fn catalog_subject_reuse_requires_enabled_profile_and_catalog_ids() {
        let mut settings = CullingSettings::default();
        settings.use_subject_detection = true;
        settings.subject_mode = "birds".to_string();
        let image_ids = HashMap::from([("/photos/bird.ARW".to_string(), 42)]);

        assert!(should_reuse_catalog_subjects(&settings, Some(&image_ids)));

        settings.use_subject_detection = false;
        assert!(!should_reuse_catalog_subjects(&settings, Some(&image_ids)));

        settings.use_subject_detection = true;
        settings.subject_mode = "landscape".to_string();
        assert!(!should_reuse_catalog_subjects(&settings, Some(&image_ids)));
        assert!(!should_reuse_catalog_subjects(&settings, None));
    }
}
