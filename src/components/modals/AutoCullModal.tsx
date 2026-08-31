import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useTranslation } from 'react-i18next';
import { Loader2, Wand2, CheckCircle, XCircle, RotateCcw, Square, CheckSquare } from 'lucide-react';
import { CullingSettings, AutoCullPlan, AutoCullResult, Invokes } from '../ui/AppProperties';
import { useUIStore } from '../../store/useUIStore';
import { useProcessStore } from '../../store/useProcessStore';
import Button from '../ui/Button';
import Switch from '../ui/Switch';
import Input from '../ui/Input';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';

const DEFAULT_SETTINGS: CullingSettings = {
  similarityThreshold: 28,
  blurThreshold: 100.0,
  groupSimilar: true,
  filterBlurry: true,
  useSubjectDetection: false,
  subjectMode: 'general',
};

const CULLING_PRESETS = [
  { id: 'tight', label: 'Fewer, stronger selections', similarityThreshold: 36, blurThreshold: 150 },
  { id: 'balanced', label: 'Balanced selection', similarityThreshold: 28, blurThreshold: 100 },
  { id: 'generous', label: 'More images to review', similarityThreshold: 20, blurThreshold: 65 },
] as const;

export default function AutoCullModal() {
  const { t } = useTranslation();
  const autoCullModalState = useUIStore((s) => s.autoCullModalState);
  const setUI = useUIStore((s) => s.setUI);
  const thumbnails = useProcessStore((s) => s.thumbnails);
  const { isOpen, folderPath, stage, progress, plan, result, error } = autoCullModalState;

  const [includeSubfolders, setIncludeSubfolders] = useState(false);
  const [settings, setSettings] = useState<CullingSettings>(DEFAULT_SETTINGS);
  const [rejectedFolderName, setRejectedFolderName] = useState('_rejected');
  const [deleteInsteadOfMove, setDeleteInsteadOfMove] = useState(false);
  const [feedbackItemPath, setFeedbackItemPath] = useState<string | null>(null);
  const [reviewSelection, setReviewSelection] = useState<Set<string>>(new Set());
  const [inspectedPath, setInspectedPath] = useState<string | null>(null);
  const [reviewFilter, setReviewFilter] = useState<'all' | 'selected' | 'duplicates' | 'blurry'>('all');
  const dragSelectionValue = useRef<boolean | null>(null);

  useEffect(() => {
    const stopDragSelection = () => { dragSelectionValue.current = null; };
    window.addEventListener('mouseup', stopDragSelection);
    return () => window.removeEventListener('mouseup', stopDragSelection);
  }, []);

  useEffect(() => {
    if (stage !== 'preview' || !plan) return;
    setReviewSelection(new Set(plan.items.filter((item) => !item.keep).map((item) => item.representativePath)));
    setInspectedPath((current) => current && plan.items.some((item) => item.representativePath === current)
      ? current
      : plan.items[0]?.representativePath ?? null);
    const missing = plan.items
      .filter((item) => !thumbnails[item.representativePath])
      .slice(0, 500)
      .map((item) => ({ path: item.representativePath, modified: null }));
    if (missing.length > 0) void invoke(Invokes.UpdateThumbnailQueue, { paths: missing });
  }, [stage, plan?.sessionId]);

  const close = useCallback(() => {
    setUI({
      autoCullModalState: {
        isOpen: false,
        folderPath: null,
        stage: 'rules',
        progress: null,
        plan: null,
        result: null,
        error: null,
      },
    });
  }, [setUI]);

  const runPlan = useCallback(async () => {
    if (!folderPath) return;
    setUI((s) => ({ autoCullModalState: { ...s.autoCullModalState, stage: 'analyzing', error: null } }));
    try {
      const newPlan = await invoke<AutoCullPlan>(Invokes.PlanAutoCull, {
        folderPath,
        includeSubfolders,
        settings,
        rejectedFolderName,
        deleteInsteadOfMove,
      });
      setUI((s) => ({
        autoCullModalState: { ...s.autoCullModalState, stage: 'preview', plan: newPlan, progress: null },
      }));
    } catch (err) {
      setUI((s) => ({
        autoCullModalState: { ...s.autoCullModalState, stage: 'rules', progress: null, error: String(err) },
      }));
    }
  }, [folderPath, includeSubfolders, settings, rejectedFolderName, deleteInsteadOfMove, setUI]);

  const runApply = useCallback(
    async (conflictAction?: 'skip' | 'overwrite') => {
      if (!plan) return;
      setUI((s) => ({
        autoCullModalState: { ...s.autoCullModalState, stage: 'applying', progress: null, error: null },
      }));
      try {
        const applyResult = await invoke<AutoCullResult>(Invokes.ApplyAutoCullPlan, {
          plan,
          conflictAction: conflictAction ?? null,
        });
        setUI((s) => ({ autoCullModalState: { ...s.autoCullModalState, stage: 'summary', result: applyResult } }));
      } catch (err) {
        setUI((s) => ({
          autoCullModalState: { ...s.autoCullModalState, stage: 'preview', error: String(err) },
        }));
      }
    },
    [plan, setUI],
  );

  const handleMoveClick = useCallback(() => {
    if (!plan) return;
    const hasConflicts = plan.items.some((i) => !i.keep && i.hasConflict);
    if (hasConflicts) {
      setUI((s) => ({ autoCullModalState: { ...s.autoCullModalState, stage: 'conflict' } }));
    } else {
      runApply();
    }
  }, [plan, runApply, setUI]);

  const runUndo = useCallback(async () => {
    if (!result) return;
    try {
      await invoke(Invokes.UndoAutoCull, { result });
      close();
    } catch (err) {
      setUI((s) => ({ autoCullModalState: { ...s.autoCullModalState, error: String(err) } }));
    }
  }, [result, close, setUI]);

  const rejectItems = useMemo(() => (plan ? plan.items.filter((i) => !i.keep) : []), [plan]);
  const keepItems = useMemo(() => (plan ? plan.items.filter((i) => i.keep) : []), [plan]);
  const conflictItems = useMemo(() => rejectItems.filter((i) => i.hasConflict), [rejectItems]);

  const togglePlanItem = async (representativePath: string) => {
    if (!plan) return;
    const item = plan.items.find((candidate) => candidate.representativePath === representativePath);
    if (!item) return;
    const keep = !item.keep;
    const updatedItems = plan.items.map((candidate) => candidate.representativePath === representativePath ? { ...candidate, keep } : candidate);
    const updatedPlan = { ...plan, items: updatedItems, rejectCount: updatedItems.filter((candidate) => !candidate.keep).length };
    setUI((s) => ({ autoCullModalState: { ...s.autoCullModalState, plan: updatedPlan } }));
    if (plan.sessionId) {
      try { await invoke(Invokes.UpdateCullSessionDecision, { sessionId: plan.sessionId, representativePath, keep }); }
      catch (err) { setUI((s) => ({ autoCullModalState: { ...s.autoCullModalState, plan, error: String(err) } })); }
    }
    if (item.keep === false && keep) setFeedbackItemPath(representativePath);
  };

  const recordFeedback = async (representativePath: string, feedbackReason: string) => {
    if (!plan?.sessionId) return;
    const item = plan.items.find((candidate) => candidate.representativePath === representativePath);
    if (!item) return;
    try {
      await invoke(Invokes.UpdateCullSessionDecision, {
        sessionId: plan.sessionId,
        representativePath,
        keep: item.keep,
        feedbackReason,
      });
    } catch (err) {
      setUI((s) => ({ autoCullModalState: { ...s.autoCullModalState, error: String(err) } }));
    } finally {
      setFeedbackItemPath(null);
    }
  };

  const setReviewItemSelected = (path: string, selected: boolean) => {
    setReviewSelection((previous) => {
      const next = new Set(previous);
      if (selected) next.add(path);
      else next.delete(path);
      return next;
    });
  };

  const setDecisionForPaths = async (paths: string[], keep: boolean) => {
    if (!plan || paths.length === 0) return;
    const pathSet = new Set(paths);
    const updatedItems = plan.items.map((item) => pathSet.has(item.representativePath) ? { ...item, keep } : item);
    const updatedPlan = { ...plan, items: updatedItems, rejectCount: updatedItems.filter((item) => !item.keep).length };
    setUI((state) => ({ autoCullModalState: { ...state.autoCullModalState, plan: updatedPlan } }));
    if (plan.sessionId) {
      await Promise.all(paths.map((representativePath) => invoke(Invokes.UpdateCullSessionDecision, {
        sessionId: plan.sessionId,
        representativePath,
        keep,
      })));
    }
  };

  if (!isOpen) return null;

  const renderRules = () => (
    <>
      <div className="flex items-center justify-center mb-4">
        <Wand2 className="w-12 h-12 text-accent" />
      </div>
      <Text variant={TextVariants.title} className="mb-1 text-center">
        {t('modals.autoCull.title')}
      </Text>
      <Text variant={TextVariants.small} className="mb-6 text-center break-all">
        {folderPath}
      </Text>

      {error && (
        <Text variant={TextVariants.small} className="mb-4 text-red-500 text-center">
          {error}
        </Text>
      )}

      <div className="space-y-6 text-sm">
        <Switch
          label={t('modals.autoCull.includeSubfolders')}
          checked={includeSubfolders}
          onChange={setIncludeSubfolders}
        />

        <div>
          <Text variant={TextVariants.small} className="mb-2 block">Selection preference</Text>
          <div className="grid grid-cols-3 gap-2">
            {CULLING_PRESETS.map((preset) => {
              const active = settings.similarityThreshold === preset.similarityThreshold && settings.blurThreshold === preset.blurThreshold;
              return <button key={preset.id} className={`min-h-14 rounded-md border px-2 py-2 text-xs text-left ${active ? 'border-accent bg-accent/15 text-text-primary' : 'border-border-color text-text-secondary hover:bg-surface'}`} onClick={() => setSettings((current) => ({ ...current, similarityThreshold: preset.similarityThreshold, blurThreshold: preset.blurThreshold }))}>{preset.label}</button>;
            })}
          </div>
        </div>

        <Switch label="Group similar burst frames" checked={settings.groupSimilar} onChange={(value) => setSettings((current) => ({ ...current, groupSimilar: value }))} />
        <Switch label="Flag out-of-focus images" checked={settings.filterBlurry} onChange={(value) => setSettings((current) => ({ ...current, filterBlurry: value }))} />

        <div>
          <Text variant={TextVariants.small} className="mb-1.5 block">Shooting type</Text>
          <select
            value={settings.subjectMode}
            onChange={(event) => {
              const subjectMode = event.target.value as CullingSettings['subjectMode'];
              setSettings((current) => ({
                ...current,
                subjectMode,
                useSubjectDetection: subjectMode !== 'landscape',
              }));
            }}
            className="h-9 w-full rounded-md border border-border-color bg-bg-primary px-2 text-sm text-text-primary outline-none focus:border-accent"
          >
            <option value="general">General subjects</option>
            <option value="people">People and events</option>
            <option value="wildlife">Wildlife and animals</option>
            <option value="birds">Birds</option>
            <option value="landscape">Landscape and architecture</option>
          </select>
          <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mt-1">People, wildlife, and birds use foreground-aware focus scoring. Landscape scores the complete frame.</Text>
        </div>

        <Switch
          label={t('modals.culling.useSubjectDetection')}
          checked={settings.useSubjectDetection}
          onChange={(v) => setSettings((s) => ({ ...s, useSubjectDetection: v }))}
        />

        <div className="border-t border-border-color pt-4 space-y-4">
          <div>
            <Text variant={TextVariants.small} className="mb-1.5 block">
              {t('modals.autoCull.rejectedFolderName')}
            </Text>
            <Input
              value={rejectedFolderName}
              onChange={(e) => setRejectedFolderName(e.target.value)}
              disabled={deleteInsteadOfMove}
            />
          </div>
          <Switch
            label={t('modals.autoCull.deleteInsteadOfMove')}
            checked={deleteInsteadOfMove}
            onChange={setDeleteInsteadOfMove}
          />
          {deleteInsteadOfMove && (
            <Text variant={TextVariants.small} className="text-red-400">
              {t('modals.autoCull.deleteWarning')}
            </Text>
          )}
        </div>
      </div>

      <div className="flex justify-end gap-3 mt-8">
        <button className="px-4 py-2 rounded-md text-text-secondary hover:bg-surface transition-colors" onClick={close}>
          {t('modals.culling.cancel')}
        </button>
        <Button onClick={runPlan} disabled={!rejectedFolderName.trim() && !deleteInsteadOfMove}>
          {t('modals.autoCull.analyze')}
        </Button>
      </div>
    </>
  );

  const renderProgress = (fallbackLabel: string) => (
    <div className="flex flex-col items-center justify-center min-h-56">
      <Loader2 className="w-12 h-12 text-accent animate-spin" />
      <p className="mt-4 text-text-primary">{progress?.stage || fallbackLabel}</p>
      {progress?.currentItem && (
        <p className="mt-1 text-xs text-text-secondary font-mono truncate max-w-full">{progress.currentItem}</p>
      )}
      {progress && progress.total > 0 && (
        <div className="w-full bg-surface rounded-full h-2.5 mt-2">
          <div
            className="bg-accent h-2.5 rounded-full"
            style={{ width: `${(progress.current / progress.total) * 100}%` }}
          />
        </div>
      )}
    </div>
  );

  const renderPreview = () => {
    if (!plan) return null;
    const inspectedItem = plan.items.find((item) => item.representativePath === inspectedPath) || plan.items[0];
    const selectedPaths = Array.from(reviewSelection);
    const visibleItems = plan.items.filter((item) => reviewFilter === 'all'
      || (reviewFilter === 'selected' && item.keep)
      || (reviewFilter === 'duplicates' && item.reason.startsWith('duplicate_of:'))
      || (reviewFilter === 'blurry' && !item.keep && !item.reason.startsWith('duplicate_of:')));

    return (
      <>
        <div className="flex items-center justify-between gap-3 mb-3">
          <Text variant={TextVariants.title}>{t('modals.autoCull.previewTitle')}</Text>
          <Text variant={TextVariants.small} color={TextColors.secondary}>{reviewSelection.size} selected</Text>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 mb-4 text-sm text-red-400 break-words">
            {error}
          </div>
        )}

        <div className="bg-bg-primary rounded-lg p-4 mb-4 text-sm space-y-1">
          <Text variant={TextVariants.small}>
            {t('modals.autoCull.previewFolder', { folder: plan.folderPath })}
          </Text>
          <Text variant={TextVariants.small}>
            {t('modals.autoCull.previewSettings', {
              subfolders: plan.includeSubfolders ? t('modals.autoCull.yes') : t('modals.autoCull.no'),
              subjectDetection: plan.settings.useSubjectDetection ? t('modals.autoCull.yes') : t('modals.autoCull.no'),
            })}
          </Text>
          <Text variant={TextVariants.small}>
            {plan.deleteInsteadOfMove
              ? t('modals.autoCull.previewDeleteAction')
              : t('modals.autoCull.previewMoveAction', { folder: plan.rejectedFolderName })}
          </Text>
          {plan.failedPaths.length > 0 && (
            <Text variant={TextVariants.small} className="text-yellow-500">
              {t('modals.autoCull.previewFailed', { count: plan.failedPaths.length })}
            </Text>
          )}
          {conflictItems.length > 0 && (
            <Text variant={TextVariants.small} className="text-yellow-500">
              {t('modals.autoCull.previewConflicts', { count: conflictItems.length })}
            </Text>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2 mb-3">
          {(['all', 'selected', 'duplicates', 'blurry'] as const).map((filter) => <button key={filter} className={`px-2 py-1 text-xs rounded ${reviewFilter === filter ? 'bg-surface text-text-primary' : 'text-text-secondary hover:bg-surface'}`} onClick={() => setReviewFilter(filter)}>{filter === 'all' ? 'All' : filter === 'selected' ? 'Selected' : filter === 'duplicates' ? 'Duplicates' : 'Blurry'}</button>)}
          <span className="h-4 border-l border-border-color" />
          <button className="px-2 py-1 text-xs text-text-secondary hover:bg-surface rounded" onClick={() => setReviewSelection(new Set(visibleItems.map((item) => item.representativePath)))}>Select visible</button>
          <button className="px-2 py-1 text-xs text-text-secondary hover:bg-surface rounded" onClick={() => setReviewSelection(new Set())}>Select none</button>
          <button className="px-2 py-1 text-xs text-green-300 hover:bg-surface rounded disabled:opacity-40" disabled={selectedPaths.length === 0} onClick={() => void setDecisionForPaths(selectedPaths, true)}>Keep selected</button>
          <button className="px-2 py-1 text-xs text-red-300 hover:bg-surface rounded disabled:opacity-40" disabled={selectedPaths.length === 0} onClick={() => void setDecisionForPaths(selectedPaths, false)}>Reject selected</button>
        </div>

        <div className="grid grid-cols-[minmax(0,1fr)_minmax(230px,30%)] gap-3 h-[52vh] min-h-0">
          {plan.items.length === 0 ? (
            <div className="col-span-2 flex flex-col items-center justify-center h-full bg-bg-primary rounded-md">
              <CheckCircle className="w-10 h-10 text-green-500 mb-2" />
              <Text>{t('modals.autoCull.nothingToReject')}</Text>
            </div>
          ) : (
            <>
              <div className="min-h-0 overflow-y-auto bg-bg-primary rounded-md p-2 grid grid-cols-[repeat(auto-fill,minmax(112px,1fr))] content-start gap-2">
                {visibleItems.map((item) => {
                  const selected = reviewSelection.has(item.representativePath);
                  const thumbnail = thumbnails[item.representativePath];
                  return <button key={item.representativePath} className={`relative aspect-square overflow-hidden rounded border-2 text-left ${inspectedItem?.representativePath === item.representativePath ? 'border-accent' : selected ? 'border-text-secondary' : 'border-transparent hover:border-surface'}`} onMouseDown={() => { dragSelectionValue.current = !selected; setReviewItemSelected(item.representativePath, !selected); setInspectedPath(item.representativePath); }} onMouseEnter={() => { if (dragSelectionValue.current !== null) setReviewItemSelected(item.representativePath, dragSelectionValue.current); }}>
                    {thumbnail ? <img src={thumbnail} alt="" className="w-full h-full object-cover" /> : <div className="w-full h-full bg-surface flex items-center justify-center"><Loader2 className="w-5 h-5 text-text-secondary animate-spin" /></div>}
                    <span className={`absolute top-1 left-1 p-0.5 rounded bg-black/60 ${item.keep ? 'text-green-300' : 'text-red-300'}`}>{item.keep ? 'K' : 'R'}</span>
                    <span className="absolute top-1 right-1 text-accent">{selected ? <CheckSquare size={16} /> : <Square size={16} />}</span>
                  </button>;
                })}
              </div>
              <div className="min-h-0 overflow-y-auto bg-bg-primary rounded-md p-3">
                {inspectedItem && <>
                  <div className="aspect-square bg-surface rounded overflow-hidden mb-3">{thumbnails[inspectedItem.representativePath] ? <img src={thumbnails[inspectedItem.representativePath]} alt={inspectedItem.representativePath} className="w-full h-full object-contain" /> : <Loader2 className="m-auto mt-[45%] w-6 h-6 text-accent animate-spin" />}</div>
                  <Text variant={TextVariants.small} className="block truncate" title={inspectedItem.representativePath}>{inspectedItem.representativePath.split(/[\\/]/).pop()}</Text>
                  <div className="flex gap-2 mt-2"><button className="text-xs text-green-300" onClick={() => void setDecisionForPaths([inspectedItem.representativePath], true)}>Keep</button><button className="text-xs text-red-300" onClick={() => void setDecisionForPaths([inspectedItem.representativePath], false)}>Reject</button></div>
                  <Text variant={TextVariants.small} color={TextColors.secondary} className="mt-3 block">{inspectedItem.reason}</Text>
                  <div className="mt-2 space-y-2 text-xs text-text-secondary">{inspectedItem.decisionFactors.map((factor) => <div key={factor.id}><span className={factor.impact === 'reject' ? 'text-red-300' : 'text-text-primary'}>{factor.label}</span><div>{factor.detail}</div></div>)}</div>
                </>}
              </div>
            </>
          )}
        </div>

        <div className="flex justify-end gap-3 mt-6">
          <button className="px-4 py-2 rounded-md text-text-secondary hover:bg-surface transition-colors" onClick={close}>
            {t('modals.culling.cancel')}
          </button>
          <Button onClick={handleMoveClick} disabled={rejectItems.length === 0}>
            {t('modals.autoCull.runButton', { count: rejectItems.length })}
          </Button>
        </div>
      </>
    );
  };

  const renderSummary = () => {
    if (error) {
      return (
        <div className="flex flex-col items-center justify-center h-48">
          <XCircle className="w-16 h-16 text-red-500" />
          <Text variant={TextVariants.heading} className="mt-4 text-center">
            {t('modals.autoCull.failed')}
          </Text>
          <Text>{error}</Text>
          <div className="mt-6">
            <Button onClick={close}>{t('modals.culling.close')}</Button>
          </div>
        </div>
      );
    }

    if (!result) return null;

    return (
      <div className="flex flex-col items-center justify-center text-center py-4">
        <CheckCircle className="w-16 h-16 text-green-500" />
        <Text variant={TextVariants.heading} className="mt-4">
          {result.deleted
            ? t('modals.autoCull.summaryDeleted', { count: result.labeledPaths.length })
            : t('modals.autoCull.summaryMoved', { count: result.moved.length, folder: result.rejectedFolderPath })}
        </Text>
        {result.skippedPaths.length > 0 && (
          <Text variant={TextVariants.small} color={TextColors.secondary} className="mt-1">
            {t('modals.autoCull.summarySkipped', { count: result.skippedPaths.length })}
          </Text>
        )}
        <div className="flex gap-3 mt-6">
          {!result.deleted && (
            <button
              className="px-4 py-2 rounded-md text-text-secondary hover:bg-surface transition-colors flex items-center gap-1.5"
              onClick={runUndo}
            >
              <RotateCcw size={14} />
              {t('modals.autoCull.undo')}
            </button>
          )}
          <Button onClick={close}>{t('modals.culling.done')}</Button>
        </div>
      </div>
    );
  };

  const renderConflict = () => (
    <div className="flex flex-col items-center text-center py-4">
      <Text variant={TextVariants.heading} className="mb-2">
        {t('modals.autoCull.conflictTitle')}
      </Text>
      <Text className="mb-6">
        {t('modals.autoCull.conflictDesc', { count: conflictItems.length })}
      </Text>

      <div className="bg-bg-primary rounded-lg p-2 w-full max-h-[30vh] overflow-y-auto text-sm mb-6 text-left">
        {conflictItems.map((item) => (
          <Text
            key={item.representativePath}
            variant={TextVariants.small}
            className="block truncate px-2 py-1"
            data-tooltip={item.representativePath}
          >
            {item.representativePath.split(/[\\/]/).pop()}
          </Text>
        ))}
      </div>

      <div className="flex flex-wrap justify-center gap-3">
        <button
          className="px-4 py-2 rounded-md text-text-secondary hover:bg-surface transition-colors"
          onClick={() => setUI((s) => ({ autoCullModalState: { ...s.autoCullModalState, stage: 'preview' } }))}
        >
          {t('modals.culling.cancel')}
        </button>
        <button
          className="px-4 py-2 rounded-md text-text-secondary hover:bg-surface transition-colors"
          onClick={() => runApply('skip')}
        >
          {t('modals.autoCull.skipConflicts')}
        </button>
        <Button onClick={() => runApply('overwrite')}>{t('modals.autoCull.overwriteConflicts')}</Button>
      </div>
    </div>
  );

  const renderContent = () => {
    switch (stage) {
      case 'rules':
        return renderRules();
      case 'analyzing':
        return renderProgress(t('modals.autoCull.analyzing'));
      case 'preview':
        return renderPreview();
      case 'conflict':
        return renderConflict();
      case 'applying':
        return renderProgress(t('modals.autoCull.applying'));
      case 'summary':
        return renderSummary();
      default:
        return null;
    }
  };

  return (
    <div
      className="fixed inset-0 flex items-center justify-center z-50 bg-black/30 backdrop-blur-xs"
      onClick={stage === 'analyzing' || stage === 'applying' ? undefined : close}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="bg-surface rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {renderContent()}
      </div>
    </div>
  );
}
