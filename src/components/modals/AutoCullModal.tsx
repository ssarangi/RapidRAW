import { useCallback, useMemo, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useTranslation } from 'react-i18next';
import { Loader2, Wand2, CheckCircle, XCircle, RotateCcw } from 'lucide-react';
import { CullingSettings, AutoCullPlan, AutoCullResult, Invokes } from '../ui/AppProperties';
import { useUIStore } from '../../store/useUIStore';
import Button from '../ui/Button';
import Switch from '../ui/Switch';
import Slider from '../ui/Slider';
import Input from '../ui/Input';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';

const DEFAULT_SETTINGS: CullingSettings = {
  similarityThreshold: 28,
  blurThreshold: 100.0,
  groupSimilar: true,
  filterBlurry: true,
  useSubjectDetection: false,
};

export default function AutoCullModal() {
  const { t } = useTranslation();
  const autoCullModalState = useUIStore((s) => s.autoCullModalState);
  const setUI = useUIStore((s) => s.setUI);
  const { isOpen, folderPath, stage, progress, plan, result, error } = autoCullModalState;

  const [includeSubfolders, setIncludeSubfolders] = useState(false);
  const [settings, setSettings] = useState<CullingSettings>(DEFAULT_SETTINGS);
  const [rejectedFolderName, setRejectedFolderName] = useState('_rejected');
  const [deleteInsteadOfMove, setDeleteInsteadOfMove] = useState(false);

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
          <Switch
            label={t('modals.culling.groupSimilar')}
            checked={settings.groupSimilar}
            onChange={(v) => setSettings((s) => ({ ...s, groupSimilar: v }))}
          />
          {settings.groupSimilar && (
            <div className="mt-2 pl-4 border-l-2 border-border-color ml-1">
              <Slider
                label={t('modals.culling.similarityThreshold')}
                min={1}
                max={64}
                step={1}
                value={settings.similarityThreshold}
                defaultValue={28}
                onChange={(e) => setSettings((s) => ({ ...s, similarityThreshold: Number(e.target.value) }))}
                fillOrigin="min"
              />
            </div>
          )}
        </div>

        <div>
          <Switch
            label={t('modals.culling.filterBlurry')}
            checked={settings.filterBlurry}
            onChange={(v) => setSettings((s) => ({ ...s, filterBlurry: v }))}
          />
          {settings.filterBlurry && (
            <div className="mt-2 pl-4 border-l-2 border-border-color ml-1">
              <Slider
                label={t('modals.culling.blurThreshold')}
                min={25}
                max={500}
                step={25}
                value={settings.blurThreshold}
                defaultValue={100.0}
                onChange={(e) => setSettings((s) => ({ ...s, blurThreshold: Number(e.target.value) }))}
                fillOrigin="min"
              />
            </div>
          )}
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
    <div className="flex flex-col items-center justify-center h-48">
      <Loader2 className="w-16 h-16 text-accent animate-spin" />
      <p className="mt-4 text-text-primary">{progress?.stage || fallbackLabel}</p>
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

    return (
      <>
        <Text variant={TextVariants.title} className="mb-4">
          {t('modals.autoCull.previewTitle')}
        </Text>

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

        <Text variant={TextVariants.heading} className="mb-2">
          {t('modals.autoCull.previewSummary', {
            total: plan.totalCount,
            reject: plan.rejectCount,
            keep: keepItems.length,
          })}
        </Text>

        <div className="bg-bg-primary rounded-lg p-2 h-[40vh] overflow-y-auto text-sm">
          {rejectItems.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full">
              <CheckCircle className="w-10 h-10 text-green-500 mb-2" />
              <Text>{t('modals.autoCull.nothingToReject')}</Text>
            </div>
          ) : (
            rejectItems.map((item) => (
              <div
                key={item.representativePath}
                className="flex items-center justify-between px-2 py-1.5 border-b border-surface last:border-0"
              >
                <Text variant={TextVariants.small} className="truncate max-w-[60%]" data-tooltip={item.representativePath}>
                  {item.representativePath.split(/[\\/]/).pop()}
                </Text>
                <div className="flex items-center gap-2">
                  {item.hasConflict && (
                    <Text variant={TextVariants.small} className="text-yellow-500">
                      {t('modals.autoCull.reasonConflict')}
                    </Text>
                  )}
                  <Text variant={TextVariants.small} color={TextColors.secondary}>
                    {item.reason.startsWith('duplicate_of:')
                      ? t('modals.autoCull.reasonDuplicate')
                      : t('modals.autoCull.reasonBlurry')}
                  </Text>
                </div>
              </div>
            ))
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
