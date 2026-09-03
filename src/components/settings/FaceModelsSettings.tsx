import { useCallback, useEffect, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import { invoke } from '@tauri-apps/api/core';
import { CheckCircle2, Download, ExternalLink, Loader2, RotateCcw, ShieldCheck } from 'lucide-react';
import Button from '../ui/Button';
import Checkbox from '../ui/Checkbox';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';
import { CatalogFaceProcessingSettings, FaceModelPack, FaceModelPackId, FaceModelPackStatus, FaceModelSelection, FaceModelSelectionPolicy, Invokes } from '../ui/AppProperties';

interface DownloadProgress {
  packId: string;
  displayName: string;
  current: number;
  total: number;
  stage: string;
}

interface FaceModelsSettingsProps {
  onOpenExternal(url: string): Promise<void>;
}

function normalizeStatus(value: unknown): FaceModelPackStatus | null {
  if (!value || typeof value !== 'object') return null;
  const status = value as Record<string, unknown>;
  const pack = (status.pack && typeof status.pack === 'object' ? status.pack : status) as Record<string, unknown>;
  if (
    !Object.values(FaceModelPackId).includes(pack.id as FaceModelPackId) ||
    typeof pack.displayName !== 'string' ||
    typeof pack.description !== 'string' ||
    typeof pack.detector !== 'string' ||
    typeof pack.recognizer !== 'string' ||
    typeof pack.detectorLandmarks !== 'number' ||
    (pack.availability !== 'directDownload' && pack.availability !== 'conversionRequired') ||
    typeof pack.licenseName !== 'string' ||
    typeof pack.licenseUrl !== 'string' ||
    typeof pack.modelSourceUrl !== 'string' ||
    typeof pack.licenseAcknowledgementRequired !== 'boolean' ||
    !Array.isArray(pack.artifacts)
  ) return null;
  const artifacts = pack.artifacts
    .filter((artifact): artifact is Record<string, unknown> => Boolean(artifact && typeof artifact === 'object'))
    .filter((artifact) => typeof artifact.fileName === 'string' && typeof artifact.format === 'string' && typeof artifact.sourceUrl === 'string')
    .map((artifact) => ({
      fileName: artifact.fileName as string,
      format: artifact.format as FaceModelPack['artifacts'][number]['format'],
      sourceUrl: artifact.sourceUrl as string,
      sha256: typeof artifact.sha256 === 'string' ? artifact.sha256 : null,
    }));
  if (artifacts.length !== pack.artifacts.length) return null;
  return {
    pack: {
      id: pack.id as FaceModelPackId,
      displayName: pack.displayName,
      description: pack.description,
      detector: pack.detector,
      recognizer: pack.recognizer,
      detectorLandmarks: pack.detectorLandmarks,
      embeddingDimensions: typeof pack.embeddingDimensions === 'number' ? pack.embeddingDimensions : null,
      accuracyRank: typeof pack.accuracyRank === 'number' ? pack.accuracyRank : Number.MAX_SAFE_INTEGER,
      speedRank: typeof pack.speedRank === 'number' ? pack.speedRank : Number.MAX_SAFE_INTEGER,
      balancedRank: typeof pack.balancedRank === 'number' ? pack.balancedRank : Number.MAX_SAFE_INTEGER,
      runtimeSupport: pack.runtimeSupport === 'supported' ? 'supported' : 'adapterPending',
      availability: pack.availability,
      artifacts,
      licenseName: pack.licenseName,
      licenseUrl: pack.licenseUrl,
      licenseAcknowledgementRequired: pack.licenseAcknowledgementRequired,
      modelSourceUrl: pack.modelSourceUrl,
    } as FaceModelPack,
    installed: status.installed === true,
    installPath: typeof status.installPath === 'string' ? status.installPath : '',
    installedArtifacts: Array.isArray(status.installedArtifacts)
      ? status.installedArtifacts
          .filter((artifact): artifact is Record<string, unknown> => Boolean(artifact && typeof artifact === 'object'))
          .filter((artifact) => typeof artifact.fileName === 'string' && typeof artifact.sha256 === 'string')
          .map((artifact) => ({ fileName: artifact.fileName as string, sha256: artifact.sha256 as string }))
      : [],
  };
}

const availabilityLabel = (status: FaceModelPackStatus) =>
  status.pack.availability === 'directDownload' ? 'Ready to download' : 'Evaluation only';

export default function FaceModelsSettings({ onOpenExternal }: FaceModelsSettingsProps) {
  const [statuses, setStatuses] = useState<FaceModelPackStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingError, setLoadingError] = useState<string | null>(null);
  const [activeDownload, setActiveDownload] = useState<DownloadProgress | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [acceptedLicenses, setAcceptedLicenses] = useState<Set<string>>(() => new Set());
  const [policy, setPolicy] = useState<FaceModelSelectionPolicy>('accuracy');
  const [selection, setSelection] = useState<FaceModelSelection | null>(null);
  const [catalogAvailable, setCatalogAvailable] = useState(false);
  const [needsReprocess, setNeedsReprocess] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setLoadingError(null);
    try {
      const rawStatuses = await invoke<unknown[]>(Invokes.ListFaceModelPackStatuses);
      setStatuses(rawStatuses.map(normalizeStatus).filter((status): status is FaceModelPackStatus => status !== null));
      try {
        const settings = await invoke<CatalogFaceProcessingSettings>(Invokes.GetCatalogFaceProcessingSettings);
        const nextPolicy = settings.faceModelPolicy === 'speed' ? 'speed' : 'accuracy';
        setPolicy(nextPolicy);
        setCatalogAvailable(true);
        setSelection(await invoke<FaceModelSelection>(Invokes.ResolveFaceModelSelection, { policy: nextPolicy }));
      } catch {
        setCatalogAvailable(false);
        setSelection(null);
      }
    } catch (error) {
      setLoadingError(String(error));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const updatePolicy = async (nextPolicy: FaceModelSelectionPolicy) => {
    if (nextPolicy === policy) return;
    setPolicy(nextPolicy);
    try {
      await invoke<CatalogFaceProcessingSettings>(Invokes.SetCatalogFaceProcessingSettings, { policy: nextPolicy });
      setNeedsReprocess(true);
      try {
        setSelection(await invoke<FaceModelSelection>(Invokes.ResolveFaceModelSelection, { policy: nextPolicy }));
      } catch {
        setSelection(null);
      }
    } catch (error) {
      setActionError(`Could not save face model policy: ${String(error)}`);
    }
  };

  const reprocessFaceIndex = async () => {
    if (!window.confirm('Reprocess all faces in this catalog? Existing detected faces and face embeddings will be rebuilt using the selected processing mode.')) return;
    setActionError(null);
    setReprocessing(true);
    try {
      await invoke<string>(Invokes.ReprocessFaceIndex);
      setNeedsReprocess(false);
    } catch (error) {
      setActionError(`Could not start face reprocessing: ${String(error)}`);
      setReprocessing(false);
    }
  };

  useEffect(() => {
    const progressListener = listen<DownloadProgress>('face-model-download-progress', (event) => {
      setActiveDownload(event.payload);
      setActionError(null);
    });
    const completeListener = listen<DownloadProgress>('face-model-download-complete', (event) => {
      setActiveDownload(event.payload);
      void refresh();
    });
    const reindexListener = listen('face-reindex-complete', () => {
      setReprocessing(false);
      void refresh();
    });
    return () => {
      void progressListener.then((unlisten) => unlisten());
      void completeListener.then((unlisten) => unlisten());
      void reindexListener.then((unlisten) => unlisten());
    };
  }, [refresh]);

  useEffect(() => {
    void (async () => {
      try {
        const jobs = await invoke<{ kind: string; state: string; payloadJson?: string | null }[]>(Invokes.ListBackgroundJobs);
        const activeJob = jobs.find((job) => {
          if (job.kind !== 'model_download' || !['queued', 'running', 'paused', 'cancelling'].includes(job.state)) return false;
          try {
            const payload = job.payloadJson ? JSON.parse(job.payloadJson) : null;
            return payload?.registry === 'face';
          } catch { return false; }
        });
        if (activeJob?.payloadJson) {
          const payload = JSON.parse(activeJob.payloadJson);
          if (typeof payload.packId === 'string' && typeof payload.displayName === 'string') {
            setActiveDownload({ packId: payload.packId, displayName: payload.displayName, current: 0, total: 1, stage: 'Downloading' });
          }
        }
      } catch { /* best-effort resume of in-flight download state */ }
    })();
  }, []);

  const download = async (status: FaceModelPackStatus) => {
    setActionError(null);
    setActiveDownload({
      packId: status.pack.id,
      displayName: status.pack.displayName,
      current: 0,
      total: status.pack.artifacts.length,
      stage: 'Preparing download',
    });
    try {
      await invoke<FaceModelPackStatus>(Invokes.DownloadFaceModelPack, {
        packId: status.pack.id,
        acceptRestrictedLicense: acceptedLicenses.has(status.pack.id),
      });
      await refresh();
    } catch (error) {
      setActionError(`Could not download ${status.pack.displayName}: ${String(error)}`);
    } finally {
      setActiveDownload(null);
    }
  };

  const downloadEligible = async () => {
    const eligible = statuses.filter(
      (status) =>
        !status.installed &&
        status.pack.availability === 'directDownload' &&
        (!status.pack.licenseAcknowledgementRequired || acceptedLicenses.has(status.pack.id)),
    );
    if (eligible.length === 0) {
      setActionError('No eligible model packs are available. Accept required upstream licenses before downloading restricted packs.');
      return;
    }
    setActionError(null);
    for (const status of eligible) {
      setActiveDownload({ packId: status.pack.id, displayName: status.pack.displayName, current: 0, total: status.pack.artifacts.length, stage: 'Preparing download' });
      try {
        await invoke<FaceModelPackStatus>(Invokes.DownloadFaceModelPack, {
          packId: status.pack.id,
          acceptRestrictedLicense: acceptedLicenses.has(status.pack.id),
        });
      } catch (error) {
        setActionError(`Could not download ${status.pack.displayName}: ${String(error)}`);
        break;
      }
    }
    setActiveDownload(null);
    await refresh();
  };

  return (
    <div className="p-6 bg-surface rounded-xl shadow-md space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <Text variant={TextVariants.title} color={TextColors.accent} className="mb-2">
            Face Processing
          </Text>
          <Text variant={TextVariants.small}>
            Choose how this catalog identifies people. Processing happens locally; no photo data is sent to a service.
          </Text>
        </div>
        <Button onClick={() => void downloadEligible()} disabled={activeDownload !== null || loading} className="shrink-0">
          <Download size={16} /> Download eligible
        </Button>
      </div>

      <div className="rounded-md border border-border-color bg-bg-primary p-4 flex items-start gap-3">
        <ShieldCheck size={18} className="text-accent shrink-0 mt-0.5" />
        <Text variant={TextVariants.small}>
          Install one pack at a time for comparison. InsightFace packs require acknowledgement of their upstream license before download.
        </Text>
      </div>

      <div className="rounded-md border border-border-color bg-bg-primary p-4 space-y-3">
        <div>
          <span className="font-medium">Face processing for this catalog</span>
          <Text variant={TextVariants.small}>Use one mode consistently so matching stays reliable across the catalog.</Text>
        </div>
        {catalogAvailable ? (
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => void updatePolicy('accuracy')} className={policy === 'accuracy' ? '' : 'bg-surface text-text-primary border border-border-color shadow-none'}>
              High Accuracy <span className="text-xs opacity-75">Recommended</span>
            </Button>
            <Button onClick={() => void updatePolicy('speed')} className={policy === 'speed' ? '' : 'bg-surface text-text-primary border border-border-color shadow-none'}>
              Fast Processing
            </Button>
          </div>
        ) : (
          <Text variant={TextVariants.small}>Open a catalog to choose its face-processing mode.</Text>
        )}
        <Text variant={TextVariants.small} className="text-muted-foreground">
          {selection
            ? policy === 'accuracy'
              ? 'High Accuracy is ready: best for profiles, partial faces, and identity matching.'
              : 'Fast Processing is ready: optimized for quicker face scans on large imports.'
            : 'Install a complete face-model pair to enable the selected processing mode.'}
        </Text>
        {needsReprocess && catalogAvailable && (
          <div className="rounded border border-amber-400/40 bg-amber-900/10 p-3 flex flex-wrap items-center justify-between gap-3">
            <Text variant={TextVariants.small}>The existing face index uses the previous mode. Reprocess it to apply this choice consistently.</Text>
            <Button onClick={() => void reprocessFaceIndex()} disabled={reprocessing} className="shrink-0">
              {reprocessing ? <Loader2 size={16} className="animate-spin" /> : <RotateCcw size={16} />}
              {reprocessing ? 'Starting reprocess' : 'Reprocess face index'}
            </Button>
          </div>
        )}
      </div>

      {loading ? (
        <div className="py-10 flex justify-center"><Loader2 className="animate-spin text-accent" size={24} /></div>
      ) : loadingError ? (
        <div className="rounded-md border border-red-500/50 bg-red-900/10 p-4 space-y-3">
          <Text variant={TextVariants.small}>{loadingError}</Text>
          <Button onClick={refresh} className="bg-bg-primary text-text-primary border border-border-color shadow-none">
            <RotateCcw size={16} /> Retry
          </Button>
        </div>
      ) : (
        <div className="space-y-3">
          <Text variant={TextVariants.heading}>Advanced model downloads</Text>
          {statuses.map((status) => {
            const isDownloading = activeDownload?.packId === status.pack.id;
            const isDirect = status.pack.availability === 'directDownload';
            const needsAcknowledgement = status.pack.licenseAcknowledgementRequired && !acceptedLicenses.has(status.pack.id);
            return (
              <div key={status.pack.id} className="rounded-md border border-border-color bg-bg-primary p-4">
                <div className="flex flex-wrap gap-4 justify-between">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2 mb-1">
                      <Text variant={TextVariants.heading}>{status.pack.displayName}</Text>
                      <span className={status.installed && status.pack.runtimeSupport === 'supported' ? 'text-green-400 text-xs' : isDirect ? 'text-accent text-xs' : 'text-amber-300 text-xs'}>
                        {status.installed && status.pack.runtimeSupport === 'supported' ? 'Installed and ready' : status.pack.runtimeSupport !== 'supported' ? 'Runtime adapter pending' : isDirect ? 'Ready to download' : 'Runtime adapter pending'}
                      </span>
                    </div>
                    <Text variant={TextVariants.small}>{status.pack.description}</Text>
                    <Text variant={TextVariants.small} className="mt-2">
                      Detector: {status.pack.detector} ({status.pack.detectorLandmarks} landmarks) · Recognition: {status.pack.recognizer}
                      {status.pack.embeddingDimensions ? ` · ${status.pack.embeddingDimensions}D embedding` : ''}
                      {` · Accuracy #${status.pack.accuracyRank} · Speed #${status.pack.speedRank}`}
                    </Text>
                    <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
                      <button className="text-xs text-accent inline-flex items-center gap-1" onClick={() => void onOpenExternal(status.pack.modelSourceUrl)}>
                        Model source <ExternalLink size={12} />
                      </button>
                      <button className="text-xs text-accent inline-flex items-center gap-1" onClick={() => void onOpenExternal(status.pack.licenseUrl)}>
                        {status.pack.licenseName} <ExternalLink size={12} />
                      </button>
                    </div>
                    {status.pack.licenseAcknowledgementRequired && !status.installed && (
                      <Checkbox
                        id={`accept-license-${status.pack.id}`}
                        className="mt-3"
                        checked={acceptedLicenses.has(status.pack.id)}
                        onChange={(checked) => {
                          setAcceptedLicenses((current) => {
                            const next = new Set(current);
                            if (checked) next.add(status.pack.id);
                            else next.delete(status.pack.id);
                            return next;
                          });
                        }}
                        label="I reviewed and accept the upstream license for this model pack."
                      />
                    )}
                  </div>
                  <div className="shrink-0 flex items-start">
                    {status.installed ? (
                      <div className="text-green-400 flex items-center gap-2 text-sm py-2"><CheckCircle2 size={18} /> Installed</div>
                    ) : !isDirect ? (
                      <Text variant={TextVariants.small} color={TextColors.secondary} className="max-w-40 text-right">Visible for evaluation. This build has no compatible runtime adapter.</Text>
                    ) : (
                      <Button onClick={() => void download(status)} disabled={activeDownload !== null || needsAcknowledgement} className="px-3">
                        {isDownloading ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
                        {isDownloading ? 'Downloading' : 'Download'}
                      </Button>
                    )}
                  </div>
                </div>
                {isDownloading && (
                  <div className="mt-4 border-t border-border-color pt-3">
                    <div className="flex justify-between gap-3 text-xs text-text-secondary mb-2">
                      <span className="truncate">{activeDownload.stage}</span>
                      <span>{activeDownload.current}/{activeDownload.total}</span>
                    </div>
                    <div className="h-1.5 bg-surface rounded-full overflow-hidden">
                      <div className="h-full bg-accent transition-all" style={{ width: `${Math.max(4, (activeDownload.current / Math.max(activeDownload.total, 1)) * 100)}%` }} />
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {actionError && <div className="rounded-md border border-red-500/50 bg-red-900/10 p-3"><Text variant={TextVariants.small}>{actionError}</Text></div>}
    </div>
  );
}
