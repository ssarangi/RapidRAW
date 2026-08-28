import { useCallback, useEffect, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import { invoke } from '@tauri-apps/api/core';
import { CheckCircle2, Download, ExternalLink, Loader2, RotateCcw, ShieldCheck } from 'lucide-react';
import Button from '../ui/Button';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';
import { FaceModelPackStatus, Invokes } from '../ui/AppProperties';

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

const availabilityLabel = (status: FaceModelPackStatus) =>
  status.pack.availability === 'directDownload' ? 'Ready to download' : 'Evaluation only';

export default function FaceModelsSettings({ onOpenExternal }: FaceModelsSettingsProps) {
  const [statuses, setStatuses] = useState<FaceModelPackStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingError, setLoadingError] = useState<string | null>(null);
  const [activeDownload, setActiveDownload] = useState<DownloadProgress | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [acceptedLicenses, setAcceptedLicenses] = useState<Set<string>>(() => new Set());

  const refresh = useCallback(async () => {
    setLoading(true);
    setLoadingError(null);
    try {
      const rawStatuses = await invoke<Array<FaceModelPackStatus & { id?: string }>>(
        Invokes.ListFaceModelPackStatuses,
      );
      // Rust serializes FaceModelPackStatus with `#[serde(flatten)]`; normalize it
      // here so this component remains compatible with both payload shapes.
      setStatuses(
        rawStatuses
          .filter((status): status is FaceModelPackStatus & { id: string } => Boolean(status?.pack?.id || status?.id))
          .map((status) => (status.pack ? status : { ...status, pack: status } as FaceModelPackStatus)),
      );
    } catch (error) {
      setLoadingError(String(error));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    const progressListener = listen<DownloadProgress>('face-model-download-progress', (event) => {
      setActiveDownload(event.payload);
      setActionError(null);
    });
    const completeListener = listen<DownloadProgress>('face-model-download-complete', (event) => {
      setActiveDownload(event.payload);
      void refresh();
    });
    return () => {
      void progressListener.then((unlisten) => unlisten());
      void completeListener.then((unlisten) => unlisten());
    };
  }, [refresh]);

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
            Face Models
          </Text>
          <Text variant={TextVariants.small}>
            Manage local face detection and recognition models. Downloads stay on this device and no photo data is sent to a service.
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
                      <span className={status.installed ? 'text-green-400 text-xs' : isDirect ? 'text-accent text-xs' : 'text-amber-300 text-xs'}>
                        {status.installed ? 'Installed' : isDirect ? 'Ready to download' : 'Runtime adapter pending'}
                      </span>
                    </div>
                    <Text variant={TextVariants.small}>{status.pack.description}</Text>
                    <Text variant={TextVariants.small} className="mt-2">
                      Detector: {status.pack.detector} ({status.pack.detectorLandmarks} landmarks) · Recognition: {status.pack.recognizer}
                      {status.pack.embeddingDimensions ? ` · ${status.pack.embeddingDimensions}D embedding` : ''}
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
                      <label className="mt-3 flex items-start gap-2 text-xs text-text-secondary cursor-pointer">
                        <input
                          type="checkbox"
                          checked={acceptedLicenses.has(status.pack.id)}
                          onChange={(event) => {
                            setAcceptedLicenses((current) => {
                              const next = new Set(current);
                              if (event.target.checked) next.add(status.pack.id);
                              else next.delete(status.pack.id);
                              return next;
                            });
                          }}
                          className="mt-0.5 accent-[var(--color-accent)]"
                        />
                        I reviewed and accept the upstream license for this model pack.
                      </label>
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
