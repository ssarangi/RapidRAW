import { useCallback, useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { CheckCircle2, Download, ExternalLink, FolderOpen, Loader2, RotateCcw, Trash2 } from 'lucide-react';
import Button from '../ui/Button';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';
import { Invokes, VisualModelPackStatus } from '../ui/AppProperties';

interface VisualModelsSettingsProps { onOpenExternal(url: string): Promise<void>; }

export default function VisualModelsSettings({ onOpenExternal }: VisualModelsSettingsProps) {
  const [statuses, setStatuses] = useState<VisualModelPackStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [removeModelId, setRemoveModelId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      // VisualModelPackStatus is flattened by the Rust serializer. Accept the
      // nested shape too so this stays compatible if the command contract is
      // corrected server-side.
      const rawStatuses = await invoke<Array<VisualModelPackStatus & { id?: string }>>(
        Invokes.ListVisualModelPackStatuses,
      );
      setStatuses(
        rawStatuses
          .filter((status): status is VisualModelPackStatus & { id: string } => Boolean(status?.pack?.id || status?.id))
          .map((status) => (status.pack ? status : { ...status, pack: status } as VisualModelPackStatus)),
      );
      setError(null);
    } catch (reason) {
      setError(String(reason));
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);
  const download = async (status: VisualModelPackStatus) => { setDownloadingId(status.pack.id); setError(null); try { await invoke(Invokes.DownloadVisualModelPack, { packId: status.pack.id }); await refresh(); } catch (reason) { setError(`Could not download ${status.pack.displayName}: ${String(reason)}`); } finally { setDownloadingId(null); } };
  const installBundle = async (status: VisualModelPackStatus) => {
    const sourceDirectory = await open({ directory: true, multiple: false, title: `Choose ${status.pack.displayName} bundle folder` });
    if (!sourceDirectory || Array.isArray(sourceDirectory)) return;
    setDownloadingId(status.pack.id);
    setError(null);
    try {
      await invoke(Invokes.InstallVisualModelBundle, { packId: status.pack.id, sourceDirectory });
      await refresh();
    } catch (reason) { setError(`Could not install ${status.pack.displayName}: ${String(reason)}`); }
    finally { setDownloadingId(null); }
  };
  const remove = async (status: VisualModelPackStatus) => {
    setDownloadingId(status.pack.id);
    setError(null);
    try { await invoke(Invokes.RemoveVisualModelPack, { packId: status.pack.id }); await refresh(); }
    catch (reason) { setError(`Could not remove ${status.pack.displayName}: ${String(reason)}`); }
    finally { setDownloadingId(null); setRemoveModelId(null); }
  };

  return <div className="p-6 bg-surface rounded-xl shadow-md space-y-5"><div><Text variant={TextVariants.title} color={TextColors.accent}>Visual AI Models</Text><Text as="div" variant={TextVariants.small} className="mt-2">Local models for broad catalog tags and species classification. Image data stays on this device.</Text></div>{loading ? <div className="py-10 flex justify-center"><Loader2 size={24} className="animate-spin text-accent" /></div> : <div className="space-y-3">{statuses.map((status) => { const downloading = downloadingId === status.pack.id; const direct = status.pack.availability === 'directDownload'; return <div key={status.pack.id} className="rounded-md border border-border-color bg-bg-primary p-4"><div className="flex flex-wrap justify-between gap-3"><div className="min-w-0 flex-1"><div className="flex items-center gap-2"><Text variant={TextVariants.heading}>{status.pack.displayName}</Text><span className={status.installed ? 'text-green-400 text-xs' : direct ? 'text-accent text-xs' : 'text-text-secondary text-xs'}>{status.installed ? 'Installed' : direct ? 'Ready to download' : 'Bundle required'}</span></div><Text as="div" variant={TextVariants.small} className="mt-1">{status.pack.description}</Text><Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mt-2">{status.pack.task}</Text>{!direct && <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mt-2">Bundle files: {status.pack.artifacts.map((artifact) => artifact.fileName).join(', ')}</Text>}<div className="mt-2 flex gap-3"><button className="text-xs text-accent inline-flex items-center gap-1" onClick={() => void onOpenExternal(status.pack.modelSourceUrl)}>Model source <ExternalLink size={12} /></button><button className="text-xs text-accent inline-flex items-center gap-1" onClick={() => void onOpenExternal(status.pack.licenseUrl)}>{status.pack.licenseName} <ExternalLink size={12} /></button></div></div><div className="shrink-0">{status.installed ? removeModelId === status.pack.id ? <div className="flex items-center gap-2"><Button className="bg-red-600 text-white" onClick={() => void remove(status)} disabled={downloadingId !== null}>Remove</Button><Button className="bg-bg-primary text-text-primary border border-border-color shadow-none" onClick={() => setRemoveModelId(null)} disabled={downloadingId !== null}>Cancel</Button></div> : <div className="flex items-center gap-2"><span className="text-green-400 text-sm inline-flex items-center gap-2 py-2"><CheckCircle2 size={17} /> Installed</span><button className="p-2 text-red-300 hover:bg-red-500/10 rounded" onClick={() => setRemoveModelId(status.pack.id)} data-tooltip={`Remove ${status.pack.displayName}`}><Trash2 size={16} /></button></div> : direct ? <Button onClick={() => void download(status)} disabled={downloadingId !== null}>{downloading ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}{downloading ? 'Downloading' : 'Download'}</Button> : <Button onClick={() => void installBundle(status)} disabled={downloadingId !== null}>{downloading ? <Loader2 size={16} className="animate-spin" /> : <FolderOpen size={16} />}{downloading ? 'Installing' : 'Install bundle'}</Button>}</div></div></div>; })}</div>}{error && <div className="rounded-md border border-red-500/50 bg-red-900/10 p-3"><Text variant={TextVariants.small}>{error}</Text><Button className="mt-2 bg-bg-primary text-text-primary border border-border-color shadow-none" onClick={() => void refresh()}><RotateCcw size={15} /> Retry</Button></div>}</div>;
}
