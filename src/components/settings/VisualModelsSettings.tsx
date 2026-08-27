import { useCallback, useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { CheckCircle2, Download, ExternalLink, Loader2, RotateCcw } from 'lucide-react';
import Button from '../ui/Button';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';
import { Invokes, VisualModelPackStatus } from '../ui/AppProperties';

interface VisualModelsSettingsProps { onOpenExternal(url: string): Promise<void>; }

export default function VisualModelsSettings({ onOpenExternal }: VisualModelsSettingsProps) {
  const [statuses, setStatuses] = useState<VisualModelPackStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const refresh = useCallback(async () => { setLoading(true); try { setStatuses(await invoke<VisualModelPackStatus[]>(Invokes.ListVisualModelPackStatuses)); setError(null); } catch (reason) { setError(String(reason)); } finally { setLoading(false); } }, []);
  useEffect(() => { void refresh(); }, [refresh]);
  const download = async (status: VisualModelPackStatus) => { setDownloadingId(status.pack.id); setError(null); try { await invoke(Invokes.DownloadVisualModelPack, { packId: status.pack.id }); await refresh(); } catch (reason) { setError(`Could not download ${status.pack.displayName}: ${String(reason)}`); } finally { setDownloadingId(null); } };

  return <div className="p-6 bg-surface rounded-xl shadow-md space-y-5"><div><Text variant={TextVariants.title} color={TextColors.accent}>Visual AI Models</Text><Text as="div" variant={TextVariants.small} className="mt-2">Local models for broad catalog tags and species classification. Image data stays on this device.</Text></div>{loading ? <div className="py-10 flex justify-center"><Loader2 size={24} className="animate-spin text-accent" /></div> : <div className="space-y-3">{statuses.map((status) => { const downloading = downloadingId === status.pack.id; const direct = status.pack.availability === 'directDownload'; return <div key={status.pack.id} className="rounded-md border border-border-color bg-bg-primary p-4"><div className="flex flex-wrap justify-between gap-3"><div className="min-w-0 flex-1"><div className="flex items-center gap-2"><Text variant={TextVariants.heading}>{status.pack.displayName}</Text><span className={status.installed ? 'text-green-400 text-xs' : direct ? 'text-accent text-xs' : 'text-text-secondary text-xs'}>{status.installed ? 'Installed' : direct ? 'Ready to download' : 'Bundle required'}</span></div><Text as="div" variant={TextVariants.small} className="mt-1">{status.pack.description}</Text><Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mt-2">{status.pack.task}</Text><div className="mt-2 flex gap-3"><button className="text-xs text-accent inline-flex items-center gap-1" onClick={() => void onOpenExternal(status.pack.modelSourceUrl)}>Model source <ExternalLink size={12} /></button><button className="text-xs text-accent inline-flex items-center gap-1" onClick={() => void onOpenExternal(status.pack.licenseUrl)}>{status.pack.licenseName} <ExternalLink size={12} /></button></div></div><div className="shrink-0">{status.installed ? <span className="text-green-400 text-sm inline-flex items-center gap-2 py-2"><CheckCircle2 size={17} /> Installed</span> : direct ? <Button onClick={() => void download(status)} disabled={downloadingId !== null}>{downloading ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}{downloading ? 'Downloading' : 'Download'}</Button> : <span className="text-sm text-text-secondary">Pinned ONNX bundle required</span>}</div></div></div>; })}</div>}{error && <div className="rounded-md border border-red-500/50 bg-red-900/10 p-3"><Text variant={TextVariants.small}>{error}</Text><Button className="mt-2 bg-bg-primary text-text-primary border border-border-color shadow-none" onClick={() => void refresh()}><RotateCcw size={15} /> Retry</Button></div>}</div>;
}
