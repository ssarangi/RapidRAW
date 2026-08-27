import { useCallback, useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { BarChart3, Calendar, Camera, Loader2, RefreshCw, Tag, Users } from 'lucide-react';
import { toast } from 'react-toastify';
import { CatalogFacetValue, CatalogMetrics, CatalogSearchQuery, ImageFile, Invokes } from '../ui/AppProperties';
import Text from '../ui/Text';
import Button from '../ui/Button';
import { TextColors, TextVariants, TextWeights } from '../../types/typography';
import { useLibraryStore } from '../../store/useLibraryStore';
import { useUIStore } from '../../store/useUIStore';

type FacetKind = 'year' | 'camera' | 'lens' | 'person' | 'tag' | 'aiTag';

interface FacetPanelProps {
  title: string;
  Icon: typeof Camera;
  values: CatalogFacetValue[];
  kind: FacetKind;
  onSelect(kind: FacetKind, value: string): void;
}

function FacetPanel({ title, Icon, values, kind, onSelect }: FacetPanelProps) {
  return <section className="border border-border-color bg-bg-primary rounded-md min-w-0"><div className="flex items-center gap-2 px-3 py-2 border-b border-border-color/60"><Icon size={16} className="text-text-secondary shrink-0" /><Text variant={TextVariants.small} weight={TextWeights.semibold}>{title}</Text></div>{values.length === 0 ? <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="px-3 py-4">No catalog data yet.</Text> : <div className="divide-y divide-border-color/50">{values.slice(0, 8).map((facet) => <button key={facet.value} className="w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-surface focus:outline-hidden focus:bg-surface" onClick={() => onSelect(kind, facet.value)}><span className="min-w-0 flex-1 truncate text-sm text-text-primary">{facet.value}</span><span className="shrink-0 text-xs tabular-nums text-text-secondary">{facet.count}</span></button>)}</div>}</section>;
}

export default function InsightsView() {
  const [metrics, setMetrics] = useState<CatalogMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isOpeningResults, setIsOpeningResults] = useState(false);

  const loadMetrics = useCallback(async () => {
    setIsLoading(true);
    try { setMetrics(await invoke<CatalogMetrics>(Invokes.GetCatalogMetrics)); }
    catch (error) { console.error('Failed to load catalog metrics:', error); toast.error(`Failed to load catalog metrics: ${error}`); setMetrics(null); }
    finally { setIsLoading(false); }
  }, []);

  useEffect(() => { void loadMetrics(); }, [loadMetrics]);

  const openFacet = async (kind: FacetKind, value: string) => {
    const query: CatalogSearchQuery = { limit: 20_000 };
    if (kind === 'year') query.year = Number(value);
    if (kind === 'camera') query.camera = value;
    if (kind === 'lens') query.lens = value;
    if (kind === 'person') query.person = value;
    if (kind === 'tag') query.tags = [value];
    if (kind === 'aiTag') query.aiTags = [value];
    setIsOpeningResults(true);
    try {
      const files = await invoke<ImageFile[]>(Invokes.SearchCatalogImages, { query });
      const imageRatings: Record<string, number> = {};
      files.forEach((file) => { imageRatings[file.path] = file.rating || 0; });
      useLibraryStore.getState().setLibrary({ currentFolderPath: `Library: ${value}`, activeAlbumId: null, imageList: files, imageRatings, multiSelectedPaths: [], libraryActivePath: null, libraryScrollTop: 0 });
      useLibraryStore.getState().setSearchCriteria({ text: '', tags: [], mode: 'OR' });
      useUIStore.getState().setUI({ activeView: 'library' });
    } catch (error) { console.error('Failed to open insight results:', error); toast.error(`Failed to open catalog results: ${error}`); }
    finally { setIsOpeningResults(false); }
  };

  const summary = metrics ? [['Images', metrics.totalImages], ['Rated', metrics.ratedImages], ['Edited', metrics.editedImages], ['Missing', metrics.missingImages], ['AI suggestions', metrics.aiTagsSuggested], ['AI accepted', metrics.aiTagsAccepted], ['RAM++ analyzed', metrics.ramPlusAnalyzed], ['RAM++ pending', metrics.ramPlusPending], ['RAM++ failed', metrics.ramPlusFailed], ['Cull sessions', metrics.cullSessions], ['Overrides', metrics.cullOverrides]] : [];
  return <div className="flex-1 overflow-y-auto p-5"><div className="mb-5 flex items-start justify-between gap-4"><div><Text variant={TextVariants.title} color={TextColors.accent}>Insights</Text><Text as="div" variant={TextVariants.small} color={TextColors.secondary}>Catalog coverage, metadata, and AI analysis.</Text></div><Button className="h-9 w-9 p-0 bg-surface text-text-primary shadow-none" onClick={() => void loadMetrics()} disabled={isLoading} data-tooltip="Refresh insights"><RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} /></Button></div>{isLoading && !metrics ? <div className="min-h-64 flex flex-col items-center justify-center text-text-secondary"><Loader2 className="mb-3 animate-spin" size={28} /><Text variant={TextVariants.small}>Loading catalog metrics...</Text></div> : metrics ? <><div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-4 gap-3 mb-5">{summary.map(([label, value]) => <div key={String(label)} className="border border-border-color bg-bg-primary rounded-md p-3 min-w-0"><Text as="div" variant={TextVariants.small} color={TextColors.secondary}>{label}</Text><Text variant={TextVariants.heading}>{value}</Text></div>)}</div>{isOpeningResults && <div className="mb-3 flex items-center gap-2 text-text-secondary"><Loader2 size={15} className="animate-spin" /><Text variant={TextVariants.small}>Opening catalog results...</Text></div>}<div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3"><FacetPanel title="Years" Icon={Calendar} values={metrics.years} kind="year" onSelect={openFacet} /><FacetPanel title="Cameras" Icon={Camera} values={metrics.cameras} kind="camera" onSelect={openFacet} /><FacetPanel title="Lenses" Icon={Camera} values={metrics.lenses} kind="lens" onSelect={openFacet} /><FacetPanel title="People" Icon={Users} values={metrics.people} kind="person" onSelect={openFacet} /><FacetPanel title="Tags" Icon={Tag} values={metrics.tags} kind="tag" onSelect={openFacet} /><FacetPanel title="AI Tags" Icon={BarChart3} values={metrics.aiTags} kind="aiTag" onSelect={openFacet} /></div></> : <div className="min-h-64 flex flex-col items-center justify-center text-text-secondary"><BarChart3 className="mb-3" size={32} /><Text variant={TextVariants.small}>Catalog metrics are unavailable.</Text></div>}</div>;
}
