import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { BarChart3 } from 'lucide-react';
import { CatalogMetrics, Invokes } from '../ui/AppProperties';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';

export default function InsightsView() {
  const [metrics, setMetrics] = useState<CatalogMetrics | null>(null);
  useEffect(() => { void invoke<CatalogMetrics>(Invokes.GetCatalogMetrics).then(setMetrics).catch(console.error); }, []);
  const values = metrics ? [
    ['Images', metrics.totalImages], ['Rated', metrics.ratedImages], ['Edited', metrics.editedImages],
    ['Missing', metrics.missingImages], ['AI pending', metrics.aiTagsSuggested], ['AI accepted', metrics.aiTagsAccepted],
  ] : [];
  return <div className="flex-1 overflow-y-auto p-5"><div className="mb-6"><Text variant={TextVariants.title} color={TextColors.accent}>Insights</Text><Text variant={TextVariants.small}>Catalog coverage and AI analysis status.</Text></div>{metrics ? <div className="grid grid-cols-2 md:grid-cols-3 gap-3">{values.map(([label, value]) => <div key={String(label)} className="border border-border-color bg-bg-primary rounded-md p-4"><Text variant={TextVariants.small} color={TextColors.secondary}>{label}</Text><Text variant={TextVariants.heading}>{value}</Text></div>)}</div> : <div className="min-h-64 flex flex-col items-center justify-center"><BarChart3 className="text-text-secondary mb-3" size={32}/><Text variant={TextVariants.small}>Loading catalog metrics...</Text></div>}</div>;
}
