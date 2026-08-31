import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import clsx from 'clsx';
import { GeminiCritique, Invokes } from '../../ui/AppProperties';
import { useSettingsStore } from '../../../store/useSettingsStore';

interface GeminiCritiquePanelProps {
  imagePath: string;
  catalogImageId: number | null | undefined;
  imageWidth?: number | null;
  imageHeight?: number | null;
  previewUrl?: string | null;
}

/**
 * On-demand Gemini vision critique, separate from the local heuristic
 * decision factors used elsewhere - only runs when the user asks for it, and
 * the result is cached server-side so a photo is never re-billed on repeat
 * views. Shared by the culling review rail and the editor's AI panel so both
 * read from the exact same request/render logic.
 */
export default function GeminiCritiquePanel({ imagePath, catalogImageId, imageWidth, imageHeight, previewUrl }: GeminiCritiquePanelProps) {
  const [geminiCritique, setGeminiCritique] = useState<GeminiCritique | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const hasGeminiApiKey = useSettingsStore((s) => !!s.appSettings?.geminiApiKey?.trim());

  useEffect(() => {
    setGeminiCritique(null);
    setError(null);
  }, [imagePath]);

  const requestCritique = async () => {
    if (!catalogImageId) return;
    setIsLoading(true);
    setError(null);
    try {
      const critique = await invoke<GeminiCritique>(Invokes.GetOrGenerateGeminiCritique, {
        imageId: catalogImageId,
      });
      setGeminiCritique(critique);
    } catch (err) {
      setError(String(err));
    } finally {
      setIsLoading(false);
    }
  };

  if (!catalogImageId) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between px-0.5">
        <div className="text-xs font-semibold text-text-primary">AI Critique (Gemini)</div>
        {!geminiCritique && (
          <button
            onClick={() => void requestCritique()}
            disabled={isLoading || !hasGeminiApiKey}
            className="text-xs text-accent hover:text-accent/80 underline underline-offset-2 disabled:opacity-40 disabled:no-underline cursor-pointer disabled:cursor-not-allowed"
            title={hasGeminiApiKey ? undefined : 'Add a Gemini API key in Settings > General > Tagging first'}
          >
            {isLoading ? 'Analyzing...' : 'Get AI Critique'}
          </button>
        )}
      </div>
      {error && (
        <div className="rounded-md border border-red-500/30 bg-red-500/10 px-2.5 py-2 text-xs text-red-200">
          {error}
        </div>
      )}
      {geminiCritique && (
        <div className="space-y-2">
          <div className="rounded-md border border-border-color bg-bg-primary p-2.5">
            <div className="text-sm text-text-primary leading-relaxed">{geminiCritique.overallSummary}</div>
          </div>
          {geminiCritique.regions.length > 0 && (
            <div
              className="relative w-full overflow-hidden rounded-md border border-border-color bg-surface"
              style={{ aspectRatio: imageWidth && imageHeight ? `${imageWidth} / ${imageHeight}` : '3 / 2' }}
            >
              {previewUrl && <img src={previewUrl} alt="" className="h-full w-full object-cover" />}
              {geminiCritique.regions.map((region, index) => {
                const [x, y, w, h] = region.box;
                return (
                  <div
                    key={index}
                    className={clsx(
                      'absolute rounded border-2',
                      region.positive ? 'border-green-400' : 'border-red-400',
                    )}
                    style={{
                      left: `${x * 100}%`,
                      top: `${y * 100}%`,
                      width: `${w * 100}%`,
                      height: `${h * 100}%`,
                    }}
                  >
                    <span
                      className={clsx(
                        'absolute -left-1 -top-1 flex h-5 w-5 items-center justify-center rounded-full text-[11px] font-bold text-black shadow-xs',
                        region.positive ? 'bg-green-400' : 'bg-red-400',
                      )}
                    >
                      {index + 1}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
          <div className="space-y-1">
            {geminiCritique.regions.map((region, index) => (
              <div key={index} className="flex items-start gap-1.5 text-sm text-text-primary">
                <span
                  className={clsx(
                    'mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] font-bold text-black',
                    region.positive ? 'bg-green-400' : 'bg-red-400',
                  )}
                >
                  {index + 1}
                </span>
                <div className="min-w-0">
                  <span className="font-semibold">{region.label}</span>
                  <span className="text-text-secondary"> - {region.note}</span>
                </div>
              </div>
            ))}
          </div>
          {geminiCritique.cached && (
            <div className="text-[11px] text-text-secondary italic">Showing a previously generated critique for this photo.</div>
          )}
        </div>
      )}
    </div>
  );
}
