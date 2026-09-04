import { Aperture, SlidersHorizontal, Sparkles } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import Slider from '../ui/Slider';
import Switch from '../ui/Switch';
import Dropdown from '../ui/Dropdown';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';
import { Adjustments, DemosaicAlgorithm, RawDevelopAdjustment, SharpenMethod } from '../../utils/adjustments';

interface RawDevelopPanelProps {
  adjustments: Adjustments;
  setAdjustments(adjustments: Partial<Adjustments>): any;
  isRaw: boolean;
  onDragStateChange?: (isDragging: boolean) => void;
}

const DEFAULT_MANUAL_AMOUNT = 30;

export default function RawDevelopPanel({
  adjustments,
  setAdjustments,
  isRaw,
  onDragStateChange,
}: RawDevelopPanelProps) {
  const { t } = useTranslation();
  const demosaicOptions = Object.values(DemosaicAlgorithm).map((value) => ({
    value,
    label: t(`editor.adjustments.rawDevelop.demosaicOptions.${value}`),
  }));
  const sharpenMethodOptions = Object.values(SharpenMethod).map((value) => ({
    value,
    label: t(`editor.adjustments.rawDevelop.sharpenMethodOptions.${value}`),
  }));
  const setField = (key: string, value: unknown) =>
    setAdjustments((prev: Partial<Adjustments>) => ({ ...prev, [key]: value }));

  const renderAutoSlider = (key: string, label: string, value: number, icon: React.ReactNode) => {
    const isAuto = value < 0;
    return (
      <div
        className={`rounded-md border p-3 transition-colors ${isAuto ? 'border-border-color bg-bg-primary/60' : 'border-[#d89538]/40 bg-[#d89538]/[0.055]'}`}
      >
        <div className="mb-3 flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2">
            <span
              className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-sm ${isAuto ? 'bg-surface text-text-secondary' : 'bg-[#d89538]/15 text-[#e9ab4a]'}`}
            >
              {icon}
            </span>
            <Text variant={TextVariants.small} color={TextColors.primary} className="truncate font-medium">
              {label}
            </Text>
          </div>
          <Switch
            id={`raw-develop-auto-${key}`}
            label={t('editor.adjustments.rawDevelop.autoToggleLabel')}
            checked={isAuto}
            onChange={(checked) => setField(key, checked ? -1 : DEFAULT_MANUAL_AMOUNT)}
            className="shrink-0 scale-90"
          />
        </div>
        <Slider
          label=""
          min={0}
          max={100}
          step={1}
          value={isAuto ? 0 : value}
          disabled={isAuto}
          onChange={(event: any) => setField(key, Number(event.target.value))}
          onDragStateChange={onDragStateChange}
        />
      </div>
    );
  };

  if (!isRaw) {
    return (
      <div className="rounded-md border border-border-color bg-bg-primary/50 p-4">
        <Text variant={TextVariants.small} color={TextColors.secondary}>
          {t('editor.adjustments.rawDevelop.description')}
        </Text>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="relative overflow-hidden rounded-md border border-[#d89538]/35 bg-[linear-gradient(115deg,rgba(216,149,56,0.12),rgba(216,149,56,0.025)_48%,transparent_72%)] p-3">
        <div className="absolute inset-y-0 left-0 w-0.5 bg-[#e9ab4a]" />
        <div className="flex items-start gap-3">
          <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-sm bg-[#d89538]/15 text-[#efb34f]">
            <Aperture size={16} />
          </span>
          <div className="min-w-0">
            <Text variant={TextVariants.small} color={TextColors.primary} className="font-medium">
              {t('editor.adjustments.sections.rawDevelop')}
            </Text>
            <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mt-0.5 leading-relaxed">
              {t('editor.adjustments.rawDevelop.description')}
            </Text>
          </div>
        </div>
        <div className="mt-3 border-t border-[#d89538]/20 pt-3">
          <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mb-1.5">
            {t('editor.adjustments.rawDevelop.demosaicLabel')}
          </Text>
          <Dropdown
            value={adjustments.rawDemosaicAlgorithm}
            options={demosaicOptions}
            onChange={(value) => setField(RawDevelopAdjustment.DemosaicAlgorithm, value)}
            triggerClassName="border-[#d89538]/30 bg-bg-primary/70"
          />
        </div>
      </div>

      <div className="grid gap-2">
        <div
          className={`rounded-md border p-3 transition-colors ${adjustments.rawAiDenoiseEnabled ? 'border-[#d89538]/40 bg-[#d89538]/[0.055]' : 'border-border-color bg-bg-primary/60'}`}
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex min-w-0 items-start gap-2">
              <span
                className={`mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-sm ${adjustments.rawAiDenoiseEnabled ? 'bg-[#d89538]/15 text-[#e9ab4a]' : 'bg-surface text-text-secondary'}`}
              >
                <Sparkles size={14} />
              </span>
              <div className="min-w-0">
                <Text variant={TextVariants.small} color={TextColors.primary} className="font-medium">
                  AI sensor denoise
                </Text>
                <Text
                  as="div"
                  variant={TextVariants.small}
                  color={TextColors.secondary}
                  className="mt-0.5 leading-relaxed"
                >
                  Uses RawNIND directly on the Bayer sensor data during RAW development.
                </Text>
              </div>
            </div>
            <Switch
              id="raw-develop-ai-denoise-enabled"
              label="Enable AI sensor denoise"
              checked={adjustments.rawAiDenoiseEnabled}
              onChange={(checked) => setField(RawDevelopAdjustment.AiDenoiseEnabled, checked)}
              className="shrink-0 scale-90"
            />
          </div>
        </div>
        {renderAutoSlider(
          RawDevelopAdjustment.DenoiseAmount,
          t('editor.adjustments.rawDevelop.denoiseLabel'),
          adjustments.rawDenoiseAmount,
          <SlidersHorizontal size={14} />,
        )}
        {renderAutoSlider(
          RawDevelopAdjustment.SharpenAmount,
          t('editor.adjustments.rawDevelop.sharpenLabel'),
          adjustments.rawSharpenAmount,
          <Sparkles size={14} />,
        )}
      </div>

      <div className="rounded-md border border-border-color bg-bg-primary/50 p-3">
        <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mb-1.5">
          {t('editor.adjustments.rawDevelop.sharpenMethodLabel')}
        </Text>
        <Dropdown
          value={adjustments.rawSharpenMethod}
          options={sharpenMethodOptions}
          onChange={(value) => setField(RawDevelopAdjustment.SharpenMethod, value)}
        />
        <div className="mt-3 border-t border-border-color pt-3">
          <Switch
            id="raw-develop-preprocess-enabled"
            label={t('editor.adjustments.rawDevelop.preprocessLabel')}
            checked={adjustments.rawPreprocessEnabled}
            onChange={(checked) => setField(RawDevelopAdjustment.PreprocessEnabled, checked)}
          />
        </div>
      </div>
    </div>
  );
}
