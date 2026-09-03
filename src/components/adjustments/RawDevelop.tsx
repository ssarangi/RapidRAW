import { useTranslation } from 'react-i18next';
import { Loader2 } from 'lucide-react';
import Slider from '../ui/Slider';
import Switch from '../ui/Switch';
import Dropdown from '../ui/Dropdown';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';
import { Adjustments, DemosaicAlgorithm, RawDevelopAdjustment, SharpenMethod } from '../../utils/adjustments';
import { useEditorStore } from '../../store/useEditorStore';

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
  const isRawReprocessing = useEditorStore((s) => s.isRawReprocessing);

  const demosaicOptions = Object.values(DemosaicAlgorithm).map((value) => ({
    value,
    label: t(`editor.adjustments.rawDevelop.demosaicOptions.${value}`),
  }));

  const sharpenMethodOptions = Object.values(SharpenMethod).map((value) => ({
    value,
    label: t(`editor.adjustments.rawDevelop.sharpenMethodOptions.${value}`),
  }));

  const setField = (key: string, value: unknown) => {
    setAdjustments((prev: Partial<Adjustments>) => ({ ...prev, [key]: value }));
  };

  const renderAutoSlider = (key: string, label: string, value: number) => {
    const isAuto = value < 0;
    return (
      <div className="mb-3">
        <div className="flex items-center justify-between gap-3 mb-1">
          <Text variant={TextVariants.small} color={TextColors.secondary}>
            {label}
          </Text>
          <Switch
            id={`raw-develop-auto-${key}`}
            label={t('editor.adjustments.rawDevelop.autoToggleLabel')}
            checked={isAuto}
            onChange={(checked) => setField(key, checked ? -1 : DEFAULT_MANUAL_AMOUNT)}
            className="scale-90 shrink-0"
          />
        </div>
        <Slider
          label=""
          min={0}
          max={100}
          step={1}
          value={isAuto ? 0 : value}
          disabled={isAuto}
          onChange={(e: any) => setField(key, Number(e.target.value))}
          onDragStateChange={onDragStateChange}
        />
      </div>
    );
  };

  if (!isRaw) {
    return (
      <Text variant={TextVariants.small} color={TextColors.secondary}>
        {t('editor.adjustments.rawDevelop.description')}
      </Text>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between gap-3 mb-3">
        <Text as="div" variant={TextVariants.small} color={TextColors.secondary}>
          {t('editor.adjustments.rawDevelop.description')}
        </Text>
        {isRawReprocessing && (
          <div className="flex items-center gap-1.5 shrink-0" title={t('editor.adjustments.rawDevelop.reprocessing')}>
            <Loader2 size={13} className="animate-spin text-text-secondary" />
            <Text variant={TextVariants.small} color={TextColors.secondary}>
              {t('editor.adjustments.rawDevelop.reprocessing')}
            </Text>
          </div>
        )}
      </div>

      <div className="mb-4">
        <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mb-1">
          {t('editor.adjustments.rawDevelop.demosaicLabel')}
        </Text>
        <Dropdown
          value={adjustments.rawDemosaicAlgorithm}
          options={demosaicOptions}
          onChange={(value) => setField(RawDevelopAdjustment.DemosaicAlgorithm, value)}
        />
      </div>

      {renderAutoSlider(
        RawDevelopAdjustment.DenoiseAmount,
        t('editor.adjustments.rawDevelop.denoiseLabel'),
        adjustments.rawDenoiseAmount,
      )}
      {renderAutoSlider(
        RawDevelopAdjustment.SharpenAmount,
        t('editor.adjustments.rawDevelop.sharpenLabel'),
        adjustments.rawSharpenAmount,
      )}

      <div className="mb-4">
        <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mb-1">
          {t('editor.adjustments.rawDevelop.sharpenMethodLabel')}
        </Text>
        <Dropdown
          value={adjustments.rawSharpenMethod}
          options={sharpenMethodOptions}
          onChange={(value) => setField(RawDevelopAdjustment.SharpenMethod, value)}
        />
      </div>

      <Switch
        id="raw-develop-preprocess-enabled"
        label={t('editor.adjustments.rawDevelop.preprocessLabel')}
        checked={adjustments.rawPreprocessEnabled}
        onChange={(checked) => setField(RawDevelopAdjustment.PreprocessEnabled, checked)}
      />
    </div>
  );
}
