import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { Camera } from 'lucide-react';
import Text from '../../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../../types/typography';
import { IconAperture, IconShutter, IconIso, IconFocalLength, IconLens } from './ExifIcons';

interface CameraSetting {
  format?(value: number): string | number;
  label: string;
}

interface CameraSettings {
  [index: string]: CameraSetting;
  ExposureTime: CameraSetting;
  FNumber: CameraSetting;
  FocalLengthIn35mmFilm: CameraSetting;
  LensModel: CameraSetting;
  PhotographicSensitivity: CameraSetting;
}

const CAMERA_ICONS: Record<string, React.FC> = {
  FNumber: IconAperture,
  ExposureTime: IconShutter,
  PhotographicSensitivity: IconIso,
  FocalLengthIn35mmFilm: IconFocalLength,
  LensModel: IconLens,
};

const CAMERA_GRID_KEYS = ['ExposureTime', 'FNumber', 'PhotographicSensitivity', 'FocalLengthIn35mmFilm'];

/**
 * Condensed camera-settings summary (aperture/shutter/ISO/focal length/lens),
 * shared by MetadataPanel and the culling review rail so both read from the
 * exact same EXIF formatting rules instead of drifting apart.
 */
export default function ExifCameraSummary({ exif }: { exif: { [key: string]: string } | null | undefined }) {
  const { t } = useTranslation();

  const KEY_CAMERA_SETTINGS_MAP: CameraSettings = useMemo(() => ({
    FNumber: {
      format: (value: number) => {
        const fStr = String(value);
        return fStr.toLowerCase().startsWith('f') ? fStr : `f/${fStr}`;
      },
      label: t('editor.metadata.camera.aperture'),
    },
    ExposureTime: {
      format: (value: number) => (String(value).endsWith('s') ? value : `${value}s`),
      label: t('editor.metadata.camera.shutterSpeed'),
    },
    PhotographicSensitivity: {
      format: (value: number) => `${value}`,
      label: t('editor.metadata.camera.iso'),
    },
    FocalLengthIn35mmFilm: {
      format: (value: number) => (String(value).endsWith('mm') ? value : `${value} mm`),
      label: t('editor.metadata.camera.focalLength'),
    },
    LensModel: {
      format: (value: number) => String(value).replace(/"/g, ''),
      label: t('editor.metadata.camera.lens'),
    },
  }), [t]);

  const cameraName = useMemo(() => {
    const exifData = exif || {};
    const make = (exifData.Make || '').trim();
    const model = (exifData.Model || '').trim();
    if (!make && !model) return null;
    // Some cameras already include the brand in the model string (e.g.
    // "Canon EOS R5"), so avoid a duplicated "Canon Canon EOS R5".
    if (!make || model.toLowerCase().startsWith(make.toLowerCase())) return model || make;
    return `${make} ${model}`;
  }, [exif]);

  const { cameraGridSettings, lensSetting } = useMemo(() => {
    const exifData = exif || {};
    const cameraGridSettings = CAMERA_GRID_KEYS.map((key) => {
      const value = exifData[key];
      const hasValue = value !== undefined && value !== null && value !== '';
      return {
        key,
        label: KEY_CAMERA_SETTINGS_MAP[key].label,
        value: hasValue && KEY_CAMERA_SETTINGS_MAP[key].format
          ? KEY_CAMERA_SETTINGS_MAP[key].format!(value as unknown as number)
          : hasValue ? value : '-',
      };
    });
    const lensValue = exifData['LensModel'];
    const hasLensValue = lensValue !== undefined && lensValue !== null && lensValue !== '';
    const lensSetting = {
      key: 'LensModel',
      label: KEY_CAMERA_SETTINGS_MAP.LensModel.label,
      value: hasLensValue && KEY_CAMERA_SETTINGS_MAP.LensModel.format
        ? KEY_CAMERA_SETTINGS_MAP.LensModel.format(lensValue as unknown as number)
        : hasLensValue ? lensValue : '-',
    };
    return { cameraGridSettings, lensSetting };
  }, [exif, KEY_CAMERA_SETTINGS_MAP]);

  const LensIcon = CAMERA_ICONS.LensModel;

  return (
    <div className="flex flex-col gap-2">
      {cameraName && (
        <div
          className="flex items-center gap-2 bg-surface border border-surface px-3 py-2 rounded-xl cursor-default"
          data-tooltip={t('editor.metadata.camera.title')}
        >
          <span className="text-text-secondary opacity-90 flex items-center justify-center shrink-0">
            <Camera size={16} />
          </span>
          <Text as="span" variant={TextVariants.small} color={TextColors.primary} weight={TextWeights.medium} className="truncate">
            {cameraName}
          </Text>
        </div>
      )}
      <div className="grid grid-cols-2 gap-2">
        {cameraGridSettings.map((item) => {
          const Icon = CAMERA_ICONS[item.key];
          return (
            <div
              key={item.key}
              className="flex items-center gap-2 bg-surface border border-surface px-3 py-2 rounded-xl cursor-default"
              data-tooltip={item.label}
            >
              {Icon && (
                <span className="text-text-secondary opacity-90 flex items-center justify-center shrink-0">
                  <Icon />
                </span>
              )}
              <Text as="span" variant={TextVariants.small} color={TextColors.primary} weight={TextWeights.medium} className="truncate">
                {item.value}
              </Text>
            </div>
          );
        })}
      </div>
      <div
        className="flex items-center gap-2 bg-surface border border-surface px-3 py-2 rounded-xl cursor-default"
        data-tooltip={lensSetting.label}
      >
        {LensIcon && (
          <span className="text-text-secondary opacity-90 flex items-center justify-center shrink-0">
            <LensIcon />
          </span>
        )}
        <Text as="span" variant={TextVariants.small} weight={TextWeights.medium} color={TextColors.primary} className="truncate">
          {lensSetting.value}
        </Text>
      </div>
    </div>
  );
}
