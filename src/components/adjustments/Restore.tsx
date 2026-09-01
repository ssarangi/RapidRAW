import { useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { Loader2 } from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import { toast } from 'react-toastify';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';
import { useEditorStore } from '../../store/useEditorStore';
import { useLibraryStore } from '../../store/useLibraryStore';

type OperationKind = 'raw_denoise' | 'rgb_denoise';

export default function RestorePanel() {
  const [denoiseStrength, setDenoiseStrength] = useState(0.8);
  const [microcontrastStrength, setMicrocontrastStrength] = useState(0);
  const [detailRecovery, setDetailRecovery] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const selectedImage = useEditorStore((s) => s.selectedImage);
  const { catalogImageId, isRaw } = useLibraryStore(
    useShallow((s) => {
      const catalogImage = selectedImage
        ? s.imageList.find((img) => img.path === selectedImage.path)
        : undefined;
      return {
        catalogImageId: catalogImage?.catalog_image_id ?? null,
        isRaw: catalogImage?.is_raw ?? !!selectedImage?.isRaw,
      };
    }),
  );

  const handleRunRestore = async (operationKind: OperationKind) => {
    if (!catalogImageId) {
      toast.info('Add this image to a catalog library to run restoration.');
      return;
    }
    setIsProcessing(true);
    try {
      const recipe = {
        operationKind,
        modelId: operationKind === 'raw_denoise' ? 'rawnind-utnet2-bayer' : 'nafnet-sidd-rgb',
        modelRevision: 'v1',
        denoiseStrength,
        microcontrastStrength,
        detailRecovery,
        // rawnind-utnet2-bayer's ONNX graph has a static 512x512 input, and
        // the Bayer tiling code halves tileSize before feeding the model, so
        // this must be exactly 1024 for that model; nafnet-sidd-rgb's graph
        // is static at 768x768 and takes tileSize directly.
        tileSize: operationKind === 'raw_denoise' ? 1024 : 768,
        tileOverlap: 64,
      };
      await invoke('start_image_restoration', {
        imageId: catalogImageId,
        recipe,
      });
      toast.success(`${operationKind === 'raw_denoise' ? 'RAW Restore' : 'RGB Denoise'} job started. Check Background Jobs.`);
    } catch (error) {
      console.error('Failed to start restoration:', error);
      toast.error(`Restoration failed: ${error}`);
    } finally {
      setIsProcessing(false);
    }
  };

  if (!catalogImageId) {
    return (
      <Text variant={TextVariants.small} color={TextColors.secondary}>
        Add this image to a catalog library to run RAW Restore or RGB Denoise.
      </Text>
    );
  }

  return (
    <div>
      <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mb-3">
        {isRaw
          ? 'Runs an AI Bayer-level denoise and demosaic pass on the RAW sensor data, with optional microcontrast and detail recovery.'
          : 'Runs an AI denoise pass on the developed image, with optional microcontrast and detail recovery.'}
      </Text>

      <div className="mb-4">
        <div className="flex justify-between text-xs text-text-secondary mb-1">
          <span>Noise Reduction</span>
          <span>{Math.round(denoiseStrength * 100)}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={denoiseStrength}
          onChange={(e) => setDenoiseStrength(Number(e.target.value))}
          className="w-full accent-accent"
        />
      </div>

      <div className="mb-4">
        <div className="flex justify-between text-xs text-text-secondary mb-1">
          <span>Microcontrast</span>
          <span>{Math.round(microcontrastStrength * 100)}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={microcontrastStrength}
          onChange={(e) => setMicrocontrastStrength(Number(e.target.value))}
          className="w-full accent-accent"
        />
      </div>

      <div className="mb-4">
        <div className="flex justify-between text-xs text-text-secondary mb-1">
          <span>Detail Recovery</span>
          <span>{Math.round(detailRecovery * 100)}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={detailRecovery}
          onChange={(e) => setDetailRecovery(Number(e.target.value))}
          className="w-full accent-accent"
        />
      </div>

      {isRaw ? (
        <button
          type="button"
          className="w-full flex items-center justify-center bg-accent text-button-text py-2 text-xs font-medium rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={isProcessing}
          onClick={() => void handleRunRestore('raw_denoise')}
          data-tooltip="Run Bayer RAW denoise and demosaic restoration"
        >
          {isProcessing ? <Loader2 className="w-4 h-4 animate-spin" /> : 'RAW Restore'}
        </button>
      ) : (
        <button
          type="button"
          className="w-full flex items-center justify-center bg-surface border border-border-color text-text-primary py-2 text-xs font-medium rounded-md hover:bg-card-active disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={isProcessing}
          onClick={() => void handleRunRestore('rgb_denoise')}
          data-tooltip="Run developed-image RGB denoise"
        >
          {isProcessing ? <Loader2 className="w-4 h-4 animate-spin" /> : 'RGB Denoise'}
        </button>
      )}
    </div>
  );
}
