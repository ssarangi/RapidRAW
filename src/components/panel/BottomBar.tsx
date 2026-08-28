import { useState, useEffect, useRef } from 'react';
import {
  Star,
  Copy,
  ClipboardPaste,
  Check,
  Settings,
  Filter,
  PanelLeft,
  PanelBottom,
  PanelRight,
  Loader2,
  Database,
  X,
  Pause,
  Play,
  RotateCcw,
  Square,
} from 'lucide-react';
import { convertFileSrc, invoke } from '@tauri-apps/api/core';
import clsx from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';
import { useShallow } from 'zustand/react/shallow';
import { useTranslation } from 'react-i18next';

import Filmstrip from './Filmstrip';
import { BackgroundJob, GLOBAL_KEYS, ImageFile, Invokes, SelectedImage, ThumbnailAspectRatio } from '../ui/AppProperties';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';
import { useEditorStore } from '../../store/useEditorStore';
import { useLibraryStore } from '../../store/useLibraryStore';
import { useUIStore } from '../../store/useUIStore';
import { COLOR_LABELS } from '../../utils/adjustments';
import { useProcessStore } from '../../store/useProcessStore';

interface BottomBarProps {
  filmstripHeight?: number;
  imageList?: Array<ImageFile>;
  imageRatings?: Record<string, number> | null;
  isAndroid?: boolean;
  isCopied: boolean;
  isCopyDisabled: boolean;
  isExportDisabled?: boolean;
  isFilmstripVisible?: boolean;
  isLibraryView?: boolean;
  isLoading?: boolean;
  isPasted: boolean;
  isPasteDisabled: boolean;
  isRatingDisabled?: boolean;
  isResetDisabled?: boolean;
  isResizing?: boolean;
  multiSelectedPaths?: Array<string>;
  onClearSelection?(): void;
  onContextMenu?(event: any, path: string): void;
  onEmptyAreaContextMenu?(event: any): void;
  onCopy(): void;
  onExportClick?(): void;
  onImageSelect?(path: string, event: any): void;
  onOpenCopyPasteSettings?(): void;
  onRequestThumbnails?(paths: string[]): void;
  onPaste(): void;
  onRate(rate: number): void;
  onReset?(): void;
  onZoomChange?(zoomValue: number, fitToWindow?: boolean): void;
  rating: number;
  selectedImage?: SelectedImage;
  setIsFilmstripVisible?(isVisible: boolean): void;
  showFilmstrip?: boolean;
  showZoomControls?: boolean;
  thumbnailAspectRatio: ThumbnailAspectRatio;
  totalImages?: number;
}

interface StarRatingProps {
  disabled: boolean;
  onRate(rate: number): void;
  rating: number;
}

const StarRating = ({ rating, onRate, disabled }: StarRatingProps) => {
  const { t } = useTranslation();

  return (
    <div className={clsx('flex items-center gap-1', disabled && 'cursor-not-allowed')}>
      {[...Array(5)].map((_, index: number) => {
        const starValue = index + 1;
        return (
          <button
            className="disabled:cursor-not-allowed"
            disabled={disabled}
            key={starValue}
            onClick={() => !disabled && onRate(starValue === rating ? 0 : starValue)}
            data-tooltip={
              disabled
                ? t('ui.bottomBar.tooltips.selectToRate')
                : t('ui.bottomBar.tooltips.rateStars', { count: starValue })
            }
          >
            <Star
              size={18}
              className={clsx(
                'transition-colors duration-150',
                disabled
                  ? 'text-text-secondary opacity-40'
                  : starValue <= rating
                    ? 'fill-accent text-accent'
                    : 'text-text-secondary hover:text-accent',
              )}
            />
          </button>
        );
      })}
    </div>
  );
};

interface PanelToggleButtonProps {
  onClick: () => void;
  Icon: React.ElementType;
  tooltip: string;
  disabled?: boolean;
}

const PanelToggleButton = ({ onClick, Icon, tooltip, disabled = false }: PanelToggleButtonProps) => (
  <button
    className={clsx(
      'p-1.5 rounded-md transition-colors',
      disabled
        ? 'text-text-secondary opacity-40 cursor-not-allowed'
        : 'text-text-secondary hover:bg-surface hover:text-text-primary',
    )}
    onClick={() => !disabled && onClick()}
    disabled={disabled}
    data-tooltip={tooltip}
  >
    <Icon size={18} />
  </button>
);

export default function BottomBar({
  filmstripHeight,
  imageList = [],
  imageRatings,
  isAndroid,
  isCopied,
  isCopyDisabled,
  isFilmstripVisible,
  isLibraryView = false,
  isLoading = false,
  isPasted,
  isPasteDisabled,
  isRatingDisabled = false,
  isResizing,
  multiSelectedPaths = [],
  onClearSelection,
  onContextMenu,
  onEmptyAreaContextMenu,
  onCopy,
  onImageSelect,
  onOpenCopyPasteSettings,
  onRequestThumbnails,
  onPaste,
  onRate,
  onZoomChange = () => {},
  rating,
  selectedImage,
  setIsFilmstripVisible,
  showFilmstrip = true,
  showZoomControls = true,
  thumbnailAspectRatio,
  totalImages,
}: BottomBarProps) {
  const { t } = useTranslation();

  const { isInstantTransition, uiVisibility, setUI } = useUIStore(
    useShallow((state) => ({
      isInstantTransition: state.isInstantTransition,
      uiVisibility: state.uiVisibility,
      setUI: state.setUI,
    })),
  );

  const isLeftOpen = uiVisibility.leftPanel;
  const isRightOpen = uiVisibility.rightPanel;
  const isBottomOpen = uiVisibility.filmstrip;

  const toggleLeft = () =>
    setUI((s) => {
      const isOpening = !s.uiVisibility.leftPanel;
      return {
        uiVisibility: { ...s.uiVisibility, leftPanel: isOpening },
        leftPanelWidth: isOpening && s.leftPanelWidth < 250 ? 350 : s.leftPanelWidth,
      };
    });

  const toggleRight = () =>
    setUI((s) => {
      const isOpening = !s.uiVisibility.rightPanel;
      return {
        uiVisibility: { ...s.uiVisibility, rightPanel: isOpening },
        rightPanelWidth: isOpening && s.rightPanelWidth < 250 ? 350 : s.rightPanelWidth,
      };
    });

  const toggleBottom = () =>
    setUI((s) => ({
      uiVisibility: { ...s.uiVisibility, filmstrip: !s.uiVisibility.filmstrip },
    }));

  const { displaySize, originalSize } = useEditorStore(
    useShallow((state) => ({
      displaySize: state.displaySize,
      originalSize: state.originalSize,
    })),
  );

  const [isEditingPercent, setIsEditingPercent] = useState(false);
  const [percentInputValue, setPercentInputValue] = useState('');
  const isDraggingSlider = useRef(false);
  const [isZoomActive, setIsZoomActive] = useState(false);

  const percentInputRef = useRef<HTMLInputElement>(null);
  const [isZoomLabelHovered, setIsZoomLabelHovered] = useState(false);
  const isZoomReady = !isLoading && originalSize && originalSize.width > 0 && displaySize && displaySize.width > 0;

  const currentOriginalPercent = isZoomReady
    ? (displaySize.width * (typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1)) / originalSize.width
    : 1.0;

  const [latchedSliderValue, setLatchedSliderValue] = useState(1.0);
  const [latchedDisplayPercent, setLatchedDisplayPercent] = useState(100);

  const numSelected = multiSelectedPaths.length;
  const total = totalImages ?? 0;
  const showSelectionCounter = numSelected > 1;

  const [isFilterExpanded, setIsFilterExpanded] = useState(false);
  const [isCatalogScanModalOpen, setIsCatalogScanModalOpen] = useState(false);
  const [backgroundJobs, setBackgroundJobs] = useState<BackgroundJob[]>([]);
  const [backgroundJobsError, setBackgroundJobsError] = useState<string | null>(null);
  const [thumbError, setThumbError] = useState(false);
  const { catalogScan, catalogScanThumbnail, thumbnails } = useProcessStore(
    useShallow((state) => ({
      catalogScan: state.catalogScan,
      catalogScanThumbnail: state.catalogScan.currentPath ? state.thumbnails[state.catalogScan.currentPath] : undefined,
      thumbnails: state.thumbnails,
    })),
  );
  const { filterCriteria, setFilterCriteria } = useLibraryStore(
    useShallow((state) => ({
      filterCriteria: state.filterCriteria,
      setFilterCriteria: state.setFilterCriteria,
    })),
  );

  const isRawFile = (path: string | null | undefined): boolean => {
    if (!path) return false;
    const ext = path.split('.').pop()?.toLowerCase();
    return ['arw', 'cr2', 'cr3', 'nef', 'dng', 'orf', 'rw2', 'pef', 'raf', 'raw', 'sr2', 'srf', 'nrw', 'kdc', 'mrw'].includes(ext || '');
  };

  const allColors = [...COLOR_LABELS, { name: 'none', color: '#9ca3af' }];
  const currentHeight = filmstripHeight ?? 120;
  const isCollapsed = !isFilmstripVisible;
  const effectiveHeight = isFilmstripVisible ? currentHeight : 0;
  const shouldAnimate = !isInstantTransition && (!isResizing || isCollapsed);
  const catalogScanFileName = catalogScan.currentPath?.split(/[\\/]/).pop() || '';
  const activeBackgroundJobs = backgroundJobs.filter((job) =>
    ['queued', 'running', 'paused', 'cancelling'].includes(job.state),
  );
  const activeBackgroundJob = activeBackgroundJobs[0];
  const currentDisplayedPath =
    catalogScan.isActive
      ? catalogScan.currentPath
      : activeBackgroundJob?.currentItem || catalogScan.currentPath || null;
  const currentThumbnailSrc =
    catalogScan.isActive && catalogScanThumbnail
      ? catalogScanThumbnail
      : currentDisplayedPath
        ? thumbnails[currentDisplayedPath] || (!isRawFile(currentDisplayedPath) ? convertFileSrc(currentDisplayedPath) : undefined)
        : undefined;

  useEffect(() => {
    setThumbError(false);
    if (
      currentDisplayedPath &&
      !thumbnails[currentDisplayedPath]
    ) {
      invoke('update_thumbnail_queue', { paths: [{ path: currentDisplayedPath, modified: null }] }).catch(() => {});
    }
  }, [currentDisplayedPath, thumbnails]);
  const catalogScanPercent =
    catalogScan.isActive && catalogScan.total > 0
      ? Math.min(100, Math.round((catalogScan.current / catalogScan.total) * 100))
      : activeBackgroundJob && activeBackgroundJob.total > 0
        ? Math.min(100, Math.round((activeBackgroundJob.current / activeBackgroundJob.total) * 100))
        : null;

  const handlePauseCatalogScan = async () => {
    try {
      await invoke(Invokes.PauseCatalogScan);
      useProcessStore.getState().setProcess((state) => ({
        catalogScan: { ...state.catalogScan, isPaused: true, message: 'Indexing paused' },
      }));
    } catch (err) {
      console.error('Failed to pause catalog scan:', err);
    }
  };

  const handleResumeCatalogScan = async () => {
    try {
      await invoke(Invokes.ResumeCatalogScan);
      useProcessStore.getState().setProcess((state) => ({
        catalogScan: { ...state.catalogScan, isPaused: false, message: 'Indexing image metadata' },
      }));
    } catch (err) {
      console.error('Failed to resume catalog scan:', err);
    }
  };

  const handleCancelCatalogScan = async () => {
    try {
      await invoke(Invokes.CancelCatalogScan);
      useProcessStore.getState().setProcess((state) => ({
        catalogScan: { ...state.catalogScan, isPaused: false, message: 'Cancelling indexing...' },
      }));
    } catch (err) {
      console.error('Failed to cancel catalog scan:', err);
    }
  };

  const handleCancelBackgroundJob = async (jobId: string) => {
    try {
      await invoke(Invokes.CancelBackgroundJob, { jobId });
      setBackgroundJobs((jobs) =>
        jobs.map((job) => (job.id === jobId ? { ...job, state: 'cancelling', message: 'Cancellation requested' } : job)),
      );
    } catch (error) {
      console.error('Failed to cancel background job:', error);
    }
  };
  const handlePauseBackgroundJob = async (jobId: string, resume: boolean) => {
    try { await invoke(resume ? Invokes.ResumeBackgroundJob : Invokes.PauseBackgroundJob, { jobId }); setBackgroundJobs((jobs) => jobs.map((job) => job.id === jobId ? { ...job, state: resume ? 'running' : 'paused', message: resume ? 'Resume requested' : 'Pause requested' } : job)); } catch (error) { console.error('Failed to update background job:', error); }
  };
  const handleRetryBackgroundJob = async (jobId: string) => {
    try {
      await invoke(Invokes.RetryBackgroundJob, { jobId });
      const jobs = await invoke<BackgroundJob[]>(Invokes.ListBackgroundJobs);
      setBackgroundJobs(jobs);
    } catch (error) {
      console.error('Failed to retry background job:', error);
    }
  };
  const handleRetryAllEligibleJobs = async () => {
    const retryableKinds = new Set(['catalog_scan', 'cull_analysis', 'model_download', 'ram_plus_tagging', 'ai_tagging', 'face_detection', 'face_recognition', 'raw_denoise', 'rgb_denoise', 'thumbnail_generation', 'metadata_extraction', 'sidecar_metadata']);
    const eligible = backgroundJobs.filter((job) => retryableKinds.has(job.kind) && ['failed', 'cancelled'].includes(job.state));
    const retriedKinds = new Set<string>();
    if (eligible.length === 0) return;
    try {
      for (const job of eligible.filter((job) => {
        if (retriedKinds.has(job.kind)) return false;
        retriedKinds.add(job.kind);
        return true;
      })) {
        await invoke(Invokes.RetryBackgroundJob, { jobId: job.id });
      }
      setBackgroundJobs(await invoke<BackgroundJob[]>(Invokes.ListBackgroundJobs));
    } catch (error) {
      setBackgroundJobsError(`Could not retry all eligible jobs: ${String(error)}`);
    }
  };

  useEffect(() => {
    let active = true;
    const loadJobs = async () => {
      try {
        const jobs = await invoke<BackgroundJob[]>(Invokes.ListBackgroundJobs);
        if (active) {
          setBackgroundJobs(jobs);
          setBackgroundJobsError(null);
        }
      } catch (error) {
        if (active && isCatalogScanModalOpen) setBackgroundJobsError(String(error));
      }
    };
    void loadJobs();
    const timer = window.setInterval(() => void loadJobs(), isCatalogScanModalOpen ? 1500 : 3000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [isCatalogScanModalOpen]);

  useEffect(() => {
    if (isZoomReady && !isDraggingSlider.current) {
      setLatchedSliderValue(currentOriginalPercent);
      setLatchedDisplayPercent(Math.round(currentOriginalPercent * 100));
    }
  }, [currentOriginalPercent, isZoomReady]);

  useEffect(() => {
    const handleDragEndGlobal = () => {
      if (isZoomActive) {
        setIsZoomActive(false);
        isDraggingSlider.current = false;
        if (isZoomReady) {
          setLatchedDisplayPercent(Math.round(currentOriginalPercent * 100));
        }
      }
    };

    if (isZoomActive) {
      window.addEventListener('mouseup', handleDragEndGlobal);
      window.addEventListener('touchend', handleDragEndGlobal);
    }

    return () => {
      window.removeEventListener('mouseup', handleDragEndGlobal);
      window.removeEventListener('touchend', handleDragEndGlobal);
    };
  }, [isZoomActive, isZoomReady, currentOriginalPercent]);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newZoom = parseFloat(e.target.value);
    setLatchedSliderValue(newZoom);
    setLatchedDisplayPercent(Math.round(newZoom * 100));
    onZoomChange(newZoom);
  };

  const handleMouseDown = () => {
    isDraggingSlider.current = true;
    setIsZoomActive(true);
  };

  const handleMouseUp = () => {
    isDraggingSlider.current = false;
    setIsZoomActive(false);
    if (isZoomReady) {
      setLatchedDisplayPercent(Math.round(currentOriginalPercent * 100));
    }
  };

  const handleZoomKeyDown = (e: React.KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && ['z', 'y'].includes(e.key.toLowerCase())) {
      (e.target as HTMLElement).blur();
      return;
    }
    if (GLOBAL_KEYS.includes(e.key)) {
      (e.target as HTMLElement).blur();
    }
  };

  const handleResetZoom = () => {
    onZoomChange(0, true);
  };

  const handlePercentClick = () => {
    if (!isZoomReady) return;
    setIsEditingPercent(true);
    setPercentInputValue(latchedDisplayPercent.toString());
    setTimeout(() => {
      percentInputRef.current?.focus();
      percentInputRef.current?.select();
    }, 0);
  };

  const handlePercentSubmit = () => {
    const value = parseFloat(percentInputValue);
    if (!isNaN(value)) {
      const originalPercent = value / 100;
      const clampedPercent = Math.max(0.1, Math.min(2.0, originalPercent));
      onZoomChange(clampedPercent);
    }
    setIsEditingPercent(false);
    setPercentInputValue('');
  };

  const handlePercentKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handlePercentSubmit();
    else if (e.key === 'Escape') {
      setIsEditingPercent(false);
      setPercentInputValue('');
    }
    e.stopPropagation();
  };

  return (
    <div className="shrink-0 bg-bg-secondary rounded-lg flex flex-col">
      {!isLibraryView && showFilmstrip && (
        <div
          className={clsx(
            'overflow-hidden shrink-0 relative',
            shouldAnimate && 'transition-all duration-300 ease-in-out',
          )}
          style={{ height: `${effectiveHeight}px` }}
        >
          <div
            className={clsx(
              'w-full p-2 duration-300 ease-in-out',
              shouldAnimate ? 'transition-all' : 'transition-opacity',
              isCollapsed ? 'opacity-0 pointer-events-none' : 'opacity-100 pointer-events-auto',
            )}
            style={{ height: `${currentHeight}px` }}
          >
            <Filmstrip
              imageList={imageList}
              imageRatings={imageRatings}
              isLoading={isLoading}
              multiSelectedPaths={multiSelectedPaths}
              onClearSelection={onClearSelection}
              onContextMenu={onContextMenu}
              onEmptyAreaContextMenu={onEmptyAreaContextMenu}
              onImageSelect={onImageSelect}
              onRequestThumbnails={onRequestThumbnails}
              selectedImage={selectedImage}
              thumbnailAspectRatio={thumbnailAspectRatio}
            />
          </div>
        </div>
      )}

      <div
        className={clsx(
          'shrink-0 h-12 flex items-center justify-between px-3',
          !isLibraryView && 'border-t transition-colors duration-300',
          !isLibraryView && showFilmstrip && isFilmstripVisible ? 'border-surface' : 'border-transparent',
        )}
      >
        <div className="flex items-center gap-4">
          <StarRating rating={rating} onRate={onRate} disabled={isRatingDisabled} />
          <div className="h-5 w-px bg-surface"></div>
          <div className="flex items-center gap-2">
            <button
              className="relative w-8 h-8 flex items-center justify-center rounded-md text-text-secondary hover:bg-surface hover:text-text-primary transition-colors disabled:opacity-40 disabled:hover:bg-transparent disabled:cursor-not-allowed"
              disabled={isCopyDisabled}
              onClick={onCopy}
              data-tooltip={t('ui.bottomBar.tooltips.copySettings')}
            >
              <AnimatePresence mode="wait" initial={false}>
                {isCopied ? (
                  <motion.div
                    key="copied"
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.5 }}
                    transition={{ duration: 0.15 }}
                    className="absolute"
                  >
                    <Check size={18} className="text-green-500" />
                  </motion.div>
                ) : (
                  <motion.div
                    key="copy"
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.5 }}
                    transition={{ duration: 0.15 }}
                    className="absolute"
                  >
                    <Copy size={18} />
                  </motion.div>
                )}
              </AnimatePresence>
            </button>

            <button
              className="relative w-8 h-8 flex items-center justify-center rounded-md text-text-secondary hover:bg-surface hover:text-text-primary transition-colors disabled:opacity-40 disabled:hover:bg-transparent disabled:cursor-not-allowed"
              disabled={isPasteDisabled}
              onClick={onPaste}
              data-tooltip={t('ui.bottomBar.tooltips.pasteSettings')}
            >
              <AnimatePresence mode="wait" initial={false}>
                {isPasted ? (
                  <motion.div
                    key="pasted"
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.5 }}
                    transition={{ duration: 0.15 }}
                    className="absolute"
                  >
                    <Check size={18} className="text-green-500" />
                  </motion.div>
                ) : (
                  <motion.div
                    key="paste"
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.5 }}
                    transition={{ duration: 0.15 }}
                    className="absolute"
                  >
                    <ClipboardPaste size={18} />
                  </motion.div>
                )}
              </AnimatePresence>
            </button>

            <button
              className="w-8 h-8 flex items-center justify-center rounded-md text-text-secondary hover:bg-surface hover:text-text-primary transition-colors"
              onClick={onOpenCopyPasteSettings}
              data-tooltip={t('ui.bottomBar.tooltips.copyPasteSettings')}
            >
              <Settings size={18} />
            </button>
          </div>

          <div className="h-5 w-px bg-surface"></div>

          <div
            className={clsx(
              'flex items-center transition-all duration-300',
              isFilterExpanded ? 'bg-surface rounded-md' : 'bg-transparent',
            )}
          >
            <button
              className={clsx(
                'relative w-8 h-8 flex items-center justify-center rounded-md transition-colors shrink-0',
                isFilterExpanded ? 'text-text-primary' : 'text-text-secondary hover:bg-surface hover:text-text-primary',
              )}
              onClick={() => setIsFilterExpanded(!isFilterExpanded)}
              data-tooltip={t('ui.bottomBar.tooltips.quickFilter', 'Quick Filter')}
            >
              <Filter size={18} />
            </button>

            <div
              className={clsx(
                'flex items-center transition-all duration-300 ease-in-out overflow-hidden',
                isFilterExpanded ? 'max-w-100 opacity-100 pr-2 ml-1' : 'max-w-0 opacity-0 pr-0 ml-0',
              )}
            >
              <div className="flex items-center gap-3 whitespace-nowrap">
                <div className="flex items-center gap-0.5">
                  {[1, 2, 3, 4, 5].map((starValue) => {
                    const isFilled = filterCriteria.rating > 0 && starValue <= filterCriteria.rating;
                    return (
                      <button
                        key={`qf-star-${starValue}`}
                        onClick={() =>
                          setFilterCriteria((prev) => ({
                            ...prev,
                            rating: prev.rating === starValue ? 0 : starValue,
                          }))
                        }
                        className="p-0.5 focus:outline-none"
                      >
                        <Star
                          size={16}
                          className={clsx(
                            'transition-colors duration-150',
                            isFilled ? 'text-accent fill-accent' : 'text-text-secondary hover:text-accent',
                          )}
                        />
                      </button>
                    );
                  })}
                </div>

                <div className="h-4 w-px bg-border-color"></div>

                <div className="flex items-center gap-1.5">
                  {allColors.map((color) => {
                    const isSelected = (filterCriteria.colors || []).includes(color.name);

                    const tooltipTitle =
                      color.name === 'none'
                        ? t('library.header.viewOptions.noLabel')
                        : t(`contextMenus.colors.${color.name}`, {
                            defaultValue: color.name.charAt(0).toUpperCase() + color.name.slice(1),
                          });

                    return (
                      <button
                        key={`qf-color-${color.name}`}
                        onClick={() => {
                          const currentColors = filterCriteria.colors || [];
                          const newColors = currentColors.includes(color.name)
                            ? currentColors.filter((c) => c !== color.name)
                            : [...currentColors, color.name];
                          setFilterCriteria((prev) => ({ ...prev, colors: newColors }));
                        }}
                        className={clsx(
                          'w-4 h-4 rounded-full transition-transform hover:scale-105 flex items-center justify-center focus:outline-none',
                          isSelected ? 'ring-2 ring-accent ring-offset-1 ring-offset-bg-primary' : '',
                        )}
                        style={{ backgroundColor: color.color }}
                        data-tooltip={tooltipTitle}
                      >
                        {isSelected && <Check size={10} className="text-white drop-shadow-md" />}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          {(catalogScan.isActive || catalogScan.error || isLibraryView || activeBackgroundJobs.length > 0) && (
            <>
              <div className="h-5 w-px bg-surface"></div>
              <button
                className="flex items-center gap-2 rounded-md px-2 py-1 text-text-secondary hover:bg-surface hover:text-text-primary transition-colors max-w-[440px]"
                onClick={() => setIsCatalogScanModalOpen(true)}
                data-tooltip="Indexing details"
              >
                {catalogScanThumbnail && catalogScan.isActive ? (
                  <img
                    src={catalogScanThumbnail}
                    className="h-6 w-6 rounded object-cover border border-border-color shrink-0"
                    alt=""
                  />
                ) : catalogScan.isActive || activeBackgroundJobs.length > 0 ? (
                  <Loader2 size={16} className="animate-spin text-accent shrink-0" />
                ) : (
                  <Database size={16} className="text-red-500 shrink-0" />
                )}
                <Text as="span" variant={TextVariants.small} className="truncate">
                  {catalogScan.error
                    ? 'Indexing failed'
                    : catalogScan.isPaused
                      ? `Indexing paused ${catalogScan.current}/${catalogScan.total || '?'}`
                      : catalogScan.isActive && catalogScan.total > 0
                        ? `Indexing collection ${catalogScan.current}/${catalogScan.total}${catalogScanFileName ? ` · ${catalogScanFileName}` : ''}`
                        : activeBackgroundJobs.length > 0
                          ? activeBackgroundJobs.length === 1
                            ? `${activeBackgroundJob?.message || 'Background job running'}${
                                activeBackgroundJob && activeBackgroundJob.total > 0
                                  ? ` ${activeBackgroundJob.current}/${activeBackgroundJob.total}`
                                  : ''
                              }`
                            : `${activeBackgroundJobs.length} active background jobs`
                          : 'Background jobs'}
                </Text>
                {catalogScanPercent !== null && (
                  <span className="text-[11px] text-text-secondary tabular-nums shrink-0">{catalogScanPercent}%</span>
                )}
              </button>
            </>
          )}

          <div
            className={clsx(
              'flex items-center transition-all duration-300 ease-out overflow-hidden',
              showSelectionCounter ? 'max-w-xs opacity-100' : 'max-w-0 opacity-0',
            )}
          >
            <div className="h-5 w-px bg-surface mr-4"></div>
            <Text as="span" className="whitespace-nowrap">
              {t('ui.bottomBar.imagesSelected', { current: numSelected, total })}
            </Text>
          </div>
        </div>

        <div className="grow" />

        <div className="flex items-center gap-4">
          {!isLibraryView && showZoomControls && (
            <>
              <div className="flex items-center gap-2 w-56">
                <div
                  className="relative w-12 h-full flex items-center justify-end cursor-pointer"
                  onClick={handleResetZoom}
                  onMouseEnter={() => setIsZoomLabelHovered(true)}
                  onMouseLeave={() => setIsZoomLabelHovered(false)}
                  data-tooltip={t('ui.bottomBar.tooltips.resetZoom')}
                >
                  <span className="absolute right-0 text-xs text-text-secondary select-none text-right w-max transition-colors hover:text-text-primary">
                    {isZoomLabelHovered ? t('ui.bottomBar.zoomLabelReset') : t('ui.bottomBar.zoomLabel')}
                  </span>
                </div>

                <div className="relative flex-1 h-5">
                  <div className="absolute top-1/2 left-0 w-full h-1.5 -translate-y-1/2 bg-surface rounded-full pointer-events-none" />
                  <input
                    type="range"
                    min={0.1}
                    max={2.0}
                    step="0.05"
                    value={latchedSliderValue}
                    onChange={handleSliderChange}
                    onKeyDown={handleZoomKeyDown}
                    onMouseDown={handleMouseDown}
                    onMouseUp={handleMouseUp}
                    onTouchStart={handleMouseDown}
                    onTouchEnd={handleMouseUp}
                    onDoubleClick={handleResetZoom}
                    className={`absolute top-1/2 left-0 w-full h-1.5 mt-[-1.5px] appearance-none bg-transparent cursor-pointer p-0 slider-input z-10 ${
                      isZoomActive ? 'slider-thumb-active' : ''
                    }`}
                  />
                </div>

                <div className="relative text-xs text-text-secondary w-6 text-right flex items-center justify-end h-5 gap-1">
                  {isEditingPercent ? (
                    <input
                      ref={percentInputRef}
                      type="text"
                      value={percentInputValue}
                      onChange={(e) => setPercentInputValue(e.target.value)}
                      onKeyDown={handlePercentKeyDown}
                      onBlur={handlePercentSubmit}
                      className="w-full text-xs text-text-primary bg-bg-primary border border-border-color rounded-sm px-1 text-right"
                      style={{ fontSize: '12px', height: '18px' }}
                    />
                  ) : (
                    <span
                      onClick={handlePercentClick}
                      className="cursor-pointer hover:text-text-primary transition-colors select-none"
                      data-tooltip={t('ui.bottomBar.tooltips.customZoom')}
                    >
                      {latchedDisplayPercent}%
                    </span>
                  )}
                </div>
              </div>

              <div className="h-5 w-px bg-surface"></div>
            </>
          )}

          <div className="flex items-center gap-1">
            {!isAndroid && (
              <>
                <PanelToggleButton
                  onClick={toggleLeft}
                  Icon={PanelLeft}
                  tooltip={isLeftOpen ? t('ui.bottomBar.tooltips.collapseLeft') : t('ui.bottomBar.tooltips.expandLeft')}
                />

                {showFilmstrip && (
                  <PanelToggleButton
                    onClick={toggleBottom}
                    Icon={PanelBottom}
                    tooltip={
                      isBottomOpen
                        ? t('ui.bottomBar.tooltips.collapseFilmstrip')
                        : t('ui.bottomBar.tooltips.expandFilmstrip')
                    }
                    disabled={isLibraryView}
                  />
                )}

                <PanelToggleButton
                  onClick={toggleRight}
                  Icon={PanelRight}
                  tooltip={
                    isRightOpen ? t('ui.bottomBar.tooltips.collapseRight') : t('ui.bottomBar.tooltips.expandRight')
                  }
                />
              </>
            )}
          </div>
        </div>
      </div>

      <AnimatePresence>
        {isCatalogScanModalOpen && (
          <motion.div
            className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/50 p-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsCatalogScanModalOpen(false)}
          >
            <motion.div
              className="w-full max-w-2xl rounded-lg border border-border-color bg-bg-secondary shadow-2xl p-5"
              initial={{ scale: 0.96, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.96, opacity: 0 }}
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-4 mb-4">
                <div>
                  <Text variant={TextVariants.heading}>Background Jobs</Text>
                  <Text variant={TextVariants.small} color={TextColors.secondary}>
                    {catalogScan.rootPath || 'No active collection'}
                  </Text>
                </div>
                <button
                  className="p-2 rounded-md text-text-secondary hover:text-text-primary hover:bg-surface"
                  onClick={() => setIsCatalogScanModalOpen(false)}
                >
                  <X size={18} />
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <Text variant={TextVariants.small}>
                      {catalogScan.isPaused
                        ? 'Indexing paused'
                        : catalogScan.isActive
                          ? catalogScan.message || 'Indexing collection'
                          : activeBackgroundJob
                            ? activeBackgroundJob.message
                            : catalogScan.message || 'No active background job'}
                    </Text>
                    <Text variant={TextVariants.small} color={TextColors.secondary}>
                      {catalogScan.isActive && catalogScan.total > 0
                        ? `${catalogScan.current}/${catalogScan.total}`
                        : activeBackgroundJob && activeBackgroundJob.total > 0
                          ? `${activeBackgroundJob.current}/${activeBackgroundJob.total}`
                          : catalogScan.isActive
                            ? 'Preparing'
                            : ''}
                    </Text>
                  </div>
                  <div className="h-2 rounded-full bg-surface overflow-hidden">
                    <div
                      className="h-full bg-accent transition-all"
                      style={{
                        width:
                          catalogScan.isActive && catalogScan.total > 0
                            ? `${Math.min(100, (catalogScan.current / catalogScan.total) * 100)}%`
                            : activeBackgroundJob && activeBackgroundJob.total > 0
                              ? `${Math.min(100, (activeBackgroundJob.current / activeBackgroundJob.total) * 100)}%`
                              : catalogScan.isActive
                                ? '15%'
                                : '0%',
                      }}
                    />
                  </div>
                </div>

                {catalogScan.isActive && (
                  <div className="flex flex-wrap gap-2">
                    {catalogScan.isPaused ? (
                      <button
                        className="inline-flex items-center gap-2 rounded-md bg-accent px-3 py-2 text-sm text-white hover:bg-accent-hover"
                        onClick={handleResumeCatalogScan}
                      >
                        <Play size={16} />
                        Resume
                      </button>
                    ) : (
                      <button
                        className="inline-flex items-center gap-2 rounded-md border border-border-color bg-bg-primary px-3 py-2 text-sm text-text-primary hover:bg-surface"
                        onClick={handlePauseCatalogScan}
                      >
                        <Pause size={16} />
                        Pause
                      </button>
                    )}
                    <button
                      className="inline-flex items-center gap-2 rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm text-red-300 hover:bg-red-500/20"
                      onClick={handleCancelCatalogScan}
                    >
                      <Square size={16} />
                      Cancel
                    </button>
                  </div>
                )}

                {catalogScan.error && (
                  <div className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 select-text">
                    <Text variant={TextVariants.small} className="text-red-300 select-text break-words whitespace-pre-wrap font-mono text-xs">{catalogScan.error}</Text>
                  </div>
                )}

                <div className="rounded-md border border-border-color bg-bg-primary p-3">
                  <div className="grid grid-cols-[96px_minmax(0,1fr)] gap-4">
                    <div className="h-24 w-24 rounded-md border border-border-color bg-surface overflow-hidden flex items-center justify-center">
                      {currentThumbnailSrc && !thumbError ? (
                        <img
                          src={currentThumbnailSrc}
                          className="h-full w-full object-cover"
                          alt=""
                          onError={() => setThumbError(true)}
                        />
                      ) : (catalogScan.isActive || activeBackgroundJobs.length > 0) ? (
                        <Loader2 size={22} className="animate-spin text-accent" />
                      ) : (
                        <Database size={22} className="text-text-secondary" />
                      )}
                    </div>
                    <div className="min-w-0 space-y-3">
                      <div>
                        <Text variant={TextVariants.label}>Current Image</Text>
                        <Text variant={TextVariants.small} className="break-all select-text font-mono text-xs">
                          {currentDisplayedPath || 'Waiting for first image...'}
                        </Text>
                      </div>
                      {catalogScan.isActive ? (
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                          <div>
                            <Text as="div" variant={TextVariants.small} color={TextColors.secondary}>
                              Camera
                            </Text>
                            <Text variant={TextVariants.small}>{catalogScan.camera || '-'}</Text>
                          </div>
                          <div>
                            <Text as="div" variant={TextVariants.small} color={TextColors.secondary}>
                              Lens
                            </Text>
                            <Text variant={TextVariants.small}>{catalogScan.lens || '-'}</Text>
                          </div>
                          <div>
                            <Text as="div" variant={TextVariants.small} color={TextColors.secondary}>
                              Year
                            </Text>
                            <Text variant={TextVariants.small}>{catalogScan.year || '-'}</Text>
                          </div>
                        </div>
                      ) : activeBackgroundJob ? (
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                          <div>
                            <Text as="div" variant={TextVariants.small} color={TextColors.secondary}>
                              Job Type
                            </Text>
                            <Text variant={TextVariants.small}>
                              {({
                                catalog_scan: 'Catalog scan',
                                cull_analysis: 'Culling analysis',
                                model_download: 'Model download',
                                ram_plus_tagging: 'RAM++ tagging',
                                ai_tagging: 'AI tagging',
                                face_detection: 'Face detection',
                                face_recognition: 'Face recognition',
                                raw_denoise: 'RAW AI denoise',
                                rgb_denoise: 'RGB AI denoise',
                                thumbnail_generation: 'Thumbnail generation',
                                metadata_extraction: 'Metadata extraction',
                                sidecar_metadata: 'Metadata extraction',
                                deblur: 'AI deblur',
                                upscale: 'AI upscale',
                              } as Record<string, string>)[activeBackgroundJob.kind] || activeBackgroundJob.kind}
                            </Text>
                          </div>
                          <div>
                            <Text as="div" variant={TextVariants.small} color={TextColors.secondary}>
                              Progress
                            </Text>
                            <Text variant={TextVariants.small}>
                              {activeBackgroundJob.total > 0
                                ? `${activeBackgroundJob.current}/${activeBackgroundJob.total}`
                                : 'Working'}
                            </Text>
                          </div>
                          <div>
                            <Text as="div" variant={TextVariants.small} color={TextColors.secondary}>
                              Status
                            </Text>
                            <Text
                              variant={TextVariants.small}
                              color={
                                activeBackgroundJob.state === 'failed'
                                  ? TextColors.error
                                  : activeBackgroundJob.state === 'completed'
                                    ? TextColors.success
                                    : TextColors.accent
                              }
                            >
                              {activeBackgroundJob.state}
                            </Text>
                          </div>
                        </div>
                      ) : (
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                          <div>
                            <Text as="div" variant={TextVariants.small} color={TextColors.secondary}>
                              Camera
                            </Text>
                            <Text variant={TextVariants.small}>{catalogScan.camera || '-'}</Text>
                          </div>
                          <div>
                            <Text as="div" variant={TextVariants.small} color={TextColors.secondary}>
                              Lens
                            </Text>
                            <Text variant={TextVariants.small}>{catalogScan.lens || '-'}</Text>
                          </div>
                          <div>
                            <Text as="div" variant={TextVariants.small} color={TextColors.secondary}>
                              Year
                            </Text>
                            <Text variant={TextVariants.small}>{catalogScan.year || '-'}</Text>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Per-File Output Section */}
                <div className="rounded-md border border-border-color bg-bg-primary p-3">
                  <div className="flex items-center justify-between mb-2">
                    <Text variant={TextVariants.label}>File Processing Output</Text>
                    {activeBackgroundJob && (
                      <span className="text-[11px] px-2 py-0.5 rounded bg-surface border border-border-color text-text-secondary font-medium">
                        {activeBackgroundJob.kind.replace(/_/g, ' ')}
                      </span>
                    )}
                  </div>
                  <div className="rounded bg-surface/60 border border-border-color/60 p-2.5 select-text">
                    <Text
                      variant={TextVariants.small}
                      className="select-text break-words font-mono text-xs"
                      color={
                        activeBackgroundJob?.error || catalogScan.error
                          ? TextColors.error
                          : TextColors.primary
                      }
                    >
                      {activeBackgroundJob?.message ||
                        (catalogScan.isActive ? catalogScan.message : null) ||
                        (backgroundJobs[0]?.message) ||
                        'No background job output yet.'}
                    </Text>
                  </div>
                </div>

                <div className="rounded-md border border-border-color bg-bg-primary overflow-hidden">
                  <div className="px-3 py-2 border-b border-border-color flex justify-between items-center">
                    <Text variant={TextVariants.label}>Recent Jobs</Text>
                    <div className="flex items-center gap-2">
                      <Text variant={TextVariants.small} color={TextColors.secondary}>{backgroundJobs.length}</Text>
                      {backgroundJobs.some((job) => ['catalog_scan', 'cull_analysis', 'model_download', 'ram_plus_tagging', 'ai_tagging', 'face_detection', 'face_recognition', 'raw_denoise', 'rgb_denoise', 'thumbnail_generation', 'metadata_extraction', 'sidecar_metadata'].includes(job.kind) && ['failed', 'cancelled'].includes(job.state)) && (
                        <button className="p-1 text-accent hover:bg-surface rounded" onClick={() => void handleRetryAllEligibleJobs()} data-tooltip="Retry all eligible jobs">
                          <RotateCcw size={14} />
                        </button>
                      )}
                    </div>
                  </div>
                  {backgroundJobsError ? (
                    <Text variant={TextVariants.small} className="p-3 text-red-300">{backgroundJobsError}</Text>
                  ) : backgroundJobs.length === 0 ? (
                    <Text variant={TextVariants.small} className="p-3">No catalog jobs have run in this library.</Text>
                  ) : (
                    <div className="max-h-48 overflow-y-auto">
                      {backgroundJobs.map((job) => (
                        <div key={job.id} className="px-3 py-2 border-b border-border-color last:border-b-0">
                          <div className="flex gap-3 justify-between items-center">
                            <Text variant={TextVariants.small}>{({ catalog_scan: 'Catalog scan', cull_analysis: 'Culling analysis', model_download: 'Model download', ram_plus_tagging: 'RAM++ tagging', ai_tagging: 'AI tagging', face_detection: 'Face detection', face_recognition: 'Face recognition', raw_denoise: 'RAW AI denoise', rgb_denoise: 'RGB AI denoise', thumbnail_generation: 'Thumbnail generation', metadata_extraction: 'Metadata extraction', sidecar_metadata: 'Metadata extraction', deblur: 'AI deblur', upscale: 'AI upscale' } as Record<string, string>)[job.kind] || job.kind}</Text>
                            <div className="flex items-center gap-2">
                              <Text variant={TextVariants.small} color={job.state === 'failed' ? TextColors.error : job.state === 'completed' ? TextColors.success : TextColors.accent}>{job.state}</Text>
                              {job.kind !== 'model_download' && ['running', 'paused'].includes(job.state) && <button className="p-1 text-text-secondary hover:bg-bg-primary rounded" onClick={() => void handlePauseBackgroundJob(job.id, job.state === 'paused')} data-tooltip={job.state === 'paused' ? 'Resume job' : 'Pause job'}>{job.state === 'paused' ? <Play size={13} /> : <Pause size={13} />}</button>}
                              {['queued', 'running', 'paused'].includes(job.state) && (
                                <button className="p-1 text-red-300 hover:bg-red-500/10 rounded" onClick={() => void handleCancelBackgroundJob(job.id)} data-tooltip="Cancel job">
                                  <Square size={13} />
                                </button>
                              )}
                              {['catalog_scan', 'cull_analysis', 'model_download', 'ram_plus_tagging', 'ai_tagging', 'face_detection', 'face_recognition', 'raw_denoise', 'rgb_denoise', 'thumbnail_generation', 'metadata_extraction', 'sidecar_metadata'].includes(job.kind) && ['failed', 'cancelled'].includes(job.state) && (
                                <button className="p-1 text-accent hover:bg-bg-primary rounded" onClick={() => void handleRetryBackgroundJob(job.id)} data-tooltip="Retry job">
                                  <RotateCcw size={13} />
                                </button>
                              )}
                            </div>
                          </div>
                          <Text variant={TextVariants.small} color={TextColors.secondary} className="select-text break-words">{job.message}</Text>
                          {job.currentItem && <Text variant={TextVariants.small} color={TextColors.secondary} className="select-text break-all font-mono text-xs">{job.currentItem.split(/[\\/]/).pop()}</Text>}
                          {job.total > 0 && <Text variant={TextVariants.small} color={TextColors.secondary}>{job.current}/{job.total}</Text>}
                          {job.error && (
                            <div className="mt-1.5 rounded bg-red-500/10 border border-red-500/30 p-2 select-text">
                              <Text variant={TextVariants.small} className="text-red-300 select-text break-words whitespace-pre-wrap font-mono text-xs">
                                {job.error}
                              </Text>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
