import React, { useEffect, useState, useRef, useMemo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { convertFileSrc, invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { List, useListCallbackRef } from 'react-window';
import {
  Loader2,
  Star as StarIcon,
  ZoomIn,
  ZoomOut,
  Maximize,
  Link,
  SquarePen,
  Tag,
  X,
  Check,
  Plus,
  SlidersHorizontal,
  Info,
  History,
  Filter,
  FolderOpen,
  Play,
  Database,
  ChevronDown,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import { AutoCullPlan, AutoCullResult, CatalogRoot, CullingSettings, CullSessionDecision, CullSessionSummary, Invokes, ImageFile } from '../../ui/AppProperties';
import { Thumbnail } from './LibraryItems';
import Text from '../../ui/Text';
import Switch from '../../ui/Switch';
import { TextColors, TextVariants, TextWeights } from '../../../types/typography';
import { useProcessStore } from '../../../store/useProcessStore';
import { useLibraryStore } from '../../../store/useLibraryStore';
import { useUIStore } from '../../../store/useUIStore';
import { useSettingsStore } from '../../../store/useSettingsStore';
import { useLibraryActions } from '../../../hooks/useLibraryActions';
import { COLOR_LABELS, Color } from '../../../utils/adjustments';
import { expandGroupedPaths } from '../../../utils/imageGrouping';
import { IconAperture, IconFocalLength, IconIso, IconShutter } from '../editor/ExifIcons';

interface SyncViewport {
  isActive: boolean;
  zoom: number;
  pan: { x: number; y: number };
  isDragging: boolean;
}

interface CullFaceReviewItem {
  face: { id: number; confidence: number };
  cropPath?: string | null;
  thumbnailDataUrl?: string | null;
}

interface CullSubjectEvidence {
  aiTags: Array<{ name: string; confidence: number; reviewState: string }>;
  species: Array<{ commonName?: string | null; scientificName: string; confidence: number; reviewState: string }>;
}

const DEFAULT_CULL_SETTINGS: CullingSettings = {
  similarityThreshold: 28,
  blurThreshold: 100,
  groupSimilar: true,
  filterBlurry: true,
  useSubjectDetection: true,
  subjectMode: 'general',
};

const CULL_FEEDBACK_REASONS = [
  'Stronger moment or expression',
  'Better focus or detail',
  'Better composition',
  'Better subject visibility',
  'Prefer a different duplicate',
  'Keep despite technical issue',
  'Reject despite AI selection',
] as const;

function CullHistoryPanel({ onClose }: { onClose(): void }) {
  const [sessions, setSessions] = useState<CullSessionSummary[]>([]);
  const [decisions, setDecisions] = useState<CullSessionDecision[]>([]);
  const [selectedSession, setSelectedSession] = useState<CullSessionSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    invoke<CullSessionSummary[]>(Invokes.ListCullSessions)
      .then((items) => { if (active) setSessions(items); })
      .catch((reason) => { if (active) setError(String(reason)); })
      .finally(() => { if (active) setIsLoading(false); });
    return () => { active = false; };
  }, []);

  const selectSession = async (session: CullSessionSummary) => {
    setSelectedSession(session);
    setDecisions([]);
    try { setDecisions(await invoke<CullSessionDecision[]>(Invokes.ListCullSessionDecisions, { sessionId: session.id })); }
    catch (reason) { setError(String(reason)); }
  };

  return <div className="absolute z-50 top-12 right-3 w-80 max-h-[calc(100%-4rem)] overflow-hidden rounded-md border border-border-color bg-bg-secondary shadow-xl flex flex-col"><div className="flex items-center justify-between px-3 py-2 border-b border-border-color"><Text variant={TextVariants.small} weight={TextWeights.semibold}>{selectedSession ? 'Culling Decisions' : 'Culling History'}</Text><button className="p-1 text-text-secondary hover:text-text-primary" onClick={onClose} data-tooltip="Close culling history"><X size={16} /></button></div>{selectedSession ? <><button className="px-3 py-2 text-left text-xs text-accent hover:bg-surface" onClick={() => { setSelectedSession(null); setDecisions([]); }}>All sessions</button><div className="px-3 pb-2 text-xs text-text-secondary truncate">{selectedSession.scopePath}</div><div className="overflow-y-auto divide-y divide-border-color/60">{decisions.map((decision) => <div key={decision.id} className="px-3 py-2"><div className="flex items-center gap-2"><span className={decision.finalStatus === 'reject' || decision.proposedStatus === 'reject' ? 'text-red-300 text-xs' : 'text-green-300 text-xs'}>{decision.finalStatus === 'pending' ? decision.proposedStatus : decision.finalStatus}</span><span className="min-w-0 flex-1 truncate text-xs text-text-primary">{decision.representativePath.split(/[\\/]/).pop()}</span><span className="text-xs tabular-nums text-text-secondary">{Math.round(decision.qualityScore * 100)}</span></div><Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mt-1 truncate">{decision.reason}</Text></div>)}{decisions.length === 0 && <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="p-3">No decisions recorded.</Text>}</div></> : <div className="overflow-y-auto">{isLoading ? <div className="flex items-center gap-2 p-3 text-text-secondary"><Loader2 size={15} className="animate-spin" /><Text variant={TextVariants.small}>Loading sessions</Text></div> : error ? <Text as="div" variant={TextVariants.small} className="p-3 text-red-300">{error}</Text> : sessions.length === 0 ? <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="p-3">No catalog culling sessions yet.</Text> : sessions.map((session) => <button key={session.id} className="w-full px-3 py-2 text-left hover:bg-surface border-b border-border-color/60 last:border-b-0" onClick={() => void selectSession(session)}><div className="flex gap-2"><span className="min-w-0 flex-1 truncate text-sm text-text-primary">{session.scopePath}</span><span className={session.state === 'applied' ? 'text-green-300 text-xs' : 'text-amber-300 text-xs'}>{session.state}</span></div><Text as="div" variant={TextVariants.small} color={TextColors.secondary}>{session.rejectedCount} rejected of {session.totalCount}</Text></button>)}</div>}</div>;
}

function CullingPreview({
  image,
  rating,
  isActive,
  isSelected,
  isFullWidth,
  syncViewport,
  setSyncViewport,
  onContextMenu,
  onImageDoubleClick,
  hoveredPath,
  setHoveredCullingPath,
  showRateBar,
  setShowRateBar,
  showInfoBar,
  setShowInfoBar,
}: {
  image: ImageFile;
  rating: number;
  isActive: boolean;
  isSelected: boolean;
  isFullWidth?: boolean;
  syncViewport: SyncViewport;
  setSyncViewport: React.Dispatch<React.SetStateAction<SyncViewport>>;
  onContextMenu: (e: React.MouseEvent, path: string, forceSingleSelection?: boolean) => void;
  onImageDoubleClick: (path: string) => void;
  hoveredPath: string | null;
  setHoveredCullingPath: (path: string | null) => void;
  showRateBar: boolean;
  setShowRateBar: React.Dispatch<React.SetStateAction<boolean>>;
  showInfoBar: boolean;
  setShowInfoBar: React.Dispatch<React.SetStateAction<boolean>>;
}) {
  const { t } = useTranslation();
  const thumbUrl = useProcessStore((s) => s.thumbnails[image.path]);
  const initialPreview = useProcessStore((s) => s.previews[image.path]);
  const setPreview = useProcessStore((s) => s.setPreview);
  const safeThumbKey = thumbUrl || '';
  const [highResSrc, setHighResSrc] = useState<string | null>(
    initialPreview?.thumbKey === safeThumbKey ? initialPreview.url : null,
  );
  const [isLoading, setIsLoading] = useState(!highResSrc);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const dragStartMouse = useRef({ x: 0, y: 0 });
  const dragStartPan = useRef({ x: 0, y: 0 });
  const hasDragged = useRef(false);
  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  const [tagInputValue, setTagInputValue] = useState('');
  const [fitScale, setFitScale] = useState<number | null>(null);
  const { handleRate, handleSetColorLabel, handleTagsChanged } = useLibraryActions();
  const USER_TAG_PREFIX = 'user:';
  const getPathsToUpdate = () =>
    expandGroupedPaths(
      useLibraryStore.getState().imageList,
      [image.path],
      useSettingsStore.getState().appSettings?.grouping ?? 'off',
    );

  const currentColor = useMemo(() => {
    return image.tags?.find((t) => t.startsWith('color:'))?.substring(6) || null;
  }, [image.tags]);

  const colorLabel = useMemo(() => {
    return COLOR_LABELS.find((c: Color) => c.name === currentColor) || null;
  }, [currentColor]);

  const displayEditIcon = useSettingsStore((s) => s.appSettings?.displayEditIcon ?? true);
  const showEditIcon = image.is_edited && displayEditIcon;
  const hasAnyOverlay = showEditIcon || !!colorLabel || rating > 0;

  const currentTags = useMemo(() => {
    return (image.tags || [])
      .filter((t) => !t.startsWith('color:'))
      .map((t) => ({
        tag: t.startsWith(USER_TAG_PREFIX) ? t.substring(USER_TAG_PREFIX.length) : t,
        isUser: t.startsWith(USER_TAG_PREFIX),
      }))
      .sort((a, b) => a.tag.localeCompare(b.tag));
  }, [image.tags]);

  const { exifData, hasExif } = useMemo(() => {
    const exif = image.exif || {};

    let fNum = exif.FNumber;
    if (fNum) {
      const fStr = String(fNum);
      fNum = fStr.toLowerCase().startsWith('f') ? fStr : `f/${fStr}`;
    }

    let captureDate = null;
    let captureTime = null;

    if (exif.DateTimeOriginal) {
      const dateTimeParts = exif.DateTimeOriginal.split(' ');
      captureDate = dateTimeParts[0]?.replace(/:/g, '-') || null;
      if (dateTimeParts[1]) {
        const timeParts = dateTimeParts[1].split(':');
        captureTime = `${timeParts[0]}:${timeParts[1]}`;
      }
    }

    const data = {
      iso: exif.PhotographicSensitivity || exif.ISO,
      fNumber: fNum,
      shutter: exif.ExposureTime,
      focal: exif.FocalLengthIn35mmFilm,
      captureDate: captureDate,
      captureTime: captureTime,
    };

    const hasData = !!(data.iso || data.fNumber || data.shutter || data.focal || data.captureDate);

    return {
      exifData: data,
      hasExif: hasData,
    };
  }, [image.exif]);

  const imageWidth = (image as any).width || image.exif?.ExifImageWidth || image.exif?.PixelXDimension;
  const imageHeight = (image as any).height || image.exif?.ExifImageHeight || image.exif?.PixelYDimension;

  const handleAddTag = async (tagToAdd: string) => {
    const newTagValue = tagToAdd.trim().toLowerCase();
    if (newTagValue && !currentTags.some((t) => t.tag === newTagValue)) {
      try {
        const prefixedTag = `${USER_TAG_PREFIX}${newTagValue}`;
        const pathsToUpdate = getPathsToUpdate();
        await invoke(Invokes.AddTagForPaths, { paths: pathsToUpdate, tag: prefixedTag });
        const newTags = [...currentTags, { tag: newTagValue, isUser: true }];
        handleTagsChanged([image.path], newTags);
        setTagInputValue('');
      } catch (err) {
        console.error(`Failed to add tag: ${err}`);
      }
    }
  };

  const handleRemoveTag = async (tagToRemove: { tag: string; isUser: boolean }) => {
    try {
      const prefixedTag = tagToRemove.isUser ? `${USER_TAG_PREFIX}${tagToRemove.tag}` : tagToRemove.tag;
      const pathsToUpdate = getPathsToUpdate();
      await invoke(Invokes.RemoveTagForPaths, { paths: pathsToUpdate, tag: prefixedTag });
      const newTags = currentTags.filter((t) => t.tag !== tagToRemove.tag);
      handleTagsChanged([image.path], newTags);
    } catch (err) {
      console.error(`Failed to remove tag: ${err}`);
    }
  };

  const handleTagInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddTag(tagInputValue);
    }
    e.stopPropagation();
  };

  useEffect(() => {
    zoomRef.current = zoom;
    panRef.current = pan;
  }, [zoom, pan]);

  const fullFileName = image.path.split(/[\\/]/).pop() || '';
  const parts = fullFileName.split('?vc=');
  const baseName = parts[0];
  const isVirtualCopy = parts.length > 1;

  const updateFitScale = useCallback(() => {
    if (!containerRef.current || !imageRef.current) return;
    const { naturalWidth, naturalHeight } = imageRef.current;
    if (!naturalWidth || !naturalHeight) return;

    const { clientWidth, clientHeight } = containerRef.current;
    const scale = Math.min(clientWidth / naturalWidth, clientHeight / naturalHeight);
    setFitScale(scale);
  }, []);

  useEffect(() => {
    if (imageRef.current && imageRef.current.complete) {
      updateFitScale();
    }
  }, [highResSrc, updateFitScale]);

  useEffect(() => {
    const observer = new ResizeObserver(() => {
      updateFitScale();
    });
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }
    return () => observer.disconnect();
  }, [updateFitScale]);

  useEffect(() => {
    const currentPreview = useProcessStore.getState().previews[image.path];
    if (currentPreview && currentPreview.thumbKey === safeThumbKey) {
      setHighResSrc(currentPreview.url);
      setIsLoading(false);
      setPreview(image.path, currentPreview.url, safeThumbKey);
      return;
    }

    let active = true;
    setIsLoading(true);
    setHighResSrc(null);

    const fetchPreviewWithAdjustments = async () => {
      try {
        const metadata: any = await invoke(Invokes.LoadMetadata, { path: image.path });
        if (!active) return;

        const adjustments =
          metadata && metadata.adjustments && !metadata.adjustments.is_null ? metadata.adjustments : {};

        const bytes = await invoke<Uint8Array>(Invokes.GeneratePreviewForPath, {
          path: image.path,
          jsAdjustments: adjustments,
        });
        if (!active) return;

        const blob = new Blob([new Uint8Array(bytes)], { type: 'image/jpeg' });
        const localBlobUrl = URL.createObjectURL(blob);

        setPreview(image.path, localBlobUrl, safeThumbKey);

        if (active) {
          setHighResSrc(localBlobUrl);
          setIsLoading(false);
        }
      } catch (err) {
        console.error('Error loading culling preview with adjustments:', err);

        if (active) {
          try {
            const fallbackBytes = await invoke<Uint8Array>(Invokes.GeneratePreviewForPath, {
              path: image.path,
              jsAdjustments: {},
            });
            if (!active) return;
            const blob = new Blob([new Uint8Array(fallbackBytes)], { type: 'image/jpeg' });
            const localBlobUrl = URL.createObjectURL(blob);

            setPreview(image.path, localBlobUrl, safeThumbKey);
            setHighResSrc(localBlobUrl);
          } catch (fallbackErr) {
            console.error('Fallback preview generation also failed:', fallbackErr);
          }
          setIsLoading(false);
        }
      }
    };

    const delayTimeout = setTimeout(() => {
      fetchPreviewWithAdjustments();
    }, 200);

    return () => {
      active = false;
      clearTimeout(delayTimeout);
    };
  }, [image.path, safeThumbKey, setPreview]);

  useEffect(() => {
    if (syncViewport.isActive) {
      setZoom(syncViewport.zoom);
      setPan(syncViewport.pan);
    }
  }, [syncViewport.isActive, syncViewport.zoom, syncViewport.pan]);

  const updateViewport = (newZoom: number, newPan: { x: number; y: number }) => {
    setZoom(newZoom);
    setPan(newPan);
    if (syncViewport.isActive) {
      setSyncViewport((prev) => ({ ...prev, isActive: true, zoom: newZoom, pan: newPan }));
    }
  };

  useEffect(() => {
    if (!isDragging) return;
    const handleWindowMouseMove = (e: MouseEvent) => {
      const dx = e.clientX - dragStartMouse.current.x;
      const dy = e.clientY - dragStartMouse.current.y;

      if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
        hasDragged.current = true;
      }

      const newPan = {
        x: dragStartPan.current.x + dx,
        y: dragStartPan.current.y + dy,
      };

      updateViewport(zoomRef.current, newPan);
    };

    const handleWindowMouseUp = () => {
      setIsDragging(false);
      setSyncViewport((prev) => (prev.isActive ? { ...prev, isDragging: false } : prev));
    };

    window.addEventListener('mousemove', handleWindowMouseMove);
    window.addEventListener('mouseup', handleWindowMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleWindowMouseMove);
      window.removeEventListener('mouseup', handleWindowMouseUp);
    };
  }, [isDragging, setSyncViewport]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    setIsDragging(true);
    hasDragged.current = false;
    dragStartMouse.current = { x: e.clientX, y: e.clientY };
    dragStartPan.current = { x: panRef.current.x, y: panRef.current.y };
    setSyncViewport((prev) => (prev.isActive ? { ...prev, isDragging: true } : prev));
  };

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (hasDragged.current) return;

    if (showRateBar || showInfoBar) {
      if (showRateBar) setShowRateBar(false);
      if (showInfoBar) setShowInfoBar(false);
      return;
    }

    if (Math.abs(zoom - 1) > 0.01 || pan.x !== 0 || pan.y !== 0) {
      updateViewport(1, { x: 0, y: 0 });
    } else {
      const targetZoom = fitScale ? 1 / fitScale : 2;
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const mouseX = e.clientX - rect.left - rect.width / 2;
        const mouseY = e.clientY - rect.top - rect.height / 2;

        const newPanX = -mouseX * (targetZoom - 1);
        const newPanY = -mouseY * (targetZoom - 1);

        updateViewport(targetZoom, { x: newPanX, y: newPanY });
      } else {
        updateViewport(targetZoom, { x: 0, y: 0 });
      }
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.stopPropagation();
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left - rect.width / 2;
    const mouseY = e.clientY - rect.top - rect.height / 2;

    const zoomFactor = Math.exp(-e.deltaY * 0.002);

    const minCSSScale = fitScale ? 0.01 / fitScale : 0.1;
    const maxCSSScale = fitScale ? 10 / fitScale : 10;

    const newZoom = Math.min(Math.max(minCSSScale, zoom * zoomFactor), maxCSSScale);
    const scaleRatio = newZoom / zoom;

    const mouseFromCenterX = mouseX - pan.x;
    const mouseFromCenterY = mouseY - pan.y;
    const newPanX = mouseX - mouseFromCenterX * scaleRatio;
    const newPanY = mouseY - mouseFromCenterY * scaleRatio;

    updateViewport(newZoom, { x: newPanX, y: newPanY });
  };

  const handleZoomIn = (e: React.MouseEvent) => {
    e.stopPropagation();
    const maxCSSScale = fitScale ? 10 / fitScale : 10;
    const newZoom = Math.min(maxCSSScale, zoom * 1.25);
    const ratio = newZoom / zoom;
    updateViewport(newZoom, { x: pan.x * ratio, y: pan.y * ratio });
  };

  const handleZoomOut = (e: React.MouseEvent) => {
    e.stopPropagation();
    const minCSSScale = fitScale ? 0.01 / fitScale : 0.1;
    const newZoom = Math.max(minCSSScale, zoom / 1.25);
    const ratio = newZoom / zoom;
    updateViewport(newZoom, { x: pan.x * ratio, y: pan.y * ratio });
  };

  const handleToggle1to1 = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!fitScale) return;

    const currentAbsoluteZoom = zoom * fitScale;
    if (Math.abs(currentAbsoluteZoom - 1) < 0.05) {
      updateViewport(1, { x: 0, y: 0 });
    } else {
      updateViewport(1 / fitScale, { x: 0, y: 0 });
    }
  };

  const toggleSync = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (syncViewport.isActive) {
      setSyncViewport((prev) => ({ ...prev, isActive: false }));
    } else {
      setSyncViewport({ isActive: true, zoom, pan, isDragging: false });
    }
  };

  const ringClass = isActive
    ? 'ring-2 ring-inset ring-accent'
    : isSelected
      ? 'ring-2 ring-inset ring-gray-400'
      : 'group-hover:ring-2 group-hover:ring-inset group-hover:ring-hover-color';

  const effectiveDragging = isDragging || (syncViewport.isActive && syncViewport.isDragging);

  const SCALE_FACTOR = 4;
  const imageTransformStyle: React.CSSProperties = {
    position: 'relative',
    width: `${SCALE_FACTOR * 100}%`,
    height: `${SCALE_FACTOR * 100}%`,
    flexShrink: 0,
    transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom / SCALE_FACTOR})`,
    transition: effectiveDragging ? 'none' : 'transform 0.1s ease-out',
    transformOrigin: 'center center',
    willChange: 'transform',
  };

  const isHovered = hoveredPath === image.path;
  const isActiveFallback = isActive && !hoveredPath;
  const displayMenu = isHovered || isActiveFallback;

  const isRateMenuVisible = showRateBar && displayMenu;
  const isInfoMenuVisible = showInfoBar && displayMenu;

  return (
    <div
      ref={containerRef}
      onContextMenu={(e) => {
        e.preventDefault();
        e.stopPropagation();
        onContextMenu(e, image.path, true);
      }}
      onClick={handleClick}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseEnter={() => setHoveredCullingPath(image.path)}
      onMouseLeave={() => setHoveredCullingPath(null)}
      className={clsx(
        'relative flex items-center justify-center w-full h-full overflow-hidden group bg-bg-primary rounded-lg shadow-sm border border-border-color/10 cursor-grab active:cursor-grabbing select-none',
        isFullWidth && 'col-span-2',
      )}
    >
      <div
        className="absolute inset-0 opacity-20 pointer-events-none z-0"
        style={{
          backgroundImage: 'radial-gradient(#444 1px, transparent 1px)',
          backgroundSize: '24px 24px',
        }}
      />

      <div
        className="absolute inset-0 flex items-center justify-center pointer-events-none z-10"
        style={{ isolation: 'isolate' }}
      >
        <div style={imageTransformStyle}>
          {thumbUrl && (
            <img
              src={thumbUrl}
              className="absolute inset-0 w-full h-full object-contain"
              alt={t('library.culling.altThumbnailLoading')}
              draggable={false}
            />
          )}

          {highResSrc && (
            <motion.img
              ref={imageRef}
              onLoad={updateFitScale}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
              src={highResSrc}
              className="absolute inset-0 w-full h-full object-contain"
              alt={t('library.culling.altCullingPreviewHighRes')}
              draggable={false}
            />
          )}
        </div>
      </div>

      <AnimatePresence>
        {isInfoMenuVisible && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute bottom-[4.5rem] left-1/2 -translate-x-1/2 flex flex-col gap-4 bg-bg-primary/70 backdrop-blur-md p-4 rounded-xl border border-white/10 shadow-xl z-30 pointer-events-auto w-64 max-h-[70%] overflow-y-auto custom-scrollbar"
            onMouseDown={(e) => e.stopPropagation()}
            onWheel={(e) => e.stopPropagation()}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <Text variant={TextVariants.small} weight={TextWeights.semibold} className="text-white">
                {t('library.culling.metadata')}
              </Text>
              <button
                onClick={() => setShowInfoBar(false)}
                className="text-white/50 hover:text-white transition-colors"
              >
                <X size={14} />
              </button>
            </div>

            <div className="flex flex-col gap-4">
              {imageWidth && imageHeight && (
                <div>
                  <Text
                    variant={TextVariants.small}
                    className="text-white/50 text-[10px] uppercase tracking-wider mb-1.5 block"
                  >
                    {t('library.culling.dimensions')}
                  </Text>
                  <Text variant={TextVariants.small} className="text-white">
                    {imageWidth} × {imageHeight}
                  </Text>
                </div>
              )}

              {hasExif && (
                <div>
                  <Text
                    variant={TextVariants.small}
                    className="text-white/50 text-[10px] uppercase tracking-wider mb-1.5 block"
                  >
                    {t('library.culling.cameraSettings')}
                  </Text>
                  <div className="grid grid-cols-2 gap-3">
                    {exifData.shutter && (
                      <div
                        className="flex items-center gap-1.5 text-white/90"
                        title={t('library.culling.shutterSpeed')}
                      >
                        <span className="opacity-70">
                          <IconShutter />
                        </span>
                        <Text variant={TextVariants.small}>{exifData.shutter}</Text>
                      </div>
                    )}
                    {exifData.fNumber && (
                      <div className="flex items-center gap-1.5 text-white/90" title={t('library.culling.aperture')}>
                        <span className="opacity-70">
                          <IconAperture />
                        </span>
                        <Text variant={TextVariants.small}>{exifData.fNumber}</Text>
                      </div>
                    )}
                    {exifData.iso && (
                      <div className="flex items-center gap-1.5 text-white/90" title={t('library.culling.iso')}>
                        <span className="opacity-70">
                          <IconIso />
                        </span>
                        <Text variant={TextVariants.small}>{exifData.iso}</Text>
                      </div>
                    )}
                    {exifData.focal && (
                      <div className="flex items-center gap-1.5 text-white/90" title={t('library.culling.focalLength')}>
                        <span className="opacity-70">
                          <IconFocalLength />
                        </span>
                        <Text variant={TextVariants.small}>
                          {String(exifData.focal).endsWith('mm') ? exifData.focal : `${exifData.focal}mm`}
                        </Text>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {!hasExif && !imageWidth && !imageHeight && (
                <Text variant={TextVariants.small} className="text-white/50 italic">
                  {t('library.culling.noMetadataAvailable')}
                </Text>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {isRateMenuVisible && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute bottom-[4.5rem] left-1/2 -translate-x-1/2 flex flex-col gap-4 bg-bg-primary/70 backdrop-blur-md p-4 rounded-xl border border-white/10 shadow-xl z-30 pointer-events-auto w-64 max-h-[70%] overflow-y-auto custom-scrollbar"
            onMouseDown={(e) => e.stopPropagation()}
            onWheel={(e) => e.stopPropagation()}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <Text variant={TextVariants.small} weight={TextWeights.semibold} className="text-white">
                {t('library.culling.rateAndLabel')}
              </Text>
              <button
                onClick={() => setShowRateBar(false)}
                className="text-white/50 hover:text-white transition-colors"
              >
                <X size={14} />
              </button>
            </div>

            <div>
              <Text
                variant={TextVariants.small}
                className="text-white/50 text-[10px] uppercase tracking-wider mb-1.5 block"
              >
                {t('library.culling.rating')}
              </Text>
              <div className="flex items-center gap-1.5">
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    onClick={() => handleRate(star, [image.path])}
                    className="focus:outline-hidden transition-transform active:scale-95 hover:scale-110"
                  >
                    <StarIcon
                      size={18}
                      className={clsx(
                        'transition-colors duration-200',
                        star <= rating
                          ? 'fill-accent text-accent'
                          : 'fill-transparent text-white/30 hover:text-white/80',
                      )}
                    />
                  </button>
                ))}
              </div>
            </div>

            <div>
              <Text
                variant={TextVariants.small}
                className="text-white/50 text-[10px] uppercase tracking-wider mb-1.5 block"
              >
                {t('library.culling.colorLabel')}
              </Text>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => handleSetColorLabel(null, [image.path])}
                  className={clsx(
                    'w-5 h-5 rounded-full flex items-center justify-center transition-all hover:scale-110',
                    currentColor === null
                      ? 'ring-2 ring-white/50 ring-offset-1 ring-offset-bg-primary'
                      : 'opacity-50 hover:opacity-100 hover:ring-2 hover:ring-white/30',
                  )}
                  data-tooltip={t('library.culling.none')}
                >
                  <X size={12} className="text-white/50" />
                </button>
                {COLOR_LABELS.map((color: Color) => (
                  <button
                    key={color.name}
                    onClick={() => handleSetColorLabel(color.name, [image.path])}
                    className={clsx(
                      'w-5 h-5 rounded-full transition-all hover:scale-110',
                      currentColor === color.name
                        ? 'ring-2 ring-white ring-offset-1 ring-offset-bg-primary'
                        : 'hover:ring-2 hover:ring-white/30',
                    )}
                    style={{ backgroundColor: color.color }}
                    data-tooltip={color.name}
                  >
                    {currentColor === color.name && <Check size={12} className="text-black/50 mx-auto" />}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <Text
                variant={TextVariants.small}
                className="text-white/50 text-[10px] uppercase tracking-wider mb-1.5 block"
              >
                {t('library.culling.tags')}
              </Text>
              <div className="flex flex-wrap gap-1 mb-2">
                <AnimatePresence>
                  {currentTags.map((tagItem) => (
                    <motion.div
                      key={tagItem.tag}
                      layout
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      className="flex items-center gap-1 bg-white/10 px-2 py-0.5 rounded-md group cursor-pointer border border-transparent hover:border-white/20 transition-colors"
                      onClick={() => handleRemoveTag(tagItem)}
                    >
                      <Text as="span" variant={TextVariants.small} className="text-white/90 text-xs">
                        {tagItem.tag}
                      </Text>
                      <X size={10} className="text-white/50 group-hover:text-white" />
                    </motion.div>
                  ))}
                </AnimatePresence>
                {currentTags.length === 0 && (
                  <Text variant={TextVariants.small} className="italic text-white/40 text-xs">
                    {t('library.culling.noTagsAdded')}
                  </Text>
                )}
              </div>
              <div className="flex items-center bg-bg-primary/40 border border-white/10 rounded-md px-2 py-1.5 focus-within:border-accent/50 transition-colors">
                <input
                  type="text"
                  value={tagInputValue}
                  onChange={(e) => setTagInputValue(e.target.value)}
                  onKeyDown={handleTagInputKeyDown}
                  placeholder={t('library.culling.addTagPlaceholder')}
                  className="bg-transparent border-none outline-hidden text-xs w-full text-white placeholder-white/40"
                />
                <button
                  onClick={() => handleAddTag(tagInputValue)}
                  disabled={!tagInputValue.trim()}
                  className="text-white/50 hover:text-white disabled:opacity-30 transition-colors"
                >
                  <Plus size={14} />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div
        className={clsx(
          'absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-bg-primary/70 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/10 shadow-xl z-20 pointer-events-auto transition-opacity duration-200 max-w-[calc(100%-1.5rem)]',
          isRateMenuVisible || isInfoMenuVisible ? 'opacity-100' : 'opacity-0 group-hover:opacity-100',
        )}
        onMouseDown={(e) => e.stopPropagation()}
        onWheel={(e) => e.stopPropagation()}
      >
        <AnimatePresence>
          {isLoading && (
            <motion.div
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: 'auto' }}
              exit={{ opacity: 0, width: 0 }}
              className="flex items-center justify-center overflow-hidden"
            >
              <Loader2 className="w-4 h-4 animate-spin text-white mr-1" />
            </motion.div>
          )}
        </AnimatePresence>

        <Text
          variant={TextVariants.small}
          className="text-white truncate shrink min-w-0 max-w-20 sm:max-w-28 md:max-w-37.5"
          data-tooltip={baseName}
        >
          {baseName}
        </Text>

        {isVirtualCopy && (
          <div className="bg-white/20 text-white px-1.5 py-0.5 rounded-sm shrink-0 ml-1">
            <Text variant={TextVariants.small} weight={TextWeights.bold} className="text-[9px] leading-none">
              {t('library.culling.vc')}
            </Text>
          </div>
        )}

        {hasAnyOverlay && (
          <div className="rounded-full h-5 px-1.5 flex items-center justify-center gap-0 shadow-md bg-surface/30 pointer-events-auto shrink-0 ml-1">
            {showEditIcon && (
              <div className="text-white flex items-center shrink-0">
                <SlidersHorizontal size={12} />
              </div>
            )}

            {colorLabel && (
              <div className={clsx('flex items-center justify-center shrink-0', showEditIcon && 'ml-1.5')}>
                <div
                  className="w-3 h-3 rounded-full transition-colors duration-200"
                  style={{ backgroundColor: colorLabel.color }}
                />
              </div>
            )}

            {rating > 0 && (
              <div className={clsx('flex items-center gap-0.5 shrink-0', (showEditIcon || colorLabel) && 'ml-1.5')}>
                <Text variant={TextVariants.small} color={TextColors.white}>
                  {rating}
                </Text>
                <StarIcon size={12} className="text-white fill-white" />
              </div>
            )}
          </div>
        )}

        <div className="w-px h-5 bg-white/20 mx-1 shrink-0"></div>

        <button
          onClick={(e) => {
            e.stopPropagation();
            onImageDoubleClick(image.path);
          }}
          className="p-1.5 text-white/60 hover:bg-white/10 hover:text-white rounded-full transition-colors shrink-0"
          data-tooltip={t('library.culling.editImage')}
        >
          <SquarePen size={14} />
        </button>

        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowInfoBar((prev) => {
              if (!prev) setShowRateBar(false);
              return !prev;
            });
          }}
          className={clsx(
            'p-1.5 rounded-full transition-colors shrink-0',
            showInfoBar ? 'bg-accent text-button-text' : 'text-white/60 hover:bg-white/10 hover:text-white',
          )}
          data-tooltip={t('library.culling.metadata')}
        >
          <Info size={14} />
        </button>

        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowRateBar((prev) => {
              if (!prev) setShowInfoBar(false);
              return !prev;
            });
          }}
          className={clsx(
            'p-1.5 rounded-full transition-colors shrink-0',
            showRateBar ? 'bg-accent text-button-text' : 'text-white/60 hover:bg-white/10 hover:text-white',
          )}
          data-tooltip={t('library.culling.rateAndLabel')}
        >
          <Tag size={14} />
        </button>

        <div className="w-px h-5 bg-white/20 mx-1 shrink-0"></div>

        <button
          onClick={toggleSync}
          className={clsx(
            'p-1.5 rounded-full transition-colors shrink-0',
            syncViewport.isActive ? 'bg-accent text-button-text' : 'text-white/60 hover:bg-white/10 hover:text-white',
          )}
          data-tooltip={t('library.culling.syncZoomAndPan')}
        >
          <Link size={14} />
        </button>

        <button
          onClick={handleZoomOut}
          className="p-1.5 text-white/60 hover:bg-white/10 hover:text-white rounded-full transition-colors shrink-0"
        >
          <ZoomOut size={16} />
        </button>

        <button
          onClick={handleToggle1to1}
          className="text-xs font-mono text-white/90 w-8 text-center select-none shrink-0 hover:bg-white/10 hover:text-white rounded-md py-1 transition-colors cursor-pointer"
          data-tooltip={t('library.culling.toggleFit')}
        >
          {fitScale ? Math.round(zoom * fitScale * 100) : Math.round(zoom * 100)}%
        </button>

        <button
          onClick={handleZoomIn}
          className="p-1.5 text-white/60 hover:bg-white/10 hover:text-white rounded-full transition-colors shrink-0"
        >
          <ZoomIn size={16} />
        </button>
      </div>

      <div
        className={clsx(
          'absolute inset-0 rounded-lg pointer-events-none z-30 transition-all duration-150 ring-2 ring-inset ring-transparent',
          ringClass,
        )}
      />
    </div>
  );
}

const Row = React.memo(
  ({
    index,
    style,
    imageList,
    multiSelectedPaths,
    activePath,
    onContextMenu,
    onImageDoubleClick,
    thumbnailAspectRatio,
    imageRatings,
    onImageClick,
    queueThumbnailRequest,
    hoveredCullingPath,
  }: any) => {
    const image: ImageFile = imageList[index];
    const isSelected = multiSelectedPaths.includes(image.path);

    useEffect(() => {
      if (!image || !queueThumbnailRequest) return;
      queueThumbnailRequest(image);

      if (image.is_cloud_placeholder) {
        const interval = setInterval(() => {
          queueThumbnailRequest(image);
        }, 5000);
        return () => clearInterval(interval);
      }
    }, [image, queueThumbnailRequest]);

    return (
      <div style={style} className="p-2 box-border">
        <div className="w-full h-full">
          <Thumbnail
            path={image.path}
            isSelected={isSelected}
            isActive={activePath === image.path}
            isForcedHover={hoveredCullingPath === image.path}
            onImageClick={(path: string, e: any) => onImageClick(path, e)}
            onContextMenu={onContextMenu}
            onImageDoubleClick={onImageDoubleClick}
            onLoad={() => {}}
            rating={imageRatings?.[image.path] || 0}
            tags={image.tags}
            exif={image.exif}
            isEdited={image.is_edited}
            aspectRatio={thumbnailAspectRatio}
            isCloudPlaceholder={image.is_cloud_placeholder}
          />
        </div>
      </div>
    );
  },
);

export default function CullingView(props: any) {
  const { t } = useTranslation();
  const {
    imageList,
    multiSelectedPaths,
    activePath,
    onImageClick,
    imageRatings,
    thumbnailAspectRatio,
    onContextMenu,
    onImageDoubleClick,
    onRequestThumbnails,
    onClearSelection,
    onEmptyAreaContextMenu,
  } = props;

  const containerRef = useRef<HTMLDivElement>(null);
  const [listHeight, setListHeight] = useState(0);
  const [sidebarWidth, setSidebarWidth] = useState(224);
  const isResizing = useRef(false);

  const requestQueueRef = useRef<Map<string, { path: string; modified?: number }>>(new Map());
  const requestTimeoutRef = useRef<any>(null);

  const [hoveredCullingPath, setHoveredCullingPath] = useState<string | null>(null);

  const [syncViewport, setSyncViewport] = useState<SyncViewport>({
    isActive: false,
    zoom: 1,
    pan: { x: 0, y: 0 },
    isDragging: false,
  });

  const [showRateBar, setShowRateBar] = useState(false);
  const [showInfoBar, setShowInfoBar] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const isCatalog = useLibraryStore((state) => state.librarySource.type === 'catalog');
  const catalogRoots = useLibraryStore((state) => state.catalogRoots);
  const cullWorkspaceFolderPath = useUIStore((state) => state.cullWorkspaceFolderPath);
  const cullProgress = useUIStore((state) => state.cullWorkspaceProgress);
  const thumbnails = useProcessStore((state) => state.thumbnails);
  const [reviewFilter, setReviewFilter] = useState<'all' | 'selected' | 'highlights' | 'duplicates' | 'blurry'>('all');
  const [cullDecisions, setCullDecisions] = useState<Record<string, CullSessionDecision>>({});
  const [activeFaces, setActiveFaces] = useState<CullFaceReviewItem[]>([]);
  const [subjectEvidence, setSubjectEvidence] = useState<CullSubjectEvidence | null>(null);
  const activeImage = useMemo(() => imageList.find((image: ImageFile) => image.path === activePath), [imageList, activePath]);
  const [cullFolderPath, setCullFolderPath] = useState<string | null>(null);
  const [cullSettings, setCullSettings] = useState<CullingSettings>(DEFAULT_CULL_SETTINGS);
  const [includeSubfolders, setIncludeSubfolders] = useState(false);
  const [isPlanningCull, setIsPlanningCull] = useState(false);
  const [isConfiguringCull, setIsConfiguringCull] = useState(true);
  const [duplicateRatio, setDuplicateRatio] = useState<'more' | 'moderate' | 'less'>('moderate');
  const [highlightPercent, setHighlightPercent] = useState<number>(15);
  const [localCullPlan, setLocalCullPlan] = useState<AutoCullPlan | null>(null);
  const [cullError, setCullError] = useState<string | null>(null);
  const [isApplyingCull, setIsApplyingCull] = useState(false);
  const [decisionFeedbackReason, setDecisionFeedbackReason] = useState('');
  const [isCompareMode, setIsCompareMode] = useState(false);
  const cullGridRef = useRef<HTMLDivElement>(null);
  const marqueeStart = useRef<{ x: number; y: number; additive: boolean; initialSelection: string[] } | null>(null);
  const marqueeMoved = useRef(false);
  const [marqueeBounds, setMarqueeBounds] = useState<{ left: number; top: number; width: number; height: number } | null>(null);
  const activePlanItem = useMemo(
    () => localCullPlan?.items.find((item) => item.representativePath === activePath) || null,
    [localCullPlan, activePath],
  );
  const duplicateRepresentativePath = useMemo(() => {
    if (!activePath) return null;
    const duplicatePrefix = 'duplicate_of:';
    return activePlanItem?.reason.startsWith(duplicatePrefix)
      ? activePlanItem.reason.slice(duplicatePrefix.length)
      : activePath;
  }, [activePath, activePlanItem]);
  const relatedDuplicatePaths = useMemo(() => {
    if (!localCullPlan || !activePath) return [];
    const duplicatePrefix = 'duplicate_of:';
    const representativePath = duplicateRepresentativePath || activePath;
    const related = localCullPlan.items
      .filter((item) => item.representativePath === representativePath || item.reason === `${duplicatePrefix}${representativePath}`)
      .map((item) => item.representativePath);
    return related.length > 1 ? related : [];
  }, [activePath, duplicateRepresentativePath, localCullPlan]);
  const cullCatalogScope = useMemo(() => {
    if (!cullFolderPath || !isCatalog) return null;
    const match = /^LibraryFolder:(\d+):(.*)$/.exec(cullFolderPath);
    if (!match) return null;
    const rootId = Number(match[1]);
    const relativePath = match[2] || '.';
    const root = catalogRoots.find((candidate) => candidate.id === rootId);
    if (!root) return null;
    const absoluteFolderPath = relativePath === '.'
      ? root.absolutePath
      : `${root.absolutePath.replace(/[\\/]$/, '')}/${relativePath}`;
    return { rootId, relativePath, absoluteFolderPath, rootPath: root.absolutePath };
  }, [catalogRoots, cullFolderPath, isCatalog]);

  useEffect(() => {
    if (!cullWorkspaceFolderPath || cullWorkspaceFolderPath === cullFolderPath) return;
    setCullFolderPath(cullWorkspaceFolderPath);
    setLocalCullPlan(null);
    setCullDecisions({});
    setCullError(null);
  }, [cullWorkspaceFolderPath, cullFolderPath]);

  useEffect(() => {
    if (!isCatalog) { setCullDecisions({}); return; }
    let active = true;
    void invoke<CullSessionSummary[]>(Invokes.ListCullSessions)
      .then(async (sessions) => {
        const session = sessions.find((candidate) => candidate.state === 'planned') || sessions[0];
        if (!session) return [];
        return invoke<CullSessionDecision[]>(Invokes.ListCullSessionDecisions, { sessionId: session.id });
      })
      .then((decisions) => {
        if (!active) return;
        setCullDecisions(Object.fromEntries(decisions.map((decision) => [decision.representativePath, decision])));
      })
      .catch(() => { if (active) setCullDecisions({}); });
    return () => { active = false; };
  }, [isCatalog, imageList]);

  useEffect(() => {
    if (!localCullPlan) return;
    setCullDecisions(Object.fromEntries(localCullPlan.items.map((item, index) => [item.representativePath, {
      id: -(index + 1),
      representativePath: item.representativePath,
      proposedStatus: item.keep ? 'keep' : 'reject',
      finalStatus: 'pending',
      qualityScore: item.qualityScore,
      reason: item.reason,
    }] as const)));
  }, [localCullPlan]);

  useEffect(() => {
    if (!isCatalog || !activePath) {
      setActiveFaces([]);
      return;
    }
    let active = true;
    void invoke<CullFaceReviewItem[]>(Invokes.ListCatalogFaceReviewItemsForPath, { path: activePath })
      .then((faces) => {
        if (!active) return;
        setActiveFaces(faces);
        for (const item of faces) {
          if (!item.thumbnailDataUrl && !item.cropPath) {
            void invoke<string>('get_or_generate_face_crop', { faceId: item.face.id })
              .then((thumbnailDataUrl) => {
                if (!active || !thumbnailDataUrl) return;
                setActiveFaces((current) => current.map((face) => face.face.id === item.face.id ? { ...face, thumbnailDataUrl } : face));
              })
              .catch(() => {});
          }
        }
      })
      .catch(() => { if (active) setActiveFaces([]); });
    return () => { active = false; };
  }, [isCatalog, activePath]);

  useEffect(() => {
    if (!isCatalog || !activeImage?.catalog_image_id) {
      setSubjectEvidence(null);
      return;
    }
    let active = true;
    void invoke<CullSubjectEvidence>(Invokes.GetImageProvenance, { imageId: activeImage.catalog_image_id })
      .then((evidence) => { if (active) setSubjectEvidence(evidence); })
      .catch(() => { if (active) setSubjectEvidence(null); });
    return () => { active = false; };
  }, [isCatalog, activeImage?.catalog_image_id]);

  const [listHandle, setListHandle] = useListCallbackRef();
  const prevActivePath = useRef<string | null>(null);
  const prevListElement = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!listHandle?.element || imageList.length === 0 || !activePath) return;

    const element = listHandle.element as HTMLElement;
    const isPathChanged = activePath !== prevActivePath.current;
    const isElementChanged = element !== prevListElement.current;

    if (isPathChanged || isElementChanged) {
      const isInitial = prevActivePath.current === null || isElementChanged;
      prevActivePath.current = activePath;
      prevListElement.current = element;

      const index = imageList.findIndex((img: ImageFile) => img.path === activePath);
      if (index !== -1) {
        const rowHeight = sidebarWidth - 16;
        const targetTop = index * rowHeight;
        const clientHeight = element.clientHeight;
        const scrollTop = element.scrollTop;
        const itemBottom = targetTop + rowHeight;
        const SCROLL_OFFSET = 40;

        if (isInitial) {
          element.scrollTo({
            top: Math.max(0, targetTop - clientHeight / 2 + rowHeight / 2),
            behavior: 'instant',
          });
        } else if (itemBottom > scrollTop + clientHeight) {
          element.scrollTo({
            top: itemBottom - clientHeight + SCROLL_OFFSET,
            behavior: 'smooth',
          });
        } else if (targetTop < scrollTop) {
          element.scrollTo({
            top: Math.max(0, targetTop - SCROLL_OFFSET),
            behavior: 'smooth',
          });
        }
      }
    }
  }, [activePath, listHandle, imageList, sidebarWidth]);

  const queueThumbnailRequest = useCallback(
    (image: ImageFile) => {
      if (!onRequestThumbnails) return;
      const path = image.path;
      if (useProcessStore.getState().thumbnails[path]) return;
      requestQueueRef.current.set(path, { path, modified: image.modified });
      if (!requestTimeoutRef.current) {
        requestTimeoutRef.current = setTimeout(() => {
          const paths = Array.from(requestQueueRef.current.values());
          if (paths.length > 0) {
            onRequestThumbnails(paths);
            requestQueueRef.current.clear();
          }
          requestTimeoutRef.current = null;
        }, 50);
      }
    },
    [onRequestThumbnails],
  );

  useEffect(() => {
    const updateHeight = () => {
      if (containerRef.current) {
        setListHeight(containerRef.current.clientHeight);
      }
    };
    updateHeight();
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, []);

  const resize = useCallback((mouseMoveEvent: MouseEvent) => {
    if (!isResizing.current) return;
    const newWidth = window.innerWidth - mouseMoveEvent.clientX;
    if (newWidth >= 160 && newWidth <= 450) {
      setSidebarWidth(newWidth);
    }
  }, []);

  const stopResizing = useCallback(() => {
    isResizing.current = false;
    document.removeEventListener('mousemove', resize);
    document.removeEventListener('mouseup', stopResizing);
  }, [resize]);

  const startResizing = useCallback(
    (mouseDownEvent: React.MouseEvent) => {
      mouseDownEvent.preventDefault();
      isResizing.current = true;
      document.addEventListener('mousemove', resize);
      document.addEventListener('mouseup', stopResizing);
    },
    [resize, stopResizing],
  );

  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', resize);
      document.removeEventListener('mouseup', stopResizing);
    };
  }, [resize, stopResizing]);

  const rankedFinalists = useMemo(() => {
    const kept = Object.values(cullDecisions)
      .filter((decision) => decision.finalStatus === 'keep' || (decision.finalStatus === 'pending' && decision.proposedStatus === 'keep'))
      .sort((left, right) => right.qualityScore - left.qualityScore);
    return kept.slice(0, Math.ceil(kept.length * 0.15));
  }, [cullDecisions]);
  const highlightPaths = useMemo(
    () => new Set(rankedFinalists.map((decision) => decision.representativePath)),
    [rankedFinalists],
  );
  const finalistRanks = useMemo(
    () => new Map(rankedFinalists.map((decision, index) => [decision.representativePath, index + 1])),
    [rankedFinalists],
  );

  const cullImageList = useMemo(() => imageList.filter((image: ImageFile) => {
    const decision = cullDecisions[image.path];
    if (reviewFilter === 'all') return true;
    if (reviewFilter === 'selected') return decision?.finalStatus === 'keep' || (!decision?.finalStatus && decision?.proposedStatus === 'keep');
    if (reviewFilter === 'highlights') return highlightPaths.has(image.path);
    if (reviewFilter === 'duplicates') return decision?.reason.startsWith('duplicate_of:');
    return !!decision && !decision.reason.startsWith('duplicate_of:') && (decision.proposedStatus === 'reject' || decision.finalStatus === 'reject');
  }), [imageList, cullDecisions, highlightPaths, reviewFilter]);

  const cullFilterCounts = useMemo(() => ({
    all: imageList.length,
    selected: imageList.filter((image: ImageFile) => {
      const decision = cullDecisions[image.path];
      return decision?.finalStatus === 'keep' || (!decision?.finalStatus && decision?.proposedStatus === 'keep');
    }).length,
    highlights: imageList.filter((image: ImageFile) => highlightPaths.has(image.path)).length,
    duplicates: imageList.filter((image: ImageFile) => cullDecisions[image.path]?.reason.startsWith('duplicate_of:')).length,
    blurry: imageList.filter((image: ImageFile) => {
      const decision = cullDecisions[image.path];
      return !!decision && !decision.reason.startsWith('duplicate_of:') && (decision.proposedStatus === 'reject' || decision.finalStatus === 'reject');
    }).length,
  }), [imageList, cullDecisions, highlightPaths]);

  useEffect(() => {
    cullImageList.slice(0, 500).forEach((image: ImageFile) => queueThumbnailRequest(image));
  }, [cullImageList, queueThumbnailRequest]);

  const rowProps = useMemo(
    () => ({
      imageList: cullImageList,
      multiSelectedPaths,
      activePath,
      thumbnailAspectRatio,
      imageRatings,
      onContextMenu,
      onImageDoubleClick,
      onImageClick,
      queueThumbnailRequest,
      sidebarWidth,
      hoveredCullingPath,
    }),
    [
      cullImageList,
      multiSelectedPaths,
      activePath,
      thumbnailAspectRatio,
      imageRatings,
      onContextMenu,
      onImageDoubleClick,
      onImageClick,
      queueThumbnailRequest,
      sidebarWidth,
      hoveredCullingPath,
    ],
  );

  const displayPaths = multiSelectedPaths.slice(-6);
  const displayImages = displayPaths
    .map((p: string) => imageList.find((img: ImageFile) => img.path === p))
    .filter(Boolean);
  const displayCount = displayImages.length;

  const handleSidebarEmptyClick = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    if (!target.closest('[data-bench-id="thumbnail"]') && !target.closest('button')) {
      onClearSelection?.();
    }
  };

  const handleSidebarEmptyContextMenu = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    if (!target.closest('[data-bench-id="thumbnail"]') && !target.closest('button')) {
      onEmptyAreaContextMenu?.(e);
    }
  };

  const selectLibraryRoot = async (root: CatalogRoot) => {
    const folderKey = `LibraryFolder:${root.id}:.`;
    setCullFolderPath(folderKey);
    useUIStore.getState().setUI({ cullWorkspaceFolderPath: folderKey });
    setLocalCullPlan(null);
    setCullDecisions({});
    setCullError(null);
    try {
      const files = await invoke<ImageFile[]>(Invokes.ListCatalogImages, {
        rootId: root.id,
        recursive: includeSubfolders,
        folderPath: '.',
      });
      const ratings = Object.fromEntries(files.map((file) => [file.path, file.rating || 0]));
      useLibraryStore.getState().setLibrary({
        currentFolderPath: folderKey,
        rootPaths: [root.absolutePath],
        imageList: files,
        imageRatings: ratings,
        multiSelectedPaths: [],
        libraryActivePath: null,
        libraryScrollTop: 0,
      });
    } catch (err) {
      console.error('Failed to load library images for culling:', err);
    }
  };

  const chooseCullFolder = async () => {
    const path = await open({ directory: true, multiple: false, title: 'Choose a folder to cull' });
    if (typeof path !== 'string') return;
    setCullFolderPath(path);
    useUIStore.getState().setUI({ cullWorkspaceFolderPath: path });
    setLocalCullPlan(null);
    setCullDecisions({});
    setCullError(null);
    try {
      const files = await invoke<ImageFile[]>(
        includeSubfolders ? Invokes.ListImagesRecursive : Invokes.ListImagesInDir,
        { path },
      );
      const ratings = Object.fromEntries(files.map((file) => [file.path, file.rating || 0]));
      useLibraryStore.getState().setLibrary({
        currentFolderPath: path,
        rootPaths: [path],
        imageList: files,
        imageRatings: ratings,
        multiSelectedPaths: [],
        libraryActivePath: null,
        libraryScrollTop: 0,
      });
    } catch (err) {
      console.error('Failed to load folder images for culling:', err);
    }
  };

  const startCullFromRail = async () => {
    if (!cullFolderPath || isPlanningCull) return;
    setIsPlanningCull(true);
    setCullError(null);
    useUIStore.getState().setUI({ cullWorkspaceProgress: null });
    try {
      const files = cullCatalogScope
        ? await invoke<ImageFile[]>(Invokes.ListCatalogImages, {
          rootId: cullCatalogScope.rootId,
          recursive: includeSubfolders,
          folderPath: cullCatalogScope.relativePath,
        })
        : await invoke<ImageFile[]>(
          includeSubfolders ? Invokes.ListImagesRecursive : Invokes.ListImagesInDir,
          { path: cullFolderPath },
        );
      const imageRatings = Object.fromEntries(files.map((file) => [file.path, file.rating || 0]));
      useLibraryStore.getState().setLibrary({
        currentFolderPath: cullFolderPath,
        rootPaths: [cullCatalogScope?.rootPath || cullFolderPath],
        imageList: files,
        imageRatings,
        multiSelectedPaths: [],
        libraryActivePath: null,
        libraryScrollTop: 0,
      });
      const catalogImageIds: Record<string, number> = {};
      if (cullCatalogScope) {
        for (const file of files) {
          if (file.catalog_image_id != null) catalogImageIds[file.path] = file.catalog_image_id;
        }
      }
      const plan = await invoke<AutoCullPlan>(Invokes.PlanAutoCull, {
        folderPath: cullCatalogScope?.absoluteFolderPath || cullFolderPath,
        includeSubfolders,
        settings: cullSettings,
        rejectedFolderName: '_rejected',
        deleteInsteadOfMove: false,
        catalogImageIds: Object.keys(catalogImageIds).length > 0 ? catalogImageIds : null,
      });
      setIsCompareMode(false);
      setLocalCullPlan(plan);
    } catch (error) {
      setCullError(String(error));
    } finally {
      setIsPlanningCull(false);
      useUIStore.getState().setUI({ cullWorkspaceProgress: null });
    }
  };

  const updateLocalCullDecision = async (path: string, keep: boolean, feedbackReason?: string) => {
    if (!localCullPlan) return;
    const nextPlan = {
      ...localCullPlan,
      items: localCullPlan.items.map((item) => item.representativePath === path ? { ...item, keep } : item),
    };
    nextPlan.rejectCount = nextPlan.items.filter((item) => !item.keep).length;
    setLocalCullPlan(nextPlan);
    if (nextPlan.sessionId) {
      try {
        await invoke(Invokes.UpdateCullSessionDecision, {
          sessionId: nextPlan.sessionId,
          representativePath: path,
          keep,
          feedbackReason: feedbackReason?.trim() || null,
        });
        if (feedbackReason) setDecisionFeedbackReason('');
      } catch (error) {
        setCullError(`Could not save this review decision: ${String(error)}`);
      }
    }
  };

  const updateSelectedCullDecisions = async (keep: boolean) => {
    if (!localCullPlan) return;
    const selectedPaths = multiSelectedPaths.filter((path: string) =>
      localCullPlan.items.some((item) => item.representativePath === path),
    );
    if (selectedPaths.length === 0) return;

    const selected = new Set(selectedPaths);
    const nextPlan = {
      ...localCullPlan,
      items: localCullPlan.items.map((item) =>
        selected.has(item.representativePath) ? { ...item, keep } : item,
      ),
    };
    nextPlan.rejectCount = nextPlan.items.filter((item) => !item.keep).length;
    setLocalCullPlan(nextPlan);

    if (nextPlan.sessionId) {
      try {
        await Promise.all(selectedPaths.map((representativePath: string) => invoke(
          Invokes.UpdateCullSessionDecision,
          { sessionId: nextPlan.sessionId, representativePath, keep },
        )));
      } catch (error) {
        setCullError(`Could not save these review decisions: ${String(error)}`);
      }
    }
  };

  const selectAllCullCandidates = () => {
    if (!localCullPlan) return;
    const paths = localCullPlan.items.map((item) => item.representativePath);
    useLibraryStore.getState().setLibrary({
      multiSelectedPaths: paths,
      libraryActivePath: paths[0] || null,
      selectionAnchorPath: paths[0] || null,
    });
  };

  const beginGridSelection = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!localCullPlan || event.button !== 0) return;
    marqueeStart.current = {
      x: event.clientX,
      y: event.clientY,
      additive: event.ctrlKey || event.metaKey,
      initialSelection: multiSelectedPaths,
    };
    marqueeMoved.current = false;
    setMarqueeBounds(null);
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const extendGridSelection = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = marqueeStart.current;
    const grid = cullGridRef.current;
    if (!start || !grid) return;
    if (Math.abs(event.clientX - start.x) < 6 && Math.abs(event.clientY - start.y) < 6) return;
    marqueeMoved.current = true;
    const gridBounds = grid.getBoundingClientRect();
    setMarqueeBounds({
      left: Math.min(start.x, event.clientX) - gridBounds.left,
      top: Math.min(start.y, event.clientY) - gridBounds.top,
      width: Math.abs(event.clientX - start.x),
      height: Math.abs(event.clientY - start.y),
    });
    const left = Math.min(start.x, event.clientX);
    const right = Math.max(start.x, event.clientX);
    const top = Math.min(start.y, event.clientY);
    const bottom = Math.max(start.y, event.clientY);
    const draggedPaths = Array.from(grid.querySelectorAll<HTMLElement>('[data-cull-path]'))
      .filter((element) => {
        const bounds = element.getBoundingClientRect();
        return bounds.right >= left && bounds.left <= right && bounds.bottom >= top && bounds.top <= bottom;
      })
      .map((element) => element.dataset.cullPath)
      .filter((path): path is string => Boolean(path));
    const selectedPaths = start.additive
      ? Array.from(new Set([...start.initialSelection, ...draggedPaths]))
      : draggedPaths;
    useLibraryStore.getState().setLibrary({
      multiSelectedPaths: selectedPaths,
      libraryActivePath: selectedPaths.at(-1) || null,
      selectionAnchorPath: selectedPaths.at(-1) || null,
    });
  };

  const endGridSelection = (event: React.PointerEvent<HTMLDivElement>) => {
    marqueeStart.current = null;
    setMarqueeBounds(null);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const applyCullFromRail = async () => {
    if (!localCullPlan || isApplyingCull) return;
    if (localCullPlan.items.some((item) => !item.keep && item.hasConflict)) {
      setCullError('Some rejected files already exist in _rejected. Resolve the duplicate filenames before applying this plan.');
      return;
    }
    setIsApplyingCull(true);
    setCullError(null);
    try {
      const result = await invoke<AutoCullResult>(Invokes.ApplyAutoCullPlan, { plan: localCullPlan, conflictAction: null });
      const moved = new Set(result.moved.map((item) => item.oldPath));
      useLibraryStore.getState().setLibrary((state) => ({
        imageList: state.imageList.filter((image) => !moved.has(image.path)),
        multiSelectedPaths: state.multiSelectedPaths.filter((path) => !moved.has(path)),
        libraryActivePath: moved.has(state.libraryActivePath || '') ? null : state.libraryActivePath,
      }));
      setLocalCullPlan(null);
    } catch (error) {
      setCullError(String(error));
    } finally {
      setIsApplyingCull(false);
    }
  };

  return (
    <div className="flex w-full h-full min-h-0 bg-transparent">
      <div className="flex-1 flex overflow-hidden relative bg-transparent">
        {isCatalog && <button className="absolute z-40 top-3 right-3 h-9 w-9 flex items-center justify-center rounded-md bg-bg-secondary/90 border border-border-color text-text-primary hover:bg-surface" onClick={() => setShowHistory((current) => !current)} data-tooltip="Culling history"><History size={16} /></button>}
        {isCatalog && showHistory && <CullHistoryPanel onClose={() => setShowHistory(false)} />}
        {imageList.length === 0 ? (
          <div className="m-auto flex w-full max-w-2xl flex-col items-center px-6 py-10">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-md border border-border-color bg-bg-secondary text-text-secondary">
              <FolderOpen size={22} />
            </div>
            <Text variant={TextVariants.heading} className="text-xl font-semibold">Choose photos to cull</Text>
            <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mt-2 text-center max-w-md">
              RapidRAW analyzes duplicate groupings, focus sharpness, and subject detection so you can quickly review and organize your shoot.
            </Text>

            <div className="mt-8 grid w-full grid-cols-1 md:grid-cols-2 gap-4">
              {/* Option 1: Library Folders */}
              <div className="flex flex-col rounded-lg border border-border-color bg-bg-secondary/60 p-5 shadow-xs transition-colors hover:border-accent/40">
                <div className="flex items-center gap-3">
                  <div className="flex h-9 w-9 items-center justify-center rounded-md bg-accent/10 text-accent">
                    <Database size={18} />
                  </div>
                  <div>
                    <Text variant={TextVariants.small} weight={TextWeights.semibold} className="text-sm">
                      Library Folders
                    </Text>
                    <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="text-xs">
                      Cull from your indexed catalog roots
                    </Text>
                  </div>
                </div>

                <div className="mt-4 flex-1 space-y-2 max-h-48 overflow-y-auto pr-1">
                  {catalogRoots.length > 0 ? (
                    catalogRoots.map((root) => (
                      <button
                        key={root.id}
                        onClick={() => void selectLibraryRoot(root)}
                        className="group flex w-full items-center justify-between gap-2 rounded-md border border-border-color bg-bg-primary px-3 py-2 text-left text-xs transition-colors hover:border-accent/60 hover:bg-surface"
                      >
                        <div className="min-w-0 flex-1">
                          <div className="font-medium text-text-primary truncate">
                            {root.label || root.absolutePath.split(/[\\/]/).pop() || root.absolutePath}
                          </div>
                          <div className="text-[11px] text-text-secondary truncate font-mono mt-0.5">
                            {root.absolutePath}
                          </div>
                        </div>
                        <div className="shrink-0 flex items-center gap-2">
                          <span className="rounded bg-surface px-1.5 py-0.5 text-[10px] text-text-secondary tabular-nums border border-border-color">
                            {root.imageCount} photos
                          </span>
                          <FolderOpen size={13} className="text-text-secondary group-hover:text-text-primary transition-colors" />
                        </div>
                      </button>
                    ))
                  ) : (
                    <div className="rounded-md border border-dashed border-border-color p-4 text-center text-xs text-text-secondary">
                      No library folders configured yet.
                    </div>
                  )}
                </div>
              </div>

              {/* Option 2: External / Non-Library Folder */}
              <div className="flex flex-col justify-between rounded-lg border border-border-color bg-bg-secondary/60 p-5 shadow-xs transition-colors hover:border-accent/40">
                <div>
                  <div className="flex items-center gap-3">
                    <div className="flex h-9 w-9 items-center justify-center rounded-md bg-accent/10 text-accent">
                      <FolderOpen size={18} />
                    </div>
                    <div>
                      <Text variant={TextVariants.small} weight={TextWeights.semibold} className="text-sm">
                        External Folder
                      </Text>
                      <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="text-xs">
                        Cull from any folder on disk
                      </Text>
                    </div>
                  </div>
                  <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mt-4 text-xs">
                    Choose any local, SD card, or external drive directory without needing to add it to your catalog library first.
                  </Text>
                </div>

                <div className="mt-6">
                  <button
                    className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-accent px-4 py-2.5 text-sm font-medium text-button-text hover:brightness-110 transition-colors shadow-xs"
                    onClick={() => void chooseCullFolder()}
                  >
                    <FolderOpen size={16} /> Browse non-library folder...
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : isConfiguringCull ? (
          <div className="w-full h-full overflow-y-auto p-4 md:p-8 flex justify-center items-start">
            <div className="w-full max-w-3xl rounded-xl border border-border-color bg-bg-secondary p-6 md:p-8 shadow-2xl">
              {/* Header */}
              <div className="flex items-start justify-between border-b border-border-color pb-5">
                <div>
                  <h2 className="text-xl font-bold text-text-primary tracking-tight">Set Preferences</h2>
                  <p className="text-sm text-text-secondary mt-1">Choose your culling settings</p>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setIsConfiguringCull(false)}
                    className="text-xs text-text-secondary hover:text-text-primary underline underline-offset-4 transition-colors"
                  >
                    I want to cull manually
                  </button>
                  <button
                    onClick={() => {
                      setCullSettings(DEFAULT_CULL_SETTINGS);
                      setHighlightPercent(15);
                      setDuplicateRatio('moderate');
                    }}
                    className="rounded-full bg-surface border border-border-color px-3.5 py-1.5 text-xs font-medium text-text-primary hover:bg-card-active transition-colors"
                  >
                    Reset defaults
                  </button>
                  {localCullPlan && (
                    <button
                      onClick={() => setIsConfiguringCull(false)}
                      className="p-1 rounded-md text-text-secondary hover:text-text-primary hover:bg-surface"
                      title="Back to review grid"
                    >
                      <X size={18} />
                    </button>
                  )}
                </div>
              </div>

              {/* Selected folder badge */}
              <div className="mt-4 flex items-center justify-between rounded-lg border border-border-color bg-bg-primary px-3.5 py-2 text-xs">
                <div className="flex items-center gap-2 truncate text-text-secondary">
                  <FolderOpen size={15} className="text-accent shrink-0" />
                  <span className="font-mono text-text-primary truncate">{cullCatalogScope?.absoluteFolderPath || cullFolderPath}</span>
                  <span className="shrink-0 text-text-secondary">({cullImageList.length} photos)</span>
                </div>
                <button
                  onClick={() => void chooseCullFolder()}
                  className="shrink-0 font-medium text-accent hover:underline ml-2"
                >
                  Change folder
                </button>
              </div>

              {/* Wizard Form Rows */}
              <div className="mt-6 space-y-6">
                {/* Row 1: What type of shoot is this? */}
                <div className="grid grid-cols-1 md:grid-cols-[240px_1fr] items-center gap-3">
                  <label className="text-sm font-medium text-text-primary">
                    What type of shoot is this?
                  </label>
                  <div className="relative w-full max-w-sm">
                    <select
                      value={cullSettings.subjectMode}
                      onChange={(e) => {
                        const subjectMode = e.target.value as CullingSettings['subjectMode'];
                        setCullSettings((s) => ({
                          ...s,
                          subjectMode,
                          useSubjectDetection: subjectMode !== 'landscape',
                        }));
                      }}
                      className="h-10 w-full appearance-none rounded-lg border border-border-color bg-bg-primary pl-3.5 pr-10 text-sm text-text-primary outline-none focus:border-accent cursor-pointer shadow-xs"
                      style={{ colorScheme: 'dark' }}
                    >
                      <option value="people" className="bg-bg-secondary text-text-primary py-1.5">Family Portraits & People</option>
                      <option value="people" className="bg-bg-secondary text-text-primary py-1.5">Weddings & Engagements</option>
                      <option value="people" className="bg-bg-secondary text-text-primary py-1.5">Events & Parties</option>
                      <option value="wildlife" className="bg-bg-secondary text-text-primary py-1.5">Wildlife & Animals</option>
                      <option value="birds" className="bg-bg-secondary text-text-primary py-1.5">Birds & Nature</option>
                      <option value="landscape" className="bg-bg-secondary text-text-primary py-1.5">Landscape & Architecture</option>
                      <option value="general" className="bg-bg-secondary text-text-primary py-1.5">General / Commercial</option>
                    </select>
                    <ChevronDown size={16} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary" />
                  </div>
                </div>

                {/* Row 2: Threshold for culling blurred photos */}
                <div className="grid grid-cols-1 md:grid-cols-[240px_1fr] items-start gap-3 pt-3 border-t border-border-color/60">
                  <label className="text-sm font-medium text-text-primary pt-1">
                    Threshold for culling blurred photos
                  </label>
                  <div className="space-y-2">
                    <div className="flex flex-wrap gap-2">
                      {([
                        { label: 'Lenient', value: 70, isActive: cullSettings.blurThreshold <= 85 },
                        { label: 'Moderate', value: 100, isActive: cullSettings.blurThreshold > 85 && cullSettings.blurThreshold <= 120 },
                        { label: 'Strict', value: 140, isActive: cullSettings.blurThreshold > 120 },
                      ] as const).map((opt) => (
                        <button
                          key={opt.label}
                          onClick={() => setCullSettings((s) => ({ ...s, blurThreshold: opt.value, filterBlurry: true }))}
                          className={clsx(
                            'rounded-full px-5 py-1.5 text-xs font-semibold transition-all shadow-xs',
                            opt.isActive
                              ? 'bg-cyan-500 text-black ring-2 ring-cyan-400 font-bold'
                              : 'border border-border-color bg-bg-primary text-text-secondary hover:text-text-primary hover:bg-surface'
                          )}
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                    <p className="text-xs text-text-secondary italic">
                      {cullSettings.blurThreshold <= 85
                        ? 'Lenient: Only marks heavily blurred photos for review - recommended for fast motion or action.'
                        : cullSettings.blurThreshold > 120
                        ? 'Strict: Rejects any frames that are not tack-sharp - recommended for studio, formal portraits, and macro.'
                        : 'Moderate: Will include images that are slightly out of focus for your review - recommended for most shooting conditions.'}
                    </p>
                  </div>
                </div>

                {/* Row 3: Criteria for grouping Duplicates */}
                <div className="grid grid-cols-1 md:grid-cols-[240px_1fr] items-start gap-3 pt-3 border-t border-border-color/60">
                  <label className="text-sm font-medium text-text-primary pt-1">
                    Criteria for grouping Duplicates
                  </label>
                  <div className="space-y-2">
                    <div className="flex flex-wrap gap-2">
                      {([
                        { label: 'Identical', value: 12, isActive: cullSettings.similarityThreshold <= 18 },
                        { label: 'Similar', value: 28, isActive: cullSettings.similarityThreshold > 18 && cullSettings.similarityThreshold <= 33 },
                        { label: 'Similarish', value: 38, isActive: cullSettings.similarityThreshold > 33 && cullSettings.similarityThreshold <= 43 },
                        { label: 'Loose', value: 48, isActive: cullSettings.similarityThreshold > 43 },
                      ] as const).map((opt) => (
                        <button
                          key={opt.label}
                          onClick={() => setCullSettings((s) => ({ ...s, similarityThreshold: opt.value, groupSimilar: true }))}
                          className={clsx(
                            'rounded-full px-5 py-1.5 text-xs font-semibold transition-all shadow-xs',
                            opt.isActive
                              ? 'bg-cyan-500 text-black ring-2 ring-cyan-400 font-bold'
                              : 'border border-border-color bg-bg-primary text-text-secondary hover:text-text-primary hover:bg-surface'
                          )}
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                    <p className="text-xs text-text-secondary italic">
                      {cullSettings.similarityThreshold <= 18
                        ? 'Identical: Groups only burst sequences with near-identical camera framing and subject pose.'
                        : cullSettings.similarityThreshold <= 33
                        ? 'Similar: More Selected images - Changes in subject will create a new Duplicate set.'
                        : cullSettings.similarityThreshold <= 43
                        ? 'Similarish: Groups moderate variations in expressions and pose together.'
                        : 'Loose: Broadly groups all photos taken in the same scene/angle into duplicate sets.'}
                    </p>
                  </div>
                </div>

                {/* Row 4: Selections in each duplicate set */}
                <div className="grid grid-cols-1 md:grid-cols-[240px_1fr] items-start gap-3 pt-3 border-t border-border-color/60">
                  <label className="text-sm font-medium text-text-primary pt-1">
                    Selections in each duplicate set
                  </label>
                  <div className="space-y-2">
                    <div className="flex flex-wrap gap-2">
                      {(['More', 'Moderate', 'Less'] as const).map((opt) => (
                        <button
                          key={opt}
                          onClick={() => setDuplicateRatio(opt.toLowerCase() as any)}
                          className={clsx(
                            'rounded-full px-5 py-1.5 text-xs font-semibold transition-all shadow-xs',
                            duplicateRatio === opt.toLowerCase()
                              ? 'bg-cyan-500 text-black ring-2 ring-cyan-400'
                              : 'border border-border-color bg-bg-primary text-text-secondary hover:text-text-primary hover:bg-surface'
                          )}
                        >
                          {opt}
                        </button>
                      ))}
                    </div>
                    <p className="text-xs text-text-secondary italic">
                      {duplicateRatio === 'more'
                        ? 'More: Chooses top 35% images from each duplicate set for your selection.'
                        : duplicateRatio === 'less'
                        ? 'Less: Chooses only the single top 10% / best image from each set.'
                        : 'Moderate: Chooses top 20% images from a duplicate set.'}
                    </p>
                  </div>
                </div>

                {/* Row 5: Amount of Highlights */}
                <div className="grid grid-cols-1 md:grid-cols-[240px_1fr] items-start gap-3 pt-3 border-t border-border-color/60">
                  <label className="text-sm font-medium text-text-primary pt-1">
                    Amount of Highlights
                  </label>
                  <div className="space-y-2">
                    <div className="flex flex-wrap gap-2">
                      {([
                        { label: 'None', value: 0 },
                        { label: '10%', value: 10 },
                        { label: '15%', value: 15 },
                        { label: '20%', value: 20 },
                        { label: '25%', value: 25 },
                      ] as const).map((opt) => (
                        <button
                          key={opt.label}
                          onClick={() => setHighlightPercent(opt.value)}
                          className={clsx(
                            'rounded-full px-4 py-1.5 text-xs font-semibold transition-all shadow-xs',
                            highlightPercent === opt.value
                              ? 'bg-cyan-500 text-black ring-2 ring-cyan-400'
                              : 'border border-border-color bg-bg-primary text-text-secondary hover:text-text-primary hover:bg-surface'
                          )}
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                    <p className="text-xs text-text-secondary italic">
                      Highlights will pick the best standout images from the selected images for the chosen profile.
                    </p>
                  </div>
                </div>

                {/* Row 6: Enable / Disable AI features */}
                <div className="grid grid-cols-1 md:grid-cols-[240px_1fr] items-start gap-3 pt-3 border-t border-border-color/60">
                  <label className="text-sm font-medium text-text-primary pt-1">
                    Enable / Disable AI features
                  </label>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-1">
                    <Switch
                      label="Closed Eyes"
                      checked={cullSettings.useSubjectDetection}
                      onChange={(val) => setCullSettings((s) => ({ ...s, useSubjectDetection: val }))}
                    />
                    <Switch
                      label="Blur"
                      checked={cullSettings.filterBlurry}
                      onChange={(val) => setCullSettings((s) => ({ ...s, filterBlurry: val }))}
                    />
                    <Switch
                      label="Duplicates"
                      checked={cullSettings.groupSimilar}
                      onChange={(val) => setCullSettings((s) => ({ ...s, groupSimilar: val }))}
                    />
                  </div>
                </div>
              </div>

              {/* Progress if analyzing */}
              {isPlanningCull && cullProgress && (
                <div className="mt-6 rounded-lg border border-border-color bg-bg-primary p-4">
                  <div className="flex items-center justify-between text-xs text-text-secondary mb-1.5">
                    <span className="font-medium text-text-primary">{cullProgress.stage}</span>
                    <span className="tabular-nums font-mono">{cullProgress.current} / {cullProgress.total}</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-surface">
                    <div
                      className="h-full bg-cyan-400 transition-[width] duration-200"
                      style={{ width: `${cullProgress.total > 0 ? Math.min(100, (cullProgress.current / cullProgress.total) * 100) : 0}%` }}
                    />
                  </div>
                </div>
              )}

              {cullError && (
                <div className="mt-4 rounded-lg bg-red-500/10 border border-red-500/30 p-3 text-xs text-red-300">
                  {cullError}
                </div>
              )}

              {/* Footer Row */}
              <div className="mt-8 flex flex-wrap items-center justify-between gap-4 border-t border-border-color pt-5">
                <label className="flex items-center gap-2 text-xs text-text-secondary cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeSubfolders}
                    onChange={(e) => setIncludeSubfolders(e.target.checked)}
                    className="accent-cyan-400 h-4 w-4 rounded"
                  />
                  Include subfolders in culling analysis
                </label>
                <div className="flex items-center gap-3">
                  <button
                    disabled={isPlanningCull || (!cullFolderPath && imageList.length === 0)}
                    onClick={() => void startCullFromRail()}
                    className="inline-flex items-center justify-center gap-2 rounded-full bg-cyan-500 hover:bg-cyan-400 px-8 py-3 text-sm font-bold text-black shadow-lg hover:shadow-cyan-500/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isPlanningCull ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} fill="currentColor" />}
                    {isPlanningCull ? 'Analyzing shoot...' : 'Start Culling'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : displayCount === 0 || (localCullPlan && !isCompareMode) ? (
          <div className="w-full h-full overflow-y-auto p-3">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <Text variant={TextVariants.small} color={TextColors.secondary}>{cullImageList.length} images</Text>
                <button
                  className="rounded border border-border-color px-2.5 py-1 text-xs text-text-primary hover:bg-surface flex items-center gap-1.5 transition-colors"
                  onClick={() => setIsConfiguringCull(true)}
                >
                  <SlidersHorizontal size={13} />
                  <span>Culling Settings</span>
                </button>
              </div>
              <div className="flex items-center gap-2">
                {displayCount > 0 && (
                  <button className="rounded border border-border-color px-2 py-1 text-xs text-text-primary hover:bg-surface" onClick={() => setIsCompareMode(true)}>
                    {displayCount === 1 ? 'Inspect selected' : `Compare ${displayCount} selected`}
                  </button>
                )}
                <Text variant={TextVariants.small} color={TextColors.secondary}>Select frames to inspect or compare</Text>
              </div>
            </div>
            <div ref={cullGridRef} className="relative grid select-none grid-cols-[repeat(auto-fill,minmax(210px,1fr))] gap-3" onPointerDown={beginGridSelection} onPointerMove={extendGridSelection} onPointerUp={endGridSelection} onPointerCancel={endGridSelection} onClickCapture={(event) => { if (marqueeMoved.current) { event.preventDefault(); event.stopPropagation(); marqueeMoved.current = false; } }}>
              {marqueeBounds && <div className="pointer-events-none absolute z-20 border border-accent bg-accent/15" style={marqueeBounds} />}
              {cullImageList.map((image: ImageFile) => {
                const decision = cullDecisions[image.path];
                const finalistRank = finalistRanks.get(image.path);
                const selected = multiSelectedPaths.includes(image.path);
                const border = decision?.proposedStatus === 'keep' ? 'border-green-400/70' : decision ? 'border-red-400/60' : 'border-border-color';
                return <div key={image.path} data-cull-path={image.path} className={`relative overflow-hidden rounded-md border-2 ${selected ? 'ring-2 ring-accent' : ''} ${border}`}>{finalistRank && <span className="absolute left-2 top-2 z-20 rounded bg-cyan-500/90 px-1.5 py-0.5 text-[10px] font-medium text-white">{finalistRank === 1 ? 'Top finalist' : `Finalist #${finalistRank}`}</span>}<Thumbnail path={image.path} rating={imageRatings?.[image.path] || 0} tags={image.tags} aspectRatio={thumbnailAspectRatio} isEdited={image.is_edited} exif={image.exif} isCloudPlaceholder={image.is_cloud_placeholder} isActive={activePath === image.path} isSelected={selected} isForcedHover={false} onContextMenu={onContextMenu} onImageClick={onImageClick} onImageDoubleClick={onImageDoubleClick} onLoad={() => {}} /><div className="flex items-center justify-between gap-2 px-2 py-1.5 bg-bg-secondary text-xs"><span className="truncate">{image.path.split(/[\\/]/).pop()}</span><span className={decision?.proposedStatus === 'keep' ? 'text-green-300' : decision ? 'text-red-300' : 'text-text-secondary'}>{decision?.proposedStatus === 'keep' ? 'Selected' : decision ? 'Rejected' : ''}</span></div></div>;
              })}
            </div>
          </div>
        ) : (
          <div className="flex h-full w-full flex-col gap-2 p-2">
            <div className="flex shrink-0 items-center justify-between px-1">
              <Text variant={TextVariants.small} color={TextColors.secondary}>{displayCount === 1 ? 'Inspection view' : `Comparing ${displayCount} frames`}</Text>
              <div className="flex items-center gap-2">
                <button className="rounded border border-border-color px-2 py-1 text-xs text-text-primary hover:bg-surface flex items-center gap-1" onClick={() => setIsConfiguringCull(true)}>
                  <SlidersHorizontal size={13} />
                  <span>Settings</span>
                </button>
                {localCullPlan && <button className="rounded border border-border-color px-2 py-1 text-xs text-text-primary hover:bg-surface" onClick={() => setIsCompareMode(false)}>Back to review grid</button>}
              </div>
            </div>
            <div
              className={clsx(
              'grid min-h-0 flex-1 gap-2',
              displayCount === 1 && 'grid-cols-1 grid-rows-1',
              displayCount === 2 && 'grid-cols-2 grid-rows-1',
              displayCount === 3 && 'grid-cols-2 grid-rows-2',
              displayCount === 4 && 'grid-cols-2 grid-rows-2',
              displayCount === 5 && 'grid-cols-3 grid-rows-2',
              displayCount === 6 && 'grid-cols-3 grid-rows-2',
            )}
            >
            {displayImages.map((img: ImageFile, index: number) => (
              <CullingPreview
                key={img.path}
                image={img}
                rating={imageRatings?.[img.path] || 0}
                onContextMenu={onContextMenu}
                onImageDoubleClick={onImageDoubleClick}
                isActive={activePath === img.path}
                isSelected={true}
                isFullWidth={(displayCount === 3 && index === 2) || (displayCount === 5 && index === 4)}
                syncViewport={syncViewport}
                setSyncViewport={setSyncViewport}
                hoveredPath={hoveredCullingPath}
                setHoveredCullingPath={setHoveredCullingPath}
                showRateBar={showRateBar}
                setShowRateBar={setShowRateBar}
                showInfoBar={showInfoBar}
                setShowInfoBar={setShowInfoBar}
              />
            ))}
            </div>
          </div>
        )}
      </div>

      <div
        ref={containerRef}
        style={{ width: sidebarWidth }}
        className="relative shrink-0 border-l border-surface/50 bg-bg-secondary/50 flex flex-col"
        onClick={handleSidebarEmptyClick}
        onContextMenu={handleSidebarEmptyContextMenu}
      >
        <div className="shrink-0 border-b border-border-color p-3">
          <div className="mb-2 flex items-center justify-between gap-2">
            <Text variant={TextVariants.small} weight={TextWeights.semibold}>Culling session</Text>
            {isPlanningCull && <Loader2 size={14} className="animate-spin text-accent" />}
          </div>
          {cullFolderPath ? (
            <button
              className="w-full truncate rounded-md border border-border-color bg-bg-primary px-2 py-1.5 text-left text-xs text-text-primary hover:bg-surface"
              title={cullCatalogScope?.absoluteFolderPath || cullFolderPath}
              onClick={() => void chooseCullFolder()}
            >
              {cullCatalogScope?.absoluteFolderPath || cullFolderPath}
            </button>
          ) : (
            <div className="rounded-md border border-dashed border-border-color bg-bg-primary/50 px-2.5 py-1.5 text-xs text-text-secondary">
              Select a source from the center panel
            </div>
          )}
          {localCullPlan && (
            <>
              <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mt-2">
                {localCullPlan.rejectCount} frames marked for review out of {localCullPlan.totalCount}.
              </Text>
              <div className="mt-2 grid grid-cols-2 gap-2">
                <button className="rounded border border-border-color px-2 py-1.5 text-xs text-text-primary hover:bg-surface" onClick={selectAllCullCandidates}>
                  Select all
                </button>
                <button className="rounded border border-border-color px-2 py-1.5 text-xs text-text-secondary hover:bg-surface" onClick={onClearSelection}>
                  Clear selection
                </button>
              </div>
              {multiSelectedPaths.filter((path: string) => localCullPlan.items.some((item) => item.representativePath === path)).length > 0 && (
                <div className="mt-2 grid grid-cols-2 gap-2">
                  <button className="rounded border border-green-400/40 px-2 py-1.5 text-xs text-green-300 hover:bg-green-400/10" onClick={() => void updateSelectedCullDecisions(true)}>
                    Keep selected
                  </button>
                  <button className="rounded border border-red-400/40 px-2 py-1.5 text-xs text-red-300 hover:bg-red-400/10" onClick={() => void updateSelectedCullDecisions(false)}>
                    Reject selected
                  </button>
                </div>
              )}
              <button
                disabled={isApplyingCull || localCullPlan.rejectCount === 0}
                className="mt-2 inline-flex w-full items-center justify-center gap-2 rounded-md border border-red-500/50 px-3 py-2 text-sm text-red-200 hover:bg-red-500/10 disabled:opacity-50"
                onClick={() => void applyCullFromRail()}
              >
                {isApplyingCull ? <Loader2 size={15} className="animate-spin" /> : null}
                {isApplyingCull ? 'Moving rejected frames...' : `Move ${localCullPlan.rejectCount} rejected frames`}
              </button>
            </>
          )}
        </div>
        {activePath && cullDecisions[activePath] && (
          <div className="shrink-0 border-b border-border-color px-3 py-2">
            <div className="flex items-center justify-between gap-2">
              <Text variant={TextVariants.small} weight={TextWeights.semibold}>Cull decision</Text>
              <span className={cullDecisions[activePath].proposedStatus === 'keep' ? 'text-xs text-green-300' : 'text-xs text-red-300'}>{cullDecisions[activePath].proposedStatus === 'keep' ? 'Selected' : 'Rejected'}</span>
            </div>
            {finalistRanks.has(activePath) && <div className="mt-1 text-xs text-cyan-200">{finalistRanks.get(activePath) === 1 ? 'Top technical finalist' : `Technical finalist #${finalistRanks.get(activePath)}`}</div>}
            {localCullPlan && thumbnails[activePath] && <img className="mt-2 aspect-[4/3] w-full rounded-sm border border-border-color bg-surface object-contain" src={thumbnails[activePath]} alt="Selected culling frame" />}
            {localCullPlan && (
              <>
                <div className="mt-2 flex gap-2">
                  <button className="rounded bg-green-400/10 px-2 py-1 text-xs text-green-300 hover:bg-green-400/20" onClick={() => void updateLocalCullDecision(activePath, true, decisionFeedbackReason)}>Keep</button>
                  <button className="rounded bg-red-400/10 px-2 py-1 text-xs text-red-300 hover:bg-red-400/20" onClick={() => void updateLocalCullDecision(activePath, false, decisionFeedbackReason)}>Reject</button>
                </div>
                {localCullPlan.sessionId && (
                  <label className="mt-2 block text-xs text-text-secondary">
                    Why this choice?
                    <div className="relative mt-1">
                      <select
                        value={decisionFeedbackReason}
                        onChange={(event) => setDecisionFeedbackReason(event.target.value)}
                        className="h-8 w-full appearance-none rounded border border-border-color bg-bg-primary pl-2 pr-7 text-xs text-text-primary outline-none focus:border-accent cursor-pointer"
                        style={{ colorScheme: 'dark' }}
                      >
                        <option value="" className="bg-bg-secondary text-text-primary">No reason recorded</option>
                        {CULL_FEEDBACK_REASONS.map((reason) => (
                          <option key={reason} value={reason} className="bg-bg-secondary text-text-primary">{reason}</option>
                        ))}
                      </select>
                      <ChevronDown size={14} className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-text-secondary" />
                    </div>
                  </label>
                )}
              </>
            )}
            {activePlanItem && activePlanItem.decisionFactors.length > 0 && <div className="mt-3 space-y-2 border-t border-border-color pt-2">{activePlanItem.decisionFactors.map((factor) => <div key={factor.id} className="text-xs"><div className={factor.impact === 'reject' ? 'text-red-300' : 'text-text-primary'}>{factor.label}</div><div className="mt-0.5 text-text-secondary">{factor.detail}</div></div>)}</div>}
          </div>
        )}
        {localCullPlan && relatedDuplicatePaths.length > 0 && <div className="shrink-0 border-b border-border-color px-3 py-3"><div className="mb-2 flex items-center justify-between"><Text variant={TextVariants.small} weight={TextWeights.semibold}>Related duplicates</Text><span className="text-xs text-text-secondary">{relatedDuplicatePaths.length}</span></div><div className="grid grid-cols-3 gap-1.5">{relatedDuplicatePaths.slice(0, 6).map((path) => { const isRepresentative = path === duplicateRepresentativePath; return <button key={path} className={clsx('relative aspect-square overflow-hidden rounded-sm border bg-surface', isRepresentative ? 'border-green-400 ring-1 ring-green-400' : path === activePath ? 'border-accent ring-1 ring-accent' : 'border-border-color hover:border-text-secondary')} onClick={() => useLibraryStore.getState().setLibrary({ multiSelectedPaths: [path], libraryActivePath: path, selectionAnchorPath: path })} title={path}>{thumbnails[path] ? <img className="h-full w-full object-cover" src={thumbnails[path]} alt={isRepresentative ? 'Selected frame from duplicate group' : 'Related duplicate'} /> : <div className="h-full w-full animate-pulse bg-surface/70" />}{isRepresentative && <span className="absolute bottom-1 left-1 rounded bg-green-500/90 px-1 py-0.5 text-[10px] font-medium text-white">Selected</span>}</button>; })}</div></div>}
        {isCatalog && (activeFaces.length > 0 || (subjectEvidence?.aiTags.length ?? 0) > 0 || (subjectEvidence?.species.length ?? 0) > 0) && (
          <div className="shrink-0 border-b border-border-color px-3 py-3">
            <div className="mb-2 flex items-center justify-between">
              <Text variant={TextVariants.small} weight={TextWeights.semibold}>Key subjects</Text>
              <span className="text-xs text-text-secondary">Catalog evidence</span>
            </div>
            {(subjectEvidence?.aiTags.length || subjectEvidence?.species.length) ? <div className="mb-3 flex flex-wrap gap-1.5">
              {subjectEvidence?.species.filter((item) => item.reviewState !== 'rejected').slice(0, 3).map((item) => <span key={`${item.scientificName}-${item.confidence}`} className="rounded bg-cyan-400/15 px-1.5 py-0.5 text-xs text-cyan-200">{item.commonName || item.scientificName}</span>)}
              {subjectEvidence?.aiTags.filter((item) => item.reviewState !== 'rejected').slice(0, 6).map((item) => <span key={item.name} className="rounded bg-surface px-1.5 py-0.5 text-xs text-text-secondary">{item.name}</span>)}
            </div> : null}
            {activeFaces.length > 0 && <><Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mb-1.5">People detected</Text><div className="grid grid-cols-3 gap-1.5">
              {activeFaces.slice(0, 6).map((item) => {
                const source = item.thumbnailDataUrl || (item.cropPath ? convertFileSrc(item.cropPath) : null);
                return <div key={item.face.id} className="aspect-square overflow-hidden rounded-sm border border-border-color bg-surface">
                  {source ? <img className="h-full w-full object-cover" src={source} alt="Detected person" /> : <div className="h-full w-full animate-pulse bg-surface/70" />}
                </div>;
              })}
            </div></>}
          </div>
        )}
        <div
          onMouseDown={startResizing}
          className="absolute top-0 bottom-0 left-0 w-1 cursor-col-resize hover:bg-surface/50 active:bg-surface transition-colors z-40"
        />
        <div key={`${sidebarWidth}-${thumbnailAspectRatio}`} style={{ height: listHeight, width: '100%' }} className="min-h-0 flex-1">
          <List
            listRef={setListHandle}
            rowCount={cullImageList.length}
            rowHeight={sidebarWidth - 16}
            rowComponent={Row as unknown as (props: any) => React.ReactElement | null}
            rowProps={rowProps}
            className="custom-scrollbar"
          />
        </div>
      </div>
    </div>
  );
}
