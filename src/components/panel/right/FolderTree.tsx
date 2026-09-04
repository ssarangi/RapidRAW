import {
  Folder,
  FolderOpen,
  ChevronRight,
  ChevronUp,
  ChevronDown,
  Search,
  X,
  Album as AlbumIcon,
  Plus,
  Plane,
  Mountain,
  Sun,
  Camera,
  Map,
  Heart,
  Star,
  Users,
  User,
  Car,
  Briefcase,
  ArrowUpDown,
  Check,
  MoveRight,
  Database,
  RefreshCw,
  Loader2,
  BarChart3,
  ScanFace,
  Tags,
} from 'lucide-react';
import clsx from 'clsx';
import { motion, AnimatePresence, LayoutGroup } from 'framer-motion';
import { useState, useMemo, useEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { invoke } from '@tauri-apps/api/core';
import { useDroppable } from '@dnd-kit/core';
import { createPortal } from 'react-dom';
import { toast } from 'react-toastify';
import Text from '../../ui/Text';
import Button from '../../ui/Button';
import Checkbox from '../../ui/Checkbox';
import { TEXT_COLOR_KEYS, TextColors, TextVariants, TextWeights } from '../../../types/typography';
import { useShallow } from 'zustand/react/shallow';
import { useLibraryStore } from '../../../store/useLibraryStore';
import { useSettingsStore } from '../../../store/useSettingsStore';
import { useUIStore } from '../../../store/useUIStore';
import {
  AlbumItem,
  AlbumGroup,
  Album,
  CatalogFolderNode,
  CatalogAnalysisCoverage,
  CatalogRoot,
  CatalogSearchQuery,
  ImageFile,
  Invokes,
  FolderTreeSort,
  LibraryViewMode,
  SortDirection,
  SmartCollection,
  BackgroundJob,
} from '../../ui/AppProperties';
import { addLibraryCollection, browseCatalogRoot } from '../../../utils/libraryCollections';
import { CatalogAiAnalysisMenu, CatalogReviewQueueButton } from '../library/LibraryHeader';

export interface FolderTree {
  children: FolderTree[];
  isDir: boolean;
  name: string;
  path: string;
  imageCount?: number;
  hasSubdirs?: boolean;
  modified?: number | null;
  created?: number | null;
  faceCoverage?: CatalogAnalysisCoverage;
  ramPlusCoverage?: CatalogAnalysisCoverage;
  rootId?: number;
  relativePath?: string;
}

interface CatalogAnalysisJobs {
  face?: BackgroundJob;
  ramPlus?: BackgroundJob;
}

interface FolderTreeProps {
  isResizing: boolean;
  onContextMenu(event: any, path: string | null, isPinned?: boolean, isCatalogFolder?: boolean): void;
  onAlbumContextMenu(event: any, item: AlbumItem | null): void;
  onFolderSelect(folder: string): void;
  onSelectAlbum(albumId: string, albumName: string, images: string[]): void;
  onToggleFolder(folder: string): void;
  onOpenFolder(): void;
  style: any;
  isInstantTransition: boolean;
}

interface TreeNodeProps {
  sectionId: string;
  expandedFolders: Set<string>;
  isExpanded: boolean;
  node: FolderTree;
  onContextMenu(event: any, path: string, isPinned?: boolean, isCatalogFolder?: boolean): void;
  onFolderSelect(folder: string): void;
  onToggle(path: string): void;
  selectedPath: string | null;
  pinnedFolders: string[];
  showImageCounts: boolean;
  isInstantTransition: boolean;
  folderIcons: Record<string, string>;
  isLayoutDragging: boolean;
  isRescanning?: boolean;
  onRescan?(event: React.MouseEvent, node: FolderTree): void;
  analysisJobs?: CatalogAnalysisJobs;
  onAnalysisClick?(node: FolderTree): void;
}

const ACTIVE_JOB_STATES = new Set<BackgroundJob['state']>(['queued', 'running', 'paused', 'cancelling']);

function coverageLabel(label: string, coverage?: CatalogAnalysisCoverage, job?: BackgroundJob) {
  if (!coverage || coverage.total === 0) return null;
  const active = job && ACTIVE_JOB_STATES.has(job.state);
  const title = `${label}: ${coverage.completed}/${coverage.total} complete`;
  if (active) return { title: `${title} · ${job.message}`, className: 'text-sky-400', spinning: true };
  if (
    (job?.state === 'failed' && coverage.completed < coverage.total) ||
    (coverage.failed > 0 && coverage.completed === 0)
  ) {
    return { title: `${title} · ${coverage.failed} failed`, className: 'text-rose-400', spinning: false };
  }
  if (coverage.completed === coverage.total) return { title, className: 'text-emerald-400', spinning: false };
  if (coverage.completed > 0 || coverage.processing > 0) {
    return {
      title: `${title} · ${coverage.pending + coverage.failed} remaining`,
      className: 'text-amber-400',
      spinning: false,
    };
  }
  return { title: `${label}: not started`, className: 'text-slate-400', spinning: false };
}

function CatalogAnalysisModal({ node, onClose }: { node: FolderTree; onClose(): void }) {
  const [starting, setStarting] = useState<string | null>(null);
  const face = node.faceCoverage;
  const ramPlus = node.ramPlusCoverage;
  const scope = { rootId: node.rootId, relativePath: node.relativePath ?? '.' };
  const run = async (kind: 'face' | 'ramPlus', force: boolean) => {
    if (node.rootId == null) return;
    const key = `${kind}-${force ? 'all' : 'missing'}`;
    setStarting(key);
    try {
      if (kind === 'face') {
        await invoke(Invokes.StartFaceDetection, { ...scope, onlyPending: !force });
      } else {
        await invoke(Invokes.StartCatalogRamPlusTagging, { ...scope, force });
      }
      toast.success(`${kind === 'face' ? 'Face scan' : 'RAM++ tagging'} queued for ${node.name}.`);
      onClose();
    } catch (error) {
      toast.error(`Could not start analysis: ${error}`);
    } finally {
      setStarting(null);
    }
  };
  const coverage = (label: string, value?: CatalogAnalysisCoverage) => (
    <div className="rounded-md border border-border-color bg-bg-primary px-3 py-2.5">
      <div className="flex items-center justify-between gap-3 text-sm">
        <span>{label}</span>
        <span className="tabular-nums text-emerald-400">
          {value?.completed ?? 0}/{value?.total ?? 0}
        </span>
      </div>
      <div className="mt-1 text-xs text-text-secondary">
        {value?.processing ?? 0} running · {value?.pending ?? 0} not started · {value?.failed ?? 0} failed
      </div>
    </div>
  );
  const action = (kind: 'face' | 'ramPlus', force: boolean, label: string) => {
    const key = `${kind}-${force ? 'all' : 'missing'}`;
    return (
      <Button
        className="flex-1 bg-bg-secondary text-text-primary border border-border-color shadow-none hover:bg-card-active"
        disabled={starting !== null}
        onClick={() => void run(kind, force)}
      >
        {starting === key ? <Loader2 size={15} className="animate-spin" /> : null}
        {label}
      </Button>
    );
  };
  const content = (
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/45 p-4 backdrop-blur-xs"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={`Analysis status for ${node.name}`}
    >
      <div
        className="w-full max-w-lg rounded-lg border border-border-color bg-surface p-5 shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="mb-4">
          <Text variant={TextVariants.title}>Analysis status</Text>
          <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mt-1 truncate">
            {node.name} and its subfolders
          </Text>
        </div>
        <div className="space-y-2">
          {coverage('Face detection & recognition', face)}
          {coverage('RAM++ broad tags', ramPlus)}
        </div>
        <div className="mt-5 space-y-3">
          <div>
            <Text variant={TextVariants.small} weight={TextWeights.semibold}>
              Face detection & recognition
            </Text>
            <div className="mt-1.5 flex gap-2">
              {action('face', false, 'Scan & recognize missing')}
              {action('face', true, 'Re-scan & recognize all')}
            </div>
          </div>
          <div>
            <Text variant={TextVariants.small} weight={TextWeights.semibold}>
              RAM++ broad tags
            </Text>
            <div className="mt-1.5 flex gap-2">
              {action('ramPlus', false, 'Tag missing')}
              {action('ramPlus', true, 'Re-tag all')}
            </div>
          </div>
        </div>
        <div className="mt-5 flex justify-end">
          <Button className="bg-transparent text-text-secondary shadow-none" onClick={onClose}>
            Close
          </Button>
        </div>
      </div>
    </div>
  );
  return typeof document === 'undefined' ? content : createPortal(content, document.body);
}

interface VisibleProps {
  index: number;
  total: number;
}

const ALBUM_ICONS: Record<string, React.ElementType> = {
  plane: Plane,
  mountain: Mountain,
  sun: Sun,
  camera: Camera,
  map: Map,
  heart: Heart,
  star: Star,
  users: Users,
  user: User,
  car: Car,
  briefcase: Briefcase,
};

const filterTree = (node: FolderTree | null, query: string): FolderTree | null => {
  if (!node) {
    return null;
  }

  const lowerCaseQuery = query.toLowerCase();
  const isMatch = node.name.toLowerCase().includes(lowerCaseQuery);

  if (!node.children || node.children.length === 0) {
    return isMatch ? node : null;
  }

  const filteredChildren = node.children
    .map((child: FolderTree) => filterTree(child, query))
    .filter((child: FolderTree | null): child is FolderTree => child !== null);

  if (isMatch || filteredChildren.length > 0) {
    return { ...node, children: filteredChildren };
  }

  return null;
};

const getAutoExpandedPaths = (node: FolderTree, paths: Set<string>) => {
  if (node.children && node.children.length > 0) {
    paths.add(node.path);
    node.children.forEach((child: FolderTree) => getAutoExpandedPaths(child, paths));
  }
};

const filterAlbumTree = (node: AlbumItem | null, query: string): AlbumItem | null => {
  if (!node) return null;

  const lowerCaseQuery = query.toLowerCase();
  const isMatch = node.name.toLowerCase().includes(lowerCaseQuery);

  if (node.type === 'album') {
    return isMatch ? node : null;
  }

  if (node.type === 'group') {
    const filteredChildren = node.children
      .map((child: AlbumItem) => filterAlbumTree(child, query))
      .filter((child): child is AlbumItem => child !== null);

    if (isMatch || filteredChildren.length > 0) {
      return { ...node, children: filteredChildren };
    }
  }

  return null;
};

const getAutoExpandedAlbumGroups = (node: AlbumItem, groups: Set<string>) => {
  if (node.type === 'group' && node.children.length > 0) {
    groups.add(node.id);
    node.children.forEach((child) => getAutoExpandedAlbumGroups(child, groups));
  }
};

const sortFolderTree = (nodes: FolderTree[], sort: FolderTreeSort): FolderTree[] => {
  if (!nodes) return [];
  const sorted = [...nodes].sort((a, b) => {
    let comparison = 0;
    if (sort.key === 'name') comparison = a.name.localeCompare(b.name);
    else if (sort.key === 'modified') comparison = (a.modified || 0) - (b.modified || 0);
    else if (sort.key === 'created') comparison = (a.created || 0) - (b.created || 0);
    else if (sort.key === 'imageCount') comparison = (a.imageCount || 0) - (b.imageCount || 0);
    return sort.order === SortDirection.Ascending ? comparison : -comparison;
  });
  return sorted.map((node) => ({
    ...node,
    children: node.children && node.children.length > 0 ? sortFolderTree(node.children, sort) : node.children,
  }));
};

function FolderSortMenu({
  sort,
  onChange,
  isOpen,
  setIsOpen,
}: {
  sort: FolderTreeSort;
  onChange: (s: FolderTreeSort) => void;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}) {
  const menuRef = useRef<HTMLDivElement>(null);
  const { t } = useTranslation();

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setIsOpen(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [setIsOpen]);

  const options = [
    { key: 'name', label: t('library.folders.sort.name') },
    { key: 'created', label: t('library.folders.sort.created') },
    { key: 'modified', label: t('library.folders.sort.modified') },
    { key: 'imageCount', label: t('library.folders.sort.imageCount') },
  ];

  return (
    <div className="relative" ref={menuRef}>
      <button
        className={clsx(
          'bg-surface rounded-md hover:bg-card-active flex items-center justify-center shrink-0 overflow-hidden transition-colors w-9 h-9',
          isOpen && 'bg-card-active',
        )}
        onClick={() => setIsOpen(!isOpen)}
        data-tooltip={t('library.folders.tooltips.sortFolders')}
      >
        <ArrowUpDown size={16} className="text-text-secondary" />
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.1, ease: 'easeOut' }}
            className="absolute right-0 top-full mt-2 w-48 origin-top-right z-50"
          >
            <div className="bg-surface/90 backdrop-blur-md border border-border-color/50 rounded-lg shadow-xl p-2 flex flex-col">
              <div className="px-3 py-2 relative flex items-center">
                <Text as="div" variant={TextVariants.small} weight={TextWeights.semibold} className="uppercase">
                  {t('library.header.viewOptions.sortBy')}
                </Text>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onChange({
                      ...sort,
                      order:
                        sort.order === SortDirection.Ascending ? SortDirection.Descending : SortDirection.Ascending,
                    });
                  }}
                  data-tooltip={
                    sort.order === SortDirection.Ascending
                      ? t('library.header.viewOptions.sortDescending')
                      : t('library.header.viewOptions.sortAscending')
                  }
                  className="absolute top-1/2 right-3 -translate-y-1/2 p-1 bg-transparent border-none text-text-secondary hover:text-text-primary rounded-sm transition-colors"
                >
                  {sort.order === SortDirection.Ascending ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </button>
              </div>

              {options.map((opt) => {
                const isSelected = sort.key === opt.key;
                return (
                  <button
                    key={opt.key}
                    className={clsx(
                      'w-full text-left px-3 py-2 rounded-md flex items-center justify-between transition-colors duration-150',
                      isSelected ? 'bg-card-active' : 'hover:bg-bg-primary',
                    )}
                    onClick={() => {
                      if (sort.key !== opt.key) {
                        onChange({ key: opt.key as any, order: sort.order });
                      }
                      setIsOpen(false);
                    }}
                  >
                    <Text
                      variant={TextVariants.label}
                      color={TextColors.primary}
                      weight={isSelected ? TextWeights.semibold : TextWeights.normal}
                    >
                      {opt.label}
                    </Text>
                    {isSelected && <Check size={16} className={TEXT_COLOR_KEYS[TextColors.primary]} />}
                  </button>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function SectionHeader({
  title,
  isOpen,
  isLoading = false,
  onToggle,
  action,
}: {
  title: string;
  isOpen: boolean;
  isLoading?: boolean;
  onToggle: () => void;
  action?: ReactNode;
}) {
  const { t } = useTranslation();

  return (
    <div className="flex items-center w-full group">
      <Text
        as="div"
        variant={TextVariants.small}
        weight={TextWeights.bold}
        className="flex flex-1 min-w-0 items-center px-1 py-1.5 cursor-pointer"
        onClick={onToggle}
        data-tooltip={
          isOpen
            ? t('library.folders.collapseSection', { section: title })
            : t('library.folders.expandSection', { section: title })
        }
      >
        <div className="p-0.5 rounded-md transition-colors">
          {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </div>
        <span className="ml-1 uppercase tracking-wider select-none">{title}</span>
        {isLoading && <Loader2 size={13} className="ml-2 animate-spin text-text-secondary" />}
      </Text>
      {action && <div className="shrink-0 pr-1">{action}</div>}
    </div>
  );
}

const getAlbumImageCount = (item: any): number => {
  if (item.type === 'album' && item.images) {
    return item.images.length;
  }
  if (item.type === 'group' && item.children) {
    return item.children.reduce((sum: number, child: any) => sum + getAlbumImageCount(child), 0);
  }
  return 0;
};

function AlbumTreeNode({
  sectionId,
  item,
  expandedGroups,
  onToggle,
  onSelectAlbum,
  onContextMenu,
  selectedAlbumId,
  showImageCounts,
  isLayoutDragging,
}: {
  sectionId: string;
  item: AlbumItem;
  expandedGroups: Set<string>;
  onToggle: (id: string) => void;
  onSelectAlbum: (id: string, name: string, images: string[]) => void;
  onContextMenu: (e: any, item: AlbumItem) => void;
  selectedAlbumId: string | null;
  showImageCounts: boolean;
  isLayoutDragging: boolean;
}) {
  const isGroup = item.type === 'group';
  const isExpanded = expandedGroups.has(item.id);
  const isSelected = item.id === selectedAlbumId;
  const imageCount = getAlbumImageCount(item);

  const { setNodeRef, isOver, active } = useDroppable({
    id: `album-${sectionId}-${item.id}`,
    data: { type: 'album', id: item.id },
    disabled: isGroup || isLayoutDragging,
  });

  const isImageDrag = active?.data?.current?.type === 'library-image';
  const isDropTarget = isOver && isImageDrag && !isGroup;

  let ItemIcon: React.ElementType = isGroup ? (isExpanded ? FolderOpen : Folder) : AlbumIcon;
  if (item.icon && ALBUM_ICONS[item.icon]) {
    ItemIcon = ALBUM_ICONS[item.icon];
  }
  if (isDropTarget) {
    ItemIcon = MoveRight;
  }

  const iconKey = isDropTarget
    ? 'drop-target'
    : item.icon || (isGroup ? (isExpanded ? 'group-open' : 'group-closed') : 'album');

  return (
    <Text as="div" color={TextColors.primary} weight={TextWeights.medium}>
      <div
        ref={setNodeRef}
        className={clsx('flex items-center gap-2 p-1.5 rounded-md transition-colors cursor-pointer', {
          'bg-surface': isSelected && !isDropTarget,
          'hover:bg-card-active': !isSelected && !isDropTarget,
          'bg-accent/20': isDropTarget,
        })}
        onClick={() => (isGroup ? onToggle(item.id) : onSelectAlbum(item.id, item.name, (item as Album).images))}
        onContextMenu={(e) => onContextMenu(e, item)}
      >
        <div className="relative w-5 h-5 flex items-center justify-center p-0.5 rounded-sm text-text-secondary shrink-0">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={iconKey}
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.5 }}
              transition={{ duration: 0.15 }}
              className="absolute flex items-center justify-center"
            >
              <ItemIcon size={16} />
            </motion.div>
          </AnimatePresence>
        </div>

        <span onDoubleClick={() => isGroup && onToggle(item.id)} className="min-w-0 flex-1 select-none">
          <span className="block truncate">{item.name}</span>
        </span>

        {imageCount > 0 && (
          <Text
            as="span"
            variant={TextVariants.small}
            color={TextColors.secondary}
            className={clsx(
              'ml-auto min-w-8 shrink-0 text-right tabular-nums transition-opacity ease-in-out duration-300',
              showImageCounts ? 'opacity-100' : 'opacity-0',
            )}
          >
            {imageCount}
          </Text>
        )}

        {isGroup && (
          <div
            className="text-text-secondary p-0.5 rounded-sm hover:bg-surface/50"
            onClick={(e) => {
              e.stopPropagation();
              onToggle(item.id);
            }}
          >
            {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </div>
        )}
        {!isGroup && <div className="w-5 h-5 shrink-0" aria-hidden="true" />}
      </div>

      <AnimatePresence>
        {isGroup && isExpanded && (item as AlbumGroup).children.length > 0 && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="pl-1 border-l-[1.5px] border-border-color/50 ml-3.75 overflow-hidden"
          >
            <div className="py-1">
              <AnimatePresence>
                {(item as AlbumGroup).children.map((child) => (
                  <motion.div
                    key={`${sectionId}-${child.id}`}
                    initial={{ opacity: 0, height: 0, x: -10 }}
                    animate={{ opacity: 1, height: 'auto', x: 0 }}
                    exit={{ opacity: 0, height: 0, x: -10, overflow: 'hidden' }}
                    transition={{ duration: 0.2 }}
                  >
                    <AlbumTreeNode
                      sectionId={sectionId}
                      item={child}
                      expandedGroups={expandedGroups}
                      onToggle={onToggle}
                      onSelectAlbum={onSelectAlbum}
                      onContextMenu={onContextMenu}
                      selectedAlbumId={selectedAlbumId}
                      showImageCounts={showImageCounts}
                      isLayoutDragging={isLayoutDragging}
                    />
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Text>
  );
}

function TreeNode({
  sectionId,
  expandedFolders,
  isExpanded,
  node,
  onContextMenu,
  onFolderSelect,
  onToggle,
  selectedPath,
  pinnedFolders,
  showImageCounts,
  isInstantTransition,
  folderIcons,
  isLayoutDragging,
  isRescanning = false,
  onRescan,
  analysisJobs,
  onAnalysisClick,
}: TreeNodeProps) {
  const hasChildren = node.hasSubdirs || (node.children && node.children.length > 0);
  const isSelected = node.path === selectedPath;
  const isPinned = pinnedFolders.includes(node.path);

  const { setNodeRef, isOver, active } = useDroppable({
    id: `folder-${sectionId}-${node.path}`,
    data: { type: 'folder', path: node.path },
    disabled: isLayoutDragging,
  });

  const isImageDrag = active?.data?.current?.type === 'library-image';
  const isDropTarget = isOver && isImageDrag;

  const handleFolderIconClick = (e: any) => {
    e.stopPropagation();
    if (hasChildren) {
      onToggle(node.path);
    }
  };

  const handleNameClick = () => {
    if (hasChildren && !isExpanded && (!node.children || node.children.length === 0)) {
      onToggle(node.path);
    }
    onFolderSelect(node.path);
  };

  const handleNameDoubleClick = () => {
    if (hasChildren) {
      onToggle(node.path);
    }
  };

  const containerVariants: any = {
    closed: { height: 0, opacity: 0, transition: { duration: 0.2, ease: 'easeInOut' } },
    open: { height: 'auto', opacity: 1, transition: { duration: 0.25, ease: 'easeInOut' } },
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -15 },
    visible: ({ index, total }: VisibleProps) => ({
      opacity: 1,
      x: 0,
      transition: {
        duration: 0.25,
        delay: total < 8 ? index * 0.05 : 0,
      },
    }),
    exit: { opacity: 0, x: -15, transition: { duration: 0.2 } },
  };

  const currentFolderIconKey = folderIcons[node.path];
  let ResolvedIcon: React.ElementType = isExpanded ? FolderOpen : Folder;

  if (currentFolderIconKey && ALBUM_ICONS[currentFolderIconKey]) {
    ResolvedIcon = ALBUM_ICONS[currentFolderIconKey];
  }

  if (isDropTarget) {
    ResolvedIcon = MoveRight;
  }

  const iconKey = isDropTarget ? 'drop-target' : currentFolderIconKey || (isExpanded ? 'folder-open' : 'folder-closed');
  const faceStatus = coverageLabel('Face detection & recognition', node.faceCoverage, analysisJobs?.face);
  const ramPlusStatus = coverageLabel('RAM++ tags', node.ramPlusCoverage, analysisJobs?.ramPlus);

  return (
    <Text as="div" color={TextColors.primary} weight={TextWeights.medium}>
      <div
        ref={setNodeRef}
        className={clsx('flex items-center gap-2 p-1.5 rounded-md transition-colors cursor-pointer', {
          'bg-surface': isSelected && !isDropTarget,
          'hover:bg-card-active': !isSelected && !isDropTarget,
          'bg-accent/20': isDropTarget,
        })}
        onClick={handleNameClick}
        onContextMenu={(e: any) => onContextMenu(e, node.path, isPinned)}
      >
        <div
          className={clsx(
            'relative w-5 h-5 flex items-center justify-center p-0.5 rounded-sm text-text-secondary transition-colors shrink-0',
            {
              'hover:bg-surface-hover': !isSelected && hasChildren && !isDropTarget,
            },
          )}
          onClick={handleFolderIconClick}
        >
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={iconKey}
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.5 }}
              transition={{ duration: 0.15 }}
              className="absolute flex items-center justify-center"
            >
              <ResolvedIcon size={16} />
            </motion.div>
          </AnimatePresence>
        </div>

        <span onDoubleClick={handleNameDoubleClick} className="min-w-0 flex-1 select-none">
          <span className="block truncate">{node.name}</span>
        </span>

        {(faceStatus || ramPlusStatus) && (
          <span className="flex shrink-0 items-center gap-1" aria-label="Catalog analysis coverage">
            {faceStatus && (
              <button
                className={clsx('rounded-sm p-0.5 hover:bg-surface', faceStatus.className)}
                data-tooltip={`${faceStatus.title} · Open analysis status`}
                onClick={(event) => {
                  event.stopPropagation();
                  onAnalysisClick?.(node);
                }}
                type="button"
              >
                <ScanFace size={17} className={faceStatus.spinning ? 'animate-spin' : ''} />
              </button>
            )}
            {ramPlusStatus && (
              <button
                className={clsx('rounded-sm p-0.5 hover:bg-surface', ramPlusStatus.className)}
                data-tooltip={`${ramPlusStatus.title} · Open analysis status`}
                onClick={(event) => {
                  event.stopPropagation();
                  onAnalysisClick?.(node);
                }}
                type="button"
              >
                <Tags size={16} className={ramPlusStatus.spinning ? 'animate-spin' : ''} />
              </button>
            )}
          </span>
        )}

        {typeof node.imageCount === 'number' && node.imageCount > 0 && (
          <Text
            as="span"
            variant={TextVariants.small}
            color={TextColors.secondary}
            className={clsx(
              'ml-auto min-w-8 shrink-0 text-right tabular-nums transition-opacity ease-in-out duration-300',
              showImageCounts ? 'opacity-100' : 'opacity-0',
            )}
          >
            {node.imageCount}
          </Text>
        )}

        {onRescan && (
          <button
            aria-label={`Rescan ${node.name}`}
            className="shrink-0 rounded-sm p-1 text-text-secondary opacity-0 transition-opacity hover:bg-surface hover:text-text-primary group-hover:opacity-100 focus:opacity-100"
            data-tooltip="Rescan folder"
            disabled={isRescanning}
            onClick={(event) => {
              event.stopPropagation();
              onRescan(event, node);
            }}
            type="button"
          >
            {isRescanning ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
          </button>
        )}

        {hasChildren && (
          <Text
            as="div"
            color={TextColors.secondary}
            className="p-0.5 rounded-sm hover:bg-surface/50"
            onClick={handleFolderIconClick}
          >
            {isExpanded ? <ChevronUp size={16} className="shrink-0" /> : <ChevronDown size={16} className="shrink-0" />}
          </Text>
        )}
        {!hasChildren && <div className="w-5 h-5 shrink-0" aria-hidden="true" />}
      </div>

      <AnimatePresence initial={false}>
        {hasChildren && isExpanded && node.children && node.children.length > 0 && (
          <motion.div
            animate="open"
            className="pl-1 border-l-[1.5px] border-border-color/50 ml-3.75 overflow-hidden"
            exit="closed"
            initial={isInstantTransition ? 'open' : 'closed'}
            key="children-container"
            variants={containerVariants}
          >
            <div className="py-1">
              <AnimatePresence>
                {node?.children?.map((childNode: any, index: number) => (
                  <motion.div
                    animate="visible"
                    custom={{ index, total: node.children.length }}
                    exit="exit"
                    initial={isInstantTransition ? 'visible' : 'hidden'}
                    key={`${sectionId}-${childNode.path}`}
                    layout={isInstantTransition ? false : 'position'}
                    variants={itemVariants}
                  >
                    <TreeNode
                      sectionId={sectionId}
                      expandedFolders={expandedFolders}
                      isExpanded={expandedFolders.has(childNode.path)}
                      node={childNode}
                      onContextMenu={onContextMenu}
                      onFolderSelect={onFolderSelect}
                      onToggle={onToggle}
                      selectedPath={selectedPath}
                      pinnedFolders={pinnedFolders}
                      showImageCounts={showImageCounts}
                      isInstantTransition={isInstantTransition}
                      folderIcons={folderIcons}
                      isLayoutDragging={isLayoutDragging}
                      analysisJobs={analysisJobs}
                      onAnalysisClick={onAnalysisClick}
                    />
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Text>
  );
}

export default function FolderTree({
  isResizing,
  onContextMenu,
  onAlbumContextMenu,
  onFolderSelect,
  onSelectAlbum,
  onToggleFolder,
  onOpenFolder,
  style,
  isInstantTransition,
}: FolderTreeProps) {
  const { t } = useTranslation();
  const { appSettings, handleSettingsChange } = useSettingsStore(
    useShallow((state) => ({
      appSettings: state.appSettings,
      handleSettingsChange: state.handleSettingsChange,
    })),
  );

  const isLayoutDragging = useUIStore((state) => !!state.activeLayoutDragItem);

  const {
    folderTrees,
    pinnedFolderTrees,
    currentFolderPath: selectedPath,
    expandedFolders,
    isTreeLoading: isLoading,
    isRootFoldersLoading,
    isPinnedFoldersLoading,
    albumTree,
    activeAlbumId,
    expandedAlbumGroups,
    librarySource,
    catalogRoots,
    activeCatalogRootId,
  } = useLibraryStore(
    useShallow((state) => ({
      folderTrees: state.folderTrees,
      pinnedFolderTrees: state.pinnedFolderTrees,
      currentFolderPath: state.currentFolderPath,
      expandedFolders: state.expandedFolders,
      isTreeLoading: state.isTreeLoading,
      isRootFoldersLoading: state.isRootFoldersLoading,
      isPinnedFoldersLoading: state.isPinnedFoldersLoading,
      albumTree: state.albumTree,
      activeAlbumId: state.activeAlbumId,
      expandedAlbumGroups: state.expandedAlbumGroups,
      librarySource: state.librarySource,
      catalogRoots: state.catalogRoots,
      activeCatalogRootId: state.activeCatalogRootId,
    })),
  );

  const [searchQuery, setSearchQuery] = useState('');
  const [isHovering, setIsHovering] = useState(false);
  const [isSortMenuOpen, setIsSortMenuOpen] = useState(false);
  const [catalogFolderTrees, setCatalogFolderTrees] = useState<Record<number, CatalogFolderNode>>({});
  const [loadingCatalogTreeIds, setLoadingCatalogTreeIds] = useState<Set<number>>(new Set());
  const [catalogAnalysisJobs, setCatalogAnalysisJobs] = useState<Record<number, CatalogAnalysisJobs>>({});
  const [analysisNode, setAnalysisNode] = useState<FolderTree | null>(null);
  const [rescanningFolders, setRescanningFolders] = useState<Set<string>>(new Set());
  const [smartCollections, setSmartCollections] = useState<SmartCollection[]>([]);
  const [isLoadingSmartCollections, setIsLoadingSmartCollections] = useState(false);
  const pinnedFolders = appSettings?.pinnedFolders || [];
  const openSections = appSettings?.openTreeSections ?? ['pinned', 'current'];
  const showImageCounts = appSettings?.enableFolderImageCounts ?? false;
  const folderIcons = appSettings?.folderIcons || {};
  const folderTreeSort: FolderTreeSort = appSettings?.folderTreeSort || { key: 'name', order: SortDirection.Ascending };
  const includeSubfolderImages = appSettings?.libraryViewMode === LibraryViewMode.Recursive;
  const showHeaderButtons = isHovering || isSortMenuOpen;

  useEffect(() => {
    invoke(Invokes.GetAlbums).then((res: any) => useLibraryStore.getState().setLibrary({ albumTree: res }));
  }, []);

  const toggleSection = (section: string) => {
    if (appSettings) {
      const isOpen = openSections.includes(section);
      const newSections = isOpen ? openSections.filter((s) => s !== section) : [...openSections, section];

      handleSettingsChange({ ...appSettings, openTreeSections: newSections });
    }
  };

  const handleIncludeSubfoldersChange = async (checked: boolean) => {
    if (!appSettings) return;
    await handleSettingsChange({
      ...appSettings,
      libraryViewMode: checked ? LibraryViewMode.Recursive : LibraryViewMode.Flat,
    });
    if (selectedPath?.startsWith('LibraryFolder:')) {
      const selection = resolveCatalogFolderSelection(selectedPath);
      if (selection) {
        handleBrowseCatalogRoot(selection.root, selection.relativePath, checked);
      }
    } else if (selectedPath && !selectedPath.startsWith('Album: ') && !selectedPath.startsWith('Library: ')) {
      onFolderSelect(selectedPath);
    }
  };

  const handleEmptyAreaContextMenu = (e: any) => {
    if (e.target === e.currentTarget) {
      onContextMenu(e, null, false);
    }
  };

  const toggleAlbumGroup = (id: string) => {
    useLibraryStore.getState().setLibrary((state) => {
      const next = new Set(state.expandedAlbumGroups);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return { expandedAlbumGroups: next };
    });
  };

  const trimmedQuery = searchQuery.trim();
  const isSearching = trimmedQuery.length > 1;

  const filteredTrees = useMemo(() => {
    let base = folderTrees;
    if (isSearching) {
      base = base.map((tree: any) => filterTree(tree, trimmedQuery)).filter((t: any) => t !== null);
    }
    return sortFolderTree(base, folderTreeSort);
  }, [folderTrees, trimmedQuery, isSearching, folderTreeSort]);

  const filteredPinnedTrees = useMemo(() => {
    let base = pinnedFolderTrees;
    if (isSearching) {
      base = base.map((pinnedTree) => filterTree(pinnedTree, trimmedQuery)).filter((t): t is FolderTree => t !== null);
    }
    return sortFolderTree(base, folderTreeSort);
  }, [pinnedFolderTrees, trimmedQuery, isSearching, folderTreeSort]);

  const searchAutoExpandedFolders = useMemo(() => {
    if (!isSearching) return new Set<string>();
    const newExpanded = new Set<string>();
    filteredTrees.forEach((t: any) => getAutoExpandedPaths(t, newExpanded));
    filteredPinnedTrees.forEach((pinned) => getAutoExpandedPaths(pinned, newExpanded));
    return newExpanded;
  }, [isSearching, filteredTrees, filteredPinnedTrees]);

  const effectiveExpandedFolders = useMemo(() => {
    return new Set([...expandedFolders, ...searchAutoExpandedFolders]);
  }, [expandedFolders, searchAutoExpandedFolders]);

  const filteredAlbumTree = useMemo(() => {
    let base: AlbumItem[] = albumTree;
    if (isSearching) {
      base = base
        .map((item: any) => filterAlbumTree(item, trimmedQuery))
        .filter((item: AlbumItem | null): item is AlbumItem => item !== null);
    }
    return base;
  }, [albumTree, trimmedQuery, isSearching]);

  const searchAutoExpandedAlbumGroups = useMemo(() => {
    if (!isSearching) return new Set<string>();
    const newExpanded = new Set<string>();
    filteredAlbumTree.forEach((t: any) => getAutoExpandedAlbumGroups(t, newExpanded));
    return newExpanded;
  }, [isSearching, filteredAlbumTree]);

  const effectiveExpandedAlbumGroups = useMemo(() => {
    return new Set([...expandedAlbumGroups, ...searchAutoExpandedAlbumGroups]);
  }, [expandedAlbumGroups, searchAutoExpandedAlbumGroups]);

  useEffect(() => {
    if (isSearching && appSettings) {
      const hasPinnedResults = filteredPinnedTrees && filteredPinnedTrees.length > 0;
      const hasBaseResults = filteredTrees && filteredTrees.length > 0;
      const hasAlbumResults = filteredAlbumTree && filteredAlbumTree.length > 0;

      const newSections = [...openSections];
      let changed = false;

      if (hasPinnedResults && !newSections.includes('pinned')) {
        newSections.push('pinned');
        changed = true;
      }
      if (hasBaseResults && !newSections.includes('current')) {
        newSections.push('current');
        changed = true;
      }
      if (hasAlbumResults && !newSections.includes('albums')) {
        newSections.push('albums');
        changed = true;
      }

      if (changed) {
        handleSettingsChange({ ...appSettings, openTreeSections: newSections });
      }
    }
  }, [
    isSearching,
    filteredTrees,
    filteredPinnedTrees,
    filteredAlbumTree,
    openSections,
    handleSettingsChange,
    appSettings,
  ]);

  const isPinnedOpen = openSections.includes('pinned');
  const isCurrentOpen = openSections.includes('current');
  const isAlbumsOpen = openSections.includes('albums');
  const isCatalogOpen = openSections.includes('catalog');

  const hasVisiblePinnedTrees = filteredPinnedTrees && filteredPinnedTrees.length > 0;
  const showPinnedSection =
    hasVisiblePinnedTrees || (!isSearching && (pinnedFolders.length > 0 || isPinnedFoldersLoading));
  const hasVisibleAlbums = filteredAlbumTree && filteredAlbumTree.length > 0;
  const hasVisibleCatalogRoots = librarySource.type === 'catalog' && catalogRoots.length > 0 && !isSearching;
  // Shown whenever a catalog library is open, even with zero collections yet,
  // so there's always a way to add the first one from the sidebar.
  const showCatalogSection = librarySource.type === 'catalog' && !isSearching;
  const showAlbumsSection = hasVisibleAlbums || (!isSearching && albumTree.length === 0);

  // The currently-open image's folder can come from somewhere the tree
  // doesn't already show it - e.g. opened via "Open with RapidRAW" or the
  // external-editor protocol, which selects an image without first
  // navigating the tree to its folder. When that happens, surface it here
  // instead of silently leaving Sources pointed at whatever was last browsed.
  const adHocCurrentFolder = useMemo(() => {
    if (!selectedPath || isSearching) return null;
    if (selectedPath.startsWith('LibraryFolder:') || selectedPath.startsWith('Library: ')) return null;
    const knownRootPaths = [...folderTrees, ...pinnedFolderTrees].map((tree: any) => tree.path as string);
    const alreadyVisible = knownRootPaths.some(
      (rootPath) =>
        selectedPath === rootPath ||
        selectedPath.startsWith(`${rootPath}/`) ||
        selectedPath.startsWith(`${rootPath}\\`),
    );
    if (alreadyVisible) return null;
    const separator = selectedPath.includes('/') ? '/' : '\\';
    const name = selectedPath.split(separator).filter(Boolean).pop() || selectedPath;
    return { path: selectedPath, name };
  }, [selectedPath, isSearching, folderTrees, pinnedFolderTrees]);

  const loadCatalogFolderTree = async (root: CatalogRoot, force = false) => {
    const cachedTree = catalogFolderTrees[root.id];
    if (!force && cachedTree && cachedTree.imageCount === root.imageCount) return;
    if (loadingCatalogTreeIds.has(root.id)) return;
    setLoadingCatalogTreeIds((prev) => new Set(prev).add(root.id));
    try {
      const tree = await invoke<CatalogFolderNode>(Invokes.ListCatalogFolderTree, { rootId: root.id });
      setCatalogFolderTrees((prev) => ({ ...prev, [root.id]: tree }));
      useLibraryStore.getState().setLibrary((state) => {
        const next = new Set(state.expandedFolders);
        next.add(tree.path);
        return { expandedFolders: next };
      });
    } catch (err) {
      console.error('Failed to load catalog folder tree:', err);
      toast.error(`Failed to load catalog folders: ${err}`);
    } finally {
      setLoadingCatalogTreeIds((prev) => {
        const next = new Set(prev);
        next.delete(root.id);
        return next;
      });
    }
  };

  useEffect(() => {
    if (librarySource.type !== 'catalog') {
      setCatalogFolderTrees({});
      setLoadingCatalogTreeIds(new Set());
      return;
    }
    catalogRoots.forEach((root) => {
      loadCatalogFolderTree(root);
    });
  }, [librarySource.type, catalogRoots]);

  useEffect(() => {
    if (librarySource.type !== 'catalog') {
      setCatalogAnalysisJobs({});
      return;
    }

    let cancelled = false;
    const refreshCoverage = async () => {
      try {
        const jobs = await invoke<BackgroundJob[]>(Invokes.ListBackgroundJobs);
        if (cancelled) return;
        const next: Record<number, CatalogAnalysisJobs> = {};
        for (const root of catalogRoots) {
          const relevant = jobs.filter(
            (job) =>
              (job.kind === 'face_detection' ||
                job.kind === 'face_recognition' ||
                job.kind === 'face_reindex' ||
                job.kind === 'ram_plus_tagging') &&
              (job.rootId == null || job.rootId === root.id),
          );
          const face = relevant.find(
            (job) => job.kind === 'face_detection' || job.kind === 'face_recognition' || job.kind === 'face_reindex',
          );
          const ramPlus = relevant.find((job) => job.kind === 'ram_plus_tagging');
          if (face || ramPlus) next[root.id] = { face, ramPlus };
        }
        setCatalogAnalysisJobs(next);
        await Promise.all(catalogRoots.map((root) => loadCatalogFolderTree(root, true)));
      } catch (error) {
        console.error('Failed to refresh catalog analysis coverage:', error);
      }
    };

    void refreshCoverage();
    const interval = window.setInterval(() => void refreshCoverage(), 3000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [librarySource.type, catalogRoots]);

  const loadSmartCollections = async () => {
    if (librarySource.type !== 'catalog') {
      setSmartCollections([]);
      return;
    }
    setIsLoadingSmartCollections(true);
    try {
      setSmartCollections(await invoke<SmartCollection[]>(Invokes.ListSmartCollections));
    } catch (error) {
      console.error('Failed to load smart collections:', error);
    } finally {
      setIsLoadingSmartCollections(false);
    }
  };

  useEffect(() => {
    void loadSmartCollections();
    const refresh = () => {
      void loadSmartCollections();
    };
    window.addEventListener('smart-collections-changed', refresh);
    return () => window.removeEventListener('smart-collections-changed', refresh);
  }, [librarySource.type]);

  const resolveCatalogFolderSelection = (path: string) => {
    const match = /^LibraryFolder:(\d+):(.*)$/.exec(path);
    if (!match) return null;
    const rootId = Number(match[1]);
    const relativePath = match[2] || '.';
    const root = catalogRoots.find((candidate) => candidate.id === rootId);
    if (!root) return null;
    return { root, relativePath };
  };

  const handleBrowseCatalogRoot = async (root: CatalogRoot, relativePath = '.', recursiveOverride?: boolean) => {
    const recursive = recursiveOverride ?? appSettings?.libraryViewMode === LibraryViewMode.Recursive;
    try {
      await browseCatalogRoot(root, { relativePath, recursive });
    } catch (err) {
      console.error('Failed to load catalog images:', err);
    }
  };

  const handleCatalogFolderSelect = (path: string) => {
    const selection = resolveCatalogFolderSelection(path);
    if (!selection) return;
    handleBrowseCatalogRoot(selection.root, selection.relativePath);
  };

  const handleSmartCollectionSelect = async (collection: SmartCollection) => {
    let query: CatalogSearchQuery;
    try {
      query = JSON.parse(collection.queryJson) as CatalogSearchQuery;
    } catch {
      toast.error(`Smart collection "${collection.name}" has an invalid query.`);
      return;
    }
    useLibraryStore.getState().setLibrary({ isViewLoading: true });
    try {
      const files = await invoke<ImageFile[]>(Invokes.SearchCatalogImages, { query });
      const imageRatings: Record<string, number> = {};
      files.forEach((file) => {
        imageRatings[file.path] = file.rating || 0;
      });
      const root = catalogRoots.find((candidate) => candidate.id === query.rootId) || catalogRoots[0] || null;
      useLibraryStore.getState().setLibrary({
        rootPaths: root ? [root.absolutePath] : useLibraryStore.getState().rootPaths,
        currentFolderPath: `Library: ${collection.name}`,
        activeAlbumId: null,
        activeCatalogRootId: root?.id ?? null,
        imageList: files,
        imageRatings,
        multiSelectedPaths: [],
        libraryActivePath: null,
        libraryScrollTop: 0,
      });
      useLibraryStore.getState().setSearchCriteria({ text: '', tags: [], mode: 'OR' });
      useUIStore.getState().setUI({ activeView: 'library' });
    } catch (error) {
      console.error('Failed to apply smart collection:', error);
      toast.error(`Failed to apply smart collection: ${error}`);
    } finally {
      useLibraryStore.getState().setLibrary({ isViewLoading: false });
    }
  };

  const handleCatalogContextMenu = (event: any, virtualPath: string) => {
    const selection = resolveCatalogFolderSelection(virtualPath);
    if (!selection) {
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    const folderPath =
      selection.relativePath === '.'
        ? selection.root.absolutePath
        : `${selection.root.absolutePath}/${selection.relativePath}`;
    onContextMenu(event, folderPath, false, true);
  };

  const handleRescanCatalogRoot = async (event: React.MouseEvent, root: CatalogRoot) => {
    event.stopPropagation();
    const rescanKey = `catalog:${root.id}`;
    if (rescanningFolders.has(rescanKey)) return;
    setRescanningFolders((current) => new Set(current).add(rescanKey));
    useLibraryStore.getState().setLibrary({ isTreeLoading: true });
    try {
      await invoke(Invokes.StartCatalogScan, { rootId: root.id, recursive: true });
      toast.success(`Scanning ${root.label || root.absolutePath} for new files.`);
    } finally {
      useLibraryStore.getState().setLibrary({ isTreeLoading: false });
      setRescanningFolders((current) => {
        const next = new Set(current);
        next.delete(rescanKey);
        return next;
      });
    }
  };

  const handleRescanSourceFolder = async (event: React.MouseEvent, node: FolderTree) => {
    event.stopPropagation();
    const rescanKey = `source:${node.path}`;
    if (rescanningFolders.has(rescanKey)) return;
    setRescanningFolders((current) => new Set(current).add(rescanKey));
    try {
      await Promise.resolve(onFolderSelect(node.path));
      const refreshedTree = await invoke<FolderTree>(Invokes.GetFolderTree, {
        path: node.path,
        expandedFolders: Array.from(useLibraryStore.getState().expandedFolders),
        showImageCounts,
      });
      useLibraryStore.getState().setLibrary((state) => ({
        folderTrees: state.folderTrees.map((tree) => (tree.path === node.path ? refreshedTree : tree)),
        pinnedFolderTrees: state.pinnedFolderTrees.map((tree) => (tree.path === node.path ? refreshedTree : tree)),
      }));
      toast.success(`Rescanned ${node.name}.`);
    } catch (error) {
      toast.error(`Failed to rescan ${node.name}: ${error}`);
    } finally {
      setRescanningFolders((current) => {
        const next = new Set(current);
        next.delete(rescanKey);
        return next;
      });
    }
  };

  const [isAddingCollection, setIsAddingCollection] = useState(false);
  const handleAddCollection = async (event: React.MouseEvent) => {
    event.stopPropagation();
    if (isAddingCollection) return;
    setIsAddingCollection(true);
    try {
      const { root, cancelled } = await addLibraryCollection();
      if (!cancelled && root) {
        toast.success(`Added "${root.absolutePath}" - scanning now.`);
        if (!isCatalogOpen) toggleSection('catalog');
      }
    } catch (err) {
      toast.error(`Failed to add collection: ${err}`);
    } finally {
      setIsAddingCollection(false);
    }
  };

  return (
    <div
      className={clsx(
        'relative bg-bg-secondary rounded-lg shrink-0 flex flex-col h-full',
        !isResizing && 'transition-[width] duration-300 ease-in-out',
      )}
      style={style}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
    >
      <div className="p-3 flex justify-between items-center shrink-0 border-b border-surface">
        <Text variant={TextVariants.title}>{t('library.folders.sourcesTitle', 'Sources')}</Text>
        {librarySource.type === 'catalog' && (
          <div className="flex items-center">
            <CatalogAiAnalysisMenu compact />
            <CatalogReviewQueueButton compact />
            <button
              className="p-2 rounded-md text-text-secondary hover:bg-surface hover:text-text-primary"
              onClick={() => useUIStore.getState().setUI({ activeView: 'people' })}
              data-tooltip="People"
            >
              <Users size={17} />
            </button>
            <button
              className="p-2 rounded-md text-text-secondary hover:bg-surface hover:text-text-primary"
              onClick={() => useUIStore.getState().setUI({ activeView: 'insights' })}
              data-tooltip="Insights"
            >
              <BarChart3 size={17} />
            </button>
          </div>
        )}
      </div>

      <div className="p-2 flex flex-col flex-1 min-h-0">
        <div className="pt-1 pb-2">
          <div className="flex items-center">
            <div className="relative flex-1 min-w-0">
              <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-secondary" />
              <input
                type="text"
                placeholder={t('library.folders.searchPlaceholder')}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-surface border border-transparent rounded-md pl-9 pr-8 py-2 text-sm focus:outline-hidden truncate"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded-full hover:bg-card-active"
                  data-tooltip={t('library.folders.tooltips.clearSearch')}
                >
                  <X size={16} className="text-text-secondary" />
                </button>
              )}
            </div>

            <AnimatePresence>
              {showHeaderButtons && (
                <motion.div
                  initial={{ width: 0, opacity: 0, marginLeft: 0 }}
                  animate={{ width: 'auto', opacity: 1, marginLeft: 4 }}
                  exit={{ width: 0, opacity: 0, marginLeft: 0 }}
                  transition={{ duration: 0.2, ease: 'easeInOut' }}
                  className={clsx(
                    'flex items-center shrink-0',
                    isSortMenuOpen ? 'overflow-visible' : 'overflow-hidden',
                  )}
                >
                  <FolderSortMenu
                    sort={folderTreeSort}
                    onChange={(newSort) => {
                      if (appSettings) handleSettingsChange({ ...appSettings, folderTreeSort: newSort });
                    }}
                    isOpen={isSortMenuOpen}
                    setIsOpen={setIsSortMenuOpen}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          <div className="mt-2 px-2 py-1.5">
            <Checkbox
              label="Show images inside subfolders"
              checked={includeSubfolderImages}
              onChange={(checked) => handleIncludeSubfoldersChange(checked)}
            />
          </div>
        </div>

        <LayoutGroup id="folder-tree">
          <div className="flex-1 overflow-y-auto" onContextMenu={handleEmptyAreaContextMenu}>
            {adHocCurrentFolder && (
              <div>
                <SectionHeader
                  title={t('library.folders.sections.currentFolder', 'Current Folder')}
                  isOpen={true}
                  onToggle={() => {}}
                />
                <div className="pt-1 pb-2">
                  <div
                    onClick={() => onFolderSelect(adHocCurrentFolder.path)}
                    onContextMenu={(e) => onContextMenu(e, adHocCurrentFolder.path, false, false)}
                    className="group flex items-center gap-2 px-2 py-1.5 mx-1 rounded-md bg-card-active text-text-primary cursor-pointer"
                    data-tooltip={adHocCurrentFolder.path}
                  >
                    <Folder size={16} className="shrink-0 text-accent" />
                    <span className="truncate text-sm">{adHocCurrentFolder.name}</span>
                  </div>
                </div>
              </div>
            )}

            {showPinnedSection && (
              <>
                <div>
                  <SectionHeader
                    title={t('library.folders.sections.pinned')}
                    isOpen={isPinnedOpen}
                    isLoading={isPinnedFoldersLoading}
                    onToggle={() => toggleSection('pinned')}
                  />
                </div>
                <AnimatePresence initial={false}>
                  {isPinnedOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2, ease: 'easeInOut' }}
                      className="overflow-hidden"
                    >
                      <div className="pt-1 pb-2">
                        {isPinnedFoldersLoading && !isSearching && (
                          <Text
                            as="div"
                            variant={TextVariants.small}
                            className="flex items-center gap-2 p-2 text-text-secondary"
                          >
                            <Loader2 size={14} className="animate-spin" />
                            <span>{t('library.folders.loadingPinned', 'Loading pinned folders')}</span>
                          </Text>
                        )}
                        <AnimatePresence>
                          {filteredPinnedTrees.map((pinnedTree, index) => (
                            <motion.div
                              key={`pinned-${pinnedTree.path}`}
                              animate="visible"
                              custom={{ index, total: filteredPinnedTrees.length }}
                              exit="exit"
                              initial={isInstantTransition ? 'visible' : 'hidden'}
                              layout={isInstantTransition ? false : 'position'}
                              variants={{
                                hidden: { opacity: 0, x: -15 },
                                visible: ({ index, total }: VisibleProps) => ({
                                  opacity: 1,
                                  x: 0,
                                  transition: { duration: 0.25, delay: total < 8 ? index * 0.05 : 0 },
                                }),
                                exit: { opacity: 0, x: -15, transition: { duration: 0.2 } },
                              }}
                            >
                              <TreeNode
                                sectionId="pinned"
                                expandedFolders={effectiveExpandedFolders}
                                isExpanded={effectiveExpandedFolders.has(pinnedTree.path)}
                                node={pinnedTree}
                                onContextMenu={onContextMenu}
                                onFolderSelect={onFolderSelect}
                                onToggle={onToggleFolder}
                                selectedPath={selectedPath}
                                pinnedFolders={pinnedFolders}
                                showImageCounts={showImageCounts && isHovering}
                                isInstantTransition={isInstantTransition}
                                folderIcons={folderIcons}
                                isLayoutDragging={isLayoutDragging}
                                isRescanning={rescanningFolders.has(`source:${pinnedTree.path}`)}
                                onRescan={handleRescanSourceFolder}
                              />
                            </motion.div>
                          ))}
                        </AnimatePresence>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </>
            )}

            {showCatalogSection && (
              <>
                <div>
                  <SectionHeader
                    title="Library"
                    isOpen={isCatalogOpen}
                    onToggle={() => toggleSection('catalog')}
                    action={
                      <button
                        onClick={handleAddCollection}
                        disabled={isAddingCollection}
                        className="p-1 rounded-md text-text-secondary hover:bg-surface hover:text-text-primary disabled:opacity-50 disabled:cursor-not-allowed"
                        data-tooltip="Add a photo folder as a new collection"
                      >
                        {isAddingCollection ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
                      </button>
                    }
                  />
                </div>
                <AnimatePresence initial={false}>
                  {isCatalogOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2, ease: 'easeInOut' }}
                      className="overflow-hidden"
                    >
                      <div className="pt-1 pb-2">
                        {catalogRoots.length === 0 && (
                          <Text
                            as="div"
                            variant={TextVariants.small}
                            color={TextColors.secondary}
                            className="px-2 py-1.5"
                          >
                            No collections yet - click + above to add one.
                          </Text>
                        )}
                        {catalogRoots.map((root) => {
                          const tree = catalogFolderTrees[root.id];
                          const isLoadingTree = loadingCatalogTreeIds.has(root.id);
                          if (!tree) {
                            return (
                              <Text
                                as="div"
                                key={`catalog-${root.id}`}
                                className={clsx(
                                  'group flex items-center gap-2 p-2 rounded-md transition-colors',
                                  activeCatalogRootId === root.id ? 'bg-card-active text-text-primary' : '',
                                )}
                              >
                                {isLoadingTree ? (
                                  <Loader2 size={16} className="ml-1 shrink-0 text-text-secondary animate-spin" />
                                ) : (
                                  <Database size={16} className="ml-1 shrink-0 text-text-secondary" />
                                )}
                                <span className="truncate min-w-0 flex-1">{root.label || root.absolutePath}</span>
                                {showImageCounts && (
                                  <span className="text-xs text-text-secondary tabular-nums">{root.imageCount}</span>
                                )}
                                <button
                                  className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded-sm hover:bg-surface"
                                  data-tooltip="Rescan"
                                  onClick={(event) => handleRescanCatalogRoot(event, root)}
                                >
                                  <RefreshCw size={14} />
                                </button>
                              </Text>
                            );
                          }
                          return (
                            <TreeNode
                              key={`catalog-${root.id}`}
                              sectionId="catalog"
                              expandedFolders={effectiveExpandedFolders}
                              isExpanded={effectiveExpandedFolders.has(tree.path)}
                              node={tree}
                              onContextMenu={handleCatalogContextMenu}
                              onFolderSelect={handleCatalogFolderSelect}
                              onToggle={onToggleFolder}
                              selectedPath={selectedPath}
                              pinnedFolders={[]}
                              showImageCounts={showImageCounts && isHovering}
                              isInstantTransition={isInstantTransition}
                              folderIcons={folderIcons}
                              isLayoutDragging={isLayoutDragging}
                              isRescanning={rescanningFolders.has(`catalog:${root.id}`)}
                              onRescan={(event) => handleRescanCatalogRoot(event, root)}
                              analysisJobs={catalogAnalysisJobs[root.id]}
                              onAnalysisClick={setAnalysisNode}
                            />
                          );
                        })}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </>
            )}

            {librarySource.type === 'catalog' && (
              <>
                <div>
                  <SectionHeader title="Smart Collections" isOpen={true} onToggle={() => {}} />
                </div>
                <div className="pt-1 pb-2">
                  {isLoadingSmartCollections ? (
                    <Text
                      as="div"
                      variant={TextVariants.small}
                      className="flex items-center gap-2 p-2 text-text-secondary"
                    >
                      <Loader2 size={14} className="animate-spin" />
                      Loading collections
                    </Text>
                  ) : smartCollections.length === 0 ? (
                    <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="p-2">
                      No saved searches
                    </Text>
                  ) : (
                    smartCollections.map((collection) => (
                      <button
                        key={collection.id}
                        className="w-full flex items-center gap-2 p-2 rounded-md text-left text-text-secondary hover:bg-surface hover:text-text-primary"
                        onClick={() => void handleSmartCollectionSelect(collection)}
                      >
                        <Star size={15} className="shrink-0" />
                        <span className="truncate">{collection.name}</span>
                      </button>
                    ))
                  )}
                </div>
              </>
            )}

            {showAlbumsSection && (
              <>
                <div>
                  <SectionHeader
                    title={t('library.folders.sections.albums')}
                    isOpen={isAlbumsOpen}
                    onToggle={() => toggleSection('albums')}
                  />
                </div>
                <AnimatePresence>
                  {isAlbumsOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden"
                      onContextMenu={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        onAlbumContextMenu(e, null);
                      }}
                    >
                      <div className="pt-1 pb-2">
                        <AnimatePresence>
                          {filteredAlbumTree.map((item: any) => (
                            <motion.div
                              key={`albums-${item.id}`}
                              initial={{ opacity: 0, height: 0, x: -15 }}
                              animate={{ opacity: 1, height: 'auto', x: 0 }}
                              exit={{ opacity: 0, height: 0, x: -15, overflow: 'hidden' }}
                              transition={{ duration: 0.2 }}
                              layout="position"
                            >
                              <AlbumTreeNode
                                sectionId="albums"
                                item={item}
                                expandedGroups={effectiveExpandedAlbumGroups}
                                onToggle={toggleAlbumGroup}
                                onSelectAlbum={onSelectAlbum}
                                onContextMenu={onAlbumContextMenu}
                                selectedAlbumId={activeAlbumId}
                                showImageCounts={showImageCounts && isHovering}
                                isLayoutDragging={isLayoutDragging}
                              />
                            </motion.div>
                          ))}
                        </AnimatePresence>
                        {albumTree.length === 0 && !isSearching && (
                          <motion.div layout="position">
                            <Text variant={TextVariants.small} className="p-2 text-center">
                              {t('library.folders.albumsEmpty')}
                            </Text>
                          </motion.div>
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </>
            )}

            {filteredTrees && filteredTrees.length > 0 && (
              <>
                <div>
                  <SectionHeader
                    title={t('library.folders.sections.folders')}
                    isOpen={isCurrentOpen}
                    isLoading={isRootFoldersLoading}
                    onToggle={() => toggleSection('current')}
                  />
                </div>
                <AnimatePresence initial={false}>
                  {isCurrentOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2, ease: 'easeInOut' }}
                      className="overflow-hidden"
                    >
                      <div className="pt-1">
                        <AnimatePresence>
                          {filteredTrees.map((tree: any, index: number) => (
                            <motion.div
                              key={`current-${tree.path}`}
                              animate="visible"
                              custom={{ index, total: filteredTrees.length }}
                              exit="exit"
                              initial={isInstantTransition ? 'visible' : 'hidden'}
                              layout={isInstantTransition ? false : 'position'}
                              variants={{
                                hidden: { opacity: 0, x: -15 },
                                visible: ({ index, total }: VisibleProps) => ({
                                  opacity: 1,
                                  x: 0,
                                  transition: { duration: 0.25, delay: total < 8 ? index * 0.05 : 0 },
                                }),
                                exit: { opacity: 0, x: -15, transition: { duration: 0.2 } },
                              }}
                            >
                              <TreeNode
                                sectionId="current"
                                expandedFolders={effectiveExpandedFolders}
                                isExpanded={effectiveExpandedFolders.has(tree.path)}
                                node={tree}
                                onContextMenu={onContextMenu}
                                onFolderSelect={onFolderSelect}
                                onToggle={onToggleFolder}
                                selectedPath={selectedPath}
                                pinnedFolders={pinnedFolders}
                                showImageCounts={showImageCounts && isHovering}
                                isInstantTransition={isInstantTransition}
                                folderIcons={folderIcons}
                                isLayoutDragging={isLayoutDragging}
                                isRescanning={rescanningFolders.has(`source:${tree.path}`)}
                                onRescan={handleRescanSourceFolder}
                              />
                            </motion.div>
                          ))}
                        </AnimatePresence>

                        <AnimatePresence initial={false}>
                          {isHovering && !isSearching && (
                            <motion.div
                              layout="position"
                              initial={{ opacity: 0, height: 0, overflow: 'hidden' }}
                              animate={{ opacity: 1, height: 'auto', overflow: 'hidden' }}
                              exit={{ opacity: 0, height: 0, overflow: 'hidden' }}
                              transition={{ duration: 0.2 }}
                            >
                              <Text
                                as="div"
                                weight={TextWeights.medium}
                                className="flex items-center gap-2 p-2 mt-1 rounded-md transition-colors transition-opacity opacity-70 hover:opacity-100 hover:bg-card-active cursor-pointer hover:text-text-primary"
                                onClick={(e: React.MouseEvent) => {
                                  e.stopPropagation();
                                  onOpenFolder();
                                }}
                              >
                                <div className="relative w-4 h-4 ml-1 shrink-0 flex items-center justify-center">
                                  <Plus size={16} />
                                </div>
                                <span className="select-none">{t('library.folders.addFolder')}</span>
                              </Text>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </>
            )}

            {!filteredTrees?.length &&
              !hasVisiblePinnedTrees &&
              !hasVisibleAlbums &&
              !hasVisibleCatalogRoots &&
              isSearching && <Text className="p-2 text-center">{t('library.folders.noFoldersFound')}</Text>}

            {isRootFoldersLoading && folderTrees.length === 0 && !isSearching && (
              <>
                <div>
                  <SectionHeader
                    title={t('library.folders.sections.folders')}
                    isOpen={isCurrentOpen}
                    isLoading={isRootFoldersLoading}
                    onToggle={() => toggleSection('current')}
                  />
                </div>
                {isCurrentOpen && (
                  <Text
                    as="div"
                    variant={TextVariants.small}
                    className="flex items-center gap-2 p-2 text-text-secondary"
                  >
                    <Loader2 size={14} className="animate-spin" />
                    <span>{t('library.folders.loadingFolders', 'Loading folders')}</span>
                  </Text>
                )}
              </>
            )}

            {folderTrees.length === 0 &&
              pinnedFolderTrees.length === 0 &&
              !hasVisibleCatalogRoots &&
              !isSearching &&
              !isRootFoldersLoading &&
              !isPinnedFoldersLoading && (
                <div className="pt-1">
                  {isLoading ? (
                    <Text className="animate-pulse p-2">{t('library.folders.loading')}</Text>
                  ) : (
                    <Text className="p-2">{t('library.folders.openFolderInstruction')}</Text>
                  )}
                </div>
              )}
          </div>
        </LayoutGroup>
      </div>
      {analysisNode && <CatalogAnalysisModal node={analysisNode} onClose={() => setAnalysisNode(null)} />}
    </div>
  );
}
