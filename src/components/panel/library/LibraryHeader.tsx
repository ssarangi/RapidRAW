import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import {
  Calendar,
  Camera,
  Database,
  Search,
  Loader2,
  X,
  SlidersHorizontal,
  Check,
  Star as StarIcon,
  ChevronUp,
  ChevronDown,
  HelpCircle,
  Sparkles,
  Tags,
  User,
  Bookmark,
  Trash2,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useShallow } from 'zustand/react/shallow';
import { convertFileSrc, invoke } from '@tauri-apps/api/core';
import { toast } from 'react-toastify';
import { useLibraryStore } from '../../../store/useLibraryStore';
import {
  CatalogMetrics,
  CatalogRoot,
  CatalogSearchQuery,
  SmartCollection,
  FilterCriteria,
  ImageFile,
  Invokes,
  RawStatus,
  EditedStatus,
  LibraryViewMode,
  SortCriteria,
  SortDirection,
  ExifOverlay,
  GroupingMode,
  ThumbnailSize,
  ThumbnailAspectRatio,
} from '../../ui/AppProperties';
import { COLOR_LABELS, Color } from '../../../utils/adjustments';
import Text from '../../ui/Text';
import { TextColors, TextVariants, TextWeights, TEXT_COLOR_KEYS } from '../../../types/typography';
import Button from '../../ui/Button';
import Switch from '../../ui/Switch';
import Dropdown from '../../ui/Dropdown';
import { useSettingsStore } from '../../../store/useSettingsStore';
import { useUIStore } from '../../../store/useUIStore';
import { ADVANCED_QUERY_REGEX } from '../../../hooks/useSortedLibrary';

function DropdownMenu({ buttonContent, buttonTitle, children, contentClassName = 'w-56' }: any) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<any>(null);

  useEffect(() => {
    const handleClickOutside = (event: any) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={dropdownRef}>
      <Button
        aria-expanded={isOpen}
        aria-haspopup="true"
        className="h-12 w-12 bg-surface text-text-primary shadow-none p-0 flex items-center justify-center"
        onClick={() => setIsOpen(!isOpen)}
        data-tooltip={buttonTitle}
      >
        {buttonContent}
      </Button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className={`absolute right-0 mt-2 ${contentClassName} origin-top-right z-50`}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.1, ease: 'easeOut' }}
          >
            <div
              className="bg-surface/90 backdrop-blur-md rounded-lg shadow-xl"
              role="menu"
              aria-orientation="vertical"
            >
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

interface SegmentedSwitchProps {
  options: { id: string | number; label: string }[];
  value: string | number;
  onChange: (id: any) => void;
}

const SegmentedSwitch = ({ options, value, onChange }: SegmentedSwitchProps) => {
  const [bubbleStyle, setBubbleStyle] = useState({});
  const isInitialAnimation = useRef(true);

  const selectedIndex = options.findIndex((m) => m.id === value);
  const hasSelection = selectedIndex >= 0;

  useEffect(() => {
    const safeIndex = hasSelection ? selectedIndex : 0;
    const widthPercent = 100 / options.length;
    const targetX = `${safeIndex * 100}%`;
    const targetWidth = `${widthPercent}%`;

    if (isInitialAnimation.current) {
      setBubbleStyle({ x: targetX, width: targetWidth, opacity: hasSelection ? 1 : 0 });
      isInitialAnimation.current = false;
    } else {
      setBubbleStyle({ x: targetX, width: targetWidth, opacity: hasSelection ? 1 : 0 });
    }
  }, [value, options, hasSelection]);

  return (
    <div className="w-full bg-bg-primary p-1 rounded-md">
      <div className="relative flex w-full">
        <motion.div
          className="absolute top-0 bottom-0 left-0 z-0 bg-card-active shadow-xs"
          style={{ borderRadius: 6 }}
          animate={bubbleStyle}
          initial={false}
          transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
        />
        {options.map((option) => (
          <button
            key={option.id}
            onClick={() => onChange(option.id)}
            className={clsx(
              'relative flex-1 flex items-center justify-center px-2 py-1.5 text-xs font-medium rounded-md transition-colors truncate',
              {
                'text-text-secondary hover:text-text-primary': value !== option.id,
                'text-text-primary font-semibold': value === option.id,
              },
            )}
            style={{ WebkitTapHighlightColor: 'transparent' }}
          >
            <span className="relative z-10">{option.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

const RatingSegmentedSwitch = ({ rating, onChange, ratingFilterOptions }: any) => {
  const [bubbleStyle, setBubbleStyle] = useState({});
  const isInitialAnimation = useRef(true);

  const getActiveIndex = () => {
    if (rating === 0) return 0;
    if (rating <= -1) return 1;
    return 2;
  };

  const activeIndex = getActiveIndex();

  useEffect(() => {
    const targetX = `${activeIndex * 100}%`;
    const targetWidth = '33.333333%';

    if (isInitialAnimation.current) {
      setBubbleStyle({ x: targetX, width: targetWidth });
      isInitialAnimation.current = false;
    } else {
      setBubbleStyle({ x: targetX, width: targetWidth });
    }
  }, [activeIndex]);

  return (
    <div className="w-full bg-bg-primary p-1 rounded-md">
      <div className="relative flex w-full">
        <motion.div
          className="absolute top-0 bottom-0 left-0 z-0 bg-card-active shadow-xs"
          style={{ borderRadius: 6 }}
          animate={bubbleStyle}
          initial={false}
          transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
        />

        <button
          onClick={() => onChange(0)}
          className={clsx(
            'relative flex-1 flex items-center justify-center px-1 py-1.5 text-xs rounded-md transition-colors truncate',
            activeIndex === 0 ? 'text-text-primary font-semibold' : 'text-text-secondary hover:text-text-primary',
          )}
        >
          <span className="relative z-10">{ratingFilterOptions.find((o: any) => o.value === 0)?.label || 'All'}</span>
        </button>

        <button
          onClick={() => onChange(-1)}
          className={clsx(
            'relative flex-1 flex items-center justify-center px-1 py-1.5 text-xs rounded-md transition-colors truncate',
            activeIndex === 1 ? 'text-text-primary font-semibold' : 'text-text-secondary hover:text-text-primary',
          )}
        >
          <span className="relative z-10">
            {ratingFilterOptions.find((o: any) => o.value === -1)?.label || 'Unrated'}
          </span>
        </button>

        <div
          className={clsx(
            'relative flex-1 flex items-center justify-center gap-0.5 px-1 py-1.5 transition-colors',
            activeIndex === 2 ? 'text-text-primary' : 'text-text-secondary',
          )}
        >
          <div className="flex items-center z-10">
            {[...Array(5)].map((_, index) => {
              const starValue = index + 1;
              const isFilled = rating > 0 && starValue <= rating;
              const optionLabel = ratingFilterOptions.find((o: any) => o.value === starValue)?.label;

              return (
                <button
                  key={starValue}
                  data-tooltip={optionLabel}
                  onClick={(e) => {
                    e.stopPropagation();
                    onChange(rating === starValue ? 0 : starValue);
                  }}
                  className="focus:outline-hidden transition-transform hover:scale-110 flex items-center justify-center p-0.5"
                >
                  <StarIcon
                    size={14}
                    className={`transition-colors duration-150 ${
                      isFilled ? 'text-accent fill-accent' : 'text-text-secondary hover:text-accent'
                    }`}
                  />
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export function SearchInput({ indexingProgress, isIndexing }: any) {
  const { t } = useTranslation();
  const { searchCriteria, setSearchCriteria } = useLibraryStore(
    useShallow((state) => ({ searchCriteria: state.searchCriteria, setSearchCriteria: state.setSearchCriteria })),
  );
  const [isSearchActive, setIsSearchActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const { tags, text, mode } = searchCriteria;
  const searchFocusRequest = useUIStore((state) => state.searchFocusRequest);
  const lastSearchFocusRequest = useRef(searchFocusRequest);

  const [contentWidth, setContentWidth] = useState(0);

  useEffect(() => {
    if (isSearchActive) {
      inputRef.current?.focus();
    }
  }, [isSearchActive]);

  useEffect(() => {
    if (searchFocusRequest === lastSearchFocusRequest.current) return;
    lastSearchFocusRequest.current = searchFocusRequest;
    setIsSearchActive(true);
    inputRef.current?.focus();
  }, [searchFocusRequest]);

  useEffect(() => {
    function handleClickOutside(event: any) {
      if (containerRef.current && !containerRef.current.contains(event.target) && tags.length === 0 && !text) {
        setIsSearchActive(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [tags, text]);

  useEffect(() => {
    if (contentRef.current) {
      const timer = setTimeout(() => {
        if (contentRef.current) {
          setContentWidth(contentRef.current.scrollWidth);
        }
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [tags, text, isSearchActive]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchCriteria((prev) => ({ ...prev, text: e.target.value }));
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if ((e.key === ',' || e.key === 'Enter') && text.trim()) {
      e.preventDefault();
      setSearchCriteria((prev) => ({
        ...prev,
        tags: [...prev.tags, text.trim()],
        text: '',
      }));
    } else if (e.key === 'Backspace' && !text && tags.length > 0) {
      e.preventDefault();
      const lastTag = tags[tags.length - 1];
      setSearchCriteria((prev) => ({
        ...prev,
        tags: prev.tags.slice(0, -1),
        text: lastTag,
      }));
    }
  };

  const removeTag = (tagToRemove: string) => {
    setSearchCriteria((prev) => ({
      ...prev,
      tags: prev.tags.filter((tag) => tag !== tagToRemove),
    }));
  };

  const clearSearch = () => {
    setSearchCriteria({ tags: [], text: '', mode: 'OR' });
    setIsSearchActive(false);
    inputRef.current?.blur();
  };

  const toggleMode = () => {
    setSearchCriteria((prev) => ({
      ...prev,
      mode: prev.mode === 'AND' ? 'OR' : 'AND',
    }));
  };

  const isActive = isSearchActive || tags.length > 0 || !!text;
  const placeholderText =
    isIndexing && indexingProgress.total > 0
      ? t('library.header.search.indexingProgress', {
          current: indexingProgress.current,
          total: indexingProgress.total,
        })
      : isIndexing
        ? t('library.header.search.indexingImages')
        : tags.length > 0
          ? t('library.header.search.addFilterOrSearch')
          : t('library.header.search.searchOrQuery');

  const INACTIVE_WIDTH = 48;
  const PADDING_AND_ICONS_WIDTH = 100;
  const MAX_WIDTH = 680;

  const calculatedWidth = Math.min(MAX_WIDTH, contentWidth + PADDING_AND_ICONS_WIDTH);

  return (
    <motion.div
      animate={{ width: isActive ? calculatedWidth : INACTIVE_WIDTH }}
      className="relative flex items-center bg-surface rounded-md h-12 overflow-hidden"
      initial={false}
      transition={{ type: 'spring', stiffness: 400, damping: 35 }}
      onClick={() => inputRef.current?.focus()}
    >
      <button
        className="h-12 w-12 flex items-center justify-center text-text-primary z-10 shrink-0 bg-surface outline-hidden"
        onClick={(e) => {
          e.stopPropagation();
          if (!isActive) setIsSearchActive(true);
          inputRef.current?.focus();
        }}
        data-tooltip={t('library.header.search.tooltipSearchFilter')}
      >
        <Search className="w-4 h-4" />
      </button>
      <div
        className="flex-1 min-w-0 h-full overflow-hidden flex items-center pl-1"
        style={{ opacity: isActive ? 1 : 0, pointerEvents: isActive ? 'auto' : 'none', transition: 'opacity 0.2s' }}
      >
        <div ref={contentRef} className="flex items-center gap-2 h-full flex-nowrap min-w-[250px] pr-2">
          {tags.map((tag) => {
            const match = tag.match(ADVANCED_QUERY_REGEX);
            const isQuery = !!match;

            return (
              <motion.div
                key={tag}
                layout
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.5 }}
                className="flex items-center gap-1 bg-bg-primary px-2 py-1 rounded-sm group cursor-pointer shrink-0"
                onClick={(e) => {
                  e.stopPropagation();
                  removeTag(tag);
                }}
              >
                <Text variant={TextVariants.small} color={TextColors.primary} weight={TextWeights.medium}>
                  {isQuery ? (
                    <span className="flex gap-0.5">
                      <span className="uppercase opacity-70">{match[1]}</span>
                      <span>{match[2] || ':'}</span>
                      <span>{match[3]}</span>
                    </span>
                  ) : (
                    tag
                  )}
                </Text>
                <span className="rounded-full group-hover:bg-black/20 p-0.5 transition-colors">
                  <X size={12} />
                </span>
              </motion.div>
            );
          })}
          <input
            className="grow w-full h-full bg-transparent text-text-primary placeholder-text-secondary border-none focus:outline-hidden min-w-[150px]"
            disabled={isIndexing}
            onBlur={() => {
              if (tags.length === 0 && !text) setIsSearchActive(false);
            }}
            onChange={handleInputChange}
            onFocus={() => setIsSearchActive(true)}
            onKeyDown={handleKeyDown}
            placeholder={placeholderText}
            ref={inputRef}
            type="text"
            value={text}
          />
        </div>
      </div>
      <div
        className="shrink-0 flex items-center gap-1 pr-2 bg-surface z-10"
        style={{ opacity: isActive ? 1 : 0, pointerEvents: isActive ? 'auto' : 'none', transition: 'opacity 0.2s' }}
      >
        {tags.length > 0 && (
          <button
            onMouseDown={(e) => e.preventDefault()}
            onClick={toggleMode}
            className="p-1.5 rounded-md hover:bg-bg-primary w-10 shrink-0 flex items-center justify-center outline-hidden"
            data-tooltip={mode === 'AND' ? t('library.header.search.matchAll') : t('library.header.search.matchAny')}
          >
            <Text variant={TextVariants.small} color={TextColors.primary} weight={TextWeights.semibold}>
              {mode}
            </Text>
          </button>
        )}
        <div
          className="p-1.5 rounded-md text-text-secondary hover:text-text-primary transition-colors cursor-help shrink-0 outline-hidden"
          data-tooltip={t('library.header.search.tooltipAdvancedQueries')}
        >
          <HelpCircle size={16} />
        </div>
        {(tags.length > 0 || text) && !isIndexing && (
          <button
            onMouseDown={(e) => e.preventDefault()}
            onClick={clearSearch}
            className="p-1.5 rounded-md text-text-secondary hover:text-text-primary hover:bg-bg-primary shrink-0 outline-hidden"
            data-tooltip={t('library.header.search.tooltipClearSearch')}
          >
            <X className="h-5 w-5" />
          </button>
        )}
        {isIndexing && (
          <div className="flex items-center pr-1 pointer-events-none shrink-0">
            <Loader2 className="h-5 w-5 text-text-secondary animate-spin" />
          </div>
        )}
      </div>
    </motion.div>
  );
}

export function CatalogSearchDropdown() {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [metrics, setMetrics] = useState<CatalogMetrics | null>(null);
  const [collectionName, setCollectionName] = useState('');
  const [form, setForm] = useState({
    text: '',
    year: '',
    camera: '',
    lens: '',
    person: '',
    tags: '',
    aiTags: '',
    minRating: '',
  });
  const dropdownRef = useRef<HTMLDivElement>(null);

  const { librarySource, catalogRoots, activeCatalogRootId } = useLibraryStore(
    useShallow((state) => ({
      librarySource: state.librarySource,
      catalogRoots: state.catalogRoots,
      activeCatalogRootId: state.activeCatalogRootId,
    })),
  );

  const isCatalogAvailable = librarySource.type === 'catalog';
  const activeRoot = catalogRoots.find((root: CatalogRoot) => root.id === activeCatalogRootId) || catalogRoots[0] || null;

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    if (!isOpen || !isCatalogAvailable) return;
    invoke<CatalogMetrics>(Invokes.GetCatalogMetrics)
      .then(setMetrics)
      .catch((err) => {
        console.error('Failed to load catalog metrics:', err);
        setMetrics(null);
      });
  }, [isOpen, isCatalogAvailable]);

  const setField = (key: keyof typeof form, value: string) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const selectClassName =
    'catalog-search-select w-full appearance-none bg-transparent text-text-primary border-none outline-none pr-7 disabled:text-text-secondary disabled:opacity-70';
  const emptyOptionLabel = metrics ? 'Any' : 'Loading...';

  const runCatalogSearch = async (overrideForm = form) => {
    if (!isCatalogAvailable) {
      toast.info('Open or create a SQLite library first.');
      return;
    }

    setIsSearching(true);
    try {
      const tags = overrideForm.tags
        .split(',')
        .map((tag) => tag.trim())
        .filter(Boolean);
      const aiTags = overrideForm.aiTags
        .split(',')
        .map((tag) => tag.trim())
        .filter(Boolean);
      const query: CatalogSearchQuery = {
        rootId: activeRoot?.id ?? null,
        text: overrideForm.text.trim() || null,
        year: overrideForm.year.trim() ? Number(overrideForm.year) : null,
        camera: overrideForm.camera.trim() || null,
        lens: overrideForm.lens.trim() || null,
        person: overrideForm.person.trim() || null,
        tags: tags.length > 0 ? tags : null,
        aiTags: aiTags.length > 0 ? aiTags : null,
        tagMode: tags.length > 1 ? 'AND' : null,
        minRating: overrideForm.minRating.trim() ? Number(overrideForm.minRating) : null,
        limit: 20_000,
      };
      const files = await invoke<ImageFile[]>(Invokes.SearchCatalogImages, { query });
      const imageRatings: Record<string, number> = {};
      files.forEach((file) => {
        imageRatings[file.path] = file.rating || 0;
      });
      const overrideHasQuery = Object.values(overrideForm).some((value) => value.trim().length > 0);
      const label = overrideHasQuery
        ? 'Library: Search Results'
        : `Library: ${activeRoot?.label || activeRoot?.absolutePath || librarySource.name}`;
      useLibraryStore.getState().setLibrary({
        rootPaths: activeRoot ? [activeRoot.absolutePath] : useLibraryStore.getState().rootPaths,
        currentFolderPath: label,
        activeAlbumId: null,
        activeCatalogRootId: activeRoot?.id ?? null,
        imageList: files,
        imageRatings,
        multiSelectedPaths: [],
        libraryActivePath: null,
        libraryScrollTop: 0,
      });
      useLibraryStore.getState().setSearchCriteria({ text: '', tags: [], mode: 'OR' });
      setIsOpen(false);
    } catch (err) {
      console.error('Failed to search catalog:', err);
      toast.error(`Failed to search catalog: ${err}`);
    } finally {
      setIsSearching(false);
    }
  };

  const clearCatalogSearch = async () => {
    const emptyForm = { text: '', year: '', camera: '', lens: '', person: '', tags: '', aiTags: '', minRating: '' };
    setForm(emptyForm);
    await runCatalogSearch(emptyForm);
  };

  const saveSmartCollection = async () => {
    const name = collectionName.trim();
    if (!name) { toast.info('Enter a smart collection name.'); return; }
    const tags = form.tags.split(',').map((tag) => tag.trim()).filter(Boolean);
    const aiTags = form.aiTags.split(',').map((tag) => tag.trim()).filter(Boolean);
    const query: CatalogSearchQuery = { rootId: activeRoot?.id ?? null, text: form.text.trim() || null, year: form.year ? Number(form.year) : null, camera: form.camera || null, lens: form.lens || null, person: form.person || null, tags: tags.length ? tags : null, aiTags: aiTags.length ? aiTags : null, tagMode: tags.length > 1 ? 'AND' : null, minRating: form.minRating ? Number(form.minRating) : null, limit: 20_000 };
    try { await invoke(Invokes.SaveSmartCollection, { name, queryJson: JSON.stringify(query) }); window.dispatchEvent(new Event('smart-collections-changed')); setCollectionName(''); toast.success('Smart collection saved.'); } catch (error) { toast.error(`Failed to save smart collection: ${error}`); }
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <Button
        aria-expanded={isOpen}
        aria-haspopup="true"
        className={clsx(
          'h-12 px-3 bg-transparent text-text-primary shadow-none flex items-center justify-center gap-2',
          !isCatalogAvailable && 'opacity-50',
        )}
        onClick={() => setIsOpen((open) => !open)}
        data-tooltip={
          isCatalogAvailable
            ? t('library.header.catalogSearch.title', { defaultValue: 'Search Catalog' })
            : t('library.header.catalogSearch.unavailable', { defaultValue: 'Open a SQLite library to search catalog metadata' })
        }
      >
        <Database className="w-5 h-5" />
        <span className="text-sm font-medium whitespace-nowrap">Catalog Search</span>
      </Button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="absolute right-0 mt-2 w-[620px] max-w-[calc(100vw-2rem)] origin-top-right z-50"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.1, ease: 'easeOut' }}
          >
            <div className="bg-surface/95 backdrop-blur-md border border-border-color/50 rounded-lg shadow-xl p-4">
              <div className="flex items-start justify-between gap-4 mb-4">
                <div className="min-w-0">
                  <Text variant={TextVariants.heading} weight={TextWeights.semibold}>
                    {t('library.header.catalogSearch.title', { defaultValue: 'Search Catalog' })}
                  </Text>
                  <Text variant={TextVariants.small} color={TextColors.secondary} className="truncate">
                    {isCatalogAvailable
                      ? activeRoot?.label || activeRoot?.absolutePath || librarySource.name
                      : t('library.header.catalogSearch.noLibrary', { defaultValue: 'No SQLite library is open' })}
                  </Text>
                </div>
                <button
                  className="p-2 rounded-md text-text-secondary hover:text-text-primary hover:bg-bg-primary"
                  onClick={() => setIsOpen(false)}
                >
                  <X size={18} />
                </button>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <label className="col-span-2">
                  <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mb-1">
                    Text
                  </Text>
                    <div className="flex items-center bg-bg-primary rounded-md px-3 h-10 border border-border-color/30">
                      <Search size={16} className="text-text-secondary mr-2 shrink-0" />
                    <input
                      className="w-full bg-transparent text-text-primary border-none outline-none"
                      value={form.text}
                      onChange={(event) => setField('text', event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') runCatalogSearch();
                      }}
                    />
                  </div>
                </label>

                {[
                  { key: 'year' as const, label: 'Year', Icon: Calendar, options: metrics?.years || [] },
                  {
                    key: 'minRating' as const,
                    label: 'Minimum Rating',
                    Icon: StarIcon,
                    options: (metrics?.ratings || []).filter((item) => item.value !== '0'),
                  },
                  { key: 'lens' as const, label: 'Lens', Icon: Camera, options: metrics?.lenses || [] },
                  { key: 'camera' as const, label: 'Camera', Icon: Camera, options: metrics?.cameras || [] },
                  { key: 'person' as const, label: 'Person', Icon: User, options: metrics?.people || [] },
                  { key: 'tags' as const, label: 'Tags', Icon: Tags, options: metrics?.tags || [] },
                  { key: 'aiTags' as const, label: 'AI Tags', Icon: Tags, options: metrics?.aiTags || [] },
                ].map(({ key, label, Icon, options }) => (
                  <label key={key}>
                    <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mb-1">
                      {label}
                    </Text>
                    <div className="catalog-search-field flex items-center bg-bg-primary rounded-md px-3 h-10 border border-border-color/30">
                      <Icon size={16} className="text-text-secondary mr-2 shrink-0" />
                      <div className="relative min-w-0 flex-1">
                        <select
                          className={selectClassName}
                          disabled={!metrics || options.length === 0}
                          value={form[key]}
                          onChange={(event) => setField(key, event.target.value)}
                        >
                          <option value="">{emptyOptionLabel}</option>
                          {options.map((option) => (
                            <option key={`${key}-${option.value}`} value={option.value}>
                              {key === 'minRating' ? `${option.value}+ stars` : option.value} ({option.count})
                            </option>
                          ))}
                        </select>
                        <ChevronDown
                          size={16}
                          className="pointer-events-none absolute right-0 top-1/2 -translate-y-1/2 text-text-secondary"
                        />
                      </div>
                    </div>
                  </label>
                ))}
              </div>

              {metrics && (
                <div className="mt-4 grid grid-cols-3 gap-3">
                  {[
                    { label: 'Images', value: metrics.totalImages },
                    { label: 'Rated', value: metrics.ratedImages },
                    { label: 'Edited', value: metrics.editedImages },
                    { label: 'Missing', value: metrics.missingImages },
                    { label: 'AI suggestions', value: metrics.aiTagsSuggested },
                    { label: 'AI accepted', value: metrics.aiTagsAccepted },
                    { label: 'RAM++ analyzed', value: metrics.ramPlusAnalyzed },
                    { label: 'RAM++ pending', value: metrics.ramPlusPending },
                  ].map((item) => (
                    <div key={item.label} className="bg-bg-primary rounded-md border border-border-color/30 px-3 py-2">
                      <Text as="div" variant={TextVariants.small} color={TextColors.secondary}>
                        {item.label}
                      </Text>
                      <Text variant={TextVariants.heading}>{item.value}</Text>
                    </div>
                  ))}
                </div>
              )}

              <div className="mt-4 flex justify-end gap-2">
                <input className="h-10 w-40 bg-bg-primary text-text-primary border border-border-color rounded-md px-3 text-sm" value={collectionName} onChange={(event) => setCollectionName(event.target.value)} placeholder="Collection name" />
                <Button className="h-10 bg-bg-primary text-text-primary border border-border-color shadow-none" disabled={!isCatalogAvailable} onClick={() => void saveSmartCollection()}>Save</Button>
                <Button
                  className="h-10 bg-bg-primary text-text-primary border border-border-color shadow-none"
                  disabled={isSearching || !isCatalogAvailable}
                  onClick={clearCatalogSearch}
                >
                  Clear
                </Button>
                <Button className="h-10" disabled={isSearching || !isCatalogAvailable} onClick={() => runCatalogSearch()}>
                  {isSearching && <Loader2 size={16} className="mr-2 animate-spin" />}
                  Search
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function SmartCollectionsDropdown() {
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isApplying, setIsApplying] = useState(false);
  const [collections, setCollections] = useState<SmartCollection[]>([]);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const { librarySource, catalogRoots } = useLibraryStore(
    useShallow((state) => ({ librarySource: state.librarySource, catalogRoots: state.catalogRoots })),
  );
  const isCatalogAvailable = librarySource.type === 'catalog';

  const loadCollections = async () => {
    if (!isCatalogAvailable) return;
    setIsLoading(true);
    try {
      setCollections(await invoke<SmartCollection[]>(Invokes.ListSmartCollections));
    } catch (error) {
      console.error('Failed to load smart collections:', error);
      toast.error(`Failed to load smart collections: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) setIsOpen(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const toggleOpen = () => {
    const nextOpen = !isOpen;
    setIsOpen(nextOpen);
    if (nextOpen) void loadCollections();
  };

  const applyCollection = async (collection: SmartCollection) => {
    let query: CatalogSearchQuery;
    try {
      query = JSON.parse(collection.queryJson) as CatalogSearchQuery;
    } catch {
      toast.error(`Smart collection "${collection.name}" has an invalid query.`);
      return;
    }

    setIsApplying(true);
    try {
      const files = await invoke<ImageFile[]>(Invokes.SearchCatalogImages, { query });
      const imageRatings: Record<string, number> = {};
      files.forEach((file) => { imageRatings[file.path] = file.rating || 0; });
      const root = catalogRoots.find((candidate: CatalogRoot) => candidate.id === query.rootId) || catalogRoots[0] || null;
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
      setIsOpen(false);
    } catch (error) {
      console.error('Failed to apply smart collection:', error);
      toast.error(`Failed to apply smart collection: ${error}`);
    } finally {
      setIsApplying(false);
    }
  };

  const deleteCollection = async (collection: SmartCollection) => {
    if (!window.confirm(`Delete smart collection "${collection.name}"?`)) return;
    try {
      await invoke(Invokes.DeleteSmartCollection, { id: collection.id });
      setCollections((current) => current.filter((candidate) => candidate.id !== collection.id));
      window.dispatchEvent(new Event('smart-collections-changed'));
    } catch (error) {
      toast.error(`Failed to delete smart collection: ${error}`);
    }
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <Button
        aria-expanded={isOpen}
        aria-haspopup="true"
        className={clsx('h-12 w-12 bg-transparent text-text-primary shadow-none p-0 flex items-center justify-center', !isCatalogAvailable && 'opacity-50')}
        disabled={!isCatalogAvailable}
        onClick={toggleOpen}
        data-tooltip="Smart Collections"
      >
        <Bookmark className="w-5 h-5" />
      </Button>
      <AnimatePresence>
        {isOpen && (
          <motion.div className="absolute right-0 mt-2 w-80 max-w-[calc(100vw-2rem)] origin-top-right z-50" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }} transition={{ duration: 0.1, ease: 'easeOut' }}>
            <div className="bg-surface/95 backdrop-blur-md border border-border-color/50 rounded-lg shadow-xl p-2">
              <div className="flex items-center justify-between px-2 py-2">
                <Text variant={TextVariants.small} weight={TextWeights.semibold}>Smart Collections</Text>
                <Button className="h-7 w-7 p-0 bg-transparent text-text-secondary shadow-none" onClick={() => void loadCollections()} data-tooltip="Refresh collections">
                  <Loader2 size={15} className={clsx(isLoading && 'animate-spin')} />
                </Button>
              </div>
              {isLoading ? (
                <div className="flex items-center gap-2 px-2 py-4 text-text-secondary"><Loader2 size={16} className="animate-spin" /><Text variant={TextVariants.small}>Loading collections</Text></div>
              ) : collections.length === 0 ? (
                <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="px-2 py-4">Save a catalog search to create a smart collection.</Text>
              ) : (
                <div className="max-h-72 overflow-y-auto">
                  {collections.map((collection) => (
                    <div key={collection.id} className="group flex items-center gap-1 rounded-md hover:bg-bg-primary">
                      <button className="min-w-0 flex-1 px-2 py-2 text-left text-sm text-text-primary truncate" disabled={isApplying} onClick={() => void applyCollection(collection)}>{collection.name}</button>
                      <Button className="h-8 w-8 shrink-0 p-0 bg-transparent text-text-secondary opacity-0 group-hover:opacity-100 hover:text-red-400 shadow-none" onClick={() => void deleteCollection(collection)} data-tooltip={`Delete ${collection.name}`}>
                        <Trash2 size={15} />
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function CatalogAiTaggingButton() {
  const [isStarting, setIsStarting] = useState(false);
  const librarySource = useLibraryStore((state) => state.librarySource);
  const isCatalogAvailable = librarySource.type === 'catalog';

  const startTagging = async () => {
    if (!isCatalogAvailable) {
      toast.info('Open or create a SQLite library first.');
      return;
    }

    setIsStarting(true);
    try {
      await invoke<string>(Invokes.StartCatalogAiTagging);
      toast.info('AI tagging started. Progress is available in Background Jobs.');
    } catch (error) {
      console.error('Failed to start catalog AI tagging:', error);
      toast.error(`Failed to start AI tagging: ${error}`);
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <Button
      className="h-12 w-12 bg-transparent text-text-primary shadow-none p-0 flex items-center justify-center"
      disabled={isStarting || !isCatalogAvailable}
      onClick={() => void startTagging()}
      data-tooltip={
        isCatalogAvailable
          ? 'Analyze catalog with AI tags'
          : 'Open a SQLite library to analyze catalog images'
      }
    >
      {isStarting ? <Loader2 className="w-5 h-5 animate-spin" /> : <Sparkles className="w-5 h-5" />}
    </Button>
  );
}

export function CatalogRamPlusTaggingButton() {
  const [isStarting, setIsStarting] = useState(false);
  const librarySource = useLibraryStore((state) => state.librarySource);
  const isCatalogAvailable = librarySource.type === 'catalog';
  const startTagging = async () => {
    if (!isCatalogAvailable) { toast.info('Open or create a SQLite library first.'); return; }
    setIsStarting(true);
    try { await invoke<string>(Invokes.StartCatalogRamPlusTagging); toast.info('RAM++ tagging started. Progress is available in Background Jobs.'); }
    catch (error) { console.error('Failed to start RAM++ tagging:', error); toast.error(`Failed to start RAM++ tagging: ${error}`); }
    finally { setIsStarting(false); }
  };
  return <Button className="h-12 w-12 bg-transparent text-text-primary shadow-none p-0 flex items-center justify-center" disabled={isStarting || !isCatalogAvailable} onClick={() => void startTagging()} data-tooltip={isCatalogAvailable ? 'Analyze catalog with RAM++ broad tags' : 'Open a SQLite library to analyze catalog images'}>{isStarting ? <Loader2 className="w-5 h-5 animate-spin" /> : <Tags className="w-5 h-5" />}</Button>;
}

export function CatalogAiTagReviewButton() {
  const [items, setItems] = useState<Array<{ id: number; tag: string; imagePath: string; confidence: number }>>([]);
  const [open, setOpen] = useState(false);
  const load = async () => { try { setItems(await invoke(Invokes.ListSuggestedAiTags)); } catch (error) { console.error('Failed to load AI tag suggestions:', error); toast.error(`Failed to load AI tag suggestions: ${error}`); } };
  const review = async (id: number, reviewState: 'accepted' | 'rejected') => { try { await invoke(Invokes.ReviewAiTag, { id, reviewState }); await load(); } catch (error) { console.error('Failed to review AI tag:', error); toast.error(`Failed to review AI tag: ${error}`); } };
  const reviewAll = async (reviewState: 'accepted' | 'rejected') => { try { await invoke(Invokes.ReviewAiTags, { ids: items.map((item) => item.id), reviewState }); await load(); } catch (error) { toast.error(`Failed to review AI tags: ${error}`); } };
  return <div className="relative"><Button className="h-12 w-12 bg-transparent text-text-primary shadow-none p-0 flex items-center justify-center" onClick={() => { setOpen(!open); if (!open) void load(); }} data-tooltip="Review AI tag suggestions"><Tags className="w-5 h-5" /></Button>{open && <div className="absolute right-0 mt-2 w-72 max-h-80 overflow-y-auto z-50 bg-surface border border-border-color rounded-md shadow-xl p-3"><div className="flex items-center justify-between gap-2"><Text variant={TextVariants.small}>AI tag suggestions</Text>{items.length > 0 && <div className="flex gap-2 text-xs"><button className="text-green-300" onClick={() => void reviewAll('accepted')}>Accept all</button><button className="text-red-300" onClick={() => void reviewAll('rejected')}>Reject all</button></div>}</div>{items.length === 0 ? <Text variant={TextVariants.small} color={TextColors.secondary}>No suggestions to review</Text> : items.map((item) => <div key={item.id} className="flex items-center justify-between gap-2 py-2 border-b border-border-color/40"><img src={convertFileSrc(item.imagePath)} className="w-10 h-10 object-cover rounded-sm" alt="" /><div className="min-w-0 flex-1"><Text variant={TextVariants.small}>{item.tag}</Text><Text variant={TextVariants.small} color={TextColors.secondary}>{item.confidence > 0 ? `${Math.round(item.confidence * 100)}% confidence` : 'Derived tag'}</Text></div><div className="flex gap-1"><button className="text-green-300 text-xs" onClick={() => void review(item.id, 'accepted')}>Accept</button><button className="text-red-300 text-xs" onClick={() => void review(item.id, 'rejected')}>Reject</button></div></div>)}</div>}</div>;
}

export function CatalogSpeciesReviewButton() {
  const [items, setItems] = useState<Array<{ id: number; scientificName: string; commonName?: string; imagePath: string; confidence: number }>>([]);
  const [open, setOpen] = useState(false);
  const load = async () => { try { setItems(await invoke('list_suggested_species')); } catch (error) { console.error('Failed to load species suggestions:', error); toast.error(`Failed to load species suggestions: ${error}`); } };
  const review = async (id: number, reviewState: 'accepted' | 'rejected') => { try { await invoke('review_species', { id, reviewState }); await load(); } catch (error) { console.error('Failed to review species:', error); toast.error(`Failed to review species: ${error}`); } };
  const reviewAll = async (reviewState: 'accepted' | 'rejected') => { try { await invoke('review_species_batch', { ids: items.map((item) => item.id), reviewState }); await load(); } catch (error) { toast.error(`Failed to review species: ${error}`); } };
  return <div className="relative"><Button className="h-12 w-12 bg-transparent text-text-primary shadow-none p-0 flex items-center justify-center" onClick={() => { setOpen(!open); if (!open) void load(); }} data-tooltip="Review BioCLIP species suggestions"><Bookmark className="w-5 h-5" /></Button>{open && <div className="absolute right-0 mt-2 w-80 max-h-80 overflow-y-auto z-50 bg-surface border border-border-color rounded-md shadow-xl p-3"><div className="flex items-center justify-between gap-2"><Text variant={TextVariants.small} weight={TextWeights.semibold}>Species Classifications</Text>{items.length > 0 && <div className="flex gap-2 text-xs"><button className="text-green-300" onClick={() => void reviewAll('accepted')}>Accept all</button><button className="text-red-300" onClick={() => void reviewAll('rejected')}>Reject all</button></div>}</div>{items.length === 0 ? <Text variant={TextVariants.small} color={TextColors.secondary} className="mt-2">No species suggestions to review</Text> : items.map((item) => <div key={item.id} className="flex items-center justify-between gap-2 py-2 border-b border-border-color/40"><img src={convertFileSrc(item.imagePath)} className="w-10 h-10 object-cover rounded-sm" alt="" /><div className="min-w-0 flex-1"><Text variant={TextVariants.small} weight={TextWeights.medium} className="truncate">{item.commonName || item.scientificName}</Text><Text variant={TextVariants.small} color={TextColors.secondary} className="truncate italic">{item.scientificName}</Text><Text variant={TextVariants.small} color={TextColors.secondary}>{Math.round(item.confidence * 100)}% match</Text></div><div className="flex gap-1"><button className="text-green-300 text-xs px-2 py-1 bg-green-500/10 rounded" onClick={() => void review(item.id, 'accepted')}>Accept</button><button className="text-red-300 text-xs px-2 py-1 bg-red-500/10 rounded" onClick={() => void review(item.id, 'rejected')}>Reject</button></div></div>)}</div>}</div>;
}

export function CatalogEnhanceMenu() {
  const [open, setOpen] = useState(false);
  const [denoiseStrength, setDenoiseStrength] = useState(0.8);
  const [isProcessing, setIsProcessing] = useState(false);
  const librarySource = useLibraryStore((state) => state.librarySource);
  const libraryActivePath = useLibraryStore((state) => state.libraryActivePath);
  const imageList = useLibraryStore((state) => state.imageList);
  const isCatalogAvailable = librarySource.type === 'catalog';

  const handleRunEnhance = async (operationKind: string) => {
    if (!libraryActivePath) {
      toast.info('Select an image in the catalog to enhance.');
      return;
    }
    const currentImage = imageList.find((img) => img.path === libraryActivePath);
    if (!currentImage?.id) {
      toast.info('Selected image is not indexed in the catalog.');
      return;
    }
    if (operationKind === 'raw_denoise' && !currentImage.is_raw) {
      toast.info('RAW Restore is available only for RAW source images.');
      return;
    }
    setIsProcessing(true);
    try {
      const recipe = {
        operationKind,
        modelId: operationKind === 'raw_denoise' ? 'rawnind-utnet2-bayer' : 'nafnet-sidd-rgb',
        modelRevision: 'v1',
        denoiseStrength,
        // Finish-stage controls belong to the editor, never the catalog
        // restoration derivative.
        microcontrastStrength: 0,
        detailRecovery: 0,
        // rawnind-utnet2-bayer's ONNX graph has a static 512x512 input, and the
        // Bayer tiling code halves tileSize before feeding the model, so this
        // must be exactly 1024 for that model; nafnet-sidd-rgb's graph is
        // static at 768x768 and takes tileSize directly.
        tileSize: operationKind === 'raw_denoise' ? 1024 : 768,
        tileOverlap: 64,
      };
      await invoke('start_image_restoration', {
        imageId: currentImage.id,
        recipe,
      });
      toast.success(`${operationKind === 'raw_denoise' ? 'RAW Restore' : 'RGB Denoise'} job started. Check Background Jobs.`);
      setOpen(false);
    } catch (error) {
      console.error('Failed to start restoration:', error);
      toast.error(`Restoration failed: ${error}`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="relative">
      <Button
        className="h-12 w-12 bg-transparent text-text-primary shadow-none p-0 flex items-center justify-center"
        onClick={() => setOpen(!open)}
        data-tooltip={isCatalogAvailable ? 'RAW Restore and RGB Denoise' : 'Open a SQLite library to restore catalog images'}
        disabled={!isCatalogAvailable}
      >
        <Sparkles className="w-5 h-5 text-accent" />
      </Button>
      {open && (
        <div className="absolute right-0 mt-2 w-80 z-50 bg-surface/95 backdrop-blur-md border border-border-color rounded-lg shadow-xl p-4">
          <Text variant={TextVariants.subheading} weight={TextWeights.semibold} className="mb-3">
            RAW Restore
          </Text>

          <div className="space-y-3 mb-4">
            {!currentImage?.is_raw && <div>
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
            </div>}

          </div>

          <div className="flex gap-2">
            {currentImage?.is_raw ? <Button
              className="flex-1 bg-accent text-white py-2 text-xs font-medium rounded-md"
              disabled={isProcessing}
              onClick={() => void handleRunEnhance('raw_denoise')}
              data-tooltip="Run Bayer RAW denoise and demosaic restoration"
            >
              {isProcessing ? <Loader2 className="w-4 h-4 animate-spin mx-auto" /> : 'RAW Restore'}
            </Button> : <Button
              className="flex-1 bg-surface-hover border border-border-color text-text-primary py-2 text-xs font-medium rounded-md"
              disabled={isProcessing}
              onClick={() => void handleRunEnhance('rgb_denoise')}
              data-tooltip="Run developed-image RGB denoise"
            >
              {isProcessing ? <Loader2 className="w-4 h-4 animate-spin mx-auto" /> : 'RGB Denoise'}
            </Button>}
          </div>
        </div>
      )}
    </div>
  );
}


const groupingOptionKeys = [
  { key: 'off' as GroupingMode, labelKey: 'library.header.viewOptions.groupOff' as const },
  { key: 'raw' as GroupingMode, labelKey: 'library.header.viewOptions.groupPreferRaw' as const },
  { key: 'jpeg' as GroupingMode, labelKey: 'library.header.viewOptions.groupPreferJpeg' as const },
];

interface ViewOptionsDropdownProps {
  libraryViewMode: LibraryViewMode;
  onSelectSize: (id: ThumbnailSize) => void;
  onSelectAspectRatio: (id: ThumbnailAspectRatio) => void;
  onLibraryRefresh?: () => void;
  setLibraryViewMode: (mode: LibraryViewMode) => void;
  thumbnailSize: ThumbnailSize;
  thumbnailAspectRatio: ThumbnailAspectRatio;
  thumbnailSizeOptions: Array<{ id: ThumbnailSize; label: string; size: number }>;
  thumbnailAspectRatioOptions: Array<{ id: ThumbnailAspectRatio; label: string }>;
  ratingFilterOptions: Array<{ value: number; label: string }>;
  rawStatusOptions: Array<{ key: RawStatus; label: string }>;
  editedStatusOptions: Array<{ key: EditedStatus; label: string }>;
  sortOptions: Array<{ key: string; label: string; disabled?: boolean }>;
}

export function ViewOptionsDropdown({
  libraryViewMode,
  onSelectSize,
  onSelectAspectRatio,
  onLibraryRefresh,
  setLibraryViewMode,
  thumbnailSize,
  thumbnailAspectRatio,
  thumbnailSizeOptions,
  thumbnailAspectRatioOptions,
  ratingFilterOptions,
  rawStatusOptions,
  editedStatusOptions,
  sortOptions,
}: ViewOptionsDropdownProps) {
  const { t } = useTranslation();
  const { filterCriteria, setFilterCriteria, sortCriteria, setSortCriteria } = useLibraryStore(
    useShallow((state) => ({
      filterCriteria: state.filterCriteria,
      setFilterCriteria: state.setFilterCriteria,
      sortCriteria: state.sortCriteria,
      setSortCriteria: state.setSortCriteria,
    })),
  );

  const { appSettings, handleSettingsChange } = useSettingsStore(
    useShallow((state) => ({
      appSettings: state.appSettings,
      handleSettingsChange: state.handleSettingsChange,
    })),
  );

  const groupingMode: GroupingMode = appSettings?.grouping ?? 'off';
  const requireMatchingExif = appSettings?.requireMatchingExif ?? false;

  const isFilterActive =
    filterCriteria.rating !== 0 ||
    (filterCriteria.rawStatus && filterCriteria.rawStatus !== RawStatus.All) ||
    (filterCriteria.editedStatus && filterCriteria.editedStatus !== EditedStatus.All) ||
    (filterCriteria.colors && filterCriteria.colors.length > 0);

  const [lastClickedColor, setLastClickedColor] = useState<string | null>(null);
  const allColors = useMemo(() => [...COLOR_LABELS, { name: 'none', color: '#9ca3af' }], []);

  const metadataOptions = useMemo(
    () => [
      { id: ExifOverlay.Off, label: t('library.header.viewOptions.metadataOff') },
      { id: ExifOverlay.Hover, label: t('library.header.viewOptions.metadataHover') },
      { id: ExifOverlay.Always, label: t('library.header.viewOptions.metadataAlways') },
    ],
    [t],
  );

  const handleColorClick = (colorName: string, event: any) => {
    const { ctrlKey, metaKey, shiftKey } = event;
    const isCtrlPressed = ctrlKey || metaKey;
    const currentColors = filterCriteria.colors || [];

    if (shiftKey && lastClickedColor) {
      const lastIndex = allColors.findIndex((c) => c.name === lastClickedColor);
      const currentIndex = allColors.findIndex((c) => c.name === colorName);
      if (lastIndex !== -1 && currentIndex !== -1) {
        const start = Math.min(lastIndex, currentIndex);
        const end = Math.max(lastIndex, currentIndex);
        const range = allColors.slice(start, end + 1).map((c: Color) => c.name);
        const baseSelection = isCtrlPressed ? currentColors : [lastClickedColor];
        const newColors = Array.from(new Set([...baseSelection, ...range]));
        setFilterCriteria((prev: FilterCriteria) => ({ ...prev, colors: newColors }));
      }
    } else if (isCtrlPressed) {
      const newColors = currentColors.includes(colorName)
        ? currentColors.filter((c: string) => c !== colorName)
        : [...currentColors, colorName];
      setFilterCriteria((prev: FilterCriteria) => ({ ...prev, colors: newColors }));
    } else {
      const newColors = currentColors.length === 1 && currentColors[0] === colorName ? [] : [colorName];
      setFilterCriteria((prev: FilterCriteria) => ({ ...prev, colors: newColors }));
    }
    setLastClickedColor(colorName);
  };

  return (
    <DropdownMenu
      buttonContent={
        <>
          <SlidersHorizontal className="w-8 h-8" />
          {isFilterActive && <div className="absolute -top-1 -right-1 bg-accent rounded-full w-3 h-3" />}
        </>
      }
      buttonTitle={t('library.header.viewOptions.title')}
      contentClassName="library-view-options-menu w-[760px]"
    >
      <div className="library-view-options-content flex">
        <div className="library-view-options-section w-1/2 py-4 px-2 border-r border-border-color space-y-5">
          <div>
            <div className="px-3 py-1 relative flex items-center">
              <Text as="div" variant={TextVariants.small} weight={TextWeights.semibold} className="uppercase">
                {t('library.header.viewOptions.sortBy')}
              </Text>
              <button
                onClick={() =>
                  setSortCriteria((prev: SortCriteria) => ({
                    ...prev,
                    order: prev.order === SortDirection.Ascending ? SortDirection.Descending : SortDirection.Ascending,
                  }))
                }
                data-tooltip={
                  sortCriteria.order === SortDirection.Ascending
                    ? t('library.header.viewOptions.sortDescending')
                    : t('library.header.viewOptions.sortAscending')
                }
                className="absolute top-1/2 right-3 -translate-y-1/2 p-1 bg-transparent border-none text-text-secondary hover:text-text-primary rounded-sm transition-colors"
              >
                {sortCriteria.order === SortDirection.Ascending ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
            </div>
            <div className="px-3 mt-1">
              <Dropdown
                options={sortOptions.map((opt) => ({ value: opt.key, label: opt.label, disabled: opt.disabled }))}
                value={sortCriteria.key}
                onChange={(val) => setSortCriteria((prev: SortCriteria) => ({ ...prev, key: val }))}
                triggerClassName="bg-bg-primary w-full"
              />
            </div>
          </div>

          <div>
            <Text as="div" variant={TextVariants.small} weight={TextWeights.semibold} className="px-3 py-1 uppercase">
              {t('library.header.viewOptions.thumbnailSize')}
            </Text>
            <div className="px-3 mt-1">
              <SegmentedSwitch options={thumbnailSizeOptions} value={thumbnailSize} onChange={onSelectSize} />
            </div>
          </div>

          <div>
            <Text as="div" variant={TextVariants.small} weight={TextWeights.semibold} className="px-3 py-1 uppercase">
              {t('library.header.viewOptions.thumbnailFit')}
            </Text>
            <div className="px-3 mt-1">
              <SegmentedSwitch
                options={thumbnailAspectRatioOptions}
                value={thumbnailAspectRatio}
                onChange={onSelectAspectRatio}
              />
            </div>
          </div>

          <div>
            <Text as="div" variant={TextVariants.small} weight={TextWeights.semibold} className="px-3 py-1 uppercase">
              {t('library.header.viewOptions.displayMode')}
            </Text>
            <div className="px-3 mt-1">
              <SegmentedSwitch
                options={[
                  { id: LibraryViewMode.Flat, label: t('library.header.viewOptions.currentFolder') },
                  { id: LibraryViewMode.Recursive, label: t('library.header.viewOptions.recursive') },
                ]}
                value={libraryViewMode}
                onChange={async (val) => {
                  setLibraryViewMode(val as LibraryViewMode);
                  if (appSettings) {
                    await handleSettingsChange({ ...appSettings, libraryViewMode: val as LibraryViewMode });
                    onLibraryRefresh?.();
                  }
                }}
              />
            </div>
          </div>

          <div>
            <Text as="div" variant={TextVariants.small} weight={TextWeights.semibold} className="px-3 py-1 uppercase">
              {t('library.header.viewOptions.showMetadata')}
            </Text>
            <div className="px-3 mt-1">
              <SegmentedSwitch
                options={metadataOptions}
                value={appSettings?.exifOverlay || ExifOverlay.Off}
                onChange={(val) => handleSettingsChange({ ...appSettings!, exifOverlay: val as ExifOverlay })}
              />
            </div>
          </div>
        </div>

        <div className="library-view-options-section w-1/2 py-4 px-2 space-y-5">
          <div>
            <Text as="div" variant={TextVariants.small} weight={TextWeights.semibold} className="px-3 py-1 uppercase">
              {t('library.header.viewOptions.filterByRating')}
            </Text>
            <div className="px-3 mt-1">
              <RatingSegmentedSwitch
                rating={filterCriteria.rating}
                onChange={(val: number) => setFilterCriteria((prev: FilterCriteria) => ({ ...prev, rating: val }))}
                ratingFilterOptions={ratingFilterOptions}
              />
            </div>
          </div>

          <div>
            <Text as="div" variant={TextVariants.small} weight={TextWeights.semibold} className="px-3 py-1 uppercase">
              {t('library.header.viewOptions.filterByFileType')}
            </Text>
            <div className="px-3 mt-1">
              <SegmentedSwitch
                options={rawStatusOptions.map((o) => ({ id: o.key, label: o.label }))}
                value={filterCriteria.rawStatus || RawStatus.All}
                onChange={(val) => setFilterCriteria((prev: FilterCriteria) => ({ ...prev, rawStatus: val }))}
              />
            </div>
          </div>

          <div>
            <Text as="div" variant={TextVariants.small} weight={TextWeights.semibold} className="px-3 py-1 uppercase">
              {t('library.header.viewOptions.filterByEdited', 'Filter by Edit Status')}
            </Text>
            <div className="px-3 mt-1">
              <SegmentedSwitch
                options={editedStatusOptions.map((o) => ({ id: o.key, label: o.label }))}
                value={filterCriteria.editedStatus || EditedStatus.All}
                onChange={(val) => setFilterCriteria((prev: FilterCriteria) => ({ ...prev, editedStatus: val }))}
              />
            </div>
          </div>

          <div>
            <Text as="div" variant={TextVariants.small} weight={TextWeights.semibold} className="px-3 py-1 uppercase">
              {t('library.header.viewOptions.groupRawJpeg')}
            </Text>
            <div className="px-3 mt-1">
              <SegmentedSwitch
                options={groupingOptionKeys.map((o) => ({ id: o.key, label: t(o.labelKey) }))}
                value={groupingMode}
                onChange={async (val) => {
                  if (appSettings) {
                    await handleSettingsChange({ ...appSettings, grouping: val as GroupingMode });
                  }
                }}
              />
              <AnimatePresence initial={false}>
                {groupingMode !== 'off' && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.2, ease: 'easeInOut' }}
                    className="overflow-hidden"
                  >
                    <div className="pt-2 space-y-2 px-1">
                      <Switch
                        checked={!requireMatchingExif}
                        id="group-ignore-metadata-toggle"
                        label={t('library.header.viewOptions.groupIgnoreMetadata')}
                        onChange={async (checked) => {
                          if (appSettings) {
                            await handleSettingsChange({ ...appSettings, requireMatchingExif: !checked });
                            onLibraryRefresh?.();
                          }
                        }}
                      />
                      <Switch
                        checked={appSettings?.groupEditedFiles ?? true}
                        id="group-edited-files-toggle"
                        label={t('library.header.viewOptions.groupEditedFiles')}
                        onChange={async (checked) => {
                          if (appSettings) {
                            await handleSettingsChange({ ...appSettings, groupEditedFiles: checked });
                            onLibraryRefresh?.();
                          }
                        }}
                      />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          <div>
            <Text as="div" variant={TextVariants.small} weight={TextWeights.semibold} className="px-3 py-1 uppercase">
              {t('library.header.viewOptions.filterByColorLabel')}
            </Text>
            <div className="flex flex-wrap gap-2.5 px-3 py-1.5">
              {allColors.map((color: Color) => {
                const isSelected = (filterCriteria.colors || []).includes(color.name);
                const title =
                  color.name === 'none'
                    ? t('library.header.viewOptions.noLabel')
                    : t(`contextMenus.colors.${color.name}`, {
                        defaultValue: color.name.charAt(0).toUpperCase() + color.name.slice(1),
                      });
                return (
                  <button
                    key={color.name}
                    data-tooltip={title}
                    onClick={(e: any) => handleColorClick(color.name, e)}
                    className="w-5 h-5 rounded-full focus:outline-hidden focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-surface transition-transform hover:scale-110"
                    role="menuitem"
                  >
                    <div className="relative w-full h-full">
                      <div className="w-full h-full rounded-full" style={{ backgroundColor: color.color }}></div>
                      {isSelected && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/30 rounded-full">
                          <Check size={12} className={TEXT_COLOR_KEYS[TextColors.white]} />
                        </div>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </DropdownMenu>
  );
}
