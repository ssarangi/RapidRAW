import { invoke } from '@tauri-apps/api/core';
import {
  Search,
  Image as ImageIcon,
  Users,
  X,
  CornerDownLeft,
  Loader2,
  Sparkles,
  SlidersHorizontal,
} from 'lucide-react';
import { type ReactNode, useEffect, useRef, useState } from 'react';

import { CatalogSearchQuery, ImageFile, Invokes } from './AppProperties';
import { useLibraryStore } from '../../store/useLibraryStore';
import { useUIStore } from '../../store/useUIStore';
import { useSettingsStore } from '../../store/useSettingsStore';

interface PersonResult {
  id: number;
  displayName: string;
  faceCount: number;
}

interface GeminiCatalogSearchResult {
  summary: string;
  query: CatalogSearchQuery;
}

interface AdvancedSearchFields {
  minRating: string;
  year: string;
  dateFrom: string;
  dateTo: string;
  camera: string;
  lens: string;
  people: string;
  excludedPeople: string;
  tags: string;
  aiTags: string;
  excludedTags: string;
  excludedAiTags: string;
  color: string;
  raw: 'any' | 'raw' | 'nonRaw';
  edited: 'any' | 'edited' | 'unedited';
  tagMode: 'AND' | 'OR';
}

const EMPTY_ADVANCED_SEARCH: AdvancedSearchFields = {
  minRating: '',
  year: '',
  dateFrom: '',
  dateTo: '',
  camera: '',
  lens: '',
  people: '',
  excludedPeople: '',
  tags: '',
  aiTags: '',
  excludedTags: '',
  excludedAiTags: '',
  color: '',
  raw: 'any',
  edited: 'any',
  tagMode: 'AND',
};

const filename = (path: string) => path.split(/[\\/]/).pop() || path;

type LibrarySearchSnapshot = Pick<
  ReturnType<typeof useLibraryStore.getState>,
  | 'currentFolderPath'
  | 'rootPaths'
  | 'librarySource'
  | 'activeCatalogRootId'
  | 'activeAlbumId'
  | 'imageList'
  | 'imageRatings'
  | 'multiSelectedPaths'
  | 'selectionAnchorPath'
  | 'libraryActivePath'
  | 'libraryScrollTop'
  | 'searchCriteria'
> & {
  activeView: string;
};

// The palette remains mounted while Library swaps its result set. Keeping the
// snapshot here lets the clear pill restore exactly what the user was viewing.
let searchSnapshot: LibrarySearchSnapshot | null = null;

/**
 * A shell-level search affordance. It deliberately lives above all views so a
 * new view inherits search without adding another header or bespoke shortcut.
 */
export default function GlobalSearchPalette() {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [people, setPeople] = useState<PersonResult[]>([]);
  const [photos, setPhotos] = useState<ImageFile[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isAiInterpreting, setIsAiInterpreting] = useState(false);
  const [aiResult, setAiResult] = useState<GeminiCatalogSearchResult | null>(null);
  const [aiError, setAiError] = useState('');
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  const [advancedFields, setAdvancedFields] = useState<AdvancedSearchFields>(EMPTY_ADVANCED_SEARCH);
  const [isGeminiHelpOpen, setIsGeminiHelpOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const activeView = useUIStore((state) => state.activeView);
  const hasGeminiApiKey = useSettingsStore((state) => Boolean(state.appSettings?.geminiApiKey?.trim()));

  const close = () => {
    setIsOpen(false);
    setQuery('');
    setPeople([]);
    setPhotos([]);
    setAiResult(null);
    setAiError('');
    setIsAdvancedOpen(false);
    setAdvancedFields(EMPTY_ADVANCED_SEARCH);
    setIsGeminiHelpOpen(false);
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        setIsOpen(true);
      }
      if (event.key === 'Escape' && isOpen) close();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isOpen]);

  useEffect(() => {
    const openPalette = () => setIsOpen(true);
    window.addEventListener('rapidraw:open-global-search', openPalette);
    return () => window.removeEventListener('rapidraw:open-global-search', openPalette);
  }, []);

  useEffect(() => {
    const clearGlobalSearch = () => {
      if (searchSnapshot) {
        const { activeView: previousView, ...librarySnapshot } = searchSnapshot;
        useLibraryStore.getState().setLibrary(librarySnapshot);
        useUIStore.getState().setUI({ activeView: previousView, globalCatalogSearchLabel: null });
        searchSnapshot = null;
      } else {
        useUIStore.getState().setUI({ globalCatalogSearchLabel: null });
      }
    };
    window.addEventListener('rapidraw:clear-global-search', clearGlobalSearch);
    return () => window.removeEventListener('rapidraw:clear-global-search', clearGlobalSearch);
  }, []);

  useEffect(() => {
    const discardGlobalSearch = () => {
      // Leaving the session intentionally discards transient results. A later
      // Continue Session restore must never inherit an old search pill or its
      // in-memory result snapshot.
      searchSnapshot = null;
      useUIStore.getState().setUI({ globalCatalogSearchLabel: null });
    };
    window.addEventListener('rapidraw:discard-global-search', discardGlobalSearch);
    return () => window.removeEventListener('rapidraw:discard-global-search', discardGlobalSearch);
  }, []);

  useEffect(() => {
    if (isOpen) window.setTimeout(() => inputRef.current?.focus(), 0);
  }, [isOpen]);

  useEffect(() => {
    const trimmedQuery = query.trim();
    if (trimmedQuery.length < 2) {
      setPeople([]);
      setPhotos([]);
      setIsSearching(false);
      return;
    }
    setAiResult(null);
    setAiError('');

    let cancelled = false;
    const timer = window.setTimeout(() => {
      setIsSearching(true);
      void Promise.all([
        invoke<PersonResult[]>(Invokes.SearchCatalogPeople, { query: trimmedQuery }).catch(() => []),
        invoke<ImageFile[]>(Invokes.SearchCatalogImages, { query: { text: trimmedQuery, limit: 8 } }).catch(() => []),
      ])
        .then(([nextPeople, nextPhotos]) => {
          if (!cancelled) {
            setPeople(nextPeople.slice(0, 5));
            setPhotos(nextPhotos.slice(0, 8));
          }
        })
        .finally(() => {
          if (!cancelled) setIsSearching(false);
        });
    }, 180);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [query]);

  const rememberSearchSnapshot = () => {
    if (searchSnapshot) return;
    const library = useLibraryStore.getState();
    searchSnapshot = {
      currentFolderPath: library.currentFolderPath,
      rootPaths: library.rootPaths,
      librarySource: library.librarySource,
      activeCatalogRootId: library.activeCatalogRootId,
      activeAlbumId: library.activeAlbumId,
      imageList: library.imageList,
      imageRatings: library.imageRatings,
      multiSelectedPaths: library.multiSelectedPaths,
      selectionAnchorPath: library.selectionAnchorPath,
      libraryActivePath: library.libraryActivePath,
      libraryScrollTop: library.libraryScrollTop,
      searchCriteria: library.searchCriteria,
      activeView: useUIStore.getState().activeView,
    };
  };

  const showCatalogResults = (files: ImageFile[], label: string) => {
    rememberSearchSnapshot();
    const imageRatings: Record<string, number> = {};
    files.forEach((file) => {
      imageRatings[file.path] = file.rating || 0;
    });
    useLibraryStore.getState().setLibrary({
      // A search is an overlay on the selected folder/catalog, not a new
      // folder. The bottom-bar pill is its single visible representation.
      imageList: files,
      imageRatings,
      multiSelectedPaths: [],
      selectionAnchorPath: null,
      libraryActivePath: null,
      libraryScrollTop: 0,
    });
    useLibraryStore.getState().setSearchCriteria({ text: '', tags: [], mode: 'OR' });
    useUIStore.getState().setUI({ activeView: 'library', globalCatalogSearchLabel: label });
    close();
  };

  const openCatalogResults = async (catalogQuery: CatalogSearchQuery, label: string) => {
    try {
      const files = await invoke<ImageFile[]>(Invokes.SearchCatalogImages, {
        query: { ...catalogQuery, limit: 20_000 },
      });
      showCatalogResults(files, label);
    } catch {
      // Filesystem-only browsing has no catalog index; retain its existing
      // in-view filter behavior instead.
      rememberSearchSnapshot();
      useLibraryStore.getState().setSearchCriteria({ text: catalogQuery.text || label, tags: [], mode: 'OR' });
      useUIStore.getState().setUI({ activeView: 'library', globalCatalogSearchLabel: label });
      close();
    }
  };

  const openPhotoResults = (text: string) => openCatalogResults({ text }, text);

  const interpretWithGemini = async () => {
    const naturalLanguage = query.trim();
    if (!naturalLanguage) return;
    if (!hasGeminiApiKey) {
      setIsGeminiHelpOpen(true);
      return;
    }
    setIsAiInterpreting(true);
    setAiError('');
    try {
      const result = await invoke<GeminiCatalogSearchResult>(Invokes.InterpretGeminiCatalogSearch, { naturalLanguage });
      if (isAdvancedOpen) {
        populateAdvancedFilters(result.query);
        setAiResult(result);
      } else {
        await openCatalogResults(result.query, result.summary);
      }
    } catch (error) {
      setAiError(String(error));
    } finally {
      setIsAiInterpreting(false);
    }
  };

  const populateAdvancedFilters = (catalogQuery: CatalogSearchQuery) => {
    const formatDate = (timestamp: number | null | undefined) =>
      timestamp ? new Date(timestamp * 1000).toISOString().slice(0, 10) : '';
    setAdvancedFields((fields) => ({
      ...fields,
      minRating: catalogQuery.minRating?.toString() || '',
      year: catalogQuery.year?.toString() || '',
      dateFrom: formatDate(catalogQuery.dateFrom),
      dateTo: formatDate(catalogQuery.dateTo),
      camera: catalogQuery.camera || '',
      lens: catalogQuery.lens || '',
      people: (catalogQuery.people || (catalogQuery.person ? [catalogQuery.person] : [])).join(', '),
      excludedPeople: (catalogQuery.excludedPeople || []).join(', '),
      tags: (catalogQuery.tags || []).join(', '),
      aiTags: (catalogQuery.aiTags || []).join(', '),
      excludedTags: (catalogQuery.excludedTags || []).join(', '),
      excludedAiTags: (catalogQuery.excludedAiTags || []).join(', '),
      color: catalogQuery.color || '',
      raw:
        catalogQuery.isRaw === null || catalogQuery.isRaw === undefined ? 'any' : catalogQuery.isRaw ? 'raw' : 'nonRaw',
      edited:
        catalogQuery.isEdited === null || catalogQuery.isEdited === undefined
          ? 'any'
          : catalogQuery.isEdited
            ? 'edited'
            : 'unedited',
      tagMode: catalogQuery.tagMode || 'AND',
    }));
  };

  const runAdvancedSearch = () => {
    const csv = (value: string) =>
      value
        .split(',')
        .map((tag) => tag.trim())
        .filter(Boolean);
    const tags = csv(advancedFields.tags);
    const aiTags = csv(advancedFields.aiTags);
    const dateStart = advancedFields.dateFrom ? new Date(`${advancedFields.dateFrom}T00:00:00`).getTime() / 1000 : null;
    const dateEnd = advancedFields.dateTo ? new Date(`${advancedFields.dateTo}T23:59:59.999`).getTime() / 1000 : null;
    const minRating = Number(advancedFields.minRating);
    const year = Number(advancedFields.year);
    const catalogQuery: CatalogSearchQuery = {
      // After Gemini fills Filters, use only the text constraint it explicitly
      // returned. The user's natural-language sentence is an instruction, not
      // a filename/metadata predicate.
      text: aiResult?.query.text ?? (query.trim() || null),
      minRating: Number.isFinite(minRating) && minRating > 0 ? minRating : null,
      year: Number.isFinite(year) && year >= 1900 ? year : null,
      camera: advancedFields.camera.trim() || null,
      lens: advancedFields.lens.trim() || null,
      people: csv(advancedFields.people),
      excludedPeople: csv(advancedFields.excludedPeople),
      tags: tags.length > 0 ? tags : null,
      aiTags: aiTags.length > 0 ? aiTags : null,
      excludedTags: csv(advancedFields.excludedTags),
      excludedAiTags: csv(advancedFields.excludedAiTags),
      tagMode: tags.length > 1 ? advancedFields.tagMode : null,
      dateFrom: dateStart !== null && Number.isFinite(dateStart) ? dateStart : null,
      dateTo: dateEnd !== null && Number.isFinite(dateEnd) ? dateEnd : null,
      color: advancedFields.color || null,
      isRaw: advancedFields.raw === 'any' ? null : advancedFields.raw === 'raw',
      isEdited: advancedFields.edited === 'any' ? null : advancedFields.edited === 'edited',
    };
    void openCatalogResults(catalogQuery, aiResult?.summary || catalogQueryLabel(catalogQuery));
  };

  const openPerson = async (person: PersonResult) => {
    try {
      const files = await invoke<ImageFile[]>(Invokes.SearchCatalogImages, {
        query: { person: person.displayName, limit: 20_000 },
      });
      showCatalogResults(files, person.displayName);
    } catch {
      // A filesystem-only source has no catalog search index. The palette stays
      // open so the user can refine their query instead of losing it.
    }
  };

  const queryHint = activeView === 'people' ? 'Search people or photos' : 'Search photos, people, tags, metadata…';

  return (
    <>
      {isOpen && (
        <div className="absolute inset-0 z-50 flex items-start justify-center bg-black/50 px-4 pt-[10vh] backdrop-blur-sm">
          <div className="w-full max-w-2xl overflow-hidden rounded-2xl border border-border-color bg-bg-primary shadow-2xl">
            <div className="flex items-center gap-3 border-b border-border-color px-4 py-3">
              {isSearching ? (
                <Loader2 size={19} className="animate-spin text-accent" />
              ) : (
                <Search size={19} className="text-accent" />
              )}
              <span
                className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-accent/15 text-accent"
                title="Gemini-enabled search"
                aria-label="Gemini-enabled search"
              >
                <Sparkles size={14} />
              </span>
              <input
                ref={inputRef}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && query.trim()) {
                    event.preventDefault();
                    if (hasGeminiApiKey) void interpretWithGemini();
                    else void openPhotoResults(query.trim());
                  }
                }}
                className="min-w-0 flex-1 bg-transparent text-base text-text-primary outline-none placeholder:text-text-secondary"
                placeholder={queryHint}
              />
              <button
                type="button"
                onClick={() => setIsAdvancedOpen((open) => !open)}
                className={`inline-flex shrink-0 items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition ${
                  isAdvancedOpen
                    ? 'bg-surface text-text-primary'
                    : 'text-text-secondary hover:bg-surface hover:text-text-primary'
                }`}
                title="Advanced catalog filters"
              >
                <SlidersHorizontal size={14} />
                Filters
              </button>
              {query.trim().length >= 2 && (
                <button
                  type="button"
                  onClick={() => void interpretWithGemini()}
                  disabled={isAiInterpreting}
                  className="inline-flex shrink-0 items-center gap-1.5 rounded-md bg-accent/15 px-2.5 py-1.5 text-xs font-medium text-accent transition hover:bg-accent/25 disabled:cursor-wait disabled:opacity-60"
                  title="Interpret this request with Gemini"
                >
                  {isAiInterpreting ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
                  Ask Gemini
                </button>
              )}
              <button
                type="button"
                onClick={close}
                className="rounded-md p-1.5 text-text-secondary transition hover:bg-surface hover:text-text-primary"
                aria-label="Close search"
              >
                <X size={17} />
              </button>
            </div>

            <div className="max-h-[56vh] overflow-y-auto p-2">
              {isAiInterpreting && (
                <div className="m-1 mb-3 flex items-center gap-3 rounded-xl border border-accent/35 bg-accent/10 p-3 text-sm text-text-primary">
                  <Loader2 size={17} className="shrink-0 animate-spin text-accent" />
                  <div className="font-medium">Gemini is interpreting your request…</div>
                </div>
              )}
              {isGeminiHelpOpen && (
                <div className="m-1 mb-3 rounded-xl border border-accent/35 bg-accent/10 p-3 text-sm">
                  <div className="flex items-center gap-2 font-medium text-text-primary">
                    <Sparkles size={15} className="text-accent" /> Gemini search needs an API key
                  </div>
                  <ol className="mt-2 list-decimal space-y-1 pl-5 text-xs text-text-secondary">
                    <li>Open Google AI Studio and create an API key.</li>
                    <li>In RapidRAW, open Settings and paste it into the Gemini API key field.</li>
                    <li>Use Test key, then return here and ask Gemini to interpret your search.</li>
                  </ol>
                  <div className="mt-3 flex items-center gap-2">
                    <a
                      className="text-xs font-medium text-accent hover:underline"
                      href="https://aistudio.google.com/app/apikey"
                      target="_blank"
                      rel="noreferrer"
                    >
                      Get a Gemini API key
                    </a>
                    <button
                      type="button"
                      className="rounded-md border border-accent/40 px-2 py-1 text-xs font-medium text-accent hover:bg-accent/10"
                      onClick={() => {
                        useUIStore.getState().setUI({ isSettingsOpen: true });
                        close();
                      }}
                    >
                      Open Settings
                    </button>
                  </div>
                </div>
              )}
              {isAdvancedOpen && (
                <div className="m-1 mb-3 rounded-xl border border-border-color bg-surface/60 p-3">
                  <div className="mb-3 flex items-center gap-2 text-sm font-medium text-text-primary">
                    <SlidersHorizontal size={15} className="text-accent" /> Advanced catalog filters
                  </div>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                    <FilterField label="At least rating">
                      <select
                        className="catalog-search-select"
                        value={advancedFields.minRating}
                        onChange={(event) =>
                          setAdvancedFields((fields) => ({ ...fields, minRating: event.target.value }))
                        }
                      >
                        {['', '1', '2', '3', '4', '5'].map((value) => (
                          <option key={value} value={value}>
                            {value ? `${value} stars` : 'Any rating'}
                          </option>
                        ))}
                      </select>
                    </FilterField>
                    <FilterField label="Year">
                      <input
                        value={advancedFields.year}
                        onChange={(event) => setAdvancedFields((fields) => ({ ...fields, year: event.target.value }))}
                        placeholder="e.g. 2025"
                        inputMode="numeric"
                      />
                    </FilterField>
                    <FilterField label="From date">
                      <input
                        type="date"
                        value={advancedFields.dateFrom}
                        onChange={(event) =>
                          setAdvancedFields((fields) => ({ ...fields, dateFrom: event.target.value }))
                        }
                      />
                    </FilterField>
                    <FilterField label="To date">
                      <input
                        type="date"
                        value={advancedFields.dateTo}
                        onChange={(event) => setAdvancedFields((fields) => ({ ...fields, dateTo: event.target.value }))}
                      />
                    </FilterField>
                    <FilterField label="People together">
                      <input
                        value={advancedFields.people}
                        onChange={(event) => setAdvancedFields((fields) => ({ ...fields, people: event.target.value }))}
                        placeholder="Comma-separated"
                      />
                    </FilterField>
                    <FilterField label="Exclude people">
                      <input
                        value={advancedFields.excludedPeople}
                        onChange={(event) =>
                          setAdvancedFields((fields) => ({ ...fields, excludedPeople: event.target.value }))
                        }
                        placeholder="Comma-separated"
                      />
                    </FilterField>
                    <FilterField label="Camera">
                      <input
                        value={advancedFields.camera}
                        onChange={(event) => setAdvancedFields((fields) => ({ ...fields, camera: event.target.value }))}
                        placeholder="Camera model"
                      />
                    </FilterField>
                    <FilterField label="Lens">
                      <input
                        value={advancedFields.lens}
                        onChange={(event) => setAdvancedFields((fields) => ({ ...fields, lens: event.target.value }))}
                        placeholder="Lens"
                      />
                    </FilterField>
                    <FilterField label="Tags">
                      <input
                        value={advancedFields.tags}
                        onChange={(event) => setAdvancedFields((fields) => ({ ...fields, tags: event.target.value }))}
                        placeholder="Comma-separated"
                      />
                    </FilterField>
                    <FilterField label="AI tags">
                      <input
                        value={advancedFields.aiTags}
                        onChange={(event) => setAdvancedFields((fields) => ({ ...fields, aiTags: event.target.value }))}
                        placeholder="Comma-separated"
                      />
                    </FilterField>
                    <FilterField label="Exclude tags">
                      <input
                        value={advancedFields.excludedTags}
                        onChange={(event) =>
                          setAdvancedFields((fields) => ({ ...fields, excludedTags: event.target.value }))
                        }
                        placeholder="Comma-separated"
                      />
                    </FilterField>
                    <FilterField label="Exclude AI tags">
                      <input
                        value={advancedFields.excludedAiTags}
                        onChange={(event) =>
                          setAdvancedFields((fields) => ({ ...fields, excludedAiTags: event.target.value }))
                        }
                        placeholder="Comma-separated"
                      />
                    </FilterField>
                    <FilterField label="Tags match">
                      <select
                        className="catalog-search-select"
                        value={advancedFields.tagMode}
                        onChange={(event) =>
                          setAdvancedFields((fields) => ({
                            ...fields,
                            tagMode: event.target.value as AdvancedSearchFields['tagMode'],
                          }))
                        }
                      >
                        <option value="AND">All tags</option>
                        <option value="OR">Any tag</option>
                      </select>
                    </FilterField>
                    <FilterField label="Color label">
                      <input
                        value={advancedFields.color}
                        onChange={(event) => setAdvancedFields((fields) => ({ ...fields, color: event.target.value }))}
                        placeholder="e.g. red"
                      />
                    </FilterField>
                    <FilterField label="File type">
                      <select
                        className="catalog-search-select"
                        value={advancedFields.raw}
                        onChange={(event) =>
                          setAdvancedFields((fields) => ({
                            ...fields,
                            raw: event.target.value as AdvancedSearchFields['raw'],
                          }))
                        }
                      >
                        <option value="any">Any file</option>
                        <option value="raw">RAW only</option>
                        <option value="nonRaw">Non-RAW only</option>
                      </select>
                    </FilterField>
                    <FilterField label="Edit state">
                      <select
                        className="catalog-search-select"
                        value={advancedFields.edited}
                        onChange={(event) =>
                          setAdvancedFields((fields) => ({
                            ...fields,
                            edited: event.target.value as AdvancedSearchFields['edited'],
                          }))
                        }
                      >
                        <option value="any">Any state</option>
                        <option value="edited">Edited</option>
                        <option value="unedited">Unedited</option>
                      </select>
                    </FilterField>
                  </div>
                  <div className="mt-3 flex items-center justify-between">
                    <button
                      type="button"
                      className="text-xs text-text-secondary hover:text-text-primary"
                      onClick={() => {
                        setAdvancedFields(EMPTY_ADVANCED_SEARCH);
                        setAiResult(null);
                      }}
                    >
                      Clear filters
                    </button>
                    <button
                      type="button"
                      className="rounded-md bg-accent px-3 py-1.5 text-xs font-semibold text-button-text hover:brightness-110"
                      onClick={runAdvancedSearch}
                    >
                      Search catalog
                    </button>
                  </div>
                </div>
              )}
              {aiResult && (
                <div className="m-1 mb-3 rounded-xl border border-accent/35 bg-accent/10 p-3">
                  <div className="flex items-center gap-2 text-sm font-medium text-text-primary">
                    <Sparkles size={15} className="text-accent" />
                    Gemini filled the filters: {aiResult.summary}
                  </div>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {describeCatalogQuery(aiResult.query).map((filter) => (
                      <span key={filter} className="rounded bg-bg-primary/80 px-2 py-1 text-[11px] text-text-secondary">
                        {filter}
                      </span>
                    ))}
                  </div>
                  <p className="mt-2 text-xs text-text-secondary">Review the filters, then choose Search catalog.</p>
                </div>
              )}
              {aiError && (
                <div className="m-1 mb-3 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">
                  {aiError}
                </div>
              )}
              {query.trim().length < 2 ? (
                <div className="px-3 py-8 text-center">
                  <Search size={24} className="mx-auto mb-3 text-accent/80" />
                  <p className="text-sm font-medium text-text-primary">Find your library from anywhere</p>
                  <p className="mt-1 text-xs text-text-secondary">
                    Search locally by filename, tags, metadata, or a person’s name — or ask Gemini to interpret a
                    natural-language request.
                  </p>
                </div>
              ) : (
                <>
                  {people.length > 0 && (
                    <SearchSection title="People" icon={<Users size={14} />}>
                      {people.map((person) => (
                        <ResultButton key={person.id} onClick={() => void openPerson(person)}>
                          <Users size={16} className="text-accent" />
                          <span className="min-w-0 flex-1 truncate">{person.displayName}</span>
                          <span className="text-xs text-text-secondary">{person.faceCount} photos</span>
                        </ResultButton>
                      ))}
                    </SearchSection>
                  )}
                  {photos.length > 0 && (
                    <SearchSection title="Photos" icon={<ImageIcon size={14} />}>
                      {photos.map((photo) => (
                        <ResultButton key={photo.path} onClick={() => void openPhotoResults(filename(photo.path))}>
                          <ImageIcon size={16} className="text-text-secondary" />
                          <span className="min-w-0 flex-1 truncate">{filename(photo.path)}</span>
                          {photo.is_raw && (
                            <span className="rounded bg-surface px-1.5 py-0.5 text-[10px] text-text-secondary">
                              RAW
                            </span>
                          )}
                        </ResultButton>
                      ))}
                    </SearchSection>
                  )}
                  {!isSearching && people.length === 0 && photos.length === 0 && (
                    <div className="px-3 py-8 text-center text-sm text-text-secondary">
                      No matching people or photos for “{query.trim()}”.
                    </div>
                  )}
                </>
              )}
            </div>
            {query.trim() && (
              <button
                type="button"
                className="flex w-full items-center gap-2 border-t border-border-color px-4 py-3 text-left text-sm text-text-secondary transition hover:bg-surface hover:text-text-primary"
                onClick={() => void openPhotoResults(query.trim())}
              >
                <CornerDownLeft size={15} className="text-accent" />
                See all photo results for “{query.trim()}”
              </button>
            )}
          </div>
        </div>
      )}
    </>
  );
}

function SearchSection({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
  return (
    <section className="py-1">
      <div className="flex items-center gap-2 px-3 pb-1 pt-2 text-[11px] font-semibold uppercase tracking-wider text-text-secondary">
        {icon}
        {title}
      </div>
      {children}
    </section>
  );
}

function ResultButton({ children, onClick }: { children: ReactNode; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm text-text-primary transition hover:bg-surface"
    >
      {children}
    </button>
  );
}

function FilterField({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block text-[11px] font-medium text-text-secondary">
      {label}
      <span className="mt-1 block [&_input]:h-8 [&_input]:w-full [&_input]:rounded-md [&_input]:border [&_input]:border-border-color [&_input]:bg-bg-primary [&_input]:px-2 [&_input]:text-xs [&_input]:text-text-primary [&_input]:outline-none [&_input:focus]:border-accent [&_select]:h-8 [&_select]:w-full [&_select]:rounded-md [&_select]:border [&_select]:border-border-color [&_select]:bg-bg-primary [&_select]:px-2 [&_select]:text-xs [&_select]:text-text-primary [&_select]:outline-none [&_select:focus]:border-accent">
        {children}
      </span>
    </label>
  );
}

function describeCatalogQuery(query: CatalogSearchQuery): string[] {
  const filters: string[] = [];
  if (query.people?.length) filters.push(`With: ${query.people.join(', ')}`);
  else if (query.person) filters.push(`With: ${query.person}`);
  if (query.excludedPeople?.length) filters.push(`Without: ${query.excludedPeople.join(', ')}`);
  if (query.tags?.length) filters.push(`Tags: ${query.tags.join(', ')}`);
  if (query.excludedTags?.length) filters.push(`Without tags: ${query.excludedTags.join(', ')}`);
  if (query.aiTags?.length) filters.push(`AI tags: ${query.aiTags.join(', ')}`);
  if (query.excludedAiTags?.length) filters.push(`Without AI tags: ${query.excludedAiTags.join(', ')}`);
  if (query.dateFrom || query.dateTo) filters.push('Date range');
  if (query.minRating) filters.push(`${query.minRating}+ stars`);
  if (query.text) filters.push(`Text: ${query.text}`);
  return filters.length > 0 ? filters : ['No catalog filters returned'];
}

function catalogQueryLabel(query: CatalogSearchQuery): string {
  if (query.people?.length) return query.people.join(' + ');
  if (query.person) return query.person;
  if (query.tags?.length) return `Tags: ${query.tags.join(', ')}`;
  if (query.aiTags?.length) return `AI tags: ${query.aiTags.join(', ')}`;
  if (query.text) return query.text;
  if (query.dateFrom || query.dateTo) return 'Date range';
  return 'Filtered catalog';
}
