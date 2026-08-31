import { useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { homeDir } from '@tauri-apps/api/path';
import { toast } from 'react-toastify';
import { useLibraryStore } from '../store/useLibraryStore';
import { useSettingsStore } from '../store/useSettingsStore';
import { useEditorStore } from '../store/useEditorStore';
import { useUIStore } from '../store/useUIStore';
import { useProcessStore } from '../store/useProcessStore';
import { CatalogRoot, Invokes, LibraryViewMode, ImageFile, LibraryDisplayMode } from '../components/ui/AppProperties';
import { INITIAL_ADJUSTMENTS, normalizeLoadedAdjustments } from '../utils/adjustments';
import { globalImageCache } from '../utils/ImageLRUCache';
import { debouncedSave, debouncedSetHistory } from './useEditorActions';
import { createFolderTreePlaceholders } from '../utils/folderTreePlaceholders';

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export interface AppNavigationProps {
  clearThumbnailQueue: () => void;
  refs: {
    transformWrapperRef: React.RefObject<any>;
    preloadedDataRef: React.RefObject<any>;
    cachedEditStateRef: React.RefObject<any>;
    selectedImagePathRef: React.RefObject<string | null>;
    isBackendReadyRef: React.RefObject<boolean>;
    latestRenderedJobIdRef: React.RefObject<number>;
    previewJobIdRef: React.RefObject<number>;
    currentResRef: React.RefObject<number>;
    prevAdjustmentsRef: React.RefObject<any>;
  };
}

export function useAppNavigation({ clearThumbnailQueue, refs }: AppNavigationProps) {
  const {
    transformWrapperRef,
    preloadedDataRef,
    cachedEditStateRef,
    selectedImagePathRef,
    isBackendReadyRef,
    latestRenderedJobIdRef,
    previewJobIdRef,
    currentResRef,
    prevAdjustmentsRef,
  } = refs;

  const handleGoHome = useCallback(() => {
    useLibraryStore.getState().setLibrary({
      rootPaths: [],
      currentFolderPath: null,
      activeAlbumId: null,
      imageList: [],
      imageRatings: {},
      folderTrees: [],
      multiSelectedPaths: [],
      libraryActivePath: null,
      expandedFolders: new Set(),
    });
    useUIStore.getState().setUI({
      isLibraryExportPanelVisible: false,
      libraryDisplayMode: LibraryDisplayMode.Grid,
      cullWorkspaceFolderPath: null,
    });
  }, []);

  const handleBackToLibrary = useCallback(() => {
    const { selectedImage } = useEditorStore.getState();
    const { setLibrary } = useLibraryStore.getState();
    const { setUI } = useUIStore.getState();

    if (selectedImage?.path && cachedEditStateRef.current) {
      globalImageCache.set(selectedImage.path, cachedEditStateRef.current);
    }
    if (transformWrapperRef.current) {
      transformWrapperRef.current.resetTransform(0);
    }
    useEditorStore.getState().setEditor({ zoom: 1 });

    debouncedSave.flush();
    debouncedSetHistory.cancel();

    const lastActivePath = selectedImage?.path ?? null;

    setLibrary({ libraryActivePath: lastActivePath });
    setUI({ activeView: 'library', slideDirection: 1 });
  }, [refs]);

  const handleImageSelect = useCallback(
    async (path: string, openInEditor: boolean = true) => {
      const { selectedImage, isSliderDragging, resetHistory, setEditor } = useEditorStore.getState();
      const { setLibrary, multiSelectedPaths } = useLibraryStore.getState();
      const { setUI } = useUIStore.getState();

      if (openInEditor) {
        setUI({ activeView: 'editor' });
      }

      if (selectedImage?.path === path) return;

      useEditorStore.getState().patchesSentToBackend.clear();
      debouncedSave.flush();
      debouncedSetHistory.cancel();

      if (selectedImage?.path && cachedEditStateRef.current) {
        globalImageCache.set(selectedImage.path, cachedEditStateRef.current);
      }

      const cached = globalImageCache.get(path);
      const cachedEntry = cached?.selectedImage?.isReady ? cached : null;
      const isFrontendCached = Boolean(cachedEntry);
      const isCachedInBackend = isFrontendCached
        ? await invoke<boolean>('is_image_cached', { path }).catch(() => false)
        : false;

      const hasDifferentResolution =
        cached &&
        (useEditorStore.getState().originalSize.width !== cached.originalSize.width ||
          useEditorStore.getState().originalSize.height !== cached.originalSize.height);

      if (!isCachedInBackend || hasDifferentResolution) {
        setEditor({ hasRenderedFirstFrame: false });
      }

      selectedImagePathRef.current = path;

      const newMultiSelectedPaths = multiSelectedPaths.includes(path) ? multiSelectedPaths : [path];

      setLibrary({
        multiSelectedPaths: newMultiSelectedPaths,
        libraryActivePath: path,
        selectionAnchorPath: path,
      });

      setEditor({
        showOriginal: false,
        activeMaskId: null,
        activeMaskContainerId: null,
        activeAiPatchContainerId: null,
        activeAiSubMaskId: null,
        isWbPickerActive: false,
        transformedOriginalUrl: null,
      });

      setUI({
        isLibraryExportPanelVisible: false,
        compactEditorPanelHeightOverride: null,
      });

      if (cachedEntry) {
        setEditor({
          selectedImage: {
            ...cachedEntry.selectedImage,
            thumbnailUrl: useProcessStore.getState().thumbnails[path] || cachedEntry.selectedImage.thumbnailUrl,
          },
          originalSize: cachedEntry.originalSize,
          previewSize: cachedEntry.previewSize,
          histogram: cachedEntry.histogram,
          waveform: cachedEntry.waveform,
          finalPreviewUrl: cachedEntry.finalPreviewUrl,
          uncroppedAdjustedPreviewUrl: cachedEntry.uncroppedPreviewUrl,
        });

        setEditor({ adjustments: cachedEntry.adjustments });
        resetHistory(cachedEntry.adjustments);
        prevAdjustmentsRef.current = { path, adjustments: cachedEntry.adjustments };

        setLibrary({ isViewLoading: false });

        latestRenderedJobIdRef.current = previewJobIdRef.current;
        isBackendReadyRef.current = false;
        currentResRef.current = Infinity;

        invoke(Invokes.LoadImage, { path })
          .then((_result: any) => {
            if (selectedImagePathRef.current !== path) return;
            isBackendReadyRef.current = true;
            currentResRef.current = 0;
            setEditor({ originalSize: { width: _result.width, height: _result.height } });
          })
          .catch((err: any) => {
            if (String(err).includes('cancelled')) return;
            console.error('Background load_image failed on cache hit:', err);
            isBackendReadyRef.current = true;
            currentResRef.current = 0;
          });

        invoke(Invokes.LoadMetadata, { path })
          .then((metadata: any) => {
            if (selectedImagePathRef.current !== path) return;
            let freshAdjustments: any;
            if (metadata.adjustments && !metadata.adjustments.is_null) {
              freshAdjustments = normalizeLoadedAdjustments(metadata.adjustments);
            } else {
              freshAdjustments = { ...INITIAL_ADJUSTMENTS };
            }
            if (!isSliderDragging && JSON.stringify(cachedEntry.adjustments) !== JSON.stringify(freshAdjustments)) {
              setEditor({ adjustments: freshAdjustments });
              resetHistory(freshAdjustments);
              prevAdjustmentsRef.current = { path, adjustments: freshAdjustments };
              globalImageCache.set(path, { ...cachedEntry, adjustments: freshAdjustments });
            }
          })
          .catch((err) => console.error('Failed background metadata sync on cache hit:', err));

        return;
      }

      isBackendReadyRef.current = true;

      const imageFile = useLibraryStore.getState().imageList.find((img) => img.path === path);
      setEditor({
        selectedImage: {
          exif: null,
          group_id: imageFile?.group_id ?? null,
          height: 0,
          isRaw: false,
          isReady: false,
          metadata: null,
          originalUrl: null,
          path,
          thumbnailUrl: useProcessStore.getState().thumbnails[path],
          width: 0,
        },
        originalSize: { width: 0, height: 0 },
        previewSize: { width: 0, height: 0 },
        histogram: null,
        waveform: null,
        uncroppedAdjustedPreviewUrl: null,
      });

      setLibrary({ isViewLoading: true });

      setEditor((state) => {
        const prev = state.finalPreviewUrl;
        if (prev?.startsWith('blob:') && !globalImageCache.isProtected(prev)) {
          setTimeout(() => {
            if (!globalImageCache.isProtected(prev)) {
              URL.revokeObjectURL(prev);
            }
          }, 250);
        }
        return { finalPreviewUrl: null };
      });

      setEditor((state) => {
        if (state.interactivePatch?.url) URL.revokeObjectURL(state.interactivePatch.url);
        return { interactivePatch: null };
      });
    },
    [refs],
  );

  const handleSelectSubfolder = useCallback(
    async (
      path: string | null,
      isNewRoot = false,
      preloadedImages?: ImageFile[],
      expandParents = true,
      preserveEditor = false,
    ) => {
      const { appSettings, handleSettingsChange } = useSettingsStore.getState();
      const { pinnedFolders } = appSettings || { pinnedFolders: [] };
      const { setLibrary, sortCriteria } = useLibraryStore.getState();
      const { setUI } = useUIStore.getState();
      const { setProcess } = useProcessStore.getState();
      const { selectedImage, resetHistory, setEditor } = useEditorStore.getState();
      const libraryViewMode = appSettings?.libraryViewMode;

      if (!preserveEditor) {
        await invoke('cancel_thumbnail_generation');
        clearThumbnailQueue();
        setLibrary({ isViewLoading: true, activeAlbumId: null, libraryScrollTop: 0 });
        setProcess({ thumbnails: {} });
        globalImageCache.clear();
        setUI({ activeView: 'library' });
      } else {
        setLibrary({ isViewLoading: true });
      }

      try {
        const { rootPaths, expandedFolders: currentExpandedFolders } = useLibraryStore.getState();
        let newExpandedFolders = new Set(currentExpandedFolders);

        if (isNewRoot && path) {
          newExpandedFolders = new Set([path]);
          if (appSettings) {
            handleSettingsChange({ ...appSettings, lastRootPath: path } as any);
          }
        } else if (path && expandParents) {
          const allRoots = [...(rootPaths || []), ...(pinnedFolders || [])].filter(Boolean) as string[];
          const relevantRoot = allRoots.find((r) => path.startsWith(r));

          if (relevantRoot) {
            const separator = path.includes('/') ? '/' : '\\';
            const parentSeparatorIndex = path.lastIndexOf(separator);

            if (parentSeparatorIndex > -1 && path.length > relevantRoot.length) {
              let current = path.substring(0, parentSeparatorIndex);
              while (current && current.length >= relevantRoot.length) {
                newExpandedFolders.add(current);
                const nextParentIndex = current.lastIndexOf(separator);
                if (nextParentIndex === -1 || current === relevantRoot) break;
                current = current.substring(0, nextParentIndex);
              }
            }
            newExpandedFolders.add(relevantRoot);
          }
        }

        setLibrary({
          currentFolderPath: path,
          expandedFolders: newExpandedFolders,
          ...(preserveEditor ? {} : { imageList: [], multiSelectedPaths: [], libraryActivePath: null }),
        });

        if (!preserveEditor && selectedImage) {
          debouncedSave.flush();
          debouncedSetHistory.cancel();
          setEditor({ selectedImage: null, finalPreviewUrl: null, uncroppedAdjustedPreviewUrl: null, histogram: null });
          setEditor({ adjustments: INITIAL_ADJUSTMENTS });
          resetHistory(INITIAL_ADJUSTMENTS);
          useEditorStore.getState().patchesSentToBackend.clear();
        }

        const command =
          libraryViewMode === LibraryViewMode.Recursive ? Invokes.ListImagesRecursive : Invokes.ListImagesInDir;

        let files: ImageFile[] = [];
        if (preloadedImages) {
          files = preloadedImages;
        } else {
          let lastError: unknown = null;
          for (let attempt = 0; attempt < 4; attempt += 1) {
            try {
              files = await invoke<ImageFile[]>(command, { path });
              lastError = null;
              break;
            } catch (err) {
              lastError = err;
              const message = String(err).toLowerCase();
              if (!message.includes('directory does not exist') || attempt === 3) {
                throw err;
              }
              await delay(750 * (attempt + 1));
            }
          }
          if (lastError) throw lastError;
        }

        const initialRatings: Record<string, number> = {};
        files.forEach((f) => {
          if (f.rating !== undefined) {
            initialRatings[f.path] = f.rating;
          }
        });
        setLibrary({ imageRatings: initialRatings });

        const exifSortKeys = ['date_taken', 'iso', 'shutter_speed', 'aperture', 'focal_length'];
        const isExifSortActive = exifSortKeys.includes(sortCriteria.key);

        if (files.length > 0) {
          const paths = files.map((f: ImageFile) => f.path);

          if (isExifSortActive) {
            let combinedExifMap: Record<string, any> = {};
            const chunkSize = 100;

            for (let i = 0; i < paths.length; i += chunkSize) {
              const chunk = paths.slice(i, i + chunkSize);
              try {
                const chunkExif: any = await invoke(Invokes.ReadExifForPaths, { paths: chunk });
                combinedExifMap = { ...combinedExifMap, ...chunkExif };
              } catch (err) {
                console.error('Failed to read EXIF chunk:', err);
              }
            }

            const finalImageList = files.map((image) => ({
              ...image,
              exif: combinedExifMap[image.path] || image.exif || null,
            }));
            setLibrary({ imageList: finalImageList });
          } else {
            setLibrary({ imageList: files });

            setTimeout(() => {
              const fetchExifInChunks = async () => {
                const chunkSize = 50;
                for (let i = 0; i < paths.length; i += chunkSize) {
                  if (useLibraryStore.getState().currentFolderPath !== path) break;

                  const chunk = paths.slice(i, i + chunkSize);
                  try {
                    const chunkExif: any = await invoke(Invokes.ReadExifForPaths, { paths: chunk });
                    setLibrary((state) => ({
                      imageList: state.imageList.map((image) => ({
                        ...image,
                        exif: chunkExif[image.path] || image.exif || null,
                      })),
                    }));
                    await new Promise((resolve) => setTimeout(resolve, 50));
                  } catch (err) {
                    console.error('Failed to read EXIF chunk:', err);
                  }
                }
              };
              fetchExifInChunks();
            }, 500);
          }
        } else {
          setLibrary({ imageList: files });
        }

        if (!preserveEditor) {
          invoke(Invokes.StartBackgroundIndexing, { folderPath: path }).catch((err) => {
            console.error('Failed to start background indexing:', err);
          });
        }
      } catch (err) {
        console.error('Failed to load folder contents:', err);
        toast.error('Failed to load images from the selected folder.');
      } finally {
        useLibraryStore.getState().setLibrary({ isViewLoading: false });
      }
    },
    [clearThumbnailQueue, refs],
  );

  const handleSelectAlbum = useCallback(
    async (albumId: string, albumName: string, imagePaths: string[], preserveEditor = false) => {
      const { setLibrary } = useLibraryStore.getState();
      const { setUI } = useUIStore.getState();

      if (!preserveEditor) {
        await invoke('cancel_thumbnail_generation');
        clearThumbnailQueue();
        setLibrary({ libraryScrollTop: 0 });
        globalImageCache.clear();
        setUI({ activeView: 'library' });
      }

      setLibrary({
        isViewLoading: true,
        currentFolderPath: `Album: ${albumName}`,
        activeAlbumId: albumId,
      });

      try {
        const files: ImageFile[] = await invoke(Invokes.GetAlbumImages, { paths: imagePaths });

        const initialRatings: Record<string, number> = {};
        files.forEach((f) => {
          if (f.rating !== undefined) initialRatings[f.path] = f.rating;
        });

        setLibrary({
          imageList: files,
          imageRatings: initialRatings,
          ...(preserveEditor ? {} : { multiSelectedPaths: [], libraryActivePath: null }),
        });
      } catch (err) {
        console.error('Failed to load album images:', err);
        toast.error(`Failed to load album: ${err}`);
      } finally {
        setLibrary({ isViewLoading: false });
      }
    },
    [clearThumbnailQueue],
  );

  const handleOpenFolder = async () => {
    const { osPlatform, appSettings, handleSettingsChange } = useSettingsStore.getState();
    const { rootPaths, folderTrees, setLibrary } = useLibraryStore.getState();
    const isAndroid = osPlatform === 'android';

    try {
      let selectedPath = '';
      if (isAndroid) {
        selectedPath = await invoke<string>(Invokes.GetOrCreateInternalLibraryRoot);
      } else {
        const selected = await open({ directory: true, multiple: false, defaultPath: await homeDir() });
        if (typeof selected === 'string') {
          selectedPath = selected;
        }
      }

      if (selectedPath) {
        if (!rootPaths.includes(selectedPath)) {
          const newRootPaths = [...rootPaths, selectedPath];
          setLibrary({ rootPaths: newRootPaths });

          if (appSettings) {
            handleSettingsChange({ ...appSettings, rootFolders: newRootPaths } as any);
          }

          setLibrary({ isTreeLoading: true });
          try {
            const newTree = await invoke(Invokes.GetFolderTree, {
              path: selectedPath,
              expandedFolders: [selectedPath],
              showImageCounts:
                appSettings?.enableFolderImageCounts || appSettings?.folderTreeSort?.key === 'imageCount',
            });
            setLibrary({ folderTrees: [...folderTrees, newTree] });
          } catch (e) {
            toast.error(`Failed to load folder tree: ${e}`);
          } finally {
            setLibrary({ isTreeLoading: false });
          }
        }
        await handleSelectSubfolder(selectedPath, true);
      }
    } catch (err) {
      console.error(isAndroid ? 'Failed to open Android library root:' : 'Failed to open directory dialog:', err);
      toast.error(isAndroid ? 'Failed to open library.' : 'Failed to open folder selection dialog.');
    }
  };

  const handleContinueSession = () => {
    const restore = async () => {
      const { appSettings } = useSettingsStore.getState();
      const { setLibrary } = useLibraryStore.getState();

      const rootFolders = appSettings?.rootFolders?.length
        ? appSettings.rootFolders
        : appSettings?.lastRootPath
          ? [appSettings.lastRootPath]
          : [];

      const folderState = appSettings?.lastFolderState;
      const pathToSelect = folderState?.currentFolderPath || rootFolders[0];

      if (!pathToSelect) return;

      setLibrary({ rootPaths: rootFolders });

      if (folderState?.expandedFolders) {
        const newExpandedFolders = new Set<string>(folderState.expandedFolders);
        setLibrary({ expandedFolders: newExpandedFolders });
      } else {
        setLibrary({ expandedFolders: new Set(rootFolders) });
      }

      if (rootFolders.length > 0) {
        setLibrary({
          folderTrees: createFolderTreePlaceholders(rootFolders),
          isTreeLoading: false,
          isRootFoldersLoading: false,
        });
      }

      if (pathToSelect && pathToSelect.startsWith('Album: ')) {
        const activeAlbumId = folderState?.activeAlbumId;
        if (activeAlbumId) {
          try {
            const albumTree: any = await invoke(Invokes.GetAlbums);
            setLibrary({ albumTree });

            const findObj = (nodes: any[]): any => {
              for (const n of nodes) {
                if (n.id === activeAlbumId) return n;
                if (n.type === 'group') {
                  const f = findObj(n.children);
                  if (f) return f;
                }
              }
              return null;
            };

            const album = findObj(albumTree);
            if (album) {
              await handleSelectAlbum(album.id, album.name, album.images);
            } else {
              setLibrary({ currentFolderPath: null, activeAlbumId: null, imageList: [] });
              useUIStore.getState().setUI({ activeView: 'library' });
            }
          } catch (e) {
            console.error('Failed to restore album session:', e);
            setLibrary({ currentFolderPath: null, activeAlbumId: null, imageList: [] });
            useUIStore.getState().setUI({ activeView: 'library' });
          }
        } else {
          setLibrary({ currentFolderPath: null, activeAlbumId: null, imageList: [] });
          useUIStore.getState().setUI({ activeView: 'library' });
        }
      } else if (
        pathToSelect &&
        (pathToSelect.startsWith('Library: ') || pathToSelect.startsWith('LibraryFolder:'))
      ) {
        const activeCatalogRootId = folderState?.activeCatalogRootId;
        if (activeCatalogRootId) {
          try {
            const catalogRoots = await invoke<CatalogRoot[]>(Invokes.ListLibraryRoots);
            const activeRoot = catalogRoots.find((root) => root.id === activeCatalogRootId);
            const match = /^LibraryFolder:(\d+):(.*)$/.exec(pathToSelect);
            const folderPath = match ? match[2] || '.' : '.';
            const files = await invoke<ImageFile[]>(Invokes.ListCatalogImages, {
              rootId: activeCatalogRootId,
              recursive: appSettings?.libraryViewMode === LibraryViewMode.Recursive,
              folderPath,
            });
            const imageRatings: Record<string, number> = {};
            files.forEach((file) => {
              imageRatings[file.path] = file.rating || 0;
            });
            setLibrary({
              rootPaths: activeRoot ? [activeRoot.absolutePath] : rootFolders,
              catalogRoots,
              currentFolderPath: pathToSelect,
              activeAlbumId: null,
              activeCatalogRootId,
              imageList: files,
              imageRatings,
              multiSelectedPaths: [],
              libraryActivePath: null,
            });
            useUIStore.getState().setUI({ activeView: 'library' });
          } catch (e) {
            console.error('Failed to restore catalog session:', e);
            setLibrary({ currentFolderPath: null, activeCatalogRootId: null, imageList: [] });
            useUIStore.getState().setUI({ activeView: 'library' });
          }
        } else {
          setLibrary({ currentFolderPath: null, activeCatalogRootId: null, imageList: [] });
          useUIStore.getState().setUI({ activeView: 'library' });
        }
      } else {
        setLibrary({
          currentFolderPath: null,
          activeAlbumId: null,
          activeCatalogRootId: null,
          imageList: [],
          imageRatings: {},
          multiSelectedPaths: [],
          libraryActivePath: null,
        });
        useUIStore.getState().setUI({ activeView: 'library' });
      }
    };

    restore().catch((err) => {
      console.error('Failed to restore session:', err);
      toast.error('Failed to restore session. A folder may have been moved or deleted.');
      handleGoHome();
      useLibraryStore.getState().setLibrary({
        isTreeLoading: false,
        isRootFoldersLoading: false,
        isPinnedFoldersLoading: false,
      });
    });
  };

  return {
    handleGoHome,
    handleBackToLibrary,
    handleImageSelect,
    handleSelectSubfolder,
    handleSelectAlbum,
    handleOpenFolder,
    handleContinueSession,
  };
}
