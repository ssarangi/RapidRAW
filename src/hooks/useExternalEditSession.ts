import { useCallback, useEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { exit } from '@tauri-apps/plugin-process';

import { useProcessStore } from '../store/useProcessStore';
import { useEditorStore } from '../store/useEditorStore';
import { useLibraryStore } from '../store/useLibraryStore';
import { useSettingsStore } from '../store/useSettingsStore';
import { ImageFile, Invokes, LibraryViewMode } from '../components/ui/AppProperties';
import { ExportSettings, Status } from '../components/ui/ExportImportProperties';
import { debouncedSave } from './useEditorActions';

function parentFolderOf(filePath: string): string {
  const separator = filePath.includes('/') ? '/' : '\\';
  return filePath.substring(0, filePath.lastIndexOf(separator));
}

/**
 * A file opened via "Open with RapidRAW" or the external-editor protocol
 * selects an image directly, without going through the normal tree-click
 * navigation that would otherwise set currentFolderPath/librarySource to
 * match it - so Sources is left pointed at whatever catalog or pinned folder
 * was last browsed. This resolves the image's real folder before opening it:
 * if it falls under an already-open catalog's root, browse that catalog
 * folder; otherwise treat it as an ad-hoc filesystem folder (surfaced by
 * FolderTree's "Current Folder" section) the same way "Browse folder..." does.
 */
async function attributeSourceForFile(filePath: string): Promise<void> {
  const folderPath = parentFolderOf(filePath);
  const { librarySource, catalogRoots } = useLibraryStore.getState();

  if (librarySource.type === 'catalog') {
    const matchingRoot = catalogRoots.find(
      (root) => folderPath === root.absolutePath || folderPath.startsWith(`${root.absolutePath}/`) || folderPath.startsWith(`${root.absolutePath}\\`),
    );
    if (matchingRoot) {
      const relativePath =
        folderPath === matchingRoot.absolutePath
          ? '.'
          : folderPath.slice(matchingRoot.absolutePath.length + 1).replace(/\\/g, '/');
      useLibraryStore.getState().setLibrary({ isViewLoading: true });
      try {
        const recursive = useSettingsStore.getState().appSettings?.libraryViewMode === LibraryViewMode.Recursive;
        const files = await invoke<ImageFile[]>(Invokes.ListCatalogImages, {
          rootId: matchingRoot.id,
          recursive,
          folderPath: relativePath,
        });
        const initialRatings: Record<string, number> = {};
        files.forEach((file) => { initialRatings[file.path] = file.rating || 0; });
        useLibraryStore.getState().setLibrary({
          rootPaths: [matchingRoot.absolutePath],
          currentFolderPath: `LibraryFolder:${matchingRoot.id}:${relativePath}`,
          activeAlbumId: null,
          activeCatalogRootId: matchingRoot.id,
          imageList: files,
          imageRatings: initialRatings,
          multiSelectedPaths: [],
          libraryActivePath: null,
        });
      } catch (error) {
        console.error('Failed to attribute catalog folder for opened file:', error);
      } finally {
        useLibraryStore.getState().setLibrary({ isViewLoading: false });
      }
      return;
    }
  }

  // Not part of the open catalog (or no catalog open) - fall back to a plain
  // filesystem folder view, same as browsing a non-library folder.
  useLibraryStore.getState().setLibrary({ currentFolderPath: folderPath });
  try {
    const recursive = useSettingsStore.getState().appSettings?.libraryViewMode === LibraryViewMode.Recursive;
    const command = recursive ? Invokes.ListImagesRecursive : Invokes.ListImagesInDir;
    const files = await invoke<ImageFile[]>(command, { path: folderPath });
    useLibraryStore.getState().setLibrary({ imageList: files });
  } catch (error) {
    console.error('Failed to attribute filesystem folder for opened file:', error);
  }
}

/**
 * Handles files handed to the app from outside (OS "open with" and the
 * external editor protocol: rapidraw --edit <file> --output <file>).
 * Opens the requested image in the editor and, for edit sessions, exports
 * the result to the caller-provided output path and exits the app.
 */
export function useExternalEditSession(handleImageSelect: (path: string) => void) {
  const initialFileToOpen = useProcessStore((state) => state.initialFileToOpen);
  const externalEditSession = useProcessStore((state) => state.externalEditSession);
  const exportStatus = useProcessStore((state) => state.exportState.status);
  const [isFinishing, setIsFinishing] = useState(false);

  const handleImageSelectRef = useRef(handleImageSelect);
  useEffect(() => {
    handleImageSelectRef.current = handleImageSelect;
  });

  useEffect(() => {
    if (!initialFileToOpen) return;
    useProcessStore.getState().setProcess({ initialFileToOpen: null });
    void attributeSourceForFile(initialFileToOpen).then(() => {
      handleImageSelectRef.current(initialFileToOpen);
    });
  }, [initialFileToOpen]);

  useEffect(() => {
    if (!externalEditSession) return;
    setIsFinishing(false);
    handleImageSelectRef.current(externalEditSession.source);
  }, [externalEditSession]);

  useEffect(() => {
    if (!isFinishing) return;
    if (exportStatus === Status.Success) {
      exit(0);
    } else if (exportStatus === Status.Error || exportStatus === Status.Cancelled) {
      setIsFinishing(false);
    }
  }, [isFinishing, exportStatus]);

  const finishExternalEdit = useCallback(async () => {
    const session = useProcessStore.getState().externalEditSession;
    const { selectedImage, adjustments } = useEditorStore.getState();
    if (!session || !selectedImage) return;

    debouncedSave.flush();

    const exportSettings: ExportSettings = {
      filenameTemplate: null,
      jpegQuality: session.jpegQuality,
      keepMetadata: true,
      preserveTimestamps: false,
      preserveFolders: false,
      resize: null,
      stripGps: false,
      exportMasks: false,
      watermark: null,
    };

    setIsFinishing(true);
    useProcessStore.getState().setExportState({
      status: Status.Exporting,
      progress: { current: 0, total: 1 },
      errorMessage: '',
    });

    try {
      await invoke(Invokes.ExportImages, {
        paths: [session.source],
        outputFolderOrFile: session.output,
        isExplicitFilePath: true,
        baseOriginFolders: [],
        exportSettings,
        outputFormat: session.format,
        currentEditPath: selectedImage.path,
        currentEditAdjustments: adjustments || null,
      });
    } catch (error) {
      setIsFinishing(false);
      useProcessStore.getState().setExportState({
        status: Status.Error,
        errorMessage: typeof error === 'string' ? error : 'Export failed',
      });
    }
  }, []);

  return { externalEditSession, isFinishing, finishExternalEdit };
}
