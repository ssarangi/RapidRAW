import { invoke } from '@tauri-apps/api/core';
import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { toast } from 'react-toastify';
import { useLibraryStore } from '../store/useLibraryStore';
import { useUIStore } from '../store/useUIStore';
import { CatalogRoot, ImageFile, Invokes } from '../components/ui/AppProperties';

export interface AddCollectionOutcome {
  root: CatalogRoot | null;
  cancelled: boolean;
}

export const catalogFolderPath = (rootId: number, relativePath = '.') =>
  `LibraryFolder:${rootId}:${relativePath || '.'}`;

const waitForNextPaint = () =>
  new Promise<void>((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  });

/**
 * Adds a photo folder as a new collection root to the currently open
 * library: picks a folder, registers it, updates the shared library store,
 * and kicks off an initial scan. This is the single source of truth for
 * "add a collection" - both Settings and the Sources sidebar call this same
 * function so the two entry points can never drift out of sync with each
 * other.
 */
export async function addLibraryCollection(): Promise<AddCollectionOutcome> {
  const selected = await openDialog({ directory: true, multiple: false, title: 'Add Photo Folder to Library' });
  if (typeof selected !== 'string') {
    return { root: null, cancelled: true };
  }

  const root = await invoke<CatalogRoot>(Invokes.AddLibraryRoot, { path: selected, label: null });

  const { catalogRoots } = useLibraryStore.getState();
  const nextRoots = catalogRoots.some((existing) => existing.id === root.id)
    ? catalogRoots.map((existing) => (existing.id === root.id ? root : existing))
    : [...catalogRoots, root];
  useLibraryStore.getState().setLibrary({ catalogRoots: nextRoots });

  await waitForNextPaint();
  await invoke(Invokes.StartCatalogScan, { rootId: root.id, recursive: true });

  return { root, cancelled: false };
}

/**
 * Loads a catalog root's images into the library store and switches to the
 * library view - the single source of truth for "browse into this catalog
 * root", shared by the Sources sidebar and the splash screen.
 */
export async function browseCatalogRoot(
  root: CatalogRoot,
  options: { relativePath?: string; recursive: boolean },
): Promise<void> {
  const relativePath = options.relativePath ?? '.';
  useLibraryStore.getState().setLibrary({ isViewLoading: true });
  try {
    const files = await invoke<ImageFile[]>(Invokes.ListCatalogImages, {
      rootId: root.id,
      recursive: options.recursive,
      folderPath: relativePath,
    });
    const initialRatings: Record<string, number> = {};
    files.forEach((file) => {
      initialRatings[file.path] = file.rating || 0;
    });
    useLibraryStore.getState().setLibrary({
      rootPaths: [root.absolutePath],
      currentFolderPath: catalogFolderPath(root.id, relativePath),
      activeAlbumId: null,
      activeCatalogRootId: root.id,
      imageList: files,
      imageRatings: initialRatings,
      multiSelectedPaths: [],
      libraryActivePath: null,
      libraryScrollTop: 0,
    });
    useUIStore.getState().setUI({ activeView: 'library' });
  } catch (err) {
    toast.error(`Failed to load catalog images: ${err}`);
    useUIStore.getState().setUI({ activeView: 'library' });
    throw err;
  } finally {
    useLibraryStore.getState().setLibrary({ isViewLoading: false });
  }
}
