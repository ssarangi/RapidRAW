import { invoke } from '@tauri-apps/api/core';
import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { useLibraryStore } from '../store/useLibraryStore';
import { CatalogRoot, Invokes } from '../components/ui/AppProperties';

export interface AddCollectionOutcome {
  root: CatalogRoot | null;
  cancelled: boolean;
}

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
