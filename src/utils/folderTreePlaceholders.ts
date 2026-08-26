export interface FolderTreePlaceholder {
  children: FolderTreePlaceholder[];
  created: number;
  hasSubdirs: boolean;
  imageCount: number;
  isDir: boolean;
  modified: number;
  name: string;
  path: string;
}

export const folderNameFromPath = (path: string): string => {
  const trimmed = path.replace(/[\\/]+$/, '');
  if (!trimmed) return path;
  return trimmed.split(/[\\/]/).pop() || trimmed;
};

export const createFolderTreePlaceholder = (path: string): FolderTreePlaceholder => ({
  children: [],
  created: 0,
  hasSubdirs: true,
  imageCount: 0,
  isDir: true,
  modified: 0,
  name: folderNameFromPath(path),
  path,
});

export const createFolderTreePlaceholders = (paths: string[]): FolderTreePlaceholder[] =>
  paths.filter(Boolean).map(createFolderTreePlaceholder);
