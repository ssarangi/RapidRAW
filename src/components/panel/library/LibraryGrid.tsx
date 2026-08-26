import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { List, useListCallbackRef } from 'react-window';
import { ChevronUp, ChevronDown } from 'lucide-react';
import debounce from 'lodash.debounce';
import { useTranslation } from 'react-i18next';
import { Row } from './LibraryItems';
import { useShallow } from 'zustand/react/shallow';
import { useLibraryStore } from '../../../store/useLibraryStore';
import { LibraryViewMode, SortDirection, LibraryDisplayMode } from '../../ui/AppProperties';
import Text from '../../ui/Text';
import { TextColors, TextVariants, TextWeights, TEXT_COLOR_KEYS } from '../../../types/typography';
import { useProcessStore } from '../../../store/useProcessStore';
import { ExifOverlay } from '../../ui/AppProperties';
import { useSettingsStore } from '../../../store/useSettingsStore';

function ListHeader({ widths, setWidths, containerRef, sortCriteria, onSortChange }: any) {
  const { t } = useTranslation();
  const exifOverlay = useSettingsStore((s) => s.appSettings?.exifOverlay || ExifOverlay.Off);
  const showExifCols = exifOverlay !== ExifOverlay.Off;
  const totalRawWidth =
    widths.thumbnail +
    widths.name +
    widths.date +
    widths.rating +
    widths.color +
    (showExifCols ? widths.shutter + widths.aperture + widths.iso + widths.focal : 0);

  const handleResize = (e: React.MouseEvent, leftCol: string, rightCol: string) => {
    e.preventDefault();
    e.stopPropagation();
    const startX = e.clientX;
    const startLeftWidth = widths[leftCol];
    const startRightWidth = widths[rightCol];
    const containerWidth = containerRef.current?.clientWidth || 1000;

    const onMouseMove = (moveEvent: MouseEvent) => {
      const deltaX = moveEvent.clientX - startX;
      const deltaPercent = (deltaX / containerWidth) * 100;

      let newLeft = startLeftWidth + deltaPercent;
      let newRight = startRightWidth - deltaPercent;

      if (newLeft < 1) {
        newRight -= 1 - newLeft;
        newLeft = 1;
      }
      if (newRight < 1) {
        newLeft -= 1 - newRight;
        newRight = 1;
      }

      setWidths((prev: any) => ({
        ...prev,
        [leftCol]: newLeft,
        [rightCol]: newRight,
      }));
    };

    const onMouseUp = () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  };

  const Column = ({ title, widthKey, nextKey, sortKey }: any) => {
    const isSorted = sortCriteria.key === sortKey;
    const isAsc = sortCriteria.order === SortDirection.Ascending;
    const actualWidth = `${(widths[widthKey] / totalRawWidth) * 100}%`;

    return (
      <div
        style={{ width: actualWidth }}
        className={`relative flex items-center px-3 h-full select-none ${
          sortKey ? 'cursor-pointer hover:bg-bg-primary/50 transition-colors' : ''
        }`}
        onClick={() => sortKey && onSortChange(sortKey)}
      >
        <Text
          variant={TextVariants.small}
          weight={TextWeights.semibold}
          color={isSorted ? TextColors.primary : TextColors.secondary}
          className="uppercase tracking-wider text-[11px]"
        >
          {title}
        </Text>
        {isSorted && (
          <span className={`ml-1 flex items-center ${TEXT_COLOR_KEYS[TextColors.primary]}`}>
            {isAsc ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </span>
        )}
        {nextKey && (
          <div
            className="absolute right-[-3px] top-1.5 bottom-1.5 w-[6px] cursor-col-resize z-10 group flex items-center justify-center"
            onMouseDown={(e) => handleResize(e, widthKey, nextKey)}
          >
            <div className="w-px h-full bg-border-color/40 group-hover:bg-accent transition-colors" />
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex items-center w-full h-9 bg-bg-secondary/80 backdrop-blur-sm border-b border-border-color/50 shrink-0">
      <Column title="" widthKey="thumbnail" nextKey="name" />
      <Column title={t('library.grid.columns.name')} widthKey="name" nextKey="date" sortKey="name" />
      <Column title={t('library.grid.columns.modified')} widthKey="date" nextKey="rating" sortKey="date" />
      <Column title={t('library.grid.columns.rating')} widthKey="rating" nextKey="color" sortKey="rating" />
      {showExifCols ? (
        <>
          <Column title={t('library.grid.columns.label')} widthKey="color" nextKey="shutter" />
          <Column
            title={t('library.grid.columns.shutter')}
            widthKey="shutter"
            nextKey="aperture"
            sortKey="shutter_speed"
          />
          <Column title={t('library.grid.columns.aperture')} widthKey="aperture" nextKey="iso" sortKey="aperture" />
          <Column title={t('library.grid.columns.iso')} widthKey="iso" nextKey="focal" sortKey="iso" />
          <Column title={t('library.grid.columns.focal')} widthKey="focal" sortKey="focal_length" />
        </>
      ) : (
        <Column title={t('library.grid.columns.label')} widthKey="color" />
      )}
    </div>
  );
}

const groupImagesByFolder = (images: any[], baseFolderPath: string | null) => {
  const groups: Record<string, any[]> = {};

  images.forEach((img) => {
    const physicalPath = img.path.split('?vc=')[0];
    const separator = physicalPath.includes('/') ? '/' : '\\';
    const lastSep = physicalPath.lastIndexOf(separator);
    const dir = lastSep > -1 ? physicalPath.substring(0, lastSep) : physicalPath;

    if (!groups[dir]) {
      groups[dir] = [];
    }
    groups[dir].push(img);
  });

  const sortedKeys = Object.keys(groups).sort((a, b) => {
    if (a === baseFolderPath) return -1;
    if (b === baseFolderPath) return 1;
    return a.localeCompare(b);
  });

  return sortedKeys.map((dir) => ({
    path: dir,
    images: groups[dir],
  }));
};

// Content-space rect (y already includes scrollTop) -> paths of every image
// cell it overlaps. Walks gridData.rows directly rather than querying the DOM,
// since react-window only mounts currently-visible rows - this has to work
// for rows scrolled out of view too.
function getPathsInDragRect(gridData: any, startX: number, startY: number, endX: number, endY: number): string[] {
  const { rows, itemWidth, rowHeight, headerHeight, ITEM_GAP, OUTER_PADDING, isListView } = gridData;
  const y1 = Math.min(startY, endY);
  const y2 = Math.max(startY, endY);
  const x1 = Math.min(startX, endX);
  const x2 = Math.max(startX, endX);

  const paths: string[] = [];
  let top = 0;
  for (const row of rows) {
    const h =
      row.type === 'footer' ? (isListView ? 24 : OUTER_PADDING) : row.type === 'header' ? headerHeight : rowHeight;
    const bottom = top + h;

    if (row.type === 'images' && bottom > y1 && top < y2) {
      row.images.forEach((img: any, col: number) => {
        if (isListView) {
          paths.push(img.path);
          return;
        }
        const cellLeft = OUTER_PADDING + col * (itemWidth + ITEM_GAP);
        const cellRight = cellLeft + itemWidth;
        if (cellRight > x1 && cellLeft < x2) {
          paths.push(img.path);
        }
      });
    }
    top = bottom;
  }
  return paths;
}

const DRAG_SELECT_THRESHOLD = 4;

export default function LibraryGrid(props: any) {
  const {
    imageList,
    libraryViewMode,
    thumbnailSize,
    libraryDisplayMode,
    currentFolderPath,
    activePath,
    multiSelectedPaths,
    onContextMenu,
    onImageClick,
    onImageDoubleClick,
    thumbnailAspectRatio,
    imageRatings,
    onRequestThumbnails,
    thumbnailSizeOptions,
    onThumbnailSizeChange,
    groupBadgeInfo,
  } = props;
  const { listColumnWidths, setLibrary, sortCriteria, setSortCriteria } = useLibraryStore(
    useShallow((state) => ({
      listColumnWidths: state.listColumnWidths,
      setLibrary: state.setLibrary,
      sortCriteria: state.sortCriteria,
      setSortCriteria: state.setSortCriteria,
    })),
  );

  const [gridSize, setGridSize] = useState({ height: 0, width: 0 });
  const [listHandle, setListHandle] = useListCallbackRef();
  const [collapsedRecursiveFolders, setCollapsedRecursiveFolders] = useState<Set<string>>(new Set());
  const libraryContainerRef = useRef<HTMLDivElement>(null);
  const gridObserverRef = useRef<ResizeObserver | null>(null);
  const loadedThumbnailsRef = useRef(new Set<string>());
  const requestQueueRef = useRef<Map<string, { path: string; modified?: number }>>(new Map());
  const requestTimeoutRef = useRef<any>(null);
  const exifOverlay = useSettingsStore((s) => s.appSettings?.exifOverlay || ExifOverlay.Off);
  const showExifCols = exifOverlay !== ExifOverlay.Off;

  const [dragSelectRect, setDragSelectRect] = useState<null | {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  }>(null);
  const dragStartRef = useRef<null | {
    clientX: number;
    clientY: number;
    contentX: number;
    contentY: number;
    additive: boolean;
    baseSelection: string[];
  }>(null);
  const hasDragSelectedRef = useRef(false);

  useEffect(() => {
    const el = libraryContainerRef.current;
    if (gridObserverRef.current) {
      gridObserverRef.current.disconnect();
      gridObserverRef.current = null;
    }
    if (el) {
      const ro = new ResizeObserver((entries) => {
        const entry = entries[0];
        if (entry) {
          const height = Math.round(entry.contentRect.height);
          const width = Math.round(entry.contentRect.width);

          setGridSize((prev) => (prev.height === height && prev.width === width ? prev : { height, width }));
        }
      });
      ro.observe(el);
      gridObserverRef.current = ro;
    }
    return () => gridObserverRef.current?.disconnect();
  }, [libraryContainerRef]);

  useEffect(() => {
    const handleWheel = (event: any) => {
      const container = libraryContainerRef.current;
      if (!container || !container.contains(event.target)) {
        return;
      }

      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        const currentIndex = thumbnailSizeOptions.findIndex((o: any) => o.id === thumbnailSize);
        if (currentIndex === -1) {
          return;
        }

        const nextIndex =
          event.deltaY < 0
            ? Math.min(currentIndex + 1, thumbnailSizeOptions.length - 1)
            : Math.max(currentIndex - 1, 0);
        if (nextIndex !== currentIndex) {
          onThumbnailSizeChange(thumbnailSizeOptions[nextIndex].id);
        }
      }
    };

    window.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      window.removeEventListener('wheel', handleWheel);
    };
  }, [thumbnailSize, onThumbnailSizeChange, thumbnailSizeOptions]);

  const handleScroll = useMemo(
    () =>
      debounce((top: number) => {
        setLibrary({ libraryScrollTop: top });
      }, 200),
    [setLibrary],
  );

  useEffect(() => () => handleScroll.cancel(), [handleScroll]);

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

  const handleToggleRecursiveFolder = useCallback((path: string) => {
    setCollapsedRecursiveFolders((prev) => {
      const next = new Set(prev);
      next.has(path) ? next.delete(path) : next.add(path);
      return next;
    });
  }, []);

  const handleImageLoad = useCallback((path: string) => {
    loadedThumbnailsRef.current.add(path);
  }, []);

  const gridData = useMemo(() => {
    if (gridSize.width === 0 || imageList.length === 0) return null;

    const isListView = libraryDisplayMode === LibraryDisplayMode.List;
    const OUTER_PADDING = isListView ? 0 : 12;
    const ITEM_GAP = isListView ? 0 : 12;
    const minThumbWidth = thumbnailSizeOptions.find((o: any) => o.id === thumbnailSize)?.size || 240;

    const availableWidth = gridSize.width - OUTER_PADDING * 2;
    const columnCount = isListView
      ? 1
      : Math.max(1, Math.floor((availableWidth + ITEM_GAP) / (minThumbWidth + ITEM_GAP)));
    const itemWidth = isListView ? availableWidth : (availableWidth - ITEM_GAP * (columnCount - 1)) / columnCount;

    const totalBase =
      listColumnWidths.thumbnail +
      listColumnWidths.name +
      listColumnWidths.date +
      listColumnWidths.rating +
      listColumnWidths.color +
      (showExifCols
        ? listColumnWidths.shutter + listColumnWidths.aperture + listColumnWidths.iso + listColumnWidths.focal
        : 0);

    const listRowHeight = Math.max(36, Math.min(300, (availableWidth * listColumnWidths.thumbnail) / totalBase));
    const rowHeight = isListView ? listRowHeight : itemWidth + ITEM_GAP;
    const headerHeight = 40;

    const rows: any[] = [];

    if (libraryViewMode === LibraryViewMode.Recursive) {
      const groups = groupImagesByFolder(imageList, currentFolderPath);
      groups.forEach((group) => {
        if (group.images.length === 0) return;

        const isExpanded = !collapsedRecursiveFolders.has(group.path);
        rows.push({ type: 'header', path: group.path, count: group.images.length, isExpanded });

        if (isExpanded) {
          for (let i = 0; i < group.images.length; i += columnCount) {
            rows.push({
              type: 'images',
              images: group.images.slice(i, i + columnCount),
              startIndex: i,
            });
          }
        }
      });
    } else {
      for (let i = 0; i < imageList.length; i += columnCount) {
        rows.push({
          type: 'images',
          images: imageList.slice(i, i + columnCount),
          startIndex: i,
        });
      }
    }

    rows.push({ type: 'footer' });

    return {
      rows,
      itemWidth,
      rowHeight,
      listRowHeight,
      OUTER_PADDING,
      ITEM_GAP,
      columnCount,
      isListView,
      headerHeight,
    };
  }, [
    gridSize.width,
    imageList,
    libraryViewMode,
    libraryDisplayMode,
    collapsedRecursiveFolders,
    thumbnailSize,
    listColumnWidths.thumbnail,
    currentFolderPath,
    thumbnailSizeOptions,
  ]);

  const handleGridMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      const target = e.target as HTMLElement;
      if (target.closest('[data-bench-id="thumbnail"]')) return;
      const listEl = listHandle?.element as HTMLElement | undefined;
      if (!listEl) return;

      const rect = listEl.getBoundingClientRect();
      dragStartRef.current = {
        clientX: e.clientX,
        clientY: e.clientY,
        contentX: e.clientX - rect.left,
        contentY: e.clientY - rect.top + listEl.scrollTop,
        additive: e.ctrlKey || e.metaKey,
        baseSelection: multiSelectedPaths,
      };
      hasDragSelectedRef.current = false;
    },
    [listHandle, multiSelectedPaths],
  );

  const handleGridClick = useCallback(() => {
    if (hasDragSelectedRef.current) {
      hasDragSelectedRef.current = false;
      return;
    }
    props.onClearSelection();
  }, [props]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const start = dragStartRef.current;
      const listEl = listHandle?.element as HTMLElement | undefined;
      if (!start || !listEl || !gridData) return;

      const dx = e.clientX - start.clientX;
      const dy = e.clientY - start.clientY;
      if (!hasDragSelectedRef.current && Math.abs(dx) < DRAG_SELECT_THRESHOLD && Math.abs(dy) < DRAG_SELECT_THRESHOLD) {
        return;
      }
      hasDragSelectedRef.current = true;

      setDragSelectRect({
        x1: Math.min(start.clientX, e.clientX),
        y1: Math.min(start.clientY, e.clientY),
        x2: Math.max(start.clientX, e.clientX),
        y2: Math.max(start.clientY, e.clientY),
      });

      const rect = listEl.getBoundingClientRect();
      const contentX = e.clientX - rect.left;
      const contentY = e.clientY - rect.top + listEl.scrollTop;

      const covered = getPathsInDragRect(gridData, start.contentX, start.contentY, contentX, contentY);
      const nextSelection = start.additive ? Array.from(new Set([...start.baseSelection, ...covered])) : covered;
      setLibrary({ multiSelectedPaths: nextSelection });
    };

    const handleMouseUp = () => {
      dragStartRef.current = null;
      setDragSelectRect(null);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [listHandle, gridData, setLibrary]);

  useEffect(() => {
    if (!listHandle?.element || !gridData) return;

    const savedTop = useLibraryStore.getState().libraryScrollTop;
    const element = listHandle.element as HTMLElement;

    if (savedTop > 0) {
      element.scrollTop = savedTop;
    }
  }, [listHandle, currentFolderPath]);

  const prevActivePath = useRef<string | null>(null);
  const prevDisplayMode = useRef<LibraryDisplayMode | null>(null);
  const prevListElement = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!listHandle?.element || !gridData || multiSelectedPaths.length > 1) {
      prevActivePath.current = activePath;
      prevDisplayMode.current = libraryDisplayMode;
      if (listHandle?.element) prevListElement.current = listHandle.element as HTMLElement;
      return;
    }

    const element = listHandle.element as HTMLElement;
    const isPathSame = activePath === prevActivePath.current;
    const isModeSame = libraryDisplayMode === prevDisplayMode.current;
    const isElementSame = element === prevListElement.current;

    if (isPathSame && isModeSame && isElementSame) return;

    prevActivePath.current = activePath;
    prevDisplayMode.current = libraryDisplayMode;
    prevListElement.current = element;

    const { rows, rowHeight, headerHeight, columnCount } = gridData;

    let targetTop = 0;
    let found = false;

    if (libraryViewMode === LibraryViewMode.Recursive) {
      const groups = groupImagesByFolder(imageList, currentFolderPath);
      for (const group of groups) {
        if (group.images.length === 0) continue;

        targetTop += headerHeight;

        const imageIndex = group.images.findIndex((img) => img.path === activePath);
        if (imageIndex !== -1) {
          const rowIndex = Math.floor(imageIndex / columnCount);
          targetTop += rowIndex * rowHeight;
          found = true;
          break;
        }

        const rowsInGroup = Math.ceil(group.images.length / columnCount);
        targetTop += rowsInGroup * rowHeight;
      }
    } else {
      const index = imageList.findIndex((img) => img.path === activePath);
      if (index !== -1) {
        const rowIndex = Math.floor(index / columnCount);
        targetTop = rowIndex * rowHeight;
        found = true;
      }
    }

    if (found) {
      const clientHeight = element.clientHeight;
      const scrollTop = element.scrollTop;
      const itemBottom = targetTop + rowHeight;
      const SCROLL_OFFSET = 120;

      if (!isModeSame || !isElementSame) {
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
  }, [
    activePath,
    gridData,
    multiSelectedPaths.length,
    listHandle,
    currentFolderPath,
    imageList,
    libraryViewMode,
    libraryDisplayMode,
  ]);

  const memoizedRowProps = useMemo(() => {
    if (!gridData) return {};

    return {
      rows: gridData.rows,
      activePath,
      multiSelectedSet: new Set(multiSelectedPaths),
      onContextMenu,
      onImageClick,
      onImageDoubleClick,
      thumbnailAspectRatio,
      onImageLoad: handleImageLoad,
      imageRatings,
      baseFolderPath: currentFolderPath,
      itemWidth: gridData.itemWidth,
      itemHeight: gridData.isListView ? gridData.listRowHeight : gridData.itemWidth,
      outerPadding: gridData.OUTER_PADDING,
      gap: gridData.ITEM_GAP,
      isListView: gridData.isListView,
      columnWidths: listColumnWidths,
      queueThumbnailRequest,
      onToggleRecursiveFolder: handleToggleRecursiveFolder,
      groupBadgeInfo,
    };
  }, [
    gridData,
    activePath,
    multiSelectedPaths,
    onContextMenu,
    onImageClick,
    onImageDoubleClick,
    thumbnailAspectRatio,
    handleImageLoad,
    imageRatings,
    currentFolderPath,
    listColumnWidths,
    queueThumbnailRequest,
    handleToggleRecursiveFolder,
    groupBadgeInfo,
  ]);

  const getItemSize = useCallback(
    (index: number) => {
      if (!gridData) return 0;
      if (gridData.rows[index].type === 'footer') return gridData.isListView ? 24 : gridData.OUTER_PADDING;
      return gridData.rows[index].type === 'header' ? gridData.headerHeight : gridData.rowHeight;
    },
    [gridData],
  );

  if (!gridData) {
    return (
      <div
        ref={libraryContainerRef}
        className="flex-1 w-full h-full"
        onClick={props.onClearSelection}
        onContextMenu={props.onEmptyAreaContextMenu}
      />
    );
  }

  const handleHeaderSort = (key: string) => {
    props.onClearSelection();
    setSortCriteria((prev: any) => {
      if (prev.key === key) {
        if (prev.order === SortDirection.Ascending) {
          return { ...prev, order: SortDirection.Descending };
        } else {
          return { key: 'name', order: SortDirection.Ascending };
        }
      }
      return { key, order: SortDirection.Ascending };
    });
  };

  return (
    <div
      ref={libraryContainerRef}
      className="flex-1 w-full h-full"
      onClick={handleGridClick}
      onMouseDown={handleGridMouseDown}
      onContextMenu={props.onEmptyAreaContextMenu}
    >
      {dragSelectRect && (
        <div
          className="fixed border border-accent bg-accent/20 pointer-events-none z-50"
          style={{
            left: dragSelectRect.x1,
            top: dragSelectRect.y1,
            width: dragSelectRect.x2 - dragSelectRect.x1,
            height: dragSelectRect.y2 - dragSelectRect.y1,
          }}
        />
      )}
      <div className="flex flex-col w-full h-full">
        {gridData.isListView && (
          <ListHeader
            widths={listColumnWidths}
            setWidths={(w: any) => setLibrary({ listColumnWidths: typeof w === 'function' ? w(listColumnWidths) : w })}
            containerRef={libraryContainerRef}
            sortCriteria={sortCriteria}
            onSortChange={handleHeaderSort}
          />
        )}
        <div style={{ height: gridData.isListView ? gridSize.height - 36 : gridSize.height, width: gridSize.width }}>
          <List
            listRef={setListHandle}
            rowCount={gridData.rows.length}
            rowHeight={getItemSize}
            onScroll={(e: React.UIEvent<HTMLElement>) => handleScroll(e.currentTarget.scrollTop)}
            className="custom-scrollbar"
            rowComponent={Row}
            rowProps={memoizedRowProps}
          />
        </div>
      </div>
    </div>
  );
}
