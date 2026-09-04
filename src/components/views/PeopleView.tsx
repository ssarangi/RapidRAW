import { type PointerEvent as ReactPointerEvent, useEffect, useMemo, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { convertFileSrc } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { Check, Loader2, Pencil, RefreshCw, ScanFace, ScanSearch, Trash2, Users, X } from 'lucide-react';
import Button from '../ui/Button';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';
import { Invokes } from '../ui/AppProperties';
import { useLibraryStore } from '../../store/useLibraryStore';
import { useUIStore } from '../../store/useUIStore';

interface FaceItem {
  face: {
    id: number;
    confidence: number;
    imageId: number;
    personId?: number | null;
    x: number;
    y: number;
    width: number;
    height: number;
  };
  clusterId?: number | null;
  imagePath: string;
  cropPath?: string | null;
  thumbnailDataUrl?: string | null;
}
interface Person {
  id: number;
  displayName: string;
  faceCount: number;
  coverFaceId?: number | null;
  coverThumbnailDataUrl?: string | null;
  coverSelection?: 'automatic' | 'manual';
}
interface PersonCoverCandidate {
  faceId: number;
  thumbnailDataUrl?: string | null;
  confidence: number;
  frontalScore: number;
}
interface FaceCluster {
  id: number;
  faceCount: number;
  representativeImagePath: string;
  representativeCropPath?: string | null;
  representativeThumbnailDataUrl?: string | null;
}

interface FaceReviewGroup {
  id: number | null;
  label: string;
  faces: FaceItem[];
}

interface PersonNameComboboxProps {
  idPrefix: string;
  hasPeople: boolean;
  onAssign: (personId: number) => Promise<void>;
  onError: (error: unknown) => void;
}

function PersonNameCombobox({ idPrefix, hasPeople, onAssign, onError }: PersonNameComboboxProps) {
  const [query, setQuery] = useState('');
  const [matches, setMatches] = useState<Person[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [searching, setSearching] = useState(false);
  const [assigning, setAssigning] = useState(false);
  const listId = `${idPrefix}-person-options`;

  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    const timeout = window.setTimeout(() => {
      setSearching(true);
      void invoke<Person[]>(Invokes.SearchCatalogPeople, { query })
        .then((results) => {
          if (!cancelled) setMatches(results);
        })
        .catch((error) => {
          if (!cancelled) onError(error);
        })
        .finally(() => {
          if (!cancelled) setSearching(false);
        });
    }, 120);
    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
    };
  }, [isOpen, onError, query]);

  useEffect(() => {
    setActiveIndex((current) => (current >= matches.length ? -1 : current));
  }, [matches.length]);

  const assignPerson = async (person: Person) => {
    if (assigning) return;
    setAssigning(true);
    try {
      await onAssign(person.id);
      setQuery('');
      setMatches([]);
      setIsOpen(false);
      setActiveIndex(-1);
    } catch (error) {
      onError(error);
    } finally {
      setAssigning(false);
    }
  };

  const assignName = async () => {
    const displayName = query.trim();
    if (!displayName || assigning) return;
    setAssigning(true);
    try {
      // Search again at confirmation time so pressing Enter immediately after
      // typing can never create a duplicate before the suggestion request
      // has returned.
      const currentMatches = await invoke<Person[]>(Invokes.SearchCatalogPeople, { query: displayName });
      const existing = currentMatches.find(
        (person) => person.displayName.localeCompare(displayName, undefined, { sensitivity: 'accent' }) === 0,
      );
      const person = existing || (await invoke<Person>(Invokes.CreateCatalogPerson, { displayName }));
      await onAssign(person.id);
      setQuery('');
      setMatches([]);
      setIsOpen(false);
      setActiveIndex(-1);
    } catch (error) {
      onError(error);
    } finally {
      setAssigning(false);
    }
  };

  return (
    <div className="relative space-y-1">
      <div className="flex items-center gap-1">
        <input
          className="min-w-0 flex-1 bg-surface border border-border-color rounded px-2 py-1 text-xs text-text-primary placeholder:text-text-secondary focus:outline-none focus:border-accent"
          value={query}
          onFocus={() => setIsOpen(true)}
          onBlur={(event) => {
            if (!event.currentTarget.parentElement?.parentElement?.contains(event.relatedTarget)) setIsOpen(false);
          }}
          onChange={(event) => {
            setQuery(event.target.value);
            setIsOpen(true);
            setActiveIndex(-1);
          }}
          onKeyDown={(event) => {
            if (event.key === 'ArrowDown') {
              event.preventDefault();
              setIsOpen(true);
              setActiveIndex((current) => Math.min(current + 1, matches.length - 1));
            } else if (event.key === 'ArrowUp') {
              event.preventDefault();
              setActiveIndex((current) => Math.max(current - 1, -1));
            } else if (event.key === 'Escape') {
              setIsOpen(false);
              setActiveIndex(-1);
            } else if (event.key === 'Enter') {
              event.preventDefault();
              const selected = activeIndex >= 0 ? matches[activeIndex] : undefined;
              if (selected) void assignPerson(selected);
              else void assignName();
            }
          }}
          placeholder="Name"
          aria-label="Name this face"
          aria-autocomplete="list"
          aria-controls={listId}
          aria-expanded={isOpen}
          role="combobox"
        />
        <button
          className="shrink-0 rounded p-1 text-text-secondary hover:bg-surface hover:text-text-primary disabled:opacity-50"
          onClick={() => void assignName()}
          disabled={!query.trim() || assigning}
          data-tooltip={query.trim() ? `Assign ${query.trim()}` : 'Enter a name to assign'}
          aria-label="Assign name to face"
        >
          {assigning || searching ? <Loader2 size={14} className="animate-spin" /> : <Check size={14} />}
        </button>
      </div>
      {isOpen && (
        <div
          id={listId}
          role="listbox"
          className="absolute z-20 mt-1 max-h-40 w-full overflow-y-auto rounded border border-border-color bg-bg-primary p-1 shadow-lg"
        >
          {matches.map((person, index) => (
            <button
              key={person.id}
              type="button"
              role="option"
              aria-selected={activeIndex === index}
              className={`flex w-full items-center justify-between rounded px-2 py-1.5 text-left text-xs text-text-primary ${
                activeIndex === index ? 'bg-[#d89538] text-[#19130b]' : 'hover:bg-surface'
              }`}
              onMouseEnter={() => setActiveIndex(index)}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => void assignPerson(person)}
            >
              <span className="truncate">{person.displayName}</span>
              <span className="ml-2 shrink-0 opacity-70">{person.faceCount}</span>
            </button>
          ))}
          {!searching && matches.length === 0 && query.trim() && (
            <div className="px-2 py-1.5 text-xs text-text-secondary">Create “{query.trim()}” and assign</div>
          )}
          {!searching && matches.length === 0 && !query.trim() && (
            <div className="px-2 py-1.5 text-xs text-text-secondary">Type a name to find or create a person</div>
          )}
        </div>
      )}
    </div>
  );
}

export default function PeopleView() {
  const [faces, setFaces] = useState<FaceItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState('');
  const [starting, setStarting] = useState(false);
  const [recognizing, setRecognizing] = useState(false);
  const [people, setPeople] = useState<Person[]>([]);
  const [clusters, setClusters] = useState<FaceCluster[]>([]);
  const [name, setName] = useState('');
  const [mergeSourceId, setMergeSourceId] = useState('');
  const [mergeTargetId, setMergeTargetId] = useState('');
  const [editingPersonId, setEditingPersonId] = useState<number | null>(null);
  const [editingPersonName, setEditingPersonName] = useState('');
  const [removingPersonId, setRemovingPersonId] = useState<number | null>(null);
  const [selectedFaceIds, setSelectedFaceIds] = useState<Set<number>>(new Set());
  const [coverPickerPerson, setCoverPickerPerson] = useState<Person | null>(null);
  const [coverCandidates, setCoverCandidates] = useState<PersonCoverCandidate[]>([]);
  const selectedFaceIdsRef = useRef(selectedFaceIds);
  const faceDragSelectionRef = useRef<{ pointerId: number; shouldSelect: boolean } | null>(null);

  useEffect(() => {
    selectedFaceIdsRef.current = selectedFaceIds;
  }, [selectedFaceIds]);

  const load = async () => {
    try {
      const [nextFaces, nextPeople, nextClusters] = await Promise.all([
        invoke<FaceItem[]>(Invokes.ListUnreviewedCatalogFaces),
        invoke<Person[]>(Invokes.ListCatalogPeople),
        invoke<FaceCluster[]>(Invokes.ListUnreviewedFaceClusters),
      ]);
      setFaces(nextFaces);
      setPeople(nextPeople);
      setClusters(nextClusters);
      setSelectedFaceIds(
        (current) => new Set([...current].filter((faceId) => nextFaces.some((item) => item.face.id === faceId))),
      );
      setLoading(false);

      for (const person of nextPeople) {
        if (person.coverFaceId && !person.coverThumbnailDataUrl) {
          void invoke<string>('get_or_generate_face_crop', { faceId: person.coverFaceId })
            .then((coverThumbnailDataUrl) => {
              setPeople((current) =>
                current.map((entry) => (entry.id === person.id ? { ...entry, coverThumbnailDataUrl } : entry)),
              );
            })
            .catch(() => {});
        }
      }

      // Lazily request high-res crops in background for any uncropped faces
      for (const item of nextFaces) {
        if (!item.thumbnailDataUrl && !item.cropPath) {
          invoke<string>('get_or_generate_face_crop', { faceId: item.face.id })
            .then((cropResult) => {
              if (cropResult) {
                const isDataUrl = cropResult.startsWith('data:');
                setFaces((current) =>
                  current.map((f) =>
                    f.face.id === item.face.id
                      ? {
                          ...f,
                          thumbnailDataUrl: isDataUrl ? cropResult : f.thumbnailDataUrl,
                          cropPath: isDataUrl ? f.cropPath : cropResult,
                        }
                      : f,
                  ),
                );
              }
            })
            .catch(() => {});
        }
      }
    } catch (error) {
      setMessage(String(error));
      setLoading(false);
    }
  };

  const createPerson = async () => {
    if (!name.trim()) return;
    try {
      await invoke(Invokes.CreateCatalogPerson, { displayName: name });
      setName('');
      await load();
    } catch (error) {
      setMessage(String(error));
    }
  };

  const review = async (faceId: number, personId: number | null, reviewState: string) => {
    try {
      await invoke(Invokes.ReviewCatalogFace, { faceId, personId, reviewState });
      await load();
    } catch (error) {
      setMessage(String(error));
    }
  };

  const showError = (error: unknown) => {
    setMessage(String(error));
  };

  const toggleFaceSelection = (faceId: number) => {
    setSelectedFaceIds((current) => {
      const next = new Set(current);
      if (next.has(faceId)) next.delete(faceId);
      else next.add(faceId);
      selectedFaceIdsRef.current = next;
      return next;
    });
  };

  const assignSelectedFaces = async (personId: number) => {
    const faceIds = [...selectedFaceIds];
    if (faceIds.length === 0) return;
    try {
      await invoke<number>(Invokes.ReviewCatalogFaces, {
        faceIds,
        personId,
        reviewState: 'confirmed',
      });
      setSelectedFaceIds(new Set());
      await load();
    } catch (error) {
      showError(error);
      throw error;
    }
  };

  const confirmCluster = async (clusterId: number, personId: number) => {
    try {
      await invoke(Invokes.ConfirmFaceCluster, { clusterId, personId });
      await load();
    } catch (error) {
      setMessage(String(error));
    }
  };

  useEffect(() => {
    void load();
  }, []);

  useEffect(() => {
    let disposed = false;
    let sawActiveFaceJob = false;
    const refreshAfterFaceJob = async () => {
      try {
        const jobs = await invoke<Array<{ kind: string; state: string }>>(Invokes.ListBackgroundJobs);
        const active = jobs.some(
          (job) =>
            ['face_detection', 'face_recognition'].includes(job.kind) &&
            ['queued', 'running', 'paused', 'cancelling'].includes(job.state),
        );
        if (active) {
          sawActiveFaceJob = true;
          return;
        }
        if (sawActiveFaceJob && !disposed) {
          sawActiveFaceJob = false;
          await load();
          if (!disposed) setMessage('Face analysis finished. Results refreshed.');
        }
      } catch {
        /* A library may be closed while this view is unmounting. */
      }
    };
    const timer = window.setInterval(() => void refreshAfterFaceJob(), 3000);
    void refreshAfterFaceJob();
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, []);

  const scan = async () => {
    setStarting(true);
    setMessage('Starting face detection. Progress is available in Background Jobs.');
    try {
      await invoke(Invokes.StartFaceDetection, { rootId: null });
    } catch (error) {
      setMessage(String(error));
    } finally {
      setStarting(false);
    }
  };

  const recognize = async () => {
    setRecognizing(true);
    setMessage('Starting face recognition. Progress is available in Background Jobs.');
    try {
      await invoke(Invokes.StartFaceRecognition, { rootId: null });
    } catch (error) {
      setMessage(String(error));
    } finally {
      setRecognizing(false);
    }
  };

  const openPerson = async (person: Person) => {
    try {
      const query: CatalogSearchQuery = { person: person.displayName, limit: 20_000 };
      const files = await invoke<ImageFile[]>(Invokes.SearchCatalogImages, { query });
      const imageRatings: Record<string, number> = {};
      files.forEach((file) => {
        imageRatings[file.path] = file.rating || 0;
      });
      useLibraryStore.getState().setLibrary({
        currentFolderPath: `Library: ${person.displayName}`,
        activeAlbumId: null,
        imageList: files,
        imageRatings,
        multiSelectedPaths: [],
        libraryActivePath: null,
        libraryScrollTop: 0,
      });
      useLibraryStore.getState().setSearchCriteria({ text: '', tags: [], mode: 'OR' });
      useUIStore.getState().setUI({ activeView: 'library' });
    } catch (error) {
      setMessage(`Failed to open ${person.displayName}: ${String(error)}`);
    }
  };

  const mergePeople = async () => {
    if (!mergeSourceId || !mergeTargetId || mergeSourceId === mergeTargetId) return;
    try {
      await invoke(Invokes.MergeCatalogPeople, {
        sourcePersonId: Number(mergeSourceId),
        targetPersonId: Number(mergeTargetId),
      });
      setMergeSourceId('');
      setMergeTargetId('');
      await load();
    } catch (error) {
      setMessage(`Failed to merge people: ${String(error)}`);
    }
  };

  const renamePerson = async () => {
    if (editingPersonId === null || !editingPersonName.trim()) return;
    try {
      await invoke(Invokes.RenameCatalogPerson, {
        personId: editingPersonId,
        displayName: editingPersonName,
      });
      setEditingPersonId(null);
      setEditingPersonName('');
      await load();
    } catch (error) {
      setMessage(`Failed to rename person: ${String(error)}`);
    }
  };

  const removePerson = async (person: Person) => {
    try {
      await invoke(Invokes.RemoveCatalogPerson, { personId: person.id });
      setRemovingPersonId(null);
      await load();
      setMessage(`${person.displayName} was removed. Their faces are available for relabeling.`);
    } catch (error) {
      setMessage(`Failed to remove person: ${String(error)}`);
    }
  };

  const openCoverPicker = async (person: Person) => {
    setCoverPickerPerson(person);
    setCoverCandidates([]);
    try {
      const candidates = await invoke<PersonCoverCandidate[]>(Invokes.ListCatalogPersonCoverCandidates, {
        personId: person.id,
      });
      setCoverCandidates(candidates);
    } catch (error) {
      showError(error);
    }
  };

  const setPersonCover = async (faceId: number | null) => {
    if (!coverPickerPerson) return;
    try {
      await invoke(Invokes.SetCatalogPersonCover, { personId: coverPickerPerson.id, faceId });
      setCoverPickerPerson(null);
      await load();
    } catch (error) {
      showError(error);
    }
  };

  const allFacesSelected = faces.length > 0 && selectedFaceIds.size === faces.length;
  const faceReviewGroups = useMemo<FaceReviewGroup[]>(() => {
    const knownClusterIds = new Set(clusters.map((cluster) => cluster.id));
    const facesByCluster = new Map<number, FaceItem[]>();
    const individualFaces: FaceItem[] = [];
    for (const face of faces) {
      if (face.clusterId == null || !knownClusterIds.has(face.clusterId)) {
        individualFaces.push(face);
        continue;
      }
      const group = facesByCluster.get(face.clusterId) || [];
      group.push(face);
      facesByCluster.set(face.clusterId, group);
    }
    const similarityGroups = clusters.flatMap((cluster) => {
      const groupedFaces = facesByCluster.get(cluster.id) || [];
      return groupedFaces.length > 0
        ? [{ id: cluster.id, label: 'Similarity group', faces: groupedFaces }]
        : [];
    });
    return individualFaces.length > 0
      ? [...similarityGroups, { id: null, label: 'Individual faces', faces: individualFaces }]
      : similarityGroups;
  }, [clusters, faces]);

  const updateDragSelection = (faceId: number) => {
    const dragSelection = faceDragSelectionRef.current;
    if (!dragSelection) return;
    setSelectedFaceIds((current) => {
      if (current.has(faceId) === dragSelection.shouldSelect) return current;
      const next = new Set(current);
      if (dragSelection.shouldSelect) next.add(faceId);
      else next.delete(faceId);
      selectedFaceIdsRef.current = next;
      return next;
    });
  };

  const beginFaceDragSelection = (event: ReactPointerEvent<HTMLDivElement>, faceId: number) => {
    if (event.pointerType !== 'mouse' || event.button !== 0) return;
    if (event.target instanceof HTMLElement && event.target.closest('label, input, button')) return;
    event.preventDefault();
    const shouldSelect = !selectedFaceIdsRef.current.has(faceId);
    faceDragSelectionRef.current = { pointerId: event.pointerId, shouldSelect };
    event.currentTarget.setPointerCapture(event.pointerId);
    updateDragSelection(faceId);
  };

  const continueFaceDragSelection = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!faceDragSelectionRef.current) return;
    const target = document.elementFromPoint(event.clientX, event.clientY)?.closest<HTMLElement>('[data-face-id]');
    const faceId = Number(target?.dataset.faceId);
    if (Number.isInteger(faceId)) updateDragSelection(faceId);
  };

  const endFaceDragSelection = () => {
    faceDragSelectionRef.current = null;
  };

  return (
    <div className="flex-1 overflow-y-auto p-5">
      <div className="flex flex-wrap justify-between gap-4 mb-6">
        <div>
          <Text variant={TextVariants.title} color={TextColors.accent}>
            People
          </Text>
          <Text variant={TextVariants.small}>Review detected faces and build your local people library.</Text>
        </div>
        <div className="flex gap-2">
          <Button
            className="h-9 w-9 p-0 bg-surface text-text-primary shadow-none"
            onClick={() => void load()}
            data-tooltip="Refresh people"
          >
            <RefreshCw size={16} />
          </Button>
          <Button onClick={() => void recognize()} disabled={recognizing}>
            <ScanSearch size={16} />
            {recognizing ? 'Starting Recognition' : 'Recognize Faces'}
          </Button>
          <Button onClick={() => void scan()} disabled={starting}>
            <ScanFace size={16} />
            {starting ? 'Starting Scan' : 'Scan Faces'}
          </Button>
        </div>
      </div>
      {message && (
        <div className="mb-4 rounded-md border border-border-color bg-bg-primary p-3 select-text">
          <Text variant={TextVariants.small} className="select-text break-words font-mono text-xs">
            {message}
          </Text>
        </div>
      )}
      {people.length > 0 && (
        <div className="mb-8">
          <div className="mb-3 flex items-end justify-between">
            <div>
              <Text variant={TextVariants.heading}>Your people</Text>
              <Text variant={TextVariants.small} color={TextColors.secondary}>
                Portraits are chosen for frontal pose, sharpness, visibility, and detection confidence.
              </Text>
            </div>
            <Text variant={TextVariants.small} color={TextColors.secondary}>
              {people.length} people
            </Text>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6">
            {people.map((person) => (
              <div
                key={person.id}
                className="group relative overflow-hidden rounded-xl border border-border-color bg-bg-primary shadow-[0_12px_30px_rgba(0,0,0,0.10)] transition-transform duration-200 hover:-translate-y-0.5"
              >
                <button className="block w-full text-left" onClick={() => void openPerson(person)}>
                  <div className="aspect-[5/4] overflow-hidden bg-surface">
                    {person.coverThumbnailDataUrl ? (
                      <img
                        src={person.coverThumbnailDataUrl}
                        alt={`${person.displayName} cover portrait`}
                        className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
                      />
                    ) : (
                      <div className="flex h-full items-center justify-center bg-gradient-to-br from-surface to-bg-secondary">
                        <Users size={34} className="text-text-secondary" />
                      </div>
                    )}
                  </div>
                </button>
                <div className="min-w-0 p-3">
                  {editingPersonId === person.id ? (
                    <input
                      className="w-full bg-surface border border-border-color rounded px-1.5 py-0.5 text-sm text-text-primary focus:outline-none focus:border-accent"
                      value={editingPersonName}
                      onClick={(event) => event.stopPropagation()}
                      onChange={(event) => setEditingPersonName(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') void renamePerson();
                        if (event.key === 'Escape') setEditingPersonId(null);
                      }}
                      autoFocus
                    />
                  ) : (
                    <Text variant={TextVariants.body} className="truncate font-semibold">
                      {person.displayName}
                    </Text>
                  )}
                  <Text variant={TextVariants.small} color={TextColors.secondary}>
                    {person.faceCount} confirmed photos
                  </Text>
                  <button
                    type="button"
                    className="mt-2 rounded-md border border-border-color px-2 py-1 text-[11px] text-text-secondary transition-colors hover:border-text-secondary hover:bg-surface hover:text-text-primary"
                    onClick={(event) => {
                      event.stopPropagation();
                      void openCoverPicker(person);
                    }}
                  >
                    Change portrait
                  </button>
                </div>
                <div className="absolute inset-x-0 top-0 flex items-start justify-between p-2 opacity-0 transition-opacity group-hover:opacity-100">
                  <button
                    className="rounded-full bg-black/65 px-2 py-1 text-[11px] text-white backdrop-blur hover:bg-black/80"
                    onClick={() => void openCoverPicker(person)}
                  >
                    Choose portrait
                  </button>
                  <div className="flex gap-1">
                    <button
                      className="rounded-full bg-black/65 p-1.5 text-white backdrop-blur hover:bg-black/80"
                      onClick={() => {
                        setEditingPersonId(person.id);
                        setEditingPersonName(person.displayName);
                      }}
                      aria-label="Rename person"
                    >
                      <Pencil size={13} />
                    </button>
                    <button
                      className="rounded-full bg-black/65 p-1.5 text-red-200 backdrop-blur hover:bg-black/80"
                      onClick={() => setRemovingPersonId(person.id)}
                      aria-label="Remove person"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
                {person.coverSelection === 'manual' && (
                  <span className="absolute bottom-12 right-2 rounded-full border border-[#d89538]/45 bg-[#21180e]/90 px-2 py-0.5 text-[10px] font-medium text-[#f6ce83] shadow-sm">
                    Chosen portrait
                  </span>
                )}
                {removingPersonId === person.id && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/80 p-4 text-center text-white backdrop-blur-sm">
                    <Text variant={TextVariants.small}>Remove {person.displayName}?</Text>
                    <div className="flex gap-2">
                      <button
                        className="rounded-md bg-red-500 px-3 py-1.5 text-xs"
                        onClick={() => void removePerson(person)}
                      >
                        Remove
                      </button>
                      <button
                        className="rounded-md bg-white/15 px-3 py-1.5 text-xs"
                        onClick={() => setRemovingPersonId(null)}
                      >
                        Keep
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      {coverPickerPerson && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 p-5 backdrop-blur-sm">
          <div className="w-full max-w-3xl rounded-2xl border border-border-color bg-bg-primary p-5 shadow-2xl">
            <div className="mb-4 flex items-start justify-between gap-4">
              <div>
                <Text variant={TextVariants.heading}>Choose {coverPickerPerson.displayName}’s portrait</Text>
                <Text variant={TextVariants.small} color={TextColors.secondary}>
                  Pick the face that best represents this person. The automatic choice favors frontal, sharp portraits.
                </Text>
                <button
                  type="button"
                  className="mt-3 inline-flex items-center gap-2 rounded-lg border border-accent/50 bg-accent/10 px-3 py-2 text-sm font-medium text-accent transition hover:border-accent hover:bg-accent/20"
                  onClick={() => void setPersonCover(null)}
                >
                  <RefreshCw size={15} />
                  Automatically choose the best portrait
                </button>
              </div>
              <button
                className="rounded p-1 text-text-secondary hover:bg-surface"
                onClick={() => setCoverPickerPerson(null)}
                aria-label="Close"
              >
                <X size={18} />
              </button>
            </div>
            {coverCandidates.length === 0 ? (
              <div className="flex h-48 items-center justify-center text-text-secondary">
                <Loader2 size={24} className="animate-spin" />
              </div>
            ) : (
              <div className="grid max-h-[52vh] grid-cols-3 gap-3 overflow-y-auto sm:grid-cols-4 md:grid-cols-5">
                {coverCandidates.map((candidate) => {
                  const isCurrent = candidate.faceId === coverPickerPerson.coverFaceId;
                  return (
                    <button
                      key={candidate.faceId}
                      className={`group overflow-hidden rounded-xl border text-left transition ${
                        isCurrent
                          ? 'border-accent ring-2 ring-accent/40'
                          : 'border-border-color hover:border-text-secondary'
                      }`}
                      onClick={() => void setPersonCover(candidate.faceId)}
                    >
                      <div className="aspect-square bg-surface">
                        {candidate.thumbnailDataUrl ? (
                          <img
                            src={candidate.thumbnailDataUrl}
                            alt="Portrait candidate"
                            className="h-full w-full object-cover"
                          />
                        ) : (
                          <div className="flex h-full items-center justify-center">
                            <Users size={22} className="text-text-secondary" />
                          </div>
                        )}
                      </div>
                      <div className="flex items-center justify-between px-2 py-1.5 text-[10px] text-text-secondary">
                        <span>{Math.round(candidate.frontalScore * 100)}% frontal</span>
                        {isCurrent && <Check size={12} className="text-accent" />}
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
            <div className="mt-5 flex justify-end">
              <Button className="bg-surface text-text-primary shadow-none" onClick={() => setCoverPickerPerson(null)}>
                Done
              </Button>
            </div>
          </div>
        </div>
      )}
      {clusters.length > 0 && (
        <section className="mb-6 rounded-lg border border-border-color bg-bg-primary/65 p-3">
          <div className="mb-3 flex items-end justify-between gap-3">
            <div>
              <Text variant={TextVariants.heading}>Review & name similarity groups</Text>
              <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mt-0.5">
                Name a group once to confirm every matching face in it.
              </Text>
            </div>
            <Text as="span" variant={TextVariants.small} color={TextColors.secondary} className="shrink-0">
              {clusters.length} group{clusters.length === 1 ? '' : 's'}
            </Text>
          </div>
          <div className="grid grid-cols-[repeat(auto-fill,minmax(148px,1fr))] gap-3">
            {clusters.map((cluster) => {
              const cropSrc =
                cluster.representativeThumbnailDataUrl ||
                (cluster.representativeCropPath ? convertFileSrc(cluster.representativeCropPath) : null);
              return (
                <div key={cluster.id} className="min-w-0 rounded-md border border-border-color bg-bg-secondary p-2.5">
                  <div className="aspect-square w-full overflow-hidden rounded-md border border-border-color bg-surface flex items-center justify-center">
                    {cropSrc ? (
                      <img src={cropSrc} className="w-full h-full object-cover select-none" alt="Face cluster" />
                    ) : (
                      <Loader2 size={16} className="animate-spin text-text-secondary" />
                    )}
                  </div>
                  <Text as="div" variant={TextVariants.small} color={TextColors.secondary} className="mt-2">
                    {cluster.faceCount} face{cluster.faceCount === 1 ? '' : 's'}
                  </Text>
                  <div className="mt-1.5 w-full">
                    <PersonNameCombobox
                      idPrefix={`face-cluster-${cluster.id}`}
                      hasPeople={people.length > 0}
                      onAssign={(personId) => confirmCluster(cluster.id, personId)}
                      onError={showError}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}
      <div className="mb-5 flex flex-wrap gap-2">
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') void createPerson();
          }}
          placeholder="Add person"
          className="bg-bg-primary border border-border-color rounded-md px-3 py-2 text-sm"
        />
        <Button onClick={() => void createPerson()} disabled={!name.trim()}>
          Add Person
        </Button>
        {people.length > 1 && (
          <>
            <select
              className="bg-bg-primary text-black border border-border-color rounded-md px-2 py-2 text-sm"
              value={mergeSourceId}
              onChange={(event) => setMergeSourceId(event.target.value)}
            >
              <option value="">Merge person...</option>
              {people.map((person) => (
                <option key={person.id} value={person.id}>
                  {person.displayName}
                </option>
              ))}
            </select>
            <select
              className="bg-bg-primary text-black border border-border-color rounded-md px-2 py-2 text-sm"
              value={mergeTargetId}
              onChange={(event) => setMergeTargetId(event.target.value)}
            >
              <option value="">Into person...</option>
              {people.map((person) => (
                <option key={person.id} value={person.id}>
                  {person.displayName}
                </option>
              ))}
            </select>
            <Button
              className="bg-bg-primary text-text-primary border border-border-color shadow-none"
              onClick={() => void mergePeople()}
              disabled={!mergeSourceId || !mergeTargetId || mergeSourceId === mergeTargetId}
            >
              Merge
            </Button>
          </>
        )}
      </div>
      {loading && faces.length === 0 ? (
        <div className="min-h-64 flex flex-col items-center justify-center text-center">
          <Loader2 size={32} className="animate-spin text-accent mb-3" />
          <Text variant={TextVariants.body}>Loading faces...</Text>
        </div>
      ) : faces.length === 0 ? (
        <div className="min-h-64 flex flex-col items-center justify-center text-center">
          <Users size={32} className="text-text-secondary mb-3" />
          <Text variant={TextVariants.heading}>No unknown faces to review</Text>
          <Text variant={TextVariants.small}>Run Scan Faces after installing the YuNet + SFace model pack.</Text>
        </div>
      ) : (
        <>
          <div className="mb-3 flex flex-wrap items-center gap-2 rounded-md border border-border-color bg-bg-primary p-2">
            <button
              type="button"
              className="rounded px-2 py-1 text-xs text-text-primary hover:bg-surface"
              onClick={() =>
                setSelectedFaceIds(allFacesSelected ? new Set() : new Set(faces.map((item) => item.face.id)))
              }
            >
              {allFacesSelected ? 'Clear selection' : `Select all ${faces.length}`}
            </button>
            {selectedFaceIds.size > 0 && (
              <div className="flex min-w-56 flex-1 items-center gap-2">
                <Text variant={TextVariants.small} color={TextColors.secondary} className="shrink-0">
                  {selectedFaceIds.size} selected
                </Text>
                <div className="min-w-48 flex-1">
                  <PersonNameCombobox
                    idPrefix="bulk-face-assignment"
                    hasPeople={people.length > 0}
                    onAssign={assignSelectedFaces}
                    onError={showError}
                  />
                </div>
              </div>
            )}
          </div>
          <div
            className="space-y-5 select-none"
            onPointerMove={continueFaceDragSelection}
            onPointerUp={endFaceDragSelection}
            onPointerCancel={endFaceDragSelection}
          >
            {faceReviewGroups.map((group) => (
              <section key={group.id ?? 'individual'}>
                <div className="mb-2 flex items-center justify-between gap-3">
                  <Text variant={TextVariants.small} color={TextColors.secondary}>
                    {group.label}
                  </Text>
                  <Text variant={TextVariants.small} color={TextColors.secondary} className="shrink-0">
                    {group.faces.length} face{group.faces.length === 1 ? '' : 's'}
                  </Text>
                </div>
                <div className="grid grid-cols-[repeat(auto-fill,minmax(148px,1fr))] gap-3">
                  {group.faces.map((item) => {
                    const suggested = people.find((person) => person.id === item.face.personId);
                    const cropSrc = item.thumbnailDataUrl || (item.cropPath ? convertFileSrc(item.cropPath) : null);

                    return (
                      <div
                        key={item.face.id}
                        data-face-id={item.face.id}
                        className="relative min-w-0 overflow-visible rounded-md border border-border-color bg-bg-secondary p-2.5"
                      >
                        <div
                          className="relative flex aspect-square w-full cursor-crosshair items-center justify-center overflow-hidden rounded-md border border-border-color bg-surface"
                          onPointerDown={(event) => beginFaceDragSelection(event, item.face.id)}
                        >
                          <label className="absolute left-2 top-2 z-10 flex h-5 w-5 cursor-pointer items-center justify-center rounded bg-black/60 text-white shadow">
                            <input
                              type="checkbox"
                              className="h-3.5 w-3.5 accent-[#d89538]"
                              checked={selectedFaceIds.has(item.face.id)}
                              onChange={() => toggleFaceSelection(item.face.id)}
                              aria-label="Select face for bulk assignment"
                            />
                          </label>
                          {cropSrc ? (
                            <img
                              src={cropSrc}
                              className="h-full w-full object-cover select-none"
                              alt="Detected face"
                              draggable={false}
                              onError={() => {
                                void invoke<string>('get_or_generate_face_crop', { faceId: item.face.id })
                                  .then((freshCrop) => {
                                    if (freshCrop) {
                                      const isDataUrl = freshCrop.startsWith('data:');
                                      setFaces((prev) =>
                                        prev.map((f) =>
                                          f.face.id === item.face.id
                                            ? {
                                                ...f,
                                                thumbnailDataUrl: isDataUrl ? freshCrop : f.thumbnailDataUrl,
                                                cropPath: isDataUrl ? f.cropPath : freshCrop,
                                              }
                                            : f,
                                        ),
                                      );
                                    }
                                  })
                                  .catch(() => {});
                              }}
                            />
                          ) : (
                            <Loader2 size={24} className="animate-spin text-text-secondary" />
                          )}
                        </div>
                        <div className="mt-2 space-y-2 select-text">
                          <Text as="div" variant={TextVariants.small}>
                            {suggested ? 'Suggested match' : 'Unknown face'}
                          </Text>
                          <Text as="div" variant={TextVariants.small} color={TextColors.secondary}>
                            {Math.round(item.face.confidence * 100)}% confidence
                          </Text>
                          <PersonNameCombobox
                            idPrefix={`face-${item.face.id}`}
                            hasPeople={people.length > 0}
                            onAssign={(personId) => review(item.face.id, personId, 'confirmed')}
                            onError={showError}
                          />
                          <button
                            className="inline-flex items-center gap-1 text-xs text-red-300 hover:text-red-200"
                            onClick={() => void review(item.face.id, null, 'rejected')}
                          >
                            <X size={13} /> Not a face
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
