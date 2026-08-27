import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { convertFileSrc } from '@tauri-apps/api/core';
import { ScanFace, ScanSearch, Users, X } from 'lucide-react';
import Button from '../ui/Button';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';
import { CatalogSearchQuery, ImageFile, Invokes } from '../ui/AppProperties';
import { useLibraryStore } from '../../store/useLibraryStore';
import { useUIStore } from '../../store/useUIStore';

interface FaceItem { face: { id: number; confidence: number; imageId: number; personId?: number | null; x: number; y: number; width: number; height: number }; imagePath: string; }
interface Person { id: number; displayName: string; faceCount: number; }
interface FaceCluster { id: number; faceCount: number; representativeImagePath: string; }

export default function PeopleView() {
  const [faces, setFaces] = useState<FaceItem[]>([]);
  const [message, setMessage] = useState('');
  const [starting, setStarting] = useState(false);
  const [recognizing, setRecognizing] = useState(false);
  const [people, setPeople] = useState<Person[]>([]);
  const [clusters, setClusters] = useState<FaceCluster[]>([]);
  const [name, setName] = useState('');
  const [mergeSourceId, setMergeSourceId] = useState('');
  const [mergeTargetId, setMergeTargetId] = useState('');
  const load = async () => {
    try { const [nextFaces, nextPeople, nextClusters] = await Promise.all([invoke<FaceItem[]>(Invokes.ListUnreviewedCatalogFaces), invoke<Person[]>(Invokes.ListCatalogPeople), invoke<FaceCluster[]>(Invokes.ListUnreviewedFaceClusters)]); setFaces(nextFaces); setPeople(nextPeople); setClusters(nextClusters); } catch (error) { setMessage(String(error)); }
  };
  const createPerson = async () => { if (!name.trim()) return; try { await invoke(Invokes.CreateCatalogPerson, { displayName: name }); setName(''); await load(); } catch (error) { setMessage(String(error)); } };
  const review = async (faceId: number, personId: number | null, reviewState: string) => { try { await invoke(Invokes.ReviewCatalogFace, { faceId, personId, reviewState }); await load(); } catch (error) { setMessage(String(error)); } };
  const confirmCluster = async (clusterId: number, personId: number) => { try { await invoke(Invokes.ConfirmFaceCluster, { clusterId, personId }); await load(); } catch (error) { setMessage(String(error)); } };
  useEffect(() => { void load(); }, []);
  const scan = async () => {
    setStarting(true); setMessage('Starting face detection. Progress is available in Background Jobs.');
    try { await invoke(Invokes.StartFaceDetection, { rootId: null }); } catch (error) { setMessage(String(error)); }
    finally { setStarting(false); }
  };
  const recognize = async () => {
    setRecognizing(true); setMessage('Starting face recognition. Progress is available in Background Jobs.');
    try { await invoke(Invokes.StartFaceRecognition, { rootId: null }); } catch (error) { setMessage(String(error)); }
    finally { setRecognizing(false); }
  };
  const openPerson = async (person: Person) => {
    try {
      const query: CatalogSearchQuery = { person: person.displayName, limit: 20_000 };
      const files = await invoke<ImageFile[]>(Invokes.SearchCatalogImages, { query });
      const imageRatings: Record<string, number> = {};
      files.forEach((file) => { imageRatings[file.path] = file.rating || 0; });
      useLibraryStore.getState().setLibrary({ currentFolderPath: `Library: ${person.displayName}`, activeAlbumId: null, imageList: files, imageRatings, multiSelectedPaths: [], libraryActivePath: null, libraryScrollTop: 0 });
      useLibraryStore.getState().setSearchCriteria({ text: '', tags: [], mode: 'OR' });
      useUIStore.getState().setUI({ activeView: 'library' });
    } catch (error) { setMessage(`Failed to open ${person.displayName}: ${String(error)}`); }
  };
  const mergePeople = async () => {
    if (!mergeSourceId || !mergeTargetId || mergeSourceId === mergeTargetId) return;
    try { await invoke(Invokes.MergeCatalogPeople, { sourcePersonId: Number(mergeSourceId), targetPersonId: Number(mergeTargetId) }); setMergeSourceId(''); setMergeTargetId(''); await load(); }
    catch (error) { setMessage(`Failed to merge people: ${String(error)}`); }
  };
  return <div className="flex-1 overflow-y-auto p-5">
    <div className="flex flex-wrap justify-between gap-4 mb-6">
      <div><Text variant={TextVariants.title} color={TextColors.accent}>People</Text><Text variant={TextVariants.small}>Review detected faces and build your local people library.</Text></div>
      <div className="flex gap-2"><Button onClick={() => void recognize()} disabled={recognizing}><ScanSearch size={16} />{recognizing ? 'Starting Recognition' : 'Recognize Faces'}</Button><Button onClick={() => void scan()} disabled={starting}><ScanFace size={16} />{starting ? 'Starting Scan' : 'Scan Faces'}</Button></div>
    </div>
    {message && <div className="mb-4 rounded-md border border-border-color bg-bg-primary p-3"><Text variant={TextVariants.small}>{message}</Text></div>}
    {people.length > 0 && <div className="mb-5"><Text variant={TextVariants.small} color={TextColors.secondary}>People</Text><div className="mt-2 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">{people.map((person) => <button key={person.id} className="rounded-md border border-border-color bg-bg-primary px-3 py-2 text-left hover:bg-surface" onClick={() => void openPerson(person)}><Text variant={TextVariants.small}>{person.displayName}</Text><Text as="div" variant={TextVariants.small} color={TextColors.secondary}>{person.faceCount} confirmed faces</Text></button>)}</div></div>}
    {clusters.length > 0 && <div className="mb-5"><Text variant={TextVariants.small} color={TextColors.secondary}>Unknown clusters</Text><div className="mt-2 flex gap-2 overflow-x-auto">{clusters.map((cluster) => <div key={cluster.id} className="w-28 shrink-0"><img src={convertFileSrc(cluster.representativeImagePath)} className="w-28 h-24 object-cover rounded-md border border-border-color" alt="Face cluster" /><Text variant={TextVariants.small}>{cluster.faceCount} faces</Text><select className="mt-1 w-full bg-surface border border-border-color rounded px-1 py-1 text-xs" defaultValue="" onChange={(event) => { if (event.target.value) void confirmCluster(cluster.id, Number(event.target.value)); }}><option value="">Assign...</option>{people.map((person) => <option key={person.id} value={person.id}>{person.displayName}</option>)}</select></div>)}</div></div>}
    <div className="mb-5 flex flex-wrap gap-2"><input value={name} onChange={(event) => setName(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') void createPerson(); }} placeholder="Add person" className="bg-bg-primary border border-border-color rounded-md px-3 py-2 text-sm" /><Button onClick={() => void createPerson()} disabled={!name.trim()}>Add Person</Button>{people.length > 1 && <><select className="bg-bg-primary border border-border-color rounded-md px-2 py-2 text-sm" value={mergeSourceId} onChange={(event) => setMergeSourceId(event.target.value)}><option value="">Merge person...</option>{people.map((person) => <option key={person.id} value={person.id}>{person.displayName}</option>)}</select><select className="bg-bg-primary border border-border-color rounded-md px-2 py-2 text-sm" value={mergeTargetId} onChange={(event) => setMergeTargetId(event.target.value)}><option value="">Into person...</option>{people.map((person) => <option key={person.id} value={person.id}>{person.displayName}</option>)}</select><Button className="bg-bg-primary text-text-primary border border-border-color shadow-none" onClick={() => void mergePeople()} disabled={!mergeSourceId || !mergeTargetId || mergeSourceId === mergeTargetId}>Merge</Button></>}</div>
    {faces.length === 0 ? <div className="min-h-64 flex flex-col items-center justify-center text-center"><Users size={32} className="text-text-secondary mb-3" /><Text variant={TextVariants.heading}>No unknown faces to review</Text><Text variant={TextVariants.small}>Run Scan Faces after installing the YuNet + SFace model pack.</Text></div> :
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">{faces.map((item) => { const suggested = people.find((person) => person.id === item.face.personId); const zoom = Math.max(1, Math.min(5, 0.78 / Math.max(item.face.width, item.face.height))); const centerX = (item.face.x + item.face.width / 2) * 100; const centerY = (item.face.y + item.face.height / 2) * 100; return <div key={item.face.id} className="rounded-md border border-border-color bg-bg-primary overflow-hidden"><div className="w-full aspect-square overflow-hidden bg-surface"><img src={convertFileSrc(item.imagePath)} className="w-full h-full object-cover" style={{ transform: `scale(${zoom})`, transformOrigin: `${centerX}% ${centerY}%` }} alt="Detected face" /></div><div className="p-2 space-y-2"><Text variant={TextVariants.small}>{suggested ? 'Suggested match' : 'Unknown face'}</Text><Text variant={TextVariants.small} color={TextColors.secondary}>{Math.round(item.face.confidence * 100)}% confidence</Text><select className="w-full bg-surface border border-border-color rounded px-2 py-1 text-xs" value={item.face.personId || ''} onChange={(event) => { if (event.target.value) void review(item.face.id, Number(event.target.value), 'confirmed'); }}><option value="">Name as...</option>{people.map((person) => <option key={person.id} value={person.id}>{person.displayName}</option>)}</select><button className="text-red-300 hover:text-red-200 text-xs inline-flex items-center gap-1" onClick={() => void review(item.face.id, null, 'rejected')}><X size={13} /> Not a face</button></div></div>; })}</div>}
  </div>;
}
