import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { convertFileSrc } from '@tauri-apps/api/core';
import { ScanFace, Users } from 'lucide-react';
import Button from '../ui/Button';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';
import { Invokes } from '../ui/AppProperties';

interface FaceItem { face: { id: number; confidence: number; imageId: number }; imagePath: string; }

export default function PeopleView() {
  const [faces, setFaces] = useState<FaceItem[]>([]);
  const [message, setMessage] = useState('');
  const [starting, setStarting] = useState(false);
  const load = async () => {
    try { setFaces(await invoke<FaceItem[]>(Invokes.ListUnreviewedCatalogFaces)); } catch (error) { setMessage(String(error)); }
  };
  useEffect(() => { void load(); }, []);
  const scan = async () => {
    setStarting(true); setMessage('Starting face detection. Progress is available in Background Jobs.');
    try { await invoke(Invokes.StartFaceDetection, { rootId: null }); } catch (error) { setMessage(String(error)); }
    finally { setStarting(false); }
  };
  return <div className="flex-1 overflow-y-auto p-5">
    <div className="flex flex-wrap justify-between gap-4 mb-6">
      <div><Text variant={TextVariants.title} color={TextColors.accent}>People</Text><Text variant={TextVariants.small}>Review detected faces and build your local people library.</Text></div>
      <Button onClick={() => void scan()} disabled={starting}><ScanFace size={16} />{starting ? 'Starting Scan' : 'Scan Faces'}</Button>
    </div>
    {message && <div className="mb-4 rounded-md border border-border-color bg-bg-primary p-3"><Text variant={TextVariants.small}>{message}</Text></div>}
    {faces.length === 0 ? <div className="min-h-64 flex flex-col items-center justify-center text-center"><Users size={32} className="text-text-secondary mb-3" /><Text variant={TextVariants.heading}>No unknown faces to review</Text><Text variant={TextVariants.small}>Run Scan Faces after installing the YuNet + SFace model pack.</Text></div> :
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">{faces.map((item) => <div key={item.face.id} className="rounded-md border border-border-color bg-bg-primary overflow-hidden"><img src={convertFileSrc(item.imagePath)} className="w-full aspect-square object-cover" alt="Detected face" /><div className="p-2"><Text variant={TextVariants.small}>Unknown face</Text><Text variant={TextVariants.small} color={TextColors.secondary}>{Math.round(item.face.confidence * 100)}% confidence</Text></div></div>)}</div>}
  </div>;
}
