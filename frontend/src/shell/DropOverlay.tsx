import { useEffect, useState } from 'react';
import { UploadCloud } from 'lucide-react';
import { uploadFiles } from '../lib/importing';

/** Files can be dropped anywhere on the window. */
export function DropOverlay() {
  const [over, setOver] = useState(false);
  useEffect(() => {
    let depth = 0;
    const hasFiles = (e: DragEvent) => [...(e.dataTransfer?.types ?? [])].includes('Files');
    const enter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth++;
      setOver(true);
    };
    const leave = () => {
      depth = Math.max(0, depth - 1);
      if (!depth) setOver(false);
    };
    const overFn = (e: DragEvent) => hasFiles(e) && e.preventDefault();
    const drop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      setOver(false);
      const target = e.target as HTMLElement;
      if (target.closest('[data-own-drop]')) return;
      if (e.dataTransfer?.files.length) uploadFiles([...e.dataTransfer.files]);
    };
    window.addEventListener('dragenter', enter);
    window.addEventListener('dragleave', leave);
    window.addEventListener('dragover', overFn);
    window.addEventListener('drop', drop);
    return () => {
      window.removeEventListener('dragenter', enter);
      window.removeEventListener('dragleave', leave);
      window.removeEventListener('dragover', overFn);
      window.removeEventListener('drop', drop);
    };
  }, []);
  if (!over) return null;
  return (
    <div className="drop-overlay" aria-hidden>
      <div className="drop-card">
        <UploadCloud size={34} />
        <div className="drop-title">Drop to import</div>
        <div className="drop-sub">Scans, meshes, STEP / IGES CAD models and photos go into this project</div>
      </div>
    </div>
  );
}
