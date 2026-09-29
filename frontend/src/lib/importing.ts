import { api, upload, type UploadResult } from './api';
import { jobStarted } from './jobs';
import { projectParam } from './projects';
import { useStore } from '../store';

const CAD = /\.(step|stp|iges|igs|brep)$/i;

function handle(res: UploadResult) {
  res.jobs.forEach(j => jobStarted(j));
  const st = useStore.getState();
  if (res.jobs.length) st.toast({ kind: 'info', title: `Importing ${res.jobs.length} file${res.jobs.length > 1 ? 's' : ''}` });
  if (res.images.length) {
    st.toast({ kind: 'ok', title: `Added ${res.images.length} photo${res.images.length > 1 ? 's' : ''}` });
    st.refreshAssets();
  }
}

const SCAN = /\.(ply|pcd|xyz|xyzn|xyzrgb|asc|pts|txt|csv|obj|stl|off|glb|gltf)$/i;

export async function uploadFiles(files: File[]) {
  const st = useStore.getState();
  // "Run on every upload": scans go straight through autopilot; CAD models and photos import normally
  const scans = files.filter(f => SCAN.test(f.name) && !CAD.test(f.name));
  if (scans.length) {
    const auto = await api.get<{ auto_on_upload?: boolean }>('/api/autopilot/settings').catch(() => null);
    if (auto?.auto_on_upload) {
      try {
        const pid = projectParam(useStore.getState().projectId);
        const res = await upload('/api/autopilot/upload', scans, undefined, pid ? { project_id: pid } : undefined);
        if (res?.id) jobStarted(res);
        st.toast({ kind: 'info', title: `Autopilot started on ${scans.length} scan${scans.length > 1 ? 's' : ''}` });
      } catch (err) {
        st.toast({ kind: 'error', title: 'Autopilot upload failed', body: (err as Error).message });
      }
      files = files.filter(f => !scans.includes(f));
      if (!files.length) return;
    }
  }
  const id = st.toast({ kind: 'info', title: `Uploading ${files.length} file${files.length > 1 ? 's' : ''}…`, ms: 0 });
  const cad = files.filter(f => CAD.test(f.name)).length;
  try {
    const pid = projectParam(useStore.getState().projectId);
    const res = await upload('/api/upload', files, f => {
      useStore.setState(s => ({ toasts: s.toasts.map(t => (t.id === id ? { ...t, title: `Uploading ${files.length} file${files.length > 1 ? 's' : ''}… ${Math.round(f * 100)}%` } : t)) }));
    }, pid ? { project_id: pid } : undefined);
    st.dismiss(id);
    handle(res);
    if (cad) st.toast({ kind: 'info', title: 'CAD model imported', body: 'Use it as the reference in Inspect to check scan accuracy.' });
  } catch (err) {
    st.dismiss(id);
    st.toast({ kind: 'error', title: 'Upload failed', body: (err as Error).message });
  }
}

export async function importPaths(paths: string[]) {
  try {
    const pid = projectParam(useStore.getState().projectId);
    handle(await api.post<UploadResult>('/api/import-paths', pid ? { paths, project_id: pid } : { paths }));
    return true;
  } catch (err) {
    useStore.getState().toast({ kind: 'error', title: 'Import failed', body: (err as Error).message });
    return false;
  }
}

export function pickFiles(accept = '') {
  const input = document.createElement('input');
  input.type = 'file';
  input.multiple = true;
  input.accept = accept;
  input.onchange = () => input.files?.length && uploadFiles([...input.files]);
  input.click();
}
