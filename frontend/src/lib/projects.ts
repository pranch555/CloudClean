import { api, ApiError } from './api';
import type { Asset, Project } from './types';

/** Client-side stand-in when the server predates projects: one project holding every asset. */
export const ALL_PROJECT_ID = '__all__';

function virtualProject(assets: Asset[]): Project {
  const geometry = assets.filter(a => a.kind !== 'image');
  const newest = [...geometry].sort((a, b) => a.created.localeCompare(b.created)).pop();
  return {
    id: ALL_PROJECT_ID,
    name: 'My scans',
    created: assets[0]?.created ?? new Date().toISOString(),
    updated: newest?.created ?? new Date().toISOString(),
    cover_asset_id: newest?.id ?? null,
    counts: countsOf(assets),
    virtual: true,
  };
}

export function countsOf(assets: Asset[]) {
  const scans = assets.filter(a => a.kind !== 'image' && (a.operation === 'import' || a.operation === 'capture')).length;
  const meshes = assets.filter(a => a.kind === 'mesh' && a.operation !== 'import').length;
  const photos = assets.filter(a => a.kind === 'image').length;
  return { scans, meshes, photos, results: assets.length - scans - photos, total: assets.length };
}

let supported: boolean | null = null;

export const projectsSupported = () => supported !== false;

export async function loadProjects(assets: Asset[]): Promise<Project[]> {
  if (supported === false) return [virtualProject(assets)];
  try {
    const list = await api.get<Project[]>('/api/projects');
    supported = true;
    return list.length ? list : [virtualProject(assets)];
  } catch (err) {
    if (err instanceof ApiError && (err.status === 404 || err.status === 405)) {
      supported = false;
      return [virtualProject(assets)];
    }
    throw err;
  }
}

export async function createProject(name: string, description = ''): Promise<Project | null> {
  if (supported === false) return null;
  return api.post<Project>('/api/projects', { name, description });
}

export async function renameProject(id: string, name: string) {
  if (id === ALL_PROJECT_ID) return;
  await api.patch(`/api/projects/${id}`, { name });
}

export async function deleteProject(id: string, deleteAssets: boolean) {
  await api.del(`/api/projects/${id}?delete_assets=${deleteAssets ? 'true' : 'false'}`);
}

export async function moveAsset(assetId: string, projectId: string) {
  await api.post(`/api/assets/${assetId}/move`, { project_id: projectId });
}

/** project_id to send with uploads/imports/captures (none for the virtual project). */
export function projectParam(projectId: string | null): string | undefined {
  return projectId && projectId !== ALL_PROJECT_ID ? projectId : undefined;
}
