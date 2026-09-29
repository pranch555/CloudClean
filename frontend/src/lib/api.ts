import type { Asset, Job, ParamsResponse } from './types';
import { sessionEnded } from './auth';

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, headers: {} };
  if (body !== undefined) {
    (init.headers as Record<string, string>)['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch {
    throw new ApiError('Cannot reach the CloudClean server. Is it still running?', 0);
  }
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      msg = typeof j.detail === 'string' ? j.detail : Array.isArray(j.detail) ? j.detail.map((d: { msg: string }) => d.msg).join('; ') : JSON.stringify(j.detail);
    } catch {
      /* not json */
    }
    if (res.status === 401 && !url.startsWith('/api/auth/')) sessionEnded();
    throw new ApiError(msg, res.status);
  }
  const type = res.headers.get('content-type') || '';
  return (type.includes('json') ? res.json() : res.text()) as Promise<T>;
}

export const api = {
  get: <T>(url: string) => request<T>('GET', url),
  post: <T>(url: string, body?: unknown) => request<T>('POST', url, body ?? {}),
  put: <T>(url: string, body?: unknown) => request<T>('PUT', url, body ?? {}),
  patch: <T>(url: string, body?: unknown) => request<T>('PATCH', url, body ?? {}),
  del: <T>(url: string) => request<T>('DELETE', url),

  params: () => request<ParamsResponse>('GET', '/api/params'),
  assets: () => request<Asset[]>('GET', '/api/assets'),
  asset: (id: string) => request<Asset & { report: Record<string, any> | null }>('GET', `/api/assets/${id}`),
  jobs: () => request<Job[]>('GET', '/api/jobs'),
};

export interface UploadResult {
  jobs: Job[];
  images: Asset[];
}

/** Multipart upload with progress (fetch has no upload progress). */
export function upload(url: string, files: File[], onProgress?: (fraction: number) => void, extra?: Record<string, string>): Promise<any> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    files.forEach(f => form.append('files', f, f.name));
    for (const [k, v] of Object.entries(extra || {})) form.append(k, v);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.upload.onprogress = e => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
      else {
        let msg = xhr.statusText;
        try {
          msg = JSON.parse(xhr.responseText).detail;
        } catch {
          /* keep status text */
        }
        if (xhr.status === 401) sessionEnded();
        reject(new ApiError(msg, xhr.status));
      }
    };
    xhr.onerror = () => reject(new ApiError('Upload failed. Is the server running?', 0));
    xhr.send(form);
  });
}

export interface SseEvent {
  event: string;
  data: any;
}

/** POST that streams Server-Sent Events (EventSource only supports GET). */
export async function* postSse(url: string, body: unknown, signal?: AbortSignal): AsyncGenerator<SseEvent> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    let msg = res.statusText;
    try {
      msg = (await res.json()).detail;
    } catch {
      /* keep */
    }
    if (res.status === 401) sessionEnded();
    throw new ApiError(msg, res.status);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let idx: number;
    while ((idx = buffer.search(/\r?\n\r?\n/)) >= 0) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx).replace(/^\r?\n\r?\n/, '');
      let event = 'message';
      const data: string[] = [];
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith(':')) continue;
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
      }
      if (!data.length) continue;
      let parsed: unknown = data.join('\n');
      try {
        parsed = JSON.parse(parsed as string);
      } catch {
        /* plain text */
      }
      yield { event, data: parsed };
    }
  }
}
