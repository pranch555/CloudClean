import { uid } from '../../lib/uid';
export interface Attachment {
  id: string;
  url: string;      // data URL sent to the model
  caption: string;  // short label the model sees ("photo of the part", "viewer screenshot")
  name: string;     // shown in the UI
  /** the original photo from this computer: kept in the project (full resolution) when the message is sent */
  file?: File;
  /** the photo asset this picture is (already saved in the project) */
  assetId?: string;
}

export const MAX_ATTACHMENTS = 6;

/** Re-encode to a sane size: phone photos are far bigger than a vision model needs. */
export function shrinkImage(file: Blob, maxSide = 1600, quality = 0.85): Promise<string> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      const scale = Math.min(1, maxSide / Math.max(img.width, img.height));
      const w = Math.max(1, Math.round(img.width * scale));
      const h = Math.max(1, Math.round(img.height * scale));
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const g = canvas.getContext('2d');
      if (!g) return reject(new Error('Cannot read the image'));
      g.drawImage(img, 0, 0, w, h);
      // keep PNG for screenshots (sharp text), JPEG for photos
      const png = file.type === 'image/png' && w * h <= 1600 * 1200;
      resolve(canvas.toDataURL(png ? 'image/png' : 'image/jpeg', quality));
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('That file is not an image CloudClean can read'));
    };
    img.src = url;
  });
}

/** HEIC/HEIF phone photos: only Safari can show them, so the server turns them into JPEG. */
export const isHeif = (f: File) => /\.(heic|heif)$/i.test(f.name) || /image\/hei[cf]/.test(f.type);

/** Photos by type, or by name when the browser does not know the type (HEIC on Windows). */
export const isPhotoFile = (f: File) => f.type.startsWith('image/') || isHeif(f);

async function viewable(file: File): Promise<Blob> {
  if (!isHeif(file)) return file;
  const body = new FormData();
  body.append('file', file);
  const res = await fetch('/api/images/jpeg', { method: 'POST', body });
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `Cannot read ${file.name}`);
  return res.blob();
}

export async function attachmentFromFile(file: File): Promise<Attachment> {
  return { id: uid(), url: await shrinkImage(await viewable(file)), caption: `photo: ${file.name}`, name: file.name, file };
}
