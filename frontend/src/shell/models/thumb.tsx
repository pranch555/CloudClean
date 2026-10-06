import { useState } from 'react';
import { Image as ImageIcon, Shapes, Sparkle } from 'lucide-react';
import type { Asset } from '../../lib/types';
import { useThumbs } from '../../lib/thumbnails';
import { useStore } from '../../store';

export const SERIES_VARS = ['--series-1', '--series-2', '--series-3', '--series-4', '--series-5', '--series-6', '--series-7', '--series-8'];

/** Stable colour per model (order of creation among geometry assets). */
export const assetColorVar = (assets: Asset[], id: string) => {
  const geo = assets.filter(a => a.kind !== 'image');
  const i = geo.findIndex(a => a.id === id);
  return SERIES_VARS[(i < 0 ? 0 : i) % SERIES_VARS.length];
};

/** Part size in the model's own axes when known (smallest box), else the scanner-frame box. */
export function sizeText(a: Asset): string {
  const p = a.part?.dimensions;
  const d = p ? [p.length, p.width, p.height] : a.stats.oriented_dimensions ?? a.stats.dimensions;
  if (!d) return '';
  return [...d].sort((x, y) => y - x).map(v => (v >= 100 ? v.toFixed(0) : v.toFixed(1))).join(' × ');
}

export function Thumb({ asset, size = 52 }: { asset: Asset; size?: number }) {
  const local = useThumbs(s => s.urls[asset.id]);
  const assets = useStore(s => s.assets);
  const [failed, setFailed] = useState(false);
  const src = local ?? (asset.has_thumbnail && !failed ? `/api/assets/${asset.id}/thumbnail?v=${encodeURIComponent(asset.created)}` : null);
  const Icon = asset.kind === 'mesh' ? Shapes : asset.kind === 'image' ? ImageIcon : Sparkle;
  return (
    <span className="thumb" style={{ width: size, height: size }}>
      {asset.kind === 'image' ? <img src={`/api/assets/${asset.id}/image`} alt="" loading="lazy" /> : src ? <img src={src} alt="" onError={() => setFailed(true)} /> : <Icon size={20} aria-hidden />}
      {asset.kind !== 'image' && <span className="swatch" style={{ background: `var(${assetColorVar(assets, asset.id)})` }} aria-hidden />}
    </span>
  );
}
