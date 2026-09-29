"""Strip all-black vertex colours from the .ply files already stored in a workspace.

Poisson reconstruction always emits one colour per vertex, zero-filled when the scan had no texture. Meshes built
before that array was dropped at save time still carry it, and a viewer multiplies the lit surface by black, so the
part shows up as an unlit silhouette. This rewrites only the files that are affected; everything else is left alone.

    python tools/repair_colors.py ~/cloudclean-workspace [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloudclean.io import drop_blank_colors, has_blank_colors, load, save  # noqa: E402


def asset_name(ply: Path) -> str:
    meta = ply.parent / "meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text()).get("name", ply.parent.name)
        except (OSError, ValueError):
            pass
    return ply.parent.name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workspace", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="report what would change without writing")
    args = ap.parse_args()

    if not args.workspace.is_dir():
        print(f"No such workspace: {args.workspace}")
        return 2

    repaired = 0
    for ply in sorted(args.workspace.rglob("*.ply")):
        try:
            geom = load(ply, repair=False)  # the file exactly as stored
        except Exception as exc:  # a half-written or unrelated file should not stop the sweep
            print(f"  skipped {ply}: {exc}")
            continue
        if not has_blank_colors(geom):
            continue
        drop_blank_colors(geom)
        repaired += 1
        print(f"  {'would repair' if args.dry_run else 'repaired'} {asset_name(ply)} -> {ply.name}")
        if not args.dry_run:
            save(geom, ply)

    print(f"{repaired} file(s) {'need repair' if args.dry_run else 'repaired'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
