"""Job bodies for geometry editing and exporting to the server's disk."""
from __future__ import annotations

from pathlib import Path

from ..edit import MESH, PC, apply_edits, validate_ops
from ..io import is_cloud, save
from .workspace import Workspace, safe_name


def job_edit(ws: Workspace, payload: dict, log) -> list[str]:
    asset_id = payload["asset_id"]
    meta = ws.get(asset_id)
    geom = ws.load_geometry(asset_id)
    ops = validate_ops(payload["ops"], PC if is_cloud(geom) else MESH)
    progress = getattr(log, "progress", None)
    on_op = (lambda i, n, name: progress(i / n, name)) if progress else None
    log(f"Editing '{meta['name']}' with {len(ops)} operation(s): {', '.join(op['op'] for op in ops)}")
    edited, report = apply_edits(geom, ops, log, on_op)
    if progress:
        progress(1.0, "saving")
    name = payload.get("name") or f"{meta['name']} · edit"
    return [ws.add_geometry(edited, name, "edit", [asset_id], {"ops": ops}, report)["id"]]


def export_target(ws: Workspace, asset_id: str, fmt: str, folder: str, filename: str | None = None,
                  overwrite: bool = False) -> Path:
    """Resolve and check an export path. Raises KeyError (unknown asset), ValueError (bad request) or
    FileExistsError (target exists and overwrite is False)."""
    from .server import EXPORT_FORMATS  # lazy: server imports the feature modules

    meta = ws.get(asset_id)
    kind = meta["kind"]
    if kind == "image":
        raise ValueError(f"'{meta['name']}' is a photo and cannot be exported as a 3D file")
    fmt = fmt.lower().lstrip(".")
    if fmt not in EXPORT_FORMATS[kind]:
        raise ValueError(f"A {kind} cannot be exported as '{fmt}'. Choose from: {', '.join(EXPORT_FORMATS[kind])}")
    if not folder or not folder.strip():
        raise ValueError("Give a folder to export to")
    directory = Path(folder.strip().strip('"')).expanduser()
    if not directory.is_absolute():
        raise ValueError(f"The export folder must be an absolute path on the server, got '{folder}'")
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"'{directory}' exists and is not a folder")
    stem = filename.strip() if filename and filename.strip() else meta["name"]
    if Path(stem).name != stem:
        raise ValueError("The file name must not contain folders - put those in 'folder'")
    if stem.lower().endswith(f".{fmt}"):
        stem = stem[: -len(fmt) - 1]
    target = directory / f"{safe_name(stem)}.{fmt}"
    if target.exists() and not overwrite:
        raise FileExistsError(f"'{target}' already exists - set overwrite to replace it")
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"Cannot create folder '{directory}': {exc}") from exc
    return target


def export_asset(ws: Workspace, asset_id: str, fmt: str, folder: str, filename: str | None = None,
                 overwrite: bool = False) -> Path:
    target = export_target(ws, asset_id, fmt, folder, filename, overwrite)
    try:
        save(ws.load_geometry(asset_id), target)
    except Exception:
        target.unlink(missing_ok=True)  # never leave a half-written file behind
        raise
    return target


def job_export(ws: Workspace, payload: dict, log) -> list[str]:
    path = export_asset(ws, payload["asset_id"], payload["format"], payload["folder"],
                        payload.get("filename"), bool(payload.get("overwrite")))
    log(f"Exported to {path}")
    if hasattr(log, "output"):
        log.output(path=str(path))
    return []


JOBS = {"edit": job_edit, "export": job_export}
