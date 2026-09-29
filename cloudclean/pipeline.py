"""End-to-end pipeline: clean each scan -> check and merge -> mesh -> colour from photos."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

import open3d as o3d

from .clean import CleanParams, clean_point_cloud
from .io import describe, is_cloud, load, save
from .mesh import MeshParams, cleanup_mesh, reconstruct_mesh
from .register import MergeParams, assess_merge, merge_geometries
from .texture import CameraView, TextureParams, colorize_mesh, save_textured

Log = Callable[[str], None]


def make_logger(prefix: str = "") -> Log:
    start = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - start:7.1f}s] {prefix}{msg}", flush=True)

    return log


def export_mesh(mesh: o3d.geometry.TriangleMesh, out_dir: Path, stem: str, formats) -> list[str]:
    paths = []
    for fmt in formats:
        paths.append(str(save(mesh, out_dir / f"{stem}.{fmt.strip('.').lower()}")))
    return paths


MERGE_MODES = ("auto", "ask", "always", "never")
SAFE_STEM = re.compile(r"[^\w\-. ()]+")


def run_pipeline(inputs: list, out_dir, clean_params: CleanParams | None = None,
                 merge_params: MergeParams | None = None, mesh_params: MeshParams | None = None,
                 texture_params: TextureParams | None = None, views: list[CameraView] | None = None,
                 skip_clean: bool = False, remesh_meshes: bool = True, pairs: dict | None = None,
                 formats=("ply", "stl", "obj", "glb"), merge_mode: str = "auto",
                 decide: Callable[[dict], str] | None = None, log: Log = print) -> dict:
    """Clean -> (check and) merge -> mesh -> colour.

    merge_mode (2+ scans): "auto" assesses first (`register.assess_merge`) and merges only on a "merge"
    recommendation (with the assessed transforms); on "use_best" only the best scan is meshed; on "ask" nothing
    is merged and every scan is meshed separately. "ask" calls `decide(assessment) -> "merge"|"best"|"separate"`
    when given (interactive terminal), otherwise it behaves like "auto". "always" merges without checking,
    "never" meshes every scan separately. Manual point pairs mean the user wants the merge: always.
    Separate meshes are written as mesh_<input stem>.<fmt>."""
    if merge_mode not in MERGE_MODES:
        raise ValueError(f"Unknown merge mode '{merge_mode}'. Choose from: {', '.join(MERGE_MODES)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"inputs": [str(i) for i in inputs], "outputs": {}}

    geoms = []
    for path in inputs:
        log(f"Loading {path}")
        g = load(path)
        info = describe(g)
        log(f"  {info['kind']}: {info.get('points', info.get('vertices')):,} points, size {info['dimensions']}")
        if is_cloud(g) and not skip_clean:
            g, rep = clean_point_cloud(g, clean_params, log)
            cleaned_path = save(g, out / f"{Path(path).stem}_clean.ply")
            report.setdefault("clean", []).append({"input": str(path), **rep})
            report["outputs"].setdefault("cleaned", []).append(str(cleaned_path))
        geoms.append(g)

    # (stem, geometry) to mesh
    sources: list[tuple[str, object]] = [("mesh", geoms[0])]
    if len(geoms) > 1:
        choice, transforms = "merge", None
        if pairs or merge_mode == "always":
            if merge_mode != "always":
                log("Manual point pairs given - merging without an assessment")
        elif merge_mode == "never":
            choice = "separate"
        else:
            assessment = assess_merge(geoms, merge_params, log)
            report["merge_assessment"] = assessment
            for reason in assessment["reasons"]:
                log(f"  {reason}")
            rec = assessment["recommendation"]
            decided = merge_mode == "ask" and decide is not None
            if decided:
                choice = decide(assessment)
                if choice not in ("merge", "best", "separate"):
                    raise ValueError(f"Unknown merge decision '{choice}'")
            else:
                choice = {"merge": "merge", "use_best": "best"}.get(rec, "separate")
            transforms = assessment["transforms"] if choice == "merge" else None
            best = assessment["best_index"]
            why = "as decided" if decided else "they cover the same surface" if choice == "best" else                 "the merge is uncertain (see merge_assessment reasons)"
            if choice == "best":
                note = f"Scans were not merged ({why}). Only {Path(inputs[best]).name} (lowest noise) was meshed."
            elif choice == "separate":
                note = f"Scans were not merged ({why}); each scan was meshed separately."
            else:
                note = None
            if note:
                report.setdefault("notes", []).append(note)
                log(f"NOTE: {note}")
        if choice == "merge":
            merged, _, rep = merge_geometries(geoms, merge_params, pairs, log, transforms=transforms)
            report["merge"] = rep
            report["outputs"]["merged"] = str(save(merged, out / "merged.ply"))
            sources = [("mesh", merged)]
        elif choice == "best":
            best = report["merge_assessment"]["best_index"]
            report["merge"] = {"skipped": "use_best", "best_index": best, "best_input": str(inputs[best])}
            sources = [("mesh", geoms[best])]
        else:
            report["merge"] = {"skipped": "separate"}
            stems: list[str] = []
            for path in inputs:
                stem = f"mesh_{SAFE_STEM.sub('_', Path(path).stem) or 'scan'}"
                while stem in stems:
                    stem += "_"
                stems.append(stem)
            sources = list(zip(stems, geoms))

    report["outputs"]["mesh"] = []
    for (stem, source), path in zip(sources, inputs if len(sources) > 1 else [None]):
        if is_cloud(source) or remesh_meshes:
            mesh, rep = reconstruct_mesh(source, mesh_params, log)
        else:
            ratio = (mesh_params or MeshParams()).min_component_ratio
            mesh, rep = cleanup_mesh(o3d.geometry.TriangleMesh(source), ratio, log), None
        report["outputs"]["mesh"] += export_mesh(mesh, out, stem, formats)
        if len(sources) > 1:
            report.setdefault("meshes", []).append({"input": str(path), "stem": stem, "report": rep,
                                                    "final": describe(mesh)})
            log(f"Mesh size of {Path(path).name} (x, y, z): {describe(mesh)['dimensions']}")
            continue
        if rep is not None:
            report["mesh"] = rep

        if views:
            colored, rep, textured = colorize_mesh(mesh, views, texture_params, log)
            report["texture"] = rep
            report["outputs"]["colored"] = export_mesh(colored, out, "mesh_colored", [f for f in formats if f != "stl"])
            if textured is not None:
                report["outputs"]["textured"] = [str(p) for p in save_textured(textured, out / "mesh_textured.glb")]
                report["outputs"]["textured"] += [str(p) for p in save_textured(textured, out / "mesh_textured.obj")]
        report["final"] = describe(mesh)
    if len(sources) > 1 and views:
        log("Photo colouring skipped: the scans were meshed separately")

    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"Pipeline complete -> {out.resolve()}")
    if "final" in report:
        log(f"Final mesh size (x, y, z): {report['final']['dimensions']}")
    return report
