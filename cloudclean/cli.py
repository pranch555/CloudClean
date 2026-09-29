"""Command line interface: `cloudclean <command> ...`"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .clean import CleanParams
from .mesh import MeshParams
from .register import MergeParams
from .texture import TextureParams


def _views(pairs, texture_dir=None):
    from .texture import CameraView

    views = []
    for image, camera in pairs or []:
        data = json.loads(Path(camera).read_text())
        views.append(CameraView.from_dict(data, image_path=image, base_dir=Path(camera).parent))
    return views


def _load_config(path):
    return json.loads(Path(path).read_text()) if path else {}


def _clean_params(args, config=None) -> CleanParams:
    p = CleanParams.preset(getattr(args, "preset", "standard") or "standard")
    p.update((config or {}).get("clean", {}))
    if getattr(args, "remove_plane", False):
        p.remove_plane = True
    if getattr(args, "scale", None):
        p.scale = args.scale
    if getattr(args, "voxel", None):
        p.voxel_size = args.voxel
    if getattr(args, "crop_min", None):
        p.crop_min = args.crop_min
    if getattr(args, "crop_max", None):
        p.crop_max = args.crop_max
    p.update_from_strings(getattr(args, "clean_set", None))
    return p


def _merge_params(args, config=None) -> MergeParams:
    p = MergeParams.from_dict((config or {}).get("merge", {}))
    if getattr(args, "method", None):
        p.method = args.method
    if getattr(args, "dedupe", None):
        p.dedupe_voxel = args.dedupe
    p.update_from_strings(getattr(args, "merge_set", None))
    return p


def _mesh_params(args, config=None) -> MeshParams:
    p = MeshParams.from_dict((config or {}).get("mesh", {}))
    if getattr(args, "mesh_method", None):
        p.method = args.mesh_method
    if getattr(args, "depth", None):
        p.depth = args.depth
    if getattr(args, "watertight", False):
        p.watertight = True
    if getattr(args, "smooth", None):
        p.smooth_iterations = args.smooth
    if getattr(args, "target_triangles", None):
        p.target_triangles = args.target_triangles
    p.update_from_strings(getattr(args, "mesh_set", None))
    return p


def _texture_params(args, config=None) -> TextureParams:
    p = TextureParams.from_dict((config or {}).get("texture", {}))
    if getattr(args, "texture_size", None):
        p.texture_size = args.texture_size
    p.update_from_strings(getattr(args, "texture_set", None))
    return p


def _out_path(input_path, suffix, ext=".ply"):
    src = Path(input_path)
    return src.with_name(f"{src.stem}_{suffix}{ext}")


# --------------------------------------------------------------------------- commands
def cmd_info(args):
    from .io import describe, load

    for f in args.files:
        info = describe(load(f))
        if args.json:
            print(json.dumps({"file": f, **info}, indent=2))
            continue
        print(f"\n{f}")
        for k, v in info.items():
            print(f"  {k:<20} {v}")


def cmd_clean(args):
    from .clean import clean_point_cloud
    from .io import is_cloud, load, save, to_cloud
    from .pipeline import make_logger

    log = make_logger()
    geom = load(args.input)
    if not is_cloud(geom):
        log("Input is a mesh - cleaning its vertices as a point cloud")
        geom = to_cloud(geom)
    params = _clean_params(args)
    cleaned, report = clean_point_cloud(geom, params, log)
    out = Path(args.output or _out_path(args.input, "clean"))
    save(cleaned, out)
    log(f"Saved {out}")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))


def cmd_merge(args):
    from .clean import clean_point_cloud
    from .io import is_cloud, load, save
    from .pipeline import export_mesh, make_logger
    from .register import merge_geometries, transform_meshes

    log = make_logger()
    geoms = []
    for f in args.inputs:
        g = load(f)
        if args.clean and is_cloud(g):
            g, _ = clean_point_cloud(g, _clean_params(args), log)
        geoms.append(g)
    pairs = json.loads(Path(args.pairs).read_text()) if args.pairs else None
    merged, transforms, report = merge_geometries(geoms, _merge_params(args), pairs, log)
    out = Path(args.output or _out_path(args.inputs[0], "merged"))
    save(merged, out)
    log(f"Saved merged point cloud {out}")
    if args.aligned_mesh and any(not is_cloud(g) for g in geoms):
        mesh_out = Path(args.aligned_mesh)
        save(transform_meshes(geoms, transforms), mesh_out)
        log(f"Saved aligned meshes (combined, not remeshed) {mesh_out}")
    report_path = out.with_suffix(".merge.json")
    report_path.write_text(json.dumps(report, indent=2))
    log(f"Transforms and fitness written to {report_path}")


def cmd_mesh(args):
    from .io import load, save
    from .mesh import reconstruct_mesh
    from .pipeline import make_logger

    log = make_logger()
    mesh, report = reconstruct_mesh(load(args.input), _mesh_params(args), log)
    out = Path(args.output or _out_path(args.input, "mesh"))
    save(mesh, out)
    for ext in args.also or []:
        save(mesh, out.with_suffix("." + ext.strip(".")))
    log(f"Saved {out}")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=str))


def cmd_texture(args):
    from .io import is_cloud, load, save
    from .pipeline import make_logger
    from .texture import colorize_mesh, save_textured

    log = make_logger()
    mesh = load(args.mesh)
    if is_cloud(mesh):
        raise SystemExit("texture needs a mesh - run `cloudclean mesh` first")
    colored, report, textured = colorize_mesh(mesh, _views(args.view), _texture_params(args), log)
    out = Path(args.output or _out_path(args.mesh, "colored", ".glb"))
    save(colored, out)
    log(f"Saved vertex-coloured mesh {out}")
    if textured is not None:
        tex_out = out.with_name(out.stem + "_textured.glb")
        for p in save_textured(textured, tex_out) + save_textured(textured, tex_out.with_suffix(".obj")):
            log(f"Saved {p}")


def cmd_run(args):
    from .pipeline import make_logger, run_pipeline

    config = _load_config(args.config)
    pairs = json.loads(Path(args.pairs).read_text()) if args.pairs else None
    run_pipeline(args.inputs, args.output, _clean_params(args, config), _merge_params(args, config),
                 _mesh_params(args, config), _texture_params(args, config), _views(args.view),
                 skip_clean=args.no_clean, pairs=pairs, formats=args.formats.split(","), merge_mode=args.merge,
                 log=make_logger())


def cmd_watch(args):
    from .autopilot import AutopilotSettings, run_autopilot_cli

    s = AutopilotSettings.load(_load_config(args.config).get("autopilot", {}))
    if args.preset:
        s.preset = args.preset
    if args.keep_plane:
        s.remove_plane = False
    if args.method:
        s.merge_method = args.method
    if args.merge:
        s.merge_mode = args.merge
    for key, value in (("method", args.mesh_method), ("depth", args.depth), ("smooth_iterations", args.smooth),
                       ("target_triangles", args.target_triangles)):
        if value is not None:
            s.mesh[key] = value
    if args.watertight:
        s.mesh["watertight"] = True
    if args.formats:
        s.formats = args.formats.split(",")
    if args.settle is not None:
        s.settle_seconds = args.settle
    if args.group is not None:
        s.group_seconds = args.group
    if args.no_recursive:
        s.recursive = False
    run_autopilot_cli(args.folder, args.output, s, once=args.once)


def cmd_bridge(args):
    from .capture.bridge import main as bridge_main

    raise SystemExit(bridge_main(args.bridge_args))


def cmd_params(args):
    print(json.dumps({"clean": CleanParams().to_dict(), "merge": MergeParams().to_dict(),
                      "mesh": MeshParams().to_dict(), "texture": TextureParams().to_dict(),
                      "clean_presets": CleanParams.PRESETS}, indent=2))


def cmd_serve(args):
    from .web.server import serve

    serve(host=args.host, port=args.port, workspace=args.workspace, open_browser=not args.no_browser,
          accounts=not args.no_accounts)


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cloudclean",
        description="Clean, merge, mesh and colour 3D scanner point clouds.",
        epilog="Distances in parameters ending in _multiplier are relative to the scan's point spacing. "
               "Run `cloudclean params` to list every tunable parameter.")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_clean_opts(p):
        g = p.add_argument_group("cleaning")
        g.add_argument("--preset", choices=list(CleanParams.PRESETS), default="standard")
        g.add_argument("--remove-plane", action="store_true", help="remove the table/turntable surface")
        g.add_argument("--scale", type=float, help="multiply coordinates (e.g. 1000 converts m to mm)")
        g.add_argument("--voxel", type=float, help="downsample to this voxel size (default: keep all points)")
        g.add_argument("--crop-min", type=float, nargs=3, metavar=("X", "Y", "Z"))
        g.add_argument("--crop-max", type=float, nargs=3, metavar=("X", "Y", "Z"))
        g.add_argument("--clean-set", action="append", metavar="KEY=VALUE", help="any clean parameter")

    def add_merge_opts(p):
        g = p.add_argument_group("merging")
        g.add_argument("--method", choices=["auto", "markers", "icp", "none"], help="alignment method (default auto)")
        g.add_argument("--pairs", help="JSON with manual point pairs per scan index")
        g.add_argument("--dedupe", type=float, help="thin overlapping regions to this voxel size")
        g.add_argument("--merge-set", action="append", metavar="KEY=VALUE")

    def add_mesh_opts(p):
        g = p.add_argument_group("meshing")
        g.add_argument("--mesh-method", choices=["poisson", "bpa"])
        g.add_argument("--depth", type=int, help="Poisson depth (auto if omitted; 9 coarse .. 12 very fine)")
        g.add_argument("--watertight", action="store_true", help="close holes (keep full Poisson surface)")
        g.add_argument("--smooth", type=int, help="Taubin smoothing iterations")
        g.add_argument("--target-triangles", type=int, help="decimate to this many triangles")
        g.add_argument("--mesh-set", action="append", metavar="KEY=VALUE")

    def add_texture_opts(p):
        g = p.add_argument_group("colour from photos")
        g.add_argument("--view", nargs=2, action="append", metavar=("PHOTO", "CAMERA_JSON"),
                       help="photo + camera pose JSON (export it from the web app's Colour tab); repeatable")
        g.add_argument("--texture-size", type=int, help="also bake a UV texture of this size, e.g. 4096")
        g.add_argument("--texture-set", action="append", metavar="KEY=VALUE")

    p = sub.add_parser("info", help="show size, point count, spacing of files")
    p.add_argument("files", nargs="+")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("clean", help="clean a point cloud")
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--report", help="write a JSON report")
    add_clean_opts(p)
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("merge", help="align and merge 2+ point clouds or meshes of the same item")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--output", help="merged point cloud (.ply)")
    p.add_argument("--clean", action="store_true", help="clean point cloud inputs before merging")
    p.add_argument("--aligned-mesh", help="for mesh inputs: also save the aligned meshes combined")
    add_clean_opts(p)
    add_merge_opts(p)
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("mesh", help="build a mesh from a point cloud")
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--also", nargs="*", metavar="EXT", help="extra formats, e.g. --also stl glb obj")
    p.add_argument("--report")
    add_mesh_opts(p)
    p.set_defaults(func=cmd_mesh)

    p = sub.add_parser("texture", help="colour a mesh from photos")
    p.add_argument("mesh")
    p.add_argument("-o", "--output", help="output mesh (.glb/.ply/.obj)")
    add_texture_opts(p)
    p.set_defaults(func=cmd_texture)

    p = sub.add_parser("run", help="full pipeline: clean each scan, merge, mesh, colour")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--output", required=True, help="output folder")
    p.add_argument("--config", help="JSON config with clean/merge/mesh/texture sections")
    p.add_argument("--no-clean", action="store_true")
    p.add_argument("--formats", default="ply,stl,obj,glb")
    p.add_argument("--merge", choices=["auto", "always", "never"], default="auto",
                   help="auto (default): check first and merge only when the scans add coverage and align "
                        "reliably; always: merge without checking; never: mesh every scan separately")
    add_clean_opts(p)
    add_merge_opts(p)
    add_mesh_opts(p)
    add_texture_opts(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("watch", help="hands-free: watch a folder, clean/merge/mesh every new item, export mesh files")
    p.add_argument("folder", help="folder to watch (each sub-folder, or files arriving together, is one item)")
    p.add_argument("-o", "--output", required=True, help="results go to <output>/<item>/")
    p.add_argument("--once", action="store_true", help="process what is in the folder now, then exit")
    p.add_argument("--config", help="JSON with an 'autopilot' section")
    p.add_argument("--formats", help="mesh formats, comma separated (default stl,ply)")
    p.add_argument("--settle", type=float, help="seconds a file must stay unchanged (default 5)")
    p.add_argument("--group", type=float, help="files arriving within this many seconds form one item (default 30)")
    p.add_argument("--no-recursive", action="store_true", help="ignore sub-folders")
    p.add_argument("--preset", choices=list(CleanParams.PRESETS))
    p.add_argument("--keep-plane", action="store_true", help="do not remove the table/turntable surface")
    p.add_argument("--method", choices=["auto", "markers", "icp", "none"], help="merge alignment method")
    p.add_argument("--merge", choices=["auto", "always", "never"],
                   help="when to merge an item's scans: auto (default) checks first, always, never")
    p.add_argument("--mesh-method", choices=["poisson", "bpa"])
    p.add_argument("--depth", type=int)
    p.add_argument("--watertight", action="store_true")
    p.add_argument("--smooth", type=int)
    p.add_argument("--target-triangles", type=int)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("bridge", add_help=False,
                       help="on the scanning PC: upload every new Revo Metro export to a CloudClean server "
                            "(run `cloudclean bridge --help`)")
    p.add_argument("bridge_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_bridge)

    p = sub.add_parser("params", help="print all parameters and defaults as JSON")
    p.set_defaults(func=cmd_params)

    p = sub.add_parser("serve", help="start the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--workspace", default="workspace", help="folder where uploads and results are stored")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--no-accounts", action="store_true",
                   help="no sign-in (single user on this computer); by default everyone signs in (docs/accounts.md)")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["bridge"]:  # the bridge has its own parser (and runs where only httpx is installed)
        from .capture.bridge import main as bridge_main

        sys.exit(bridge_main(argv[1:]))
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        if args.debug:
            traceback.print_exc()
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
