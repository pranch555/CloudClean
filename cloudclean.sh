#!/usr/bin/env bash
# Start CloudClean on Linux, macOS or an NVIDIA DGX Spark. It opens at http://localhost:8765
# The first start sets everything up (a few minutes; about 800 MB of disk); later starts take seconds.
#
#   ./cloudclean.sh                    start the web app
#   ./cloudclean.sh --host 0.0.0.0     ... reachable from other machines on your network (options go to "cloudclean serve")
#   ./cloudclean.sh setup              only set up / update (used by deploy/install-service.sh)
set -euo pipefail
cd "$(dirname "$0")"
VENV=.venv
PY="$VENV/bin/python"

find_python() {  # Open3D needs Python 3.10 - 3.12
  local v
  for v in python3.12 python3.11 python3.10 python3; do
    if command -v "$v" >/dev/null 2>&1 &&
       "$v" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 12))' 2>/dev/null; then
      echo "$v"; return 0
    fi
  done
  return 1
}

make_venv() {
  local py
  if py=$(find_python) && "$py" -m venv "$VENV" >/dev/null 2>&1; then return 0; fi
  # No suitable Python, or no venv module (Ubuntu without python3-venv): uv fetches Python 3.12, no sudo needed.
  rm -rf "$VENV"
  if ! command -v uv >/dev/null 2>&1; then
    echo "Python 3.10 - 3.12 with venv was not found. Installing uv, which fetches Python 3.12 for CloudClean..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv venv --seed --python 3.12 "$VENV"
}

fix_open3d() {
  # Open3D's Linux ARM64 wheel (DGX Spark, Jetson, Raspberry Pi) needs libgfortran.so.5, which DGX OS does not ship.
  # numpy bundles a copy; put it in Open3D's own folder, where Open3D looks first, so no sudo or LD_LIBRARY_PATH is needed.
  "$PY" -c 'import open3d' >/dev/null 2>&1 && return 0
  local site lib
  site=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  lib=$(ls "$site"/numpy.libs/libgfortran*.so.5* "$site"/scipy.libs/libgfortran*.so.5* 2>/dev/null | head -n 1 || true)
  if [ -n "$lib" ] && [ -d "$site/open3d" ]; then
    cp "$lib" "$site/open3d/libgfortran.so.5"
  fi
  if ! "$PY" -c 'import open3d' >/dev/null; then
    echo "Open3D does not load (see above). On Ubuntu / Debian, this installs what it needs:" >&2
    echo "  sudo apt install libgfortran5 libgomp1 libidn2-0 libgl1 libegl1" >&2
    return 1
  fi
}

setup() {
  if [ ! -x "$PY" ]; then
    echo "Setting up CloudClean for the first time. This takes a few minutes..."
    make_venv
  fi
  "$PY" -m pip install --disable-pip-version-check -q --upgrade pip
  "$PY" -m pip install --disable-pip-version-check -e ".[all]"
  fix_open3d
  cp pyproject.toml "$VENV/cloudclean-installed.toml"
}

# set up again only when the requirements changed since last time (for example after "git pull")
if [ "${1:-}" = setup ] || ! cmp -s pyproject.toml "$VENV/cloudclean-installed.toml"; then
  setup
fi
if [ "${1:-}" = setup ]; then
  echo "CloudClean is set up. Start it with ./cloudclean.sh"
  exit 0
fi
exec "$VENV/bin/cloudclean" serve "$@"
