#!/usr/bin/env bash
# Run CloudClean as a background service on Linux (systemd user service, the way it runs on the DGX Spark):
# it starts at boot, restarts if it stops, and is reachable from other machines on your network.
#
#   deploy/install-service.sh                                   port 8765, workspace ~/cloudclean-workspace
#   deploy/install-service.sh --port 9000 --workspace /data/cc  other port / data folder
#   deploy/install-service.sh --print                           only show the service file
#
# The assistant's LLM server: export CLOUDCLEAN_LLM_BASE_URL (and CLOUDCLEAN_LLM_MODEL, CLOUDCLEAN_LLM_API_KEY) before
# running this and they are written into the service, or set them later in CloudClean -> Settings -> Assistant.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PORT=8765
WORKSPACE="$HOME/cloudclean-workspace"
NAME=cloudclean
PRINT=0
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT=$2; shift 2 ;;
    --workspace) WORKSPACE=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --print) PRINT=1; shift ;;
    --force) FORCE=1; shift ;;
    *) echo "Unknown option $1 (see the top of $0)" >&2; exit 2 ;;
  esac
done

unit() {
  cat <<EOF
[Unit]
Description=CloudClean scan processing (web UI on :$PORT)
After=network-online.target

[Service]
WorkingDirectory=$REPO
Environment=PYTHONUNBUFFERED=1
EOF
  local v
  for v in CLOUDCLEAN_LLM_BASE_URL CLOUDCLEAN_LLM_MODEL CLOUDCLEAN_LLM_API_KEY; do
    if [ -n "${!v:-}" ]; then echo "Environment=\"$v=${!v}\""; fi
  done
  cat <<EOF
ExecStart="$REPO/.venv/bin/cloudclean" serve --host 0.0.0.0 --port $PORT --workspace "$WORKSPACE" --no-browser
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF
}

if [ "$PRINT" = 1 ]; then unit; exit 0; fi

FILE="$HOME/.config/systemd/user/$NAME.service"
if [ -e "$FILE" ] && [ "$FORCE" = 0 ]; then
  echo "$FILE already exists. Add --force to replace it, or --name <other> for a second instance." >&2
  exit 1
fi

"$REPO/cloudclean.sh" setup
mkdir -p "$(dirname "$FILE")" "$WORKSPACE"
unit > "$FILE"
chmod 600 "$FILE"   # it may hold the LLM API key
systemctl --user daemon-reload
systemctl --user enable --now "$NAME"
ME=$(id -un)
loginctl enable-linger "$ME" 2>/dev/null ||
  echo "Note: run 'sudo loginctl enable-linger $ME' so CloudClean keeps running when you are logged out."

echo
echo "CloudClean is running: http://$(hostname):$PORT  (or this machine's IP address)"
echo "The first account you create there is the admin. Logs: journalctl --user -u $NAME -f"
