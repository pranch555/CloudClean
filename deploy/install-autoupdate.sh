#!/usr/bin/env bash
# Keep the CloudClean service (deploy/install-service.sh) up to date with GitHub by itself: a systemd user timer runs
# deploy/update.sh every 2 minutes. It pulls new code and restarts CloudClean only while nothing is running (see the
# top of update.sh), so pushing to GitHub from the computer you develop on is all it takes to update the server.
#
#   deploy/install-autoupdate.sh                     every 2 min; service cloudclean on :8765, ~/cloudclean-workspace
#   deploy/install-autoupdate.sh --every 10min       another interval (a systemd time span)
#   deploy/install-autoupdate.sh --port 9000 --workspace /data/cc --name cc2    a service installed with those options
#   deploy/install-autoupdate.sh --off               stop updating by itself
#
#   journalctl --user -u cloudclean-update           what it did;  deploy/update.sh --check   whether one is waiting
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PORT=8765
WORKSPACE="$HOME/cloudclean-workspace"
NAME=cloudclean
EVERY=2min
OFF=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT=$2; shift 2 ;;
    --workspace) WORKSPACE=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --every) EVERY=$2; shift 2 ;;
    --off) OFF=1; shift ;;
    *) echo "Unknown option $1 (see the top of $0)" >&2; exit 2 ;;
  esac
done

DIR="$HOME/.config/systemd/user"
UNIT="$NAME-update"
if [ "$OFF" = 1 ]; then
  systemctl --user disable --now "$UNIT.timer" 2>/dev/null || true
  rm -f "$DIR/$UNIT.timer" "$DIR/$UNIT.service"
  systemctl --user daemon-reload
  echo "CloudClean no longer updates by itself. Update by hand with $REPO/deploy/update.sh"
  exit 0
fi

git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 ||
  { echo "$REPO is not a git checkout: clone it from GitHub first (docs/dgx-spark.md)." >&2; exit 1; }
chmod +x "$REPO/deploy/update.sh"
mkdir -p "$DIR"
cat > "$DIR/$UNIT.service" <<EOF
[Unit]
Description=Update CloudClean ($NAME) from GitHub when it is idle

[Service]
Type=oneshot
WorkingDirectory=$REPO
Environment=CLOUDCLEAN_SERVICE=$NAME
Environment=CLOUDCLEAN_PORT=$PORT
Environment="CLOUDCLEAN_WORKSPACE=$WORKSPACE"
ExecStart="$REPO/deploy/update.sh"
EOF
cat > "$DIR/$UNIT.timer" <<EOF
[Unit]
Description=Check GitHub for CloudClean updates every $EVERY

[Timer]
OnActiveSec=30s
OnBootSec=3min
OnUnitInactiveSec=$EVERY

[Install]
WantedBy=timers.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now "$UNIT.timer"
echo "CloudClean ($NAME) now updates itself from GitHub every $EVERY when it is idle."
echo "What it did: journalctl --user -u $UNIT    Turn off: $0 --off"
