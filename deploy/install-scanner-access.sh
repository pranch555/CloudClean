#!/usr/bin/env bash
# Let CloudClean use a Revopoint MetroY / MetroY Ultra plugged into this Linux machine (DGX Spark, Ubuntu) directly,
# without Revo Metro: its HID command channel (/dev/hidraw*) and its camera stream (/dev/video*). Run it once:
#
#   sudo deploy/install-scanner-access.sh                 for the user who runs sudo (the one CloudClean runs as)
#   sudo deploy/install-scanner-access.sh --user dgx      for another user
#
# Why: by default Linux gives a USB device like this only to whoever is logged in at the machine's own screen. A
# server with nobody at its screen (CloudClean running as a service, used from a browser elsewhere) then cannot open
# the scanner. This rule puts the scanner in the plugdev group (Ubuntu's group for plugged-in devices) and makes
# sure the user is in it. It applies at once to a scanner that is plugged in; no replugging needed.
set -euo pipefail

USER_NAME=${SUDO_USER:-}
while [ $# -gt 0 ]; do
  case "$1" in
    --user) USER_NAME=$2; shift 2 ;;
    *) echo "Unknown option $1 (see the top of $0)" >&2; exit 2 ;;
  esac
done
if [ "$(id -u)" != 0 ]; then
  echo "Run it with sudo: sudo $0" >&2
  exit 1
fi
if [ -z "$USER_NAME" ] || ! id "$USER_NAME" >/dev/null 2>&1; then
  echo "Which user runs CloudClean? sudo $0 --user <name>" >&2
  exit 1
fi

RULES=/etc/udev/rules.d/70-metroy.rules
cat > "$RULES" <<'EOF'
# Revopoint MetroY scanners (USB 2207:110c) for CloudClean's native driver (deploy/install-scanner-access.sh):
# the HID command channel and the camera stream. plugdev: works with nobody logged in at the screen (a server);
# uaccess: also the user logged in at the screen.
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="2207", ATTRS{idProduct}=="110c", MODE="0660", GROUP="plugdev", TAG+="uaccess"
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="2207", ATTRS{idProduct}=="110c", MODE="0660", GROUP="plugdev", TAG+="uaccess"
EOF
echo "Wrote $RULES"

getent group plugdev >/dev/null || groupadd plugdev
NEW_GROUP=0
if ! id -nG "$USER_NAME" | tr ' ' '\n' | grep -qx plugdev; then
  usermod -aG plugdev "$USER_NAME"
  NEW_GROUP=1
  echo "Added $USER_NAME to the plugdev group"
fi

udevadm control --reload-rules
udevadm trigger --action=change --subsystem-match=hidraw --subsystem-match=video4linux
udevadm settle

# check what the user can open now
nodes=()
for dev in /sys/class/hidraw/hidraw* /sys/class/video4linux/video*; do
  [ -e "$dev" ] || continue
  if readlink -f "$dev/device" | grep -qi "2207:110c" || udevadm info -q property -p "$dev" 2>/dev/null | grep -qi "ID_VENDOR_ID=2207"; then
    nodes+=("/dev/$(basename "$dev")")
  fi
done
if [ ${#nodes[@]} -eq 0 ]; then
  echo "The rule is in place. No MetroY is plugged in right now: plug it in and CloudClean can use it."
  exit 0
fi
ok=1
for n in "${nodes[@]}"; do
  if [ "$NEW_GROUP" = 1 ]; then
    echo "  $n: group $(stat -c %G "$n"), mode $(stat -c %a "$n")"
  elif sudo -u "$USER_NAME" test -r "$n" -a -w "$n"; then
    echo "  $n: $USER_NAME can use it"
  else
    echo "  $n: $USER_NAME still cannot open it" >&2
    ok=0
  fi
done
if [ "$NEW_GROUP" = 1 ]; then
  echo "The new group counts for $USER_NAME's processes started from now on: restart CloudClean"
  echo "(systemctl --user restart cloudclean, as $USER_NAME, or log out and in again)."
elif [ "$ok" = 1 ]; then
  echo "Done: CloudClean can use the scanner. In CloudClean: Scan -> MetroY by USB -> Connect the MetroY."
else
  exit 1
fi
