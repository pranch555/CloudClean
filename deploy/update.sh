#!/usr/bin/env bash
# Update this CloudClean checkout to the newest code on GitHub and restart the service, but only while CloudClean is
# idle: no capture session, no job queued or running, no assistant reply being written. When it is busy nothing
# happens, and the next run (the timer from deploy/install-autoupdate.sh, or running this again) updates it later.
#
#   deploy/update.sh            update now if GitHub has newer code (what the timer runs)
#   deploy/update.sh --check    only say whether an update is waiting and whether CloudClean is busy
#
# Before each update the code is saved to ~/cc-backups/code-<time>.tgz (the 20 newest are kept). If CloudClean does
# not answer after the restart, the previous code is put back and that commit is skipped until a newer one arrives.
# Files changed by hand in this folder are never overwritten: the update stops and says so.
#
# Settings (environment; deploy/install-autoupdate.sh writes them into the timer's service):
#   CLOUDCLEAN_SERVICE (cloudclean)  CLOUDCLEAN_PORT (8765)  CLOUDCLEAN_WORKSPACE (~/cloudclean-workspace)
#   CLOUDCLEAN_BRANCH (main)         CLOUDCLEAN_BACKUPS (~/cc-backups)
set -euo pipefail

main() {   # everything in a function: bash has read the whole script before git replaces this file
  REPO=$(cd "$(dirname "$0")/.." && pwd)
  cd "$REPO"
  NAME=${CLOUDCLEAN_SERVICE:-cloudclean}
  PORT=${CLOUDCLEAN_PORT:-8765}
  WORKSPACE=${CLOUDCLEAN_WORKSPACE:-$HOME/cloudclean-workspace}
  BRANCH=${CLOUDCLEAN_BRANCH:-main}
  BACKUPS=${CLOUDCLEAN_BACKUPS:-$HOME/cc-backups}
  PY="$REPO/.venv/bin/python"
  local check=0
  [ "${1:-}" = --check ] && check=1

  if ! git rev-parse --git-dir >/dev/null 2>&1; then
    say "$REPO is not a git checkout. Clone it: git clone https://github.com/pranch555/CloudClean.git"
    exit 1
  fi
  exec 9>"$(git rev-parse --git-dir)/cloudclean-update.lock"
  flock -n 9 || { say "Another update is running"; exit 0; }

  git fetch -q origin "$BRANCH"
  local old new skip why n
  old=$(git rev-parse HEAD)
  new=$(git rev-parse "origin/$BRANCH")
  if [ "$old" = "$new" ]; then
    [ "$check" = 1 ] && say "Up to date: $(git log -1 --format='%h %s')"
    exit 0
  fi
  skip=$(cat "$(git rev-parse --git-dir)/cloudclean-bad-commit" 2>/dev/null || true)
  if [ "$new" = "$skip" ]; then
    [ "$check" = 1 ] && say "GitHub's newest commit ${new:0:7} did not start last time; waiting for a newer one"
    exit 0
  fi
  if ! git merge-base --is-ancestor "$old" "$new"; then
    say "This checkout has commits that are not on GitHub ($BRANCH); not updating. See: git log origin/$BRANCH..HEAD"
    exit 1
  fi
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    say "Files here were changed by hand, so the update would overwrite them; not updating:"
    git status --short --untracked-files=no
    exit 1
  fi

  n=$(git rev-list --count "$old..$new")
  why=$(busy_reason)
  if [ "$check" = 1 ]; then
    say "Update waiting: $n new commit(s), newest: $(git log -1 --format='%h %s' "$new")${why:+. CloudClean is busy: $why}"
    exit 0
  fi
  if [ -n "$why" ]; then
    say "Update waiting ($n new commit(s)), but CloudClean is busy: $why. Trying again later."
    exit 0
  fi

  mkdir -p "$BACKUPS"
  tar czf "$BACKUPS/code-$(date +%Y%m%d-%H%M%S).tgz" --exclude=./.venv --exclude=./.git --exclude=./workspace \
      --exclude=node_modules --exclude=__pycache__ -C "$REPO" .
  ls -1t "$BACKUPS"/code-*.tgz | tail -n +21 | xargs -r rm --

  say "Updating ${old:0:7} -> ${new:0:7} ($n commit(s))"
  git merge -q --ff-only "$new"
  if ! cmp -s pyproject.toml .venv/cloudclean-installed.toml; then
    say "The requirements changed: installing"
    ./cloudclean.sh setup
  fi
  systemctl --user restart "$NAME"
  if wait_up; then
    say "CloudClean is running $(git log -1 --format='%h %s')"
    exit 0
  fi
  say "CloudClean did not answer on :$PORT after the update; putting ${old:0:7} back"
  echo "$new" > "$(git rev-parse --git-dir)/cloudclean-bad-commit"
  git reset -q --hard "$old"
  systemctl --user restart "$NAME"
  wait_up && say "CloudClean is running the previous code again" || say "CloudClean does not start: journalctl --user -u $NAME"
  exit 1
}

say() { echo "$(date '+%F %T') $*"; }

busy_reason() {  # why CloudClean must not restart now; prints nothing when it is idle or not running
  systemctl --user is-active --quiet "$NAME" || return 0
  local key cap jobs chat
  key=$(cat "$WORKSPACE/auth/machine_key" 2>/dev/null || true)
  api() { curl -fsS -m 5 -H "X-CloudClean-Key: $key" "http://127.0.0.1:$PORT/api/$1" 2>/dev/null; }
  if ! cap=$(api capture/status) || ! jobs=$(api jobs) || ! chat=$(api assistant/busy); then
    echo "it does not answer on :$PORT (starting up, or a different port/workspace)"
    return 0
  fi
  if "$PY" - "$cap" "$jobs" "$chat" <<'EOF'
import json, sys
cap, jobs, chat = (json.loads(a) for a in sys.argv[1:])
why = []
if cap.get("active"):
    why.append(f"a capture session is open ({cap.get('state')})")
n = sum(1 for j in jobs if j.get("status") in ("queued", "running"))
if n:
    why.append(f"{n} job(s) queued or running")
if chat.get("active_turns"):
    why.append("the assistant is answering")
print(", ".join(why))
EOF
  then :; else echo "its state could not be read"; fi
}

wait_up() {  # up to 90 s for the web app to answer again
  local i
  for i in $(seq 90); do
    sleep 1
    systemctl --user is-active --quiet "$NAME" || continue
    curl -fsS -m 3 -o /dev/null "http://127.0.0.1:$PORT/" 2>/dev/null && return 0
  done
  return 1
}

main "$@"
