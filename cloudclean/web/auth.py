"""Accounts: who uses CloudClean, and when (docs/accounts.md).

Everyone signs in with a username and password; a new visitor creates an account on the sign-in page (the first
account becomes the admin). Everyone shares the same projects; CloudClean records who signed in when and who started
each job. The Revo Metro bridge and scripts use the machine key instead (header X-CloudClean-Key).

Stored in <workspace>/auth/:
  users.json      {key (lower-case name): {username, created, admin, pw: scrypt(salt, n, r, p, hash), last_login,
                   last_seen, logins, jobs}}
  users.json.bak  users.json as it was before the last change, for when a hand edit goes wrong
  sessions.json   {sha256(token): {username, created, last_seen, expires, agent}} - the cookie holds the token, the
                   file only its hash, so a copy of the file signs nobody in
  activity.jsonl  one line per event: register, login, failed_login, logout, job, password, removed, machine_key
  machine_key     the key for the bridge and scripts

Every read and write of these files holds self.lock (on Windows a file cannot be replaced while it is being read).
"""
from __future__ import annotations

import base64
import contextvars
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

COOKIE = "cc_session"
KEY_HEADER = "x-cloudclean-key"
MACHINE = "bridge"         # jobs started with the machine key are recorded under this name; no account can take it
SESSION_DAYS = 30          # a sign-in ends after this many days without use
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,31}$")
MIN_PASSWORD = 8
SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}
FAILS_ALLOWED = 8          # wrong passwords per name and address ...
FAIL_WINDOW = 600          # ... within this many seconds, then a pause
ACTIVITY_KEEP = 5000       # events kept in activity.jsonl

# who is making the current request (set by the server's middleware; read by the job manager)
current_user: contextvars.ContextVar[str | None] = contextvars.ContextVar("cloudclean_user", default=None)


class AuthError(ValueError):
    """A plain-language refusal; `status` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, maxmem=64 * 1024 * 1024, dklen=32)


def _new_pw(password: str) -> dict:
    salt = secrets.token_bytes(16)
    return {"salt": base64.b64encode(salt).decode(), **SCRYPT,
            "hash": base64.b64encode(_hash(password, salt, **SCRYPT)).decode()}


def _token_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def set_session_cookie(response, token: str, secure: bool) -> None:
    """The sign-in cookie. Sent again each time the server pushes the session's end back, so the browser keeps it
    as long as the server does."""
    response.set_cookie(COOKIE, token, max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax", secure=secure,
                        path="/")


class AuthStore:
    def __init__(self, workspace_root: Path):
        self.dir = Path(workspace_root) / "auth"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.fails: dict[str, list[float]] = {}
        self._dummy = {"salt": base64.b64encode(b"0" * 16).decode(), **SCRYPT,
                       "hash": base64.b64encode(b"0" * 32).decode()}
        self.machine_key()   # made on the first start

    # ------------------------------------------------------------------ files
    def _write(self, path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        for attempt in range(40):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:   # Windows: a virus scanner or an editor has the file open for a moment
                if attempt == 39:
                    raise
                time.sleep(0.05)

    def _load(self, name: str) -> dict:
        path = self.dir / name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("it is not a list of accounts")
            return data
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            if name == "sessions.json":   # harmless to lose: everyone signs in again
                return {}
            # never read a damaged users.json as "no accounts": the next visitor would become the admin and the
            # accounts would be overwritten
            raise AuthError(f"CloudClean cannot read its list of accounts (auth/{name} in the workspace: {exc}), "
                            f"so nobody can sign in until it is fixed. Fix the file, or put back {name}.bak from "
                            f"the same folder (docs/accounts.md).", 503) from exc

    def _save(self, name: str, data: dict) -> None:
        path = self.dir / name
        if name == "users.json" and path.exists():   # the version before this change, in case a hand edit goes wrong
            self._write(self.dir / "users.json.bak", path.read_text(encoding="utf-8"))
        self._write(path, json.dumps(data, indent=1))

    def log(self, event: str, username: str | None, detail: str = "") -> None:
        line = json.dumps({"time": _now(), "event": event, "user": username, "detail": detail[:200]})
        path = self.dir / "activity.jsonl"
        with self.lock:
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "a", encoding="utf-8") as f:
                f.write(line + "\n")
            if path.stat().st_size > 2_000_000:   # keep the log small: the newest events
                lines = path.read_text(encoding="utf-8").splitlines()[-ACTIVITY_KEEP:]
                self._write(path, "\n".join(lines) + "\n")

    # ------------------------------------------------------------------ accounts
    def users_exist(self) -> bool:
        with self.lock:
            return bool(self._load("users.json"))

    def register(self, username: str, password: str) -> dict:
        username = (username or "").strip()
        if not NAME.match(username):
            raise AuthError("Usernames are 3 to 32 letters, digits, dots, dashes or underscores, starting with a "
                            "letter or digit")
        if username.lower() == MACHINE:
            raise AuthError(f"'{MACHINE}' is kept for the Revo Metro bridge - pick another name")
        self._check_password(password)
        pw = _new_pw(password)   # the slow part, before taking the lock
        with self.lock:
            users = self._load("users.json")
            if username.lower() in users:
                raise AuthError("That username is taken - sign in, or pick another name", 409)
            first = not users
            users[username.lower()] = {
                "username": username, "created": _now(), "admin": first, "last_login": None, "last_seen": None,
                "logins": 0, "jobs": 0, "pw": pw}
            self._save("users.json", users)
        self.log("register", username, "first account: admin" if first else "")
        return self.public(users[username.lower()])

    def _check_password(self, password: str) -> None:
        if not isinstance(password, str) or len(password) < MIN_PASSWORD:
            raise AuthError(f"Use a password of at least {MIN_PASSWORD} characters")
        if len(password) > 256:
            raise AuthError("That password is too long (256 characters at most)")

    def verify(self, username: str, password: str, address: str = "") -> dict:
        """The account for these credentials; AuthError (401) when wrong, (429) after too many tries."""
        key = (username or "").strip().lower()
        slot = f"{key}|{address}"
        now = time.time()
        with self.lock:
            if len(self.fails) > 1000:   # forget the names nobody has tried lately
                self.fails = {k: v for k, v in self.fails.items() if v and now - v[-1] < FAIL_WINDOW}
            recent = [t for t in self.fails.get(slot, []) if now - t < FAIL_WINDOW]
            if len(recent) >= FAILS_ALLOWED:
                self.fails[slot] = recent
                raise AuthError("Too many wrong passwords - wait a few minutes and try again", 429)
            # counted as wrong until checked: tries sent at the same moment cannot slip past the limit while the
            # slow check runs
            self.fails[slot] = recent + [now]
            user = self._load("users.json").get(key)
        pw = (user or {}).get("pw") or self._dummy   # the same work for unknown names: no way to probe for names
        ok = hmac.compare_digest(_hash(password or "", base64.b64decode(pw["salt"]), pw["n"], pw["r"], pw["p"]),
                                 base64.b64decode(pw["hash"]))
        if not ok or user is None:
            # the name only when it is an account: a password typed into the name box must not reach the log
            self.log("failed_login", user["username"] if user else None, address)
            raise AuthError("Wrong username or password", 401)
        with self.lock:
            self.fails.pop(slot, None)
        return self.public(user)

    def change_password(self, username: str, current: str, new: str) -> None:
        self.verify(username, current)
        self._check_password(new)
        pw = _new_pw(new)
        with self.lock:
            users = self._load("users.json")
            if username.lower() not in users:
                raise AuthError("That account was removed", 404)
            users[username.lower()]["pw"] = pw
            self._save("users.json", users)
            sessions = {k: v for k, v in self._load("sessions.json").items() if v["username"] != username.lower()}
            self._save("sessions.json", sessions)   # other devices must sign in again
        self.log("password", username)

    def remove(self, username: str, by: str) -> None:
        with self.lock:
            users = self._load("users.json")
            if username.lower() not in users:
                raise AuthError(f"No account '{username}'", 404)
            if username.lower() == by.lower():
                raise AuthError("You cannot remove your own account", 400)
            del users[username.lower()]
            self._save("users.json", users)
            self._save("sessions.json", {k: v for k, v in self._load("sessions.json").items()
                                         if v["username"] != username.lower()})
        self.log("removed", username, f"by {by}")

    def list(self) -> list[dict]:
        with self.lock:
            users = self._load("users.json")
            sessions = self._load("sessions.json")
        now = time.time()
        out = []
        for key, u in sorted(users.items(), key=lambda kv: kv[1]["created"]):
            active = [s for s in sessions.values() if s["username"] == key and s["expires"] > now]
            out.append({**self.public(u), "signed_in_on": len(active)})
        return out

    def activity(self, limit: int = 100) -> list[dict]:
        path = self.dir / "activity.jsonl"
        with self.lock:
            if not path.exists():
                return []
            lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in reversed(lines) if line.strip()]

    @staticmethod
    def public(user: dict) -> dict:
        return {k: user.get(k) for k in ("username", "created", "admin", "last_login", "last_seen", "logins", "jobs")}

    def note_job(self, username: str, title: str) -> None:
        with self.lock:
            users = self._load("users.json")
            if username.lower() in users:
                users[username.lower()]["jobs"] = users[username.lower()].get("jobs", 0) + 1
                self._save("users.json", users)
        self.log("job", username, title)

    # ------------------------------------------------------------------ sessions
    def start_session(self, username: str, agent: str = "", address: str = "", sign_in: bool = True) -> str:
        """A new sign-in token for this account. sign_in=False when the person already was signed in (a password
        change): then it is not counted as a sign-in."""
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self.lock:
            users = self._load("users.json")
            sessions = {k: v for k, v in self._load("sessions.json").items() if v["expires"] > now}
            mine = sorted((k for k, v in sessions.items() if v["username"] == username.lower()),
                          key=lambda k: sessions[k]["last_seen"])
            for k in mine[:-19]:   # at most 20 signed-in devices per account
                del sessions[k]
            sessions[_token_key(token)] = {"username": username.lower(), "created": now, "last_seen": now,
                                           "expires": now + SESSION_DAYS * 86400, "agent": agent[:120]}
            self._save("sessions.json", sessions)
            u = users[username.lower()]
            u["last_seen"] = _now()
            if sign_in:
                u["last_login"] = u["last_seen"]
                u["logins"] = u.get("logins", 0) + 1
            self._save("users.json", users)
        if sign_in:
            self.log("login", username, address)
        return token

    def session(self, token: str | None) -> tuple[dict | None, bool]:
        """The account signed in with this cookie token (or None), and whether the session's end was just pushed back
        (a sign-in lasts SESSION_DAYS from its last use): then the cookie should be sent again."""
        if not token:
            return None, False
        key = _token_key(token)
        now = time.time()
        with self.lock:
            sessions = self._load("sessions.json")
            s = sessions.get(key)
            if s is None or s["expires"] <= now:
                return None, False
            users = self._load("users.json")
            user = users.get(s["username"])
            if user is None:
                return None, False
            if now - s["last_seen"] <= 60:   # record use at most once a minute
                return self.public(user), False
            s["last_seen"], s["expires"] = now, now + SESSION_DAYS * 86400
            self._save("sessions.json", sessions)
            user["last_seen"] = _now()
            self._save("users.json", users)
        return self.public(user), True

    def session_user(self, token: str | None) -> dict | None:
        return self.session(token)[0]

    def identify(self, token: str | None, key: str | None) -> tuple[dict | None, bool]:
        """Who is calling: the account signed in with this cookie token, else the bridge or a script with the machine
        key, else None; and whether the cookie should be sent again (see session)."""
        user, renewed = self.session(token)
        if user is None and self.key_ok(key):
            user = {"username": MACHINE, "admin": False, "machine": True}
        return user, renewed

    def end_session(self, token: str | None) -> str | None:
        if not token:
            return None
        with self.lock:
            sessions = self._load("sessions.json")
            s = sessions.pop(_token_key(token), None)
            self._save("sessions.json", sessions)
        if s:
            self.log("logout", s["username"])
        return s["username"] if s else None

    # ------------------------------------------------------------------ machine key
    def machine_key(self) -> str:
        path = self.dir / "machine_key"
        with self.lock:
            try:
                return path.read_text(encoding="utf-8").strip()
            except FileNotFoundError:   # first start, or deleted by hand: a new one
                key = secrets.token_urlsafe(32)
                self._write(path, key)
                return key

    def new_machine_key(self, by: str) -> str:
        key = secrets.token_urlsafe(32)
        with self.lock:
            self._write(self.dir / "machine_key", key)
        self.log("machine_key", by, "new key made")
        return key

    def key_ok(self, key: str | None) -> bool:
        return bool(key) and hmac.compare_digest(key.encode(), self.machine_key().encode())
