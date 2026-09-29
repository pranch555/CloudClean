"""Accounts (cloudclean/web/auth.py, routes_auth.py, docs/accounts.md).

The sign-in wall in front of the API, creating accounts (the first one is the admin), signing in and out, the pause
after wrong passwords, password changes, the admin's view, the machine key for the bridge and scripts, who started
each job, the live-capture websocket, the server without accounts, the bridge's --key and what is kept on disk.

Jobs never run here (the job runner is switched off): these tests check who started a job, not what it does.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import threading
import time
from datetime import datetime

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocketDisconnect

from cloudclean.capture import bridge
from cloudclean.web import auth
from cloudclean.web.jobs import JobManager
from cloudclean.web.server import STATIC_DIR, create_app
from cloudclean.web.workspace import Workspace

PASSWORD = "correct horse"
KEY = "X-CloudClean-Key"          # the header the bridge and scripts send
PUBLIC = {"/api/auth/status", "/api/auth/login", "/api/auth/register", "/api/auth/logout"}
TINY_PLY = (b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
            b"end_header\n0 0 0\n1 0 0\n0 1 0\n")


@pytest.fixture(autouse=True)
def no_job_runner(monkeypatch):
    """Submitted jobs stay queued: no worker processes (and no Open3D work) in these tests."""
    monkeypatch.setattr(JobManager, "_loop", lambda self: None)


@pytest.fixture
def app(tmp_path, no_job_runner):
    return create_app(tmp_path / "ws", auth=True)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def device(app, **kw) -> TestClient:
    """Another browser or computer, with its own cookies (the `client` fixture runs the app's start-up/shut-down)."""
    return TestClient(app, **kw)


def register(c, name, password=PASSWORD) -> dict:
    r = c.post("/api/auth/register", json={"username": name, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["user"]


def login(c, name, password=PASSWORD):
    return c.post("/api/auth/login", json={"username": name, "password": password})


def detail(r) -> str:
    return r.json()["detail"]


def events(admin) -> list[tuple[str, str | None]]:
    """(event, user) of the recent activity, newest first, as the admin sees it."""
    r = admin.get("/api/auth/users")
    assert r.status_code == 200, r.text
    return [(e["event"], e["user"]) for e in r.json()["activity"]]


def assert_no_secrets(app, text: str) -> None:
    users = json.loads((app.state.auth.dir / "users.json").read_text(encoding="utf-8"))
    for u in users.values():
        assert u["pw"]["hash"] not in text and u["pw"]["salt"] not in text
    assert PASSWORD not in text
    assert not re.search(r'"(pw|hash|salt|password)"', text), text


def refused_code(c, **kw) -> int:
    """Open the live-capture websocket and expect the server to close it before accepting."""
    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect("/api/capture/stream", **kw):
            pass
    return exc.value.code


# --------------------------------------------------------------------------- the sign-in wall
def test_signed_out_visitors_get_the_page_but_not_the_api(client):
    assert client.get("/api/auth/status").json() == {"enabled": True, "signed_in": False, "user": None,
                                                    "users_exist": False}
    for method, path in (("GET", "/api/assets"), ("GET", "/api/projects"), ("GET", "/api/jobs"),
                         ("POST", "/api/clean"), ("GET", "/api/auth/me"), ("GET", "/api/auth/users")):
        r = client.request(method, path)
        assert r.status_code == 401 and detail(r) == "Sign in first", (method, path, r.text)
    if (STATIC_DIR / "index.html").exists():    # the page loads and shows the sign-in screen itself
        page = client.get("/")
        assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert client.post("/api/auth/logout").json() == {"ok": True}     # harmless when nobody is signed in


def test_every_api_route_is_behind_the_sign_in(app, client):
    """The middleware only guards /api/ paths, and websockets must check for themselves: a new route outside
    /api/, or a new websocket without its own check, would be open to anyone on the network."""
    paths = app.openapi()["paths"]
    checked = 0
    for path, item in paths.items():
        assert path.startswith("/api/"), f"{path} is outside /api/: it needs no sign-in"
        if path in PUBLIC:
            continue
        for method in item:
            if method in ("get", "post", "put", "patch", "delete"):
                r = client.request(method.upper(), re.sub(r"\{[^}]+\}", "x", path))
                assert r.status_code == 401 and detail(r) == "Sign in first", (method, path, r.status_code)
                checked += 1
    assert checked >= 50, checked

    sockets, todo = [], list(app.routes)
    while todo:        # FastAPI >= 0.14x keeps included routers nested (original_router); older ones flatten them
        route = todo.pop()
        if isinstance(route, WebSocketRoute):
            sockets.append(route.path)
        todo.extend(getattr(getattr(route, "original_router", None), "routes", []))
    assert "/api/capture/stream" in sockets
    for path in sockets:
        assert path.startswith("/api/"), path
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(re.sub(r"\{[^}]+\}", "x", path)):
                pass
        assert exc.value.code == 4401, path


# --------------------------------------------------------------------------- accounts
def test_create_accounts(app, client):
    r = client.post("/api/auth/register", json={"username": "Alice", "password": PASSWORD})
    assert r.status_code == 200, r.text
    alice = r.json()["user"]
    assert alice["username"] == "Alice" and alice["admin"] is True and alice["logins"] == 1
    cookie = r.headers["set-cookie"].lower()
    assert cookie.startswith(f"{auth.COOKIE}=") and "httponly" in cookie and "samesite=lax" in cookie
    assert f"max-age={auth.SESSION_DAYS * 86400}" in cookie and "path=/" in cookie
    assert "secure" not in cookie       # plain http here; see test_the_cookie_is_secure_over_https
    assert_no_secrets(app, r.text)
    status = client.get("/api/auth/status").json()
    assert status["enabled"] and status["signed_in"] and status["users_exist"]
    assert status["user"]["username"] == "Alice" and status["user"]["admin"] is True
    assert client.get("/api/assets").status_code == 200
    assert client.get("/api/auth/me").json()["user"]["username"] == "Alice"

    bob = device(app)
    assert bob.get("/api/auth/status").json() == {"enabled": True, "signed_in": False, "user": None,
                                                 "users_exist": True}
    assert register(bob, "bob")["admin"] is False          # only the very first account is the admin
    assert bob.get("/api/assets").status_code == 200

    for name in ("alice", "ALICE", "  Alice  ", "Bob"):
        r = device(app).post("/api/auth/register", json={"username": name, "password": PASSWORD})
        assert r.status_code == 409 and "taken" in detail(r), (name, r.text)
    for name in ("ab", "has space", ".hidden", "-dash", "_under", "x" * 33, "", "semi;colon", "a/b", "ümlaut"):
        r = device(app).post("/api/auth/register", json={"username": name, "password": PASSWORD})
        assert r.status_code == 400 and "Usernames are 3 to 32" in detail(r), (name, r.text)
        assert "set-cookie" not in r.headers
    for password, words in (("", "at least 8"), ("seven77", "at least 8"), ("x" * 257, "too long")):
        r = device(app).post("/api/auth/register", json={"username": "carol", "password": password})
        assert r.status_code == 400 and words in detail(r), (len(password), r.text)
        assert "set-cookie" not in r.headers
    assert register(device(app), "c.d", "x" * 8)["admin"] is False       # shortest name and password allowed
    assert register(device(app), "e" * 32, "y" * 256)["admin"] is False  # longest
    names = [u["username"] for u in client.get("/api/auth/users").json()["users"]]
    assert names == ["Alice", "bob", "c.d", "e" * 32]                     # refused attempts left nothing behind


def test_sign_in_and_out(app, client):
    register(client, "Alice")
    token = client.cookies.get(auth.COOKIE)
    sessions = (app.state.auth.dir / "sessions.json").read_text(encoding="utf-8")
    assert token not in sessions                                          # only its sha256 is kept
    assert hashlib.sha256(token.encode()).hexdigest() in json.loads(sessions)

    assert client.post("/api/auth/logout").json() == {"ok": True}
    assert auth.COOKIE not in client.cookies
    assert client.get("/api/auth/status").json()["signed_in"] is False
    assert client.get("/api/assets").status_code == 401
    replay = device(app).get("/api/assets", headers={"Cookie": f"{auth.COOKIE}={token}"})
    assert replay.status_code == 401                                      # the old cookie is dead on the server

    wrong, unknown = login(client, "Alice", "not the password"), login(client, "nobody", PASSWORD)
    assert wrong.status_code == unknown.status_code == 401
    assert detail(wrong) == detail(unknown) == "Wrong username or password"   # no hint which names exist
    assert auth.COOKIE not in client.cookies

    before = datetime.now().astimezone().replace(microsecond=0)
    r = login(client, "alice")                                            # names are not case-sensitive
    assert r.status_code == 200, r.text
    user = r.json()["user"]
    assert user["username"] == "Alice" and user["logins"] == 2           # registering was the first sign-in
    assert datetime.fromisoformat(user["last_login"]) >= before and user["last_seen"]
    assert client.get("/api/assets").status_code == 200
    assert client.get("/api/auth/me").json()["user"]["logins"] == 2
    assert_no_secrets(app, r.text)


def test_too_many_wrong_passwords_pause_that_name(app, client):
    assert (auth.FAILS_ALLOWED, auth.FAIL_WINDOW) == (8, 600)             # what docs/accounts.md promises
    register(client, "alice")
    register(device(app), "bob")
    guesser = device(app)
    for _ in range(auth.FAILS_ALLOWED):
        assert login(guesser, "alice", "guess guess").status_code == 401
    for name, password in (("alice", PASSWORD), ("ALICE", PASSWORD), ("alice", "guess again")):
        r = login(guesser, name, password)                                # even the right password
        assert r.status_code == 429 and "Too many wrong passwords" in detail(r), name
    assert login(guesser, "bob").status_code == 200                       # other names are not affected
    assert client.get("/api/assets").status_code == 200                   # alice's open sessions keep working

    store = app.state.auth

    def later(seconds):
        for slot, times in store.fails.items():
            store.fails[slot] = [t - seconds for t in times]

    later(auth.FAIL_WINDOW - 30)                                          # 9.5 minutes later: still paused
    assert login(guesser, "alice").status_code == 429
    later(31)                                                             # 10 minutes later: free again
    assert login(guesser, "alice").status_code == 200
    assert login(guesser, "alice", "one typo").status_code == 401        # a good sign-in resets the count
    assert ("failed_login", "alice") in events(client)


def test_changing_the_password_signs_out_other_devices(app, client):
    register(client, "alice")
    laptop = device(app)
    assert login(laptop, "alice").status_code == 200

    r = client.post("/api/auth/password", json={"current": "wrong wrong", "new": "brand new pass"})
    assert r.status_code == 400 and "current password is not right" in detail(r)
    r = client.post("/api/auth/password", json={"current": PASSWORD, "new": "short"})
    assert r.status_code == 400 and "at least 8" in detail(r)
    assert laptop.get("/api/assets").status_code == 200                   # nothing changed yet

    r = client.post("/api/auth/password", json={"current": PASSWORD, "new": "brand new pass"})
    assert r.status_code == 200, r.text
    assert client.get("/api/assets").status_code == 200                   # this device stays signed in
    assert laptop.get("/api/assets").status_code == 401                   # every other device signs in again
    assert login(laptop, "alice").status_code == 401                      # the old password is gone
    assert login(laptop, "alice", "brand new pass").status_code == 200
    assert ("password", "alice") in events(client)
    key = client.get("/api/auth/machine-key").json()["key"]
    r = device(app).post("/api/auth/password", json={"current": PASSWORD, "new": "x" * 9}, headers={KEY: key})
    assert r.status_code == 401                                           # the machine key has no password


def test_the_admin_sees_and_removes_accounts(app, client):
    register(client, "Alice")
    bob, carol = device(app), device(app)
    register(bob, "bob")
    register(carol, "carol")
    assert login(device(app), "bob", "not bobs password").status_code == 401

    for c in (bob, carol):
        r = c.get("/api/auth/users")
        assert r.status_code == 403 and detail(r) == "Only the admin can do that"
        assert c.delete("/api/auth/users/Alice").status_code == 403
        assert c.delete("/api/auth/users/bob").status_code == 403

    r = client.get("/api/auth/users")
    assert r.status_code == 200, r.text
    assert_no_secrets(app, r.text)                                        # no password hashes, salts or passwords
    body = r.json()
    users = {u["username"]: u for u in body["users"]}
    assert list(users) == ["Alice", "bob", "carol"]                       # oldest first
    assert [u["admin"] for u in users.values()] == [True, False, False]
    for u in users.values():
        assert set(u) == {"username", "created", "admin", "last_login", "last_seen", "logins", "jobs",
                          "signed_in_on"}
        assert u["created"] and u["last_seen"] and u["logins"] == 1 and u["jobs"] == 0 and u["signed_in_on"] == 1
    seen = {(e["event"], e["user"]) for e in body["activity"]}
    assert {("register", "Alice"), ("login", "Alice"), ("register", "bob"), ("login", "carol"),
            ("failed_login", "bob")} <= seen
    assert all(e["time"] for e in body["activity"])
    failed = next(e for e in body["activity"] if e["event"] == "failed_login")
    assert failed["detail"] == "testclient"                               # the address the attempt came from

    assert client.delete("/api/auth/users/Carol").json() == {"ok": True}  # names in any case
    assert carol.get("/api/assets").status_code == 401                    # signed out at once
    assert carol.get("/api/auth/status").json()["signed_in"] is False
    assert login(carol, "carol").status_code == 401
    assert [u["username"] for u in client.get("/api/auth/users").json()["users"]] == ["Alice", "bob"]
    assert any(e == "removed" and (u or "").lower() == "carol" for e, u in events(client))

    for name in ("Alice", "alice"):
        r = client.delete(f"/api/auth/users/{name}")
        assert r.status_code == 400 and "your own account" in detail(r)
    assert client.delete("/api/auth/users/nobody").status_code == 404
    assert client.get("/api/assets").status_code == 200 and bob.get("/api/assets").status_code == 200


def test_the_machine_key(app, client):
    register(client, "alice")
    r = client.get("/api/auth/machine-key")
    assert r.status_code == 200
    key = r.json()["key"]
    assert len(key) >= 32 and key == (app.state.auth.dir / "machine_key").read_text(encoding="utf-8").strip()

    bob = device(app)
    register(bob, "bob")
    assert bob.get("/api/auth/machine-key").status_code == 403
    assert bob.post("/api/auth/machine-key").status_code == 403

    script, ping = device(app), "/api/capture/bridge/ping"               # no account, no cookie
    assert script.get(ping).status_code == 401
    assert script.get(ping, headers={KEY: "wrong"}).status_code == 401
    assert script.get(ping, headers={KEY: key[:-1]}).status_code == 401
    r = script.get(ping, headers={KEY: key})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert script.get("/api/assets", headers={KEY.lower(): key}).status_code == 200   # any API call, any case
    for method, path in (("GET", "/api/auth/me"), ("GET", "/api/auth/users"), ("GET", "/api/auth/machine-key"),
                         ("POST", "/api/auth/machine-key"), ("DELETE", "/api/auth/users/bob")):
        r = script.request(method, path, headers={KEY: key})
        assert r.status_code == 401, (method, path)                      # the key is not an account
    assert script.get("/api/auth/status", headers={KEY: key}).json()["signed_in"] is False

    new = client.post("/api/auth/machine-key").json()["key"]
    assert new != key and client.get("/api/auth/machine-key").json()["key"] == new
    assert script.get(ping, headers={KEY: key}).status_code == 401       # the old key stops at once
    assert script.get(ping, headers={KEY: new}).status_code == 200
    assert ("machine_key", "alice") in events(client)


# --------------------------------------------------------------------------- jobs
def test_jobs_remember_who_started_them(app, client, tmp_path):
    scan = tmp_path / "tiny.ply"
    scan.write_bytes(TINY_PLY)
    register(client, "Alice")
    by_path = client.post("/api/import-paths", json={"paths": [str(scan)]})            # a plain route (thread)
    upload = client.post("/api/upload", files=[("files", ("tiny.ply", TINY_PLY, "application/octet-stream"))])
    assert by_path.status_code == upload.status_code == 200, (by_path.text, upload.text)  # ... an async one
    jobs = by_path.json()["jobs"] + upload.json()["jobs"]
    assert [j["user"] for j in jobs] == ["Alice", "Alice"]
    listed = {j["id"]: j["user"] for j in client.get("/api/jobs").json()}
    assert [listed[j["id"]] for j in jobs] == ["Alice", "Alice"]

    key = client.get("/api/auth/machine-key").json()["key"]
    r = device(app).post("/api/import-paths", json={"paths": [str(scan)]}, headers={KEY: key})
    assert r.json()["jobs"][0]["user"] == "bridge"                        # the bridge and scripts show as "bridge"

    body = client.get("/api/auth/users").json()
    assert body["users"][0]["username"] == "Alice" and body["users"][0]["jobs"] == 2
    started = [(e["user"], e["detail"]) for e in body["activity"] if e["event"] == "job"]
    assert started.count(("Alice", "Import tiny.ply")) == 2 and ("bridge", "Import tiny.ply") in started


def test_the_job_manager_asks_who_is_signed_in(tmp_path):
    jobs = JobManager(Workspace(tmp_path / "ws"))
    counted = []
    jobs.on_submit = lambda user, title: counted.append((user, title))
    assert jobs.submit("import", "Import a.ply", {})["user"] is None     # outside a request: nobody
    token = auth.current_user.set("alice")
    try:
        assert jobs.submit("import", "Import b.ply", {})["user"] == "alice"
        jobs.on_submit = lambda user, title: 1 / 0                        # counting must never stop a job
        job = jobs.submit("import", "Import c.ply", {})
    finally:
        auth.current_user.reset(token)
    assert counted == [("alice", "Import b.ply")]
    assert job["user"] == "alice" and job["id"] in {j["id"] for j in jobs.list()}
    assert auth.current_user.get() is None


# --------------------------------------------------------------------------- live capture websocket
def test_the_live_capture_stream_needs_a_sign_in(app, client):
    assert refused_code(client) == 4401
    assert refused_code(client, headers={KEY: "wrong"}) == 4401
    register(client, "alice")
    with client.websocket_connect("/api/capture/stream") as ws:
        assert ws.receive_json()["type"] == "status"
    key = client.get("/api/auth/machine-key").json()["key"]
    with device(app).websocket_connect("/api/capture/stream", headers={KEY: key}) as ws:
        assert ws.receive_json()["type"] == "status"
    client.post("/api/auth/logout")
    assert refused_code(client) == 4401


# --------------------------------------------------------------------------- sessions and cookies
def test_sessions_last_30_days_on_the_server_since_last_use(app, client):
    assert auth.SESSION_DAYS == 30                                        # what docs/accounts.md promises
    register(client, "alice")
    path = app.state.auth.dir / "sessions.json"
    (key, session), = json.loads(path.read_text(encoding="utf-8")).items()
    assert session["expires"] - session["created"] == pytest.approx(30 * 86400)

    session["last_seen"] -= 3600                                          # last used an hour ago ...
    session["expires"] -= 3600
    path.write_text(json.dumps({key: session}), encoding="utf-8")
    assert client.get("/api/assets").status_code == 200                   # ... used again: 30 days from now
    session = json.loads(path.read_text(encoding="utf-8"))[key]
    assert session["expires"] == pytest.approx(time.time() + 30 * 86400, abs=120)

    session["expires"] = time.time() - 1                                  # not used for 30 days
    path.write_text(json.dumps({key: session}), encoding="utf-8")
    assert client.get("/api/assets").status_code == 401
    assert client.get("/api/auth/status").json()["signed_in"] is False


def test_the_cookie_is_secure_over_https(app):
    browser = device(app, base_url="https://testserver")
    r = browser.post("/api/auth/register", json={"username": "alice", "password": PASSWORD})
    assert r.status_code == 200 and "secure" in r.headers["set-cookie"].lower()
    assert browser.get("/api/assets").status_code == 200


# --------------------------------------------------------------------------- accounts switched off
def test_without_accounts_nothing_changes(tmp_path):
    app = create_app(tmp_path / "ws")                                     # the default; cloudclean serve --no-accounts
    assert app.state.auth is None and app.state.jobs.on_submit is None
    scan = tmp_path / "tiny.ply"
    scan.write_bytes(TINY_PLY)
    with TestClient(app) as c:
        assert c.get("/api/auth/status").json() == {"enabled": False, "signed_in": True, "user": None,
                                                    "users_exist": True}
        assert c.get("/api/assets").status_code == 200 and c.get("/api/projects").status_code == 200
        for path in ("/api/auth/register", "/api/auth/login"):
            r = c.post(path, json={"username": "alice", "password": PASSWORD})
            assert r.status_code == 404 and "switched off" in detail(r), path
        assert c.get("/api/auth/users").status_code == 404
        assert c.get("/api/auth/machine-key").status_code == 404
        assert c.post("/api/import-paths", json={"paths": [str(scan)]}).json()["jobs"][0]["user"] is None
        with c.websocket_connect("/api/capture/stream") as ws:
            assert ws.receive_json()["type"] == "status"
    assert not (tmp_path / "ws" / "auth").exists()


def test_serve_turns_accounts_on_unless_told_not_to(monkeypatch):
    from cloudclean import cli
    from cloudclean.web import server

    calls = []
    monkeypatch.setattr(server, "serve", lambda **kw: calls.append(kw))
    cli.main(["serve", "--no-browser"])
    cli.main(["serve", "--no-browser", "--no-accounts"])
    assert [c["accounts"] for c in calls] == [True, False]


# --------------------------------------------------------------------------- the bridge on the scanning PC
def test_the_bridge_sends_the_key(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "exports"
    folder.mkdir()
    (folder / "scan.ply").write_bytes(TINY_PLY)
    seen: list[httpx.Request] = []

    def server(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.headers.get(KEY) != "s3cret-key":
            return httpx.Response(401, json={"detail": "Sign in first"})
        if request.url.path.endswith("/ping"):
            return httpx.Response(200, json={"ok": True, "server": "CloudClean", "extensions": [".ply"]})
        return httpx.Response(200, json={"results": [{"file": "scan.ply", "queued": True, "message": "added"}]})

    def run(key, state):
        return bridge.run("http://spark:8765", folder, once=True, settle=0, state_path=tmp_path / state,
                          transport=httpx.MockTransport(server), key=key)

    assert run(None, "a.json") == 3                                       # no key: stops with a hint
    assert "--key" in capsys.readouterr().out
    assert len(seen) == 1 and KEY.lower() not in seen[0].headers         # only the ping, nothing uploaded
    assert run("wrong", "a.json") == 3
    seen.clear()
    assert run("s3cret-key", "a.json") == 0
    assert [r.url.path for r in seen] == ["/api/capture/bridge/ping", "/api/capture/bridge/upload"]
    assert all(r.headers[KEY] == "s3cret-key" for r in seen)             # every request carries the key

    monkeypatch.setenv("CLOUDCLEAN_KEY", "s3cret-key")                    # the key from the environment ...
    seen.clear()
    args = ["--server", "http://spark:8765", "--watch", str(folder), "--once", "--settle", "0"]
    assert bridge.main(args + ["--state", str(tmp_path / "b.json")], transport=httpx.MockTransport(server)) == 0
    assert seen and all(r.headers[KEY] == "s3cret-key" for r in seen)
    assert bridge.main(args + ["--state", str(tmp_path / "c.json"), "--key", "other"],     # ... --key wins
                       transport=httpx.MockTransport(server)) == 3


def test_the_bridge_against_a_server_with_accounts(app, client, tmp_path, capsys):
    folder = tmp_path / "exports"
    folder.mkdir()
    args = ["--server", "http://testserver", "--watch", str(folder), "--once", "--settle", "0",
            "--state", str(tmp_path / "state.json")]
    assert bridge.main(args, transport=client._transport) == 3
    register(client, "alice")
    key = client.get("/api/auth/machine-key").json()["key"]
    capsys.readouterr()
    assert bridge.main(args + ["--key", key], transport=client._transport) == 0
    assert "Connected to http://testserver" in capsys.readouterr().out


# --------------------------------------------------------------------------- what is kept on disk
def test_what_is_kept_on_disk(app, client, tmp_path):
    password = "correct horse battery"
    register(client, "alice", password)
    token = client.cookies.get(auth.COOKIE)
    folder = tmp_path / "ws" / "auth"
    assert app.state.auth.dir == folder.resolve()
    assert {p.name for p in folder.iterdir()} == {"users.json", "users.json.bak", "sessions.json", "activity.jsonl",
                                                  "machine_key"}

    pw = json.loads((folder / "users.json").read_text(encoding="utf-8"))["alice"]["pw"]
    assert set(pw) == {"salt", "n", "r", "p", "hash"} and (pw["n"], pw["r"], pw["p"]) == (2 ** 14, 8, 1)
    rehash = hashlib.scrypt(password.encode(), salt=base64.b64decode(pw["salt"]), n=pw["n"], r=pw["r"], p=pw["p"],
                            maxmem=64 * 1024 * 1024, dklen=32)
    assert base64.b64decode(pw["hash"]) == rehash                         # a salted scrypt hash of the password

    sessions = json.loads((folder / "sessions.json").read_text(encoding="utf-8"))
    assert list(sessions) == [hashlib.sha256(token.encode()).hexdigest()]
    stolen = device(app).get("/api/assets", headers={"Cookie": f"{auth.COOKIE}={next(iter(sessions))}"})
    assert stolen.status_code == 401                                      # a copy of the file signs nobody in

    for path in (tmp_path / "ws").rglob("*"):                            # nowhere in the workspace, in plain text
        if path.is_file():
            data = path.read_bytes()
            assert password.encode() not in data and token.encode() not in data, path
    if os.name == "posix":                                               # the Spark: only the service user reads them
        for name in ("users.json", "users.json.bak", "sessions.json", "activity.jsonl", "machine_key"):
            assert stat.S_IMODE((folder / name).stat().st_mode) == 0o600, name


def test_a_damaged_users_file_is_refused_not_overwritten(app, client):
    """A typo from a hand edit must not read as "no accounts": the next visitor would become the admin and every
    account would be overwritten. Nobody signs in until it is fixed; users.json.bak holds the version before."""
    register(client, "alice")
    users = app.state.auth.dir / "users.json"
    good = users.read_text(encoding="utf-8")
    broken = good.replace("{", "{,", 1)                                   # a typo by hand
    users.write_text(broken, encoding="utf-8")
    visitor = device(app)
    r = visitor.post("/api/auth/register", json={"username": "mallory", "password": PASSWORD})
    assert r.status_code == 503 and "cannot read its list of accounts" in detail(r)
    assert users.read_text(encoding="utf-8") == broken                    # nothing overwritten
    status = visitor.get("/api/auth/status")
    assert status.status_code == 503 and "users.json.bak" in detail(status)   # the sign-in page says so
    assert login(visitor, "alice").status_code == 503
    assert client.get("/api/assets").status_code == 503                   # signed-in browsers are told too
    assert "alice" in (users.parent / "users.json.bak").read_text(encoding="utf-8")   # the way back
    users.write_text(good, encoding="utf-8")                              # fixed
    assert client.get("/api/assets").status_code == 200
    assert login(visitor, "alice").status_code == 200


def test_wrong_passwords_sent_at_once_cannot_beat_the_pause(app, client):
    """Each try counts before the slow password check, so 24 tries at the same moment still get only 8 checks."""
    register(client, "alice")
    store, results = app.state.auth, []

    def guess():
        try:
            store.verify("alice", "guess guess", "10.0.0.9")
        except auth.AuthError as exc:
            results.append(exc.status)

    threads = [threading.Thread(target=guess) for _ in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [401] * auth.FAILS_ALLOWED + [429] * (24 - auth.FAILS_ALLOWED)


def test_reading_and_writing_at_the_same_time(app, client):
    """On Windows a file cannot be replaced while another thread reads it: every read and write takes the lock."""
    register(client, "alice")
    store, errors, stop = app.state.auth, [], time.time() + 1.5

    def keep(work):
        while time.time() < stop:
            try:
                work()
            except Exception as exc:  # noqa: BLE001 - any error fails the test
                errors.append(repr(exc))

    def read():
        store.users_exist(), store.list(), store.activity(), store.key_ok("not the key")

    threads = [threading.Thread(target=keep, args=(w,))
               for w in (read, read, lambda: store.note_job("alice", "a job"), lambda: store.new_machine_key("alice"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]


def test_the_browser_keeps_the_cookie_as_long_as_the_server(app, client):
    """When the server pushes a session's end back (on use, at most once a minute) it sends the cookie again with
    30 days to go: someone who uses CloudClean every day stays signed in."""
    register(client, "alice")
    token = client.cookies.get(auth.COOKIE)
    path = app.state.auth.dir / "sessions.json"
    assert "set-cookie" not in client.get("/api/assets").headers          # used a moment ago: nothing to renew
    for url in ("/api/assets", "/api/auth/status"):
        (key, session), = json.loads(path.read_text(encoding="utf-8")).items()
        session["last_seen"] -= 3600                                      # last used an hour ago
        path.write_text(json.dumps({key: session}), encoding="utf-8")
        cookie = client.get(url).headers["set-cookie"]
        assert cookie.split(";")[0] == f"{auth.COOKIE}={token}", url     # the same sign-in ...
        assert f"Max-Age={30 * 86400}" in cookie and "HttpOnly" in cookie, url   # ... good for 30 more days


def test_failed_sign_ins_never_log_what_was_typed(app, client):
    """A password typed into the name box must not reach the activity list: a failed sign-in names the account only
    when it exists."""
    register(client, "alice")
    typed = "my secret pass"
    assert login(device(app), typed, "whatever pw").status_code == 401
    assert login(device(app), "ALICE", "whatever pw").status_code == 401
    failed = [e for e in client.get("/api/auth/users").json()["activity"] if e["event"] == "failed_login"]
    assert [e["user"] for e in failed] == ["alice", None]                 # newest first
    assert typed not in (app.state.auth.dir / "activity.jsonl").read_text(encoding="utf-8")


def test_the_api_docs_need_a_sign_in(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        r = client.get(path)
        assert r.status_code == 401 and detail(r) == "Sign in first", path
    register(client, "alice")
    assert client.get("/openapi.json").status_code == 200 and client.get("/docs").status_code == 200


def test_nobody_can_take_the_bridge_name(client):
    """Jobs started with the machine key are recorded as "bridge": no person may sign in under that name."""
    for name in ("bridge", "Bridge", "BRIDGE"):
        r = client.post("/api/auth/register", json={"username": name, "password": PASSWORD})
        assert r.status_code == 400 and "kept for the Revo Metro bridge" in detail(r), name


def test_a_password_change_is_not_a_sign_in(client):
    register(client, "alice")
    r = client.post("/api/auth/password", json={"current": PASSWORD, "new": "another password"})
    assert r.status_code == 200 and r.json()["user"]["logins"] == 1
    assert [e for e in events(client) if e[0] in ("login", "password")] == [("password", "alice"), ("login", "alice")]


def test_the_sign_in_check_sees_the_path_as_routed(client):
    """An encoded '#' must not make a path look like one of the open sign-in paths."""
    r = client.get("/api/auth/login%23x")
    assert r.status_code == 401 and detail(r) == "Sign in first"
