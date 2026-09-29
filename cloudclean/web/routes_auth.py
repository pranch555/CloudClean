"""Accounts (cloudclean.web.auth, docs/accounts.md).

GET    /api/auth/status              {enabled, signed_in, user, users_exist}  - open to everyone
POST   /api/auth/register            {username, password}  -> signed in (the first account becomes the admin)
POST   /api/auth/login               {username, password}  -> signed in
POST   /api/auth/logout
GET    /api/auth/me                  the signed-in account
POST   /api/auth/password            {current, new}  (other devices are signed out)
GET    /api/auth/users               admin: every account (created, last seen, sign-ins, jobs) + recent activity
DELETE /api/auth/users/{username}    admin
GET    /api/auth/machine-key         admin: the key for the Revo Metro bridge and scripts
POST   /api/auth/machine-key         admin: a new key (the old one stops working)
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from .auth import COOKIE, AuthError, AuthStore, set_session_cookie


class Creds(BaseModel):
    username: str
    password: str


class PasswordReq(BaseModel):
    current: str
    new: str


def _address(request: Request) -> str:
    return request.client.host if request.client else ""


def create_router(workspace, jobs) -> APIRouter:
    router = APIRouter()

    def store(request: Request) -> AuthStore:
        s = request.app.state.auth
        if s is None:
            raise HTTPException(404, "Accounts are switched off on this server")
        return s

    def signed_in(request: Request) -> dict:
        user = getattr(request.state, "user", None)
        if not user or user.get("machine"):
            raise HTTPException(401, "Sign in first")
        return user

    def admin(request: Request) -> dict:
        user = signed_in(request)
        if not user.get("admin"):
            raise HTTPException(403, "Only the admin can do that")
        return user

    def start(request: Request, response: Response, s: AuthStore, username: str, sign_in: bool = True) -> dict:
        token = s.start_session(username, request.headers.get("user-agent", ""), _address(request), sign_in)
        set_session_cookie(response, token, request.url.scheme == "https")
        return s.session_user(token) or {}

    @router.get("/api/auth/status")
    def status(request: Request, response: Response):
        s = request.app.state.auth
        if s is None:
            return {"enabled": False, "signed_in": True, "user": None, "users_exist": True}
        token = request.cookies.get(COOKIE)
        try:
            user, renewed = s.session(token)
            users_exist = s.users_exist()
        except AuthError as exc:   # the accounts file is damaged: the sign-in page says so
            raise HTTPException(exc.status, str(exc))
        if renewed:
            set_session_cookie(response, token, request.url.scheme == "https")
        return {"enabled": True, "signed_in": user is not None, "user": user, "users_exist": users_exist}

    @router.post("/api/auth/register")
    def register(req: Creds, request: Request, response: Response):
        s = store(request)
        try:
            user = s.register(req.username, req.password)
        except AuthError as exc:
            raise HTTPException(exc.status, str(exc))
        return {"user": start(request, response, s, user["username"])}

    @router.post("/api/auth/login")
    def login(req: Creds, request: Request, response: Response):
        s = store(request)
        try:
            user = s.verify(req.username, req.password, _address(request))
        except AuthError as exc:
            raise HTTPException(exc.status, str(exc))
        return {"user": start(request, response, s, user["username"])}

    @router.post("/api/auth/logout")
    def logout(request: Request, response: Response):
        s = request.app.state.auth
        if s is not None:
            s.end_session(request.cookies.get(COOKIE))
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @router.get("/api/auth/me")
    def me(request: Request):
        return {"user": signed_in(request)}

    @router.post("/api/auth/password")
    def password(req: PasswordReq, request: Request, response: Response):
        s, user = store(request), signed_in(request)
        try:
            s.change_password(user["username"], req.current, req.new)
        except AuthError as exc:
            raise HTTPException(400 if exc.status == 401 else exc.status,
                                "Your current password is not right" if exc.status == 401 else str(exc))
        return {"user": start(request, response, s, user["username"], sign_in=False)}   # this device stays signed in

    @router.get("/api/auth/users")
    def users(request: Request):
        s = store(request)
        admin(request)
        return {"users": s.list(), "activity": s.activity(150)}

    @router.delete("/api/auth/users/{username}")
    def remove(username: str, request: Request):
        s, user = store(request), admin(request)
        try:
            s.remove(username, user["username"])
        except AuthError as exc:
            raise HTTPException(exc.status, str(exc))
        return {"ok": True}

    @router.get("/api/auth/machine-key")
    def machine_key(request: Request):
        s = store(request)
        admin(request)
        return {"key": s.machine_key()}

    @router.post("/api/auth/machine-key")
    def new_machine_key(request: Request):
        s, user = store(request), admin(request)
        return {"key": s.new_machine_key(user["username"])}

    return router
