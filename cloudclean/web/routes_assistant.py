"""API routes of the chat assistant (local OpenAI-compatible LLM that operates CloudClean)."""
from __future__ import annotations

import json

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..assistant.agent import ID_RE, Assistant, clean_images, public_settings
from .settings import Settings


class ChatReq(BaseModel):
    conversation_id: str | None = None
    message: str = ""
    context: dict = {}
    image: str | None = None            # kept for older clients: one viewer screenshot
    images: list[str] | None = None     # data:image/(png|jpeg|webp);base64,... - photos, screenshots, annotations
    image_notes: list[str] | None = None  # optional caption per image, in order, may be shorter than `images`


class StopReq(BaseModel):
    conversation_id: str


def sse(event: dict) -> str:
    if event["event"] == "ping":
        return ": ping\n\n"
    return f"event: {event['event']}\ndata: {json.dumps(event.get('data', {}), default=str, ensure_ascii=False)}\n\n"


def create_router(workspace, jobs, transport=None) -> APIRouter:
    router = APIRouter(prefix="/api/assistant")
    assistant = Assistant(workspace, jobs, Settings(workspace.root), transport=transport)
    router.assistant = assistant  # handy for tests / other features

    @router.get("/settings")
    def get_settings():
        return public_settings(assistant.settings())

    @router.put("/settings")
    def put_settings(values: dict = Body(...)):
        try:
            return assistant.update_settings(values)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @router.get("/status")
    async def status():
        return await assistant.status()

    @router.get("/busy")
    def busy():
        """How many conversations are answering right now: deploys wait until this is 0."""
        return {"active_turns": assistant.active_count()}

    @router.post("/chat")
    async def chat(req: ChatReq):
        urls = list(req.images or [])
        notes = list(req.image_notes or [])
        if req.image:  # the old single-image field comes first in the merged list
            urls.insert(0, req.image)
            notes = notes or ["viewer screenshot"]
        try:
            images = clean_images(urls, notes)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if not req.message.strip() and not images:
            raise HTTPException(400, "Type a message first")
        if req.conversation_id and not ID_RE.match(req.conversation_id):
            raise HTTPException(400, "Invalid conversation id")
        if req.conversation_id and assistant.is_running(req.conversation_id):
            raise HTTPException(409, "This conversation is still answering - stop it first")

        async def stream():
            async for event in assistant.run(req.conversation_id, req.message, req.context, images=images):
                yield sse(event)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @router.post("/stop")
    def stop(req: StopReq):
        return {"stopped": assistant.stop(req.conversation_id)}

    @router.get("/conversations")
    def conversations():
        return assistant.store.list()

    @router.get("/conversations/{conversation_id}")
    def get_conversation(conversation_id: str):
        try:
            conv = assistant.store.get(conversation_id)
        except KeyError:
            raise HTTPException(404, "Conversation not found")
        return {**conv, "running": assistant.is_running(conversation_id)}

    @router.delete("/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str):
        if assistant.is_running(conversation_id):
            assistant.stop(conversation_id)
        try:
            assistant.store.delete(conversation_id)
        except KeyError:
            raise HTTPException(404, "Conversation not found")
        return {"deleted": conversation_id}

    return router
