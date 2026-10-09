"""TEMPORARY local-only chatbot test harness. Delete with app/services/harness_*.py.

Registered only when INTERNAL_DEV_HARNESS is truthy and APP_ENV=local. Every request must
come from a loopback peer and carry a staff admin token. Drives the real bot pipeline.
"""

from __future__ import annotations

import os
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from app.db import SessionDep
from app.security.deps import CurrentAdmin
from app.services import harness_chat
from app.services.harness_chat import HarnessError
from app.services.harness_store import get_store
from app.services.harness_telemetry import install_llm_call_recorder
from app.settings import AppEnvironment, get_settings

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def harness_enabled() -> bool:
    flag = os.environ.get("INTERNAL_DEV_HARNESS", "").strip().casefold()
    return flag in {"1", "true", "yes", "on"} and get_settings().app_env is AppEnvironment.LOCAL


def require_local(request: Request) -> None:
    host = request.client.host if request.client else ""
    if not harness_enabled() or host not in LOOPBACK or request.headers.get("x-forwarded-for"):
        raise HTTPException(404, "not_found")


router = APIRouter(
    prefix="/api/internal/dev/harness",
    dependencies=[Depends(require_local)],
    include_in_schema=False,
)


class NewConversation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    site_key: str = Field(min_length=1, max_length=120)
    visitor_name: str | None = Field(default=None, max_length=120)
    visitor_email: str | None = Field(default=None, max_length=254)


class SendMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4000)


def _fail(exc: HarnessError) -> HTTPException:
    return HTTPException(exc.status_code, exc.code)


@router.post("/conversations")
async def new_conversation(
    payload: NewConversation, response: Response, session: SessionDep, _admin: CurrentAdmin
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await harness_chat.create_conversation(
            session, payload.site_key, payload.visitor_name, payload.visitor_email
        )
    except HarnessError as exc:
        raise _fail(exc) from None


@router.post("/conversations/{conversation_id}/messages")
async def say(
    conversation_id: UUID,
    payload: SendMessage,
    response: Response,
    session: SessionDep,
    _admin: CurrentAdmin,
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await harness_chat.send_message(session, conversation_id, payload.text)
    except HarnessError as exc:
        raise _fail(exc) from None


@router.get("/conversations/{conversation_id}")
async def show(
    conversation_id: UUID, response: Response, session: SessionDep, _admin: CurrentAdmin
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await harness_chat.transcript(session, conversation_id)
    except HarnessError as exc:
        raise _fail(exc) from None


@router.get("/stats")
async def stats(_admin: CurrentAdmin) -> dict[str, Any]:
    return get_store().stats()


@router.get("/sites")
async def sites(session: SessionDep, _admin: CurrentAdmin) -> list[dict[str, Any]]:
    return await harness_chat.list_sites(session)


def prepare() -> APIRouter:
    install_llm_call_recorder()
    return router
