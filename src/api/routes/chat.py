"""Chat endpoint — POST /chat."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from services.chat import ChatResponse, chat_service

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    filters: dict[str, str] | None = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request) -> ChatResponse:
    return chat_service(
        query=request.query,
        filters=request.filters,
        graph=req.app.state.graph,
    )