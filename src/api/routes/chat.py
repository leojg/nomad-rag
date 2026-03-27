"""Chat endpoint — POST /chat."""

from __future__ import annotations

from fastapi import APIRouter, Request

from models.chat import ChatRequest, ChatResponse
from services.chat import chat_service

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request) -> ChatResponse:
    return chat_service(
        query=request.query,
        filters=request.filters,
        graph=req.app.state.graph,
    )
