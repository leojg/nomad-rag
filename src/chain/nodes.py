from __future__ import annotations

import json
from typing import Any, Callable

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.orm import Session

from chain.config import GraphConfig
from chain.prompts import (
    FALLBACK_RESPONSE,
    GENERATE_SYSTEM_PROMPT,
    GENERATE_USER_TEMPLATE,
    RERANK_SYSTEM_PROMPT,
    RERANK_USER_TEMPLATE,
    format_chunks_for_generate,
    format_chunks_for_rerank,
)
from chain.state import State
from retrieval.hybrid import hybrid_search
import json
from langchain_core.messages import SystemMessage, HumanMessage

def make_retrieve_node(config: GraphConfig, session: Session) -> Callable[[State], dict[str, Any]]:
    embeddings = OpenAIEmbeddings(model=config.embeddings_model)
    k = config.retrieve_k

    def retrieve(state: State) -> dict[str, Any]:
        chunks = hybrid_search(
            query=state["query"],
            k=k,
            embeddings=embeddings,
            session=session,
            filters=state["filters"],
        )
        return {"retrieved_chunks": chunks}

    return retrieve


def make_rerank_node(config: GraphConfig) -> Callable[[State], dict[str, Any]]:

    llm = ChatAnthropic(model=config.model, temperature=config.temperature)

    def rerank(state: State) -> dict[str, Any]:
        response = llm.invoke(
            [
                SystemMessage(content=RERANK_SYSTEM_PROMPT), 
                HumanMessage(
                    content=RERANK_USER_TEMPLATE.format(
                        query=state["query"], 
                        chunks=format_chunks_for_rerank(state["retrieved_chunks"])
                        )
                    ),
            ]
        )

        # Parse scores and sort
        scores = json.loads(response.content)
        scores.sort(key=lambda x: x["score"], reverse=True)

        # Map chunk_id to ChunkRecord
        chunk_index = {chunk.id: chunk for chunk in state["retrieved_chunks"]}
        reranked = [
            chunk_index[item["chunk_id"]] for item in scores if item["chunk_id"] in chunk_index
        ]

        return {"reranked_chunks": reranked[:config.rerank_k]}

    return rerank


def make_generate_node(config: GraphConfig) -> Callable[[State], dict[str, Any]]:
    llm = ChatAnthropic(model=config.model, temperature=config.temperature)

    def generate(state: State) -> dict[str, Any]:
        response = llm.invoke(
            [
                SystemMessage(content=GENERATE_SYSTEM_PROMPT),
                HumanMessage(content=GENERATE_USER_TEMPLATE.format(
                    chunks=format_chunks_for_generate(state["reranked_chunks"]), 
                    query=state["query"]
                    )
                ),
            ]
        )

        return {"response": response.content}

    return generate


def fallback(state: State) -> dict[str, Any]:
    return {"response": FALLBACK_RESPONSE}