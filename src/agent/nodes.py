from __future__ import annotations

import json
import re
from typing import Any, Callable

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.orm import Session

from agent.config import AgentConfig
from agent.prompts import (
    FALLBACK_RESPONSE,
    GENERATE_SYSTEM_PROMPT,
    GENERATE_USER_TEMPLATE,
    QUERY_ANALYSIS_SYSTEM_PROMPT,
    QUERY_ANALYSIS_USER_TEMPLATE,
    RERANK_SYSTEM_PROMPT,
    RERANK_USER_TEMPLATE,
    format_chunks_for_generate,
    format_chunks_for_rerank,
)
from agent.state import QueryIntent, State
from retrieval.hybrid import hybrid_search, reciprocal_rank_fusion


def make_query_analysis_node(config: AgentConfig) -> Callable[[State], dict[str, Any]]:
    llm = ChatAnthropic(model=config.analysis_model, temperature=config.temperature)

    def query_analysis(state: State) -> dict[str, Any]:
        response = llm.invoke(
            [
                SystemMessage(content=QUERY_ANALYSIS_SYSTEM_PROMPT),
                HumanMessage(content=QUERY_ANALYSIS_USER_TEMPLATE.format(query=state["query"])),
            ]
        )
        raw_text = response.content

        intents: list[QueryIntent] = []
        try:
            raw_text = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
            raw = json.loads(raw_text)
            if not isinstance(raw, list):
                raw = []
            for item in raw:
                q = item.get("query", "").strip()
                if not q:
                    continue
                filt = item.get("filters") or None
                intents.append({"query": q, "filters": filt})
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        if not intents:
            intents = [{"query": state["query"], "filters": state["filters"]}]

        return {"parsed_queries": intents}

    return query_analysis


def make_retrieve_node(config: AgentConfig, session: Session) -> Callable[[State], dict[str, Any]]:
    embeddings = OpenAIEmbeddings(model=config.embeddings_model)
    k = config.retrieve_k

    def retrieve(state: State) -> dict[str, Any]:
        pq = state.get("parsed_queries") or []
        intent = pq[0] if pq else None
        query = intent["query"] if intent else state["query"]
        filters = intent["filters"] if intent else state["filters"]
        chunks = hybrid_search(
            query=query,
            k=k,
            embeddings=embeddings,
            session=session,
            filters=filters,
        )
        return {"retrieved_chunks": chunks}

    return retrieve


def make_multi_retrieve_node(config: AgentConfig, session: Session) -> Callable[[State], dict[str, Any]]:
    embeddings = OpenAIEmbeddings(model=config.embeddings_model)
    k = config.retrieve_k

    def multi_retrieve(state: State) -> dict[str, Any]:
        intents = state["parsed_queries"] or []
        ranked_chunks = []

        for intent in intents:
            query = intent["query"]
            filters = intent["filters"]
            chunks = hybrid_search(
                query=query,
                k=k,
                embeddings=embeddings,
                session=session,
                filters=filters,
            )
            ranked_chunks.append(chunks)

        merged = reciprocal_rank_fusion(ranked_chunks)
        return {"retrieved_chunks": merged[: config.retrieve_k]}

    return multi_retrieve


def make_rerank_node(config: AgentConfig) -> Callable[[State], dict[str, Any]]:
    llm = ChatAnthropic(model=config.rerank_model, temperature=config.temperature)

    def _parse_rerank_response(content: str) -> list:
        content = re.sub(r"```(?:json)?\s*", "", content).strip()
        return json.loads(content)

    def rerank(state: State) -> dict[str, Any]:
        response = llm.invoke(
            [
                SystemMessage(content=RERANK_SYSTEM_PROMPT),
                HumanMessage(
                    content=RERANK_USER_TEMPLATE.format(
                        query=state["query"],
                        chunks=format_chunks_for_rerank(state["retrieved_chunks"]),
                    ),
                ),
            ]
        )

        scores = _parse_rerank_response(response.content)
        scores.sort(key=lambda x: x["score"], reverse=True)

        chunk_index = {chunk.id: chunk for chunk in state["retrieved_chunks"]}
        reranked = [
            chunk_index[item["chunk_id"]]
            for item in scores
            if item.get("chunk_id") in chunk_index
        ]

        return {"reranked_chunks": reranked[: config.rerank_k]}

    return rerank


def make_generate_node(config: AgentConfig) -> Callable[[State], dict[str, Any]]:
    llm = ChatAnthropic(model=config.generate_model, temperature=config.temperature)

    def generate(state: State) -> dict[str, Any]:
        response = llm.invoke(
            [
                SystemMessage(content=GENERATE_SYSTEM_PROMPT),
                HumanMessage(
                    content=GENERATE_USER_TEMPLATE.format(
                        chunks=format_chunks_for_generate(state["reranked_chunks"]),
                        query=state["query"],
                    ),
                ),
            ]
        )

        return {"response": response.content}

    return generate


def fallback(state: State) -> dict[str, Any]:
    return {"response": FALLBACK_RESPONSE}
