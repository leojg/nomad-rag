from sqlalchemy.orm import Session

from agent.config import AgentConfig
from agent.nodes import (
    fallback,
    make_generate_node,
    make_multi_retrieve_node,
    make_query_analysis_node,
    make_rerank_node,
    make_retrieve_node,
)
from agent.state import State
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph


def _route_after_query_analysis(state: State) -> str:
    pq = state.get("parsed_queries") or []
    if len(pq) > 1:
        return "multi_retrieve"
    return "retrieve"


def _route_after_retrieve(state: State) -> str:
    return "fallback" if not state["retrieved_chunks"] else "rerank"


def build_agent(config: AgentConfig, session: Session) -> CompiledStateGraph:
    """Compile the LangGraph retrieval agent."""

    graph = StateGraph(State)

    graph.add_node("query_analysis", make_query_analysis_node(config))
    graph.add_node("retrieve", make_retrieve_node(config, session))
    graph.add_node("multi_retrieve", make_multi_retrieve_node(config, session))
    graph.add_node("rerank", make_rerank_node(config))
    graph.add_node("generate", make_generate_node(config))
    graph.add_node("fallback", fallback)

    graph.add_edge(START, "query_analysis")
    graph.add_conditional_edges(
        "query_analysis",
        _route_after_query_analysis,
        {"multi_retrieve": "multi_retrieve", "retrieve": "retrieve"},
    )
    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"fallback": "fallback", "rerank": "rerank"},
    )
    graph.add_conditional_edges(
        "multi_retrieve",
        _route_after_retrieve,
        {"fallback": "fallback", "rerank": "rerank"},
    )
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()
