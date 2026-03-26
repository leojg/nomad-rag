from sqlalchemy.orm import Session

from chain.nodes import make_retrieve_node, make_rerank_node, make_generate_node, fallback
from chain.config import GraphConfig
from chain.state import State
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph


def _is_retrieval_empty(state: State) -> bool:
    """Check if the retrieval is empty."""
    return "fallback" if not state["retrieved_chunks"] else "rerank"


def make_graph(config: GraphConfig, session: Session) -> CompiledStateGraph:
    """Make the graph."""

    graph = StateGraph(State)

    graph.add_node("retrieve", make_retrieve_node(config, session))
    graph.add_node("rerank", make_rerank_node(config))
    graph.add_node("generate", make_generate_node(config))
    graph.add_node("fallback", fallback)

    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", _is_retrieval_empty)
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()