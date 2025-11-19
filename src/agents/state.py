"""
State management for LangGraph agents.

This module defines the state structure and helper functions for managing
agent state throughout the workflow.
"""

from typing import Dict, Any, TypedDict, List


class AgentState(TypedDict, total=False):
    """State structure for LangGraph agents with tool calling support."""
    data: Any
    current_step: str
    analysis_results: Dict[str, Any]
    errors: List[str]
    sentiment_analysis: Dict[str, Any]
    raw_sentiments: List[Any]
    topic_extraction: Dict[str, Any]


def create_initial_state() -> AgentState:
    """Create an initial state for the agent workflow."""
    return {
        "data": None,
        "analysis_results": {},
        "current_step": "starting",
        "errors": [],
        "sentiment_analysis": {},
        "raw_sentiments": [],
        "topic_extraction": {}
    }

def update_state_with_error(state: AgentState, error: str, step: str = None) -> AgentState:
    """Update state with error information."""
    state["errors"] = state.get("errors", []) + [error]
    if step:
        state["current_step"] = f"error_in_{step}"
    return state


def is_state_valid(state: AgentState) -> bool:
    """Check if the state is valid for processing."""
    return len(state.get("errors", [])) == 0 and state.get("data") is not None
