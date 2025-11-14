"""
State management for LangGraph agents.

This module defines the state structure and helper functions for managing
agent state throughout the workflow.
"""

from typing import Dict, Any, Optional, TypedDict, List
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State structure for LangGraph agents with tool calling support."""
    data: Optional[Any]
    analysis_results: Dict[str, Any]
    current_step: str
    messages: List[BaseMessage]
    tool_calls_made: List[str]
    error: Optional[str]


def create_initial_state() -> AgentState:
    """Create an initial state for the agent workflow."""
    return AgentState(
        data=None,
        analysis_results={},
        current_step="starting",
        messages=[],
        tool_calls_made=[],
        error=None
    )


def update_state_with_error(state: AgentState, error: str, step: str = None) -> AgentState:
    """Update state with error information."""
    state["error"] = error
    if step:
        state["current_step"] = f"error_in_{step}"
    return state


def is_state_valid(state: AgentState) -> bool:
    """Check if the state is valid for processing."""
    return state.get("error") is None and state.get("data") is not None
