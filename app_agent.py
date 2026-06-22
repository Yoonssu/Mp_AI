"""Backward-compatible import for older run commands.

New code lives in app.agent.graph.
"""

from app.agent.graph import app, build_graph

__all__ = ["app", "build_graph"]
