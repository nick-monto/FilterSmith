from __future__ import annotations

"""
Compatibility layer for Microsoft Agent Framework integration.

The framework is evolving; import paths can vary by version. This module keeps
project code stable and offers a soft-fail path so local development can proceed
with direct Python orchestration.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass
class MafStatus:
    available: bool
    detail: str


def get_maf_status() -> MafStatus:
    """Return whether Microsoft Agent Framework appears importable."""
    candidates = [
        "microsoft.agent_framework",
        "agent_framework",
        "microsoft_agent_framework",
    ]
    for name in candidates:
        try:
            import_module(name)
            return MafStatus(available=True, detail=f"Imported {name}")
        except Exception:
            continue
    return MafStatus(
        available=False,
        detail="Framework not importable with known module names. Coordinator will use local direct orchestration.",
    )


def build_maf_graph_or_none() -> Any | None:
    """Placeholder for wiring real MAF agent graph once API surface is finalized."""
    status = get_maf_status()
    if not status.available:
        return None
    return None
