"""Публичный API субагента анализа принципиальных схем из PDF.

Наружу доступны только функции :func:`run`, :func:`submit`,
:func:`get_result`, :func:`health_check`, :func:`describe_capabilities`
и pydantic-схемы контракта.
"""

from __future__ import annotations

from . import contract
from .contract import (
    Artifacts,
    BomItem,
    Component,
    Issue,
    Net,
    Node,
    Pin,
    PlcAnalysis,
    PlcLogicBlock,
    Result,
    Summary,
    Task,
    TaskOptions,
    TaskPayload,
)
from .pipeline import (
    describe_capabilities,
    get_result,
    health_check,
    run,
    submit,
)

__all__ = [
    # API
    "run",
    "submit",
    "get_result",
    "health_check",
    "describe_capabilities",
    # Схемы контракта
    "contract",
    "Task",
    "TaskOptions",
    "TaskPayload",
    "Summary",
    "Artifacts",
    "Result",
    "BomItem",
    "Component",
    "Net",
    "Node",
    "Pin",
    "Issue",
    "PlcAnalysis",
    "PlcLogicBlock",
]
