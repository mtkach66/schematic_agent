"""Тесты pydantic-контракта (ТЗ §7, §8, §10, §13)."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from schematic_agent.contract import (
    BomItem,
    Result,
    Summary,
    Task,
    TaskOptions,
    TaskPayload,
)

#: Пример задания из ТЗ §7.
VALID_TASK = {
    "task_id": "task-001",
    "task_type": "parse_electrical_schematic_pdf",
    "payload": {
        "pdf_path": "/data/input/schema.pdf",
        "document_id": "doc_001",
        "options": {
            "mode": "full",
            "use_vlm": True,
            "extract_bom": True,
            "extract_netlist": True,
            "analyze_plc_logic": True,
        },
    },
}


def test_valid_task_from_tz_parses() -> None:
    """Задание из ТЗ §7 валидно и поля совпадают."""
    task = Task.model_validate(VALID_TASK)

    assert task.task_id == "task-001"
    assert task.task_type == "parse_electrical_schematic_pdf"
    assert task.payload.pdf_path == "/data/input/schema.pdf"
    assert task.payload.document_id == "doc_001"
    assert task.payload.options == TaskOptions(
        mode="full",
        use_vlm=True,
        extract_bom=True,
        extract_netlist=True,
        analyze_plc_logic=True,
    )


def test_empty_task_id_rejected() -> None:
    """Пустой или пробельный task_id отклоняется."""
    for bad_id in ("", "   "):
        data = copy.deepcopy(VALID_TASK)
        data["task_id"] = bad_id
        with pytest.raises(ValidationError):
            Task.model_validate(data)


def test_bom_confidence_out_of_range_rejected() -> None:
    """confidence > 1 отклоняется у BomItem."""
    with pytest.raises(ValidationError):
        BomItem(
            ref_designation="R1",
            name="Резистор",
            confidence=1.5,
        )


def test_all_five_modes_accepted() -> None:
    """Все 5 режимов ТЗ §13 принимаются TaskOptions."""
    modes = ["full", "no_vlm", "bom_only", "netlist_only", "issue_review"]
    for mode in modes:
        options = TaskOptions(mode=mode)
        assert options.mode == mode

    with pytest.raises(ValidationError):
        TaskOptions(mode="unknown_mode")


def test_result_defaults_and_summary() -> None:
    """Result принимает пустые artifacts и нулевую сводку."""
    result = Result(
        task_id="task-001",
        status="completed_with_issues",
        document_id="doc_001",
        summary=Summary(),
    )
    assert result.summary.pages_processed == 0
    assert result.artifacts == {}
    assert result.error is None


def test_task_payload_requires_options_default() -> None:
    """TaskPayload создаёт опции по умолчанию."""
    payload = TaskPayload(pdf_path="/a.pdf", document_id="doc")
    assert payload.options.mode == "full"
