"""Тесты этапа 9: состав и счётчики отчёта report.md."""

from __future__ import annotations

import json
from pathlib import Path

from schematic_agent.contract import (
    BomItem,
    Component,
    Issue,
    Net,
    PlcAnalysis,
    PlcLogicBlock,
    Result,
    Summary,
)
from schematic_agent.report import build_report


def _result() -> Result:
    return Result(
        task_id="task-report",
        status="completed_with_issues",
        document_id="doc_report",
        summary=Summary(
            pages_processed=2,
            components_found=2,
            nets_found=1,
            bom_lines_found=2,
            issues_found=2,
            vlm_calls=2,
        ),
    )


def test_report_contains_counters_and_sections(tmp_path: Path) -> None:
    """report.md содержит счётчики, разделы, BOM-сверку и VLM-сводку."""
    log = tmp_path / "vlm_calls.log"
    log.write_text(
        "\n".join(
            [
                json.dumps({"status": "ok", "trigger": "unrecognized_symbol"}),
                json.dumps({"status": "bad_json", "trigger": "ambiguous_crossing"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    details = {
        "components": [
            Component(
                id="c001", type="resistor", ref_designation="R1",
                x=0, y=0, confidence=0.9,
            ),
            Component(
                id="c002", type="relay", ref_designation="K1",
                x=0, y=0, confidence=0.7,
            ),
        ],
        "bom": [
            BomItem(
                ref_designation="R1", name="Резистор", quantity=1,
                found_on_schematic=True, category="resistor",
            ),
            BomItem(
                ref_designation="K1", name="Реле", quantity=1,
                found_on_schematic=False, category="relay",
            ),
        ],
        "nets": [
            Net(id="net001", name="L1", nodes=[1], component_pins=[1],
                text_labels=["L1"], pages=[1]),
        ],
        "plc": PlcAnalysis(
            inputs=["SB1"],
            outputs=["KM1"],
            coils=["KM1"],
            contacts=["KA1"],
            interlocks=["SB1"],
            logic_blocks=[
                PlcLogicBlock(
                    kind="AND", elements=["SB1", "KA1"],
                    description="Пуск", confidence=0.8,
                )
            ],
        ),
        "issues": [
            Issue(type="unknown", description="x", confidence=0.2),
            Issue(
                type="vlm_bad_json", description="bad", confidence=0.4,
                required_action="retry without VLM",
            ),
        ],
        "vlm_log": str(log),
    }

    content = build_report(_result(), details)

    assert "| Страниц обработано | 2 |" in content
    assert "| Вызовов VLM | 2 |" in content
    assert "## Компоненты по типам" in content
    assert "## Перечень элементов (BOM)" in content
    assert "Сверка BOM: найдено на схеме 1 из 2, не найдено 1." in content
    assert "## Цепи (Netlist)" in content
    assert "## Анализ ПЛК" in content
    assert "| AND |" in content
    assert "## Проблемы" in content
    assert "## VLM" in content
    assert "ok=1" in content and "bad_json=1" in content
    assert "Деградаций VLM: 1" in content
