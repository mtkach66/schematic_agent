"""Интеграционные тесты этапов 2–4 в ``run()``."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from schematic_agent import run
from schematic_agent.config import OUTPUT_DIR_ENV
from schematic_agent.contract import Task, TaskOptions, TaskPayload


def _task(pdf: Path, document_id: str = "doc_stage") -> Task:
    """Задание по контракту ТЗ §7."""
    return Task(
        task_id="task-stages",
        payload=TaskPayload(
            pdf_path=str(pdf),
            document_id=document_id,
            options=TaskOptions(),
        ),
    )


@pytest.fixture
def output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Перенаправляет артефакты во временный каталог."""
    target = tmp_path / "out"
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(target))
    return target


def test_run_fills_components_and_summary(
    tmp_path: Path, output_dir: Path
) -> None:
    """Валидный PDF с резистором → components.json, BOM-missing, summary.

    Семантика этапа 10 (В1): в режиме full с extract_bom=True отсутствие
    таблицы BOM — это Issue(bom_table_missing) и completed_with_issues.
    """
    document = pymupdf.open()
    page = document.new_page(width=200, height=200)
    page.draw_rect(pymupdf.Rect(20, 20, 60, 40))
    page.insert_text((28, 55), "R1")
    pdf = tmp_path / "schema.pdf"
    document.save(str(pdf))
    document.close()

    result = run(_task(pdf))

    doc_dir = output_dir / "doc_stage"
    assert result.status == "completed_with_issues"
    assert result.summary.pages_processed == 1
    assert result.summary.components_found >= 1
    assert result.summary.issues_found >= 1

    payload = json.loads((doc_dir / "components.json").read_text(encoding="utf-8"))
    assert payload["components"]
    component = payload["components"][0]
    assert component["type"] == "resistor"
    assert component["ref_designation"] == "R1"
    assert component["pins"]
    assert component["attributes"]["page"] == 1

    issues = json.loads((doc_dir / "issues.json").read_text(encoding="utf-8"))
    assert any(
        item["type"] == "bom_table_missing" for item in issues["issues"]
    )
    # Остальные этапы остаются заглушками.
    assert json.loads((doc_dir / "netlist.json").read_text(encoding="utf-8")) == {
        "nets": []
    }


def test_run_corrupt_pdf_reports_parse_error(
    tmp_path: Path, output_dir: Path
) -> None:
    """Повреждённый PDF → failed и parse_error в issues.json."""
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"broken content" * 30)

    result = run(_task(pdf, document_id="doc_broken"))

    doc_dir = output_dir / "doc_broken"
    assert result.status == "failed"
    assert result.error is not None and result.error.startswith("parse_error:")
    assert result.summary.pages_processed == 0
    assert result.summary.issues_found >= 1

    issues = json.loads((doc_dir / "issues.json").read_text(encoding="utf-8"))
    assert any(item["type"] == "parse_error" for item in issues["issues"])


def test_multimodal_consistency_score_in_report(
    tmp_path: Path, output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-C1: скор каналов схемы и BOM попадает в report.md.

    Сценарий с расхождением: на схеме два компонента с обозначениями
    (R1, R2), в таблице BOM — одна строка (R1). Ожидаемый скор:
    ``(1 − |2−1| / 2) × 100 = 50.00 %``.
    """
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font.is_file():
        pytest.skip("нет шрифта DejaVu")

    document = pymupdf.open()
    page = document.new_page(width=500, height=300)
    page.insert_font(fontname="DejaVu", fontfile=str(font))
    # Два резистора схемы.
    page.draw_rect(pymupdf.Rect(20, 20, 60, 40))
    page.insert_text((28, 55), "R1", fontname="DejaVu")
    page.draw_rect(pymupdf.Rect(100, 20, 140, 40))
    page.insert_text((108, 55), "R2", fontname="DejaVu")
    # Таблица BOM: только R1.
    page.draw_rect(pymupdf.Rect(40, 120, 460, 240))
    page.draw_line((140, 120), (140, 240))
    page.draw_line((300, 120), (300, 240))
    page.draw_line((40, 180), (460, 180))
    for x, y, text in [
        (50, 153, "R1"),
        (150, 153, "Резистор 10 кОм"),
        (310, 153, "1"),
    ]:
        page.insert_text((x, y), text, fontname="DejaVu")
    pdf = tmp_path / "audit.pdf"
    document.save(str(pdf))
    document.close()

    result = run(_task(pdf, document_id="doc_audit"))
    assert result.status in {"completed", "completed_with_issues"}

    doc_dir = output_dir / "doc_audit"
    components = json.loads(
        (doc_dir / "components.json").read_text(encoding="utf-8")
    )["components"]
    bom = json.loads((doc_dir / "bom.json").read_text(encoding="utf-8"))
    report = (doc_dir / "report.md").read_text(encoding="utf-8")

    channel_a = sum(1 for item in components if item["ref_designation"].strip())
    channel_b = len(bom["items"])
    # Ручная сверка условий сценария и формулы.
    assert channel_a == 2
    assert channel_b == 1
    expected = (1.0 - abs(channel_a - channel_b) / max(channel_a, channel_b)) * 100.0
    assert expected == 50.0

    assert "## Сверка каналов" in report
    assert f"- A — компонентов с обозначением на схеме: {channel_a}" in report
    assert f"- B — строк BOM: {channel_b}" in report
    assert f"MULTIMODAL CONSISTENCY SCORE: {expected:.2f}%" in report
