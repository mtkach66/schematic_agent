"""Интеграционные тесты режимов ТЗ §13 и артефактов этапов 5–7."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from schematic_agent import run
from schematic_agent.config import OUTPUT_DIR_ENV
from schematic_agent.contract import Task, TaskOptions, TaskPayload

CYRILLIC_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


@pytest.fixture
def output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Перенаправляет артефакты во временный каталог."""
    target = tmp_path / "out"
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(target))
    return target


def _make_pdf(path: Path, *, with_table: bool = True) -> Path:
    """Схема с резистором R1, проводами и (опционально) таблицей BOM."""
    document = pymupdf.open()
    page = document.new_page(width=500, height=400)
    page.insert_font(fontname="DejaVu", fontfile=str(CYRILLIC_FONT))
    if with_table:
        page.draw_rect(pymupdf.Rect(40, 40, 460, 160))
        page.draw_line((140, 40), (140, 160))
        page.draw_line((300, 40), (300, 160))
        page.draw_line((40, 100), (460, 100))
        for x, y, text in [
            (50, 73, "R1"),
            (150, 73, "Резистор 10 кОм"),
            (310, 73, "1"),
            (50, 133, "K1"),
            (150, 133, "Реле 24В"),
            (310, 133, "1"),
        ]:
            page.insert_text((x, y), text, fontname="DejaVu")
    page.draw_rect(pymupdf.Rect(80, 250, 120, 262))
    page.draw_line((20, 256), (80, 256))
    page.draw_line((120, 256), (220, 256))
    page.insert_text((130, 259), "R1")
    page.insert_text((70, 259), "L1")
    document.save(str(path))
    document.close()
    return path


def _task(pdf: Path, mode: str, document_id: str) -> Task:
    """Задание с выбранным режимом."""
    return Task(
        task_id=f"task-{mode}",
        payload=TaskPayload(
            pdf_path=str(pdf),
            document_id=document_id,
            options=TaskOptions(mode=mode),
        ),
    )


@pytest.mark.skipif(not CYRILLIC_FONT.is_file(), reason="нет шрифта DejaVu")
def test_full_mode_fills_bom_netlist_and_plc(
    tmp_path: Path, output_dir: Path
) -> None:
    """full → BOM, Netlist и анализ ПЛК записаны реальными данными."""
    pdf = _make_pdf(tmp_path / "schema.pdf")

    result = run(_task(pdf, "full", "doc_full"))

    assert result.summary.bom_lines_found == 2
    assert result.summary.nets_found >= 2
    assert result.summary.vlm_calls == 0
    assert result.status == "completed_with_issues"  # K1 нет на схеме

    doc_dir = output_dir / "doc_full"
    bom = json.loads((doc_dir / "bom.json").read_text(encoding="utf-8"))
    refs = {item["ref_designation"]: item for item in bom["items"]}
    assert refs["R1"]["found_on_schematic"] is True
    assert refs["K1"]["found_on_schematic"] is False

    csv_header = (doc_dir / "bom.csv").read_text(encoding="utf-8").splitlines()[0]
    assert csv_header == (
        "ref_designation,name,quantity,note,page,normalized_name,category,"
        "parameters,found_on_schematic,confidence"
    )

    nets = json.loads((doc_dir / "netlist.json").read_text(encoding="utf-8"))["nets"]
    assert any(net["name"] == "L1" for net in nets)

    plc = json.loads((doc_dir / "plc_analysis.json").read_text(encoding="utf-8"))
    assert set(plc) == {
        "inputs", "outputs", "coils", "contacts", "interlocks",
        "logic_blocks", "warnings", "assumptions",
    }


@pytest.mark.skipif(not CYRILLIC_FONT.is_file(), reason="нет шрифта DejaVu")
def test_bom_only_skips_netlist(tmp_path: Path, output_dir: Path) -> None:
    """bom_only → BOM заполнен, Netlist пуст, ПЛК не анализируется."""
    pdf = _make_pdf(tmp_path / "schema.pdf")

    result = run(_task(pdf, "bom_only", "doc_bom"))

    assert result.summary.bom_lines_found == 2
    assert result.summary.nets_found == 0
    doc_dir = output_dir / "doc_bom"
    netlist = json.loads((doc_dir / "netlist.json").read_text(encoding="utf-8"))
    assert netlist == {"nets": []}
    plc = json.loads((doc_dir / "plc_analysis.json").read_text(encoding="utf-8"))
    assert plc["coils"] == []


@pytest.mark.skipif(not CYRILLIC_FONT.is_file(), reason="нет шрифта DejaVu")
def test_netlist_only_skips_bom(tmp_path: Path, output_dir: Path) -> None:
    """netlist_only → цепи построены, BOM пуст, bom_table_missing не шумит."""
    pdf = _make_pdf(tmp_path / "schema.pdf")

    result = run(_task(pdf, "netlist_only", "doc_net"))

    assert result.summary.nets_found >= 2
    assert result.summary.bom_lines_found == 0
    assert result.summary.issues_found == 0
    doc_dir = output_dir / "doc_net"
    bom = json.loads((doc_dir / "bom.json").read_text(encoding="utf-8"))
    assert bom["items"] == []
    assert bom["source"] == "table"


def test_bom_only_without_table_reports_missing(
    tmp_path: Path, output_dir: Path
) -> None:
    """bom_only без таблицы → bom_table_missing в issues.json."""
    pdf = _make_pdf(tmp_path / "schema.pdf", with_table=False)

    result = run(_task(pdf, "bom_only", "doc_no_bom"))

    assert result.status == "completed_with_issues"
    doc_dir = output_dir / "doc_no_bom"
    issues = json.loads((doc_dir / "issues.json").read_text(encoding="utf-8"))
    assert any(item["type"] == "bom_table_missing" for item in issues["issues"])
