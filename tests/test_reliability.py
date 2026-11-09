"""Тесты этапа 10: единый маппинг статусов и изоляция страниц."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pymupdf
import pytest

from schematic_agent import run
from schematic_agent.config import DOC_TIMEOUT_ENV, OUTPUT_DIR_ENV
from schematic_agent.contract import Task, TaskOptions, TaskPayload


def _task(pdf: Path, document_id: str = "doc") -> Task:
    return Task(
        task_id="task-rel",
        payload=TaskPayload(pdf_path=str(pdf), document_id=document_id),
    )


def _multi_page_pdf(path: Path, pages: int = 3) -> Path:
    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page(width=200, height=200)
        page.draw_rect(pymupdf.Rect(20, 20, 60, 40))
        page.insert_text((28, 55), f"R{index + 1}")
    document.save(str(path))
    document.close()
    return path


@pytest.fixture
def output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "out"
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(target))
    return target


def test_corrupt_pdf_is_failed(tmp_path: Path, output_dir: Path) -> None:
    """Повреждённый PDF → failed + parse_error."""
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"not a pdf" * 20)

    result = run(_task(pdf, "doc_broken"))

    assert result.status == "failed"
    assert result.error is not None and result.error.startswith("parse_error:")
    issues = json.loads(
        (output_dir / "doc_broken" / "issues.json").read_text(encoding="utf-8")
    )
    assert any(item["type"] == "parse_error" for item in issues["issues"])


def test_empty_file_is_failed_empty_file(
    tmp_path: Path, output_dir: Path
) -> None:
    """Пустой файл → failed error="empty_file" и Issue empty_file."""
    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"")

    result = run(_task(pdf, "doc_empty"))

    assert result.status == "failed"
    assert result.error == "empty_file"
    issues = json.loads(
        (output_dir / "doc_empty" / "issues.json").read_text(encoding="utf-8")
    )
    assert [item["type"] for item in issues["issues"]] == ["empty_file"]


def test_raster_pdf_is_completed_with_raster_issue(
    tmp_path: Path, output_dir: Path
) -> None:
    """Т4/Р3: растровый PDF без векторов → completed_with_issues."""
    document = pymupdf.open()
    page = document.new_page(width=200, height=200)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8))
    pixmap.clear_with(200)
    page.insert_image(pymupdf.Rect(20, 20, 120, 120), pixmap=pixmap)
    pdf = tmp_path / "raster.pdf"
    document.save(str(pdf))
    document.close()

    result = run(_task(pdf, "doc_raster"))

    # Р3: raster_only_page — деградация, а не полный успех (ТЗ §8/§11).
    assert result.status == "completed_with_issues"
    issues = json.loads(
        (output_dir / "doc_raster" / "issues.json").read_text(encoding="utf-8")
    )
    assert any(
        item["type"] == "raster_only_page" for item in issues["issues"]
    )


def test_page_failure_is_isolated_and_partial(
    tmp_path: Path, output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Т10: ошибка страницы 2 → partial, страницы 1 и 3 обработаны."""
    pdf = _multi_page_pdf(tmp_path / "schema.pdf", pages=3)
    import schematic_agent.parser.pdf_extract as pdf_extract

    original = pdf_extract._extract_one_page

    def flaky(page, page_number, tables, issues):
        if page_number == 2:
            raise RuntimeError("mock page failure")
        return original(page, page_number, tables, issues)

    monkeypatch.setattr(pdf_extract, "_extract_one_page", flaky)

    result = run(_task(pdf, "doc_page_fail"))

    assert result.status == "partial"
    assert result.summary.pages_processed == 2
    issues = json.loads(
        (output_dir / "doc_page_fail" / "issues.json").read_text(encoding="utf-8")
    )
    failed = [
        item for item in issues["issues"] if item["type"] == "page_failed"
    ]
    assert len(failed) == 1
    assert failed[0]["page"] == 2


def test_doc_timeout_is_soft_and_partial(
    tmp_path: Path, output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Общий таймаут между страницами → partial + doc_timeout."""
    pdf = _multi_page_pdf(tmp_path / "schema.pdf", pages=3)
    import schematic_agent.parser.pdf_extract as pdf_extract

    original = pdf_extract._extract_one_page

    def slow(page, page_number, tables, issues):
        if page_number == 1:
            time.sleep(0.05)
        return original(page, page_number, tables, issues)

    monkeypatch.setattr(pdf_extract, "_extract_one_page", slow)
    monkeypatch.setenv(DOC_TIMEOUT_ENV, "0.01")

    result = run(_task(pdf, "doc_timeout"))

    assert result.status == "partial"
    assert result.summary.pages_processed == 1
    issues = json.loads(
        (output_dir / "doc_timeout" / "issues.json").read_text(encoding="utf-8")
    )
    assert any(item["type"] == "doc_timeout" for item in issues["issues"])


def test_bom_table_missing_in_full_mode(
    tmp_path: Path, output_dir: Path
) -> None:
    """Т5/В1: full без таблицы BOM → bom_table_missing + with_issues."""
    document = pymupdf.open()
    page = document.new_page(width=200, height=200)
    page.draw_rect(pymupdf.Rect(20, 20, 60, 40))
    page.insert_text((28, 55), "R1")
    pdf = tmp_path / "no_table.pdf"
    document.save(str(pdf))
    document.close()

    result = run(_task(pdf, "doc_no_table"))

    assert result.status == "completed_with_issues"
    issues = json.loads(
        (output_dir / "doc_no_table" / "issues.json").read_text(encoding="utf-8")
    )
    assert any(
        item["type"] == "bom_table_missing" for item in issues["issues"]
    )
