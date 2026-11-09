"""Unit-тесты этапа 2: извлечение текста, примитивов, таблиц, растровой детекции."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from schematic_agent.parser.pdf_extract import (
    PdfExtraction,
    detect_raster_pdf,
    extract_page,
    extract_pdf,
)


def make_vector_pdf(path: Path) -> Path:
    """Создаёт синтетический векторный PDF с текстом и примитивами."""
    document = pymupdf.open()
    page = document.new_page(width=300, height=300)
    page.insert_text((50, 50), "R1")
    page.draw_line((10, 10), (100, 10))
    page.draw_rect(pymupdf.Rect(10, 20, 60, 40))
    page.draw_circle((150, 150), 20)
    page.draw_bezier((200, 50), (220, 30), (240, 50), (260, 70))
    document.save(str(path))
    document.close()
    return path


def make_image_pdf(path: Path) -> Path:
    """Создаёт PDF, страница которого содержит только растровое изображение."""
    document = pymupdf.open()
    page = document.new_page(width=200, height=200)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8))
    pixmap.clear_with(200)
    page.insert_image(pymupdf.Rect(20, 20, 120, 120), pixmap=pixmap)
    document.save(str(path))
    document.close()
    return path


def test_extract_words_and_primitives(tmp_path: Path) -> None:
    """Слова с bbox и типы примитивов распознаются."""
    pdf = make_vector_pdf(tmp_path / "vector.pdf")

    extraction = extract_pdf(str(pdf))

    assert extraction.opened is True
    assert extraction.page_count == 1
    page = extraction.pages[0]
    assert any(span.text == "R1" for span in page.texts)
    span = next(span for span in page.texts if span.text == "R1")
    assert span.x0 < span.x1 and span.y0 < span.y1

    kinds = {primitive.kind for primitive in page.primitives}
    assert {"line", "rect", "circle", "curve"} <= kinds

    circle = next(item for item in page.primitives if item.kind == "circle")
    assert circle.attributes["radius"] == pytest.approx(20.0, abs=0.5)
    assert circle.attributes["center"] == pytest.approx((150.0, 150.0), abs=0.5)


def test_extract_single_page_matches_full(tmp_path: Path) -> None:
    """extract_page возвращает ту же страницу, что extract_pdf."""
    pdf = make_vector_pdf(tmp_path / "vector.pdf")

    page = extract_page(str(pdf), 1)

    assert page.page == 1
    assert any(span.text == "R1" for span in page.texts)
    assert page.primitives


def test_raster_only_page_detected(tmp_path: Path) -> None:
    """Страница без векторов, но с изображением → raster_only и Issue."""
    pdf = make_image_pdf(tmp_path / "raster.pdf")

    extraction = extract_pdf(str(pdf))

    assert extraction.opened is True
    page = extraction.pages[0]
    assert page.raster_only is True
    issue = next(item for item in extraction.issues if item.type == "raster_only_page")
    assert issue.required_action == "OCR required"
    assert detect_raster_pdf(str(pdf)) is True


def test_corrupt_pdf_returns_parse_error(tmp_path: Path) -> None:
    """Повреждённый PDF не выбрасывает исключение, а даёт parse_error."""
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"not a real pdf" * 20)

    extraction = extract_pdf(str(pdf))

    assert isinstance(extraction, PdfExtraction)
    assert extraction.opened is False
    assert extraction.pages == []
    assert any(item.type == "parse_error" for item in extraction.issues)


def test_empty_file_returns_empty_extraction(tmp_path: Path) -> None:
    """Пустой файл — деградация без parse_error, с Issue empty_file."""
    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"")

    extraction = extract_pdf(str(pdf))

    assert extraction.opened is False
    assert extraction.pages == []
    # Семантика этапа 10: пустой файл явно помечается empty_file
    # (pipeline.run() переводит это в status=failed, error="empty_file").
    assert [issue.type for issue in extraction.issues] == ["empty_file"]
