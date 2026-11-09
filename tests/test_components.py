"""Unit-тесты этапа 4: шаблоны УГО и привязка обозначений."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from schematic_agent.components import detect_components, load_templates
from schematic_agent.geometry import build_geometry
from schematic_agent.parser.pdf_extract import extract_pdf


def _detect(path: Path) -> tuple[list, list]:
    """Прогоняет этапы 2–4 для PDF."""
    extraction = extract_pdf(str(path))
    geometry = build_geometry(extraction)
    return detect_components(geometry, extraction)


def test_resistor_rect_with_designation(tmp_path: Path) -> None:
    """Вытянутый прямоугольник + R1 → resistor с выводами."""
    document = pymupdf.open()
    page = document.new_page(width=200, height=200)
    page.draw_rect(pymupdf.Rect(20, 20, 60, 40))
    page.insert_text((28, 55), "R1")
    pdf = tmp_path / "resistor.pdf"
    document.save(str(pdf))
    document.close()

    components, issues = _detect(pdf)

    resistor = next(item for item in components if item.type == "resistor")
    assert resistor.ref_designation == "R1"
    assert resistor.confidence == 1.0
    assert len(resistor.pins) >= 2
    assert resistor.pins[0]["component_id"] == resistor.id
    assert issues == []


def test_terminal_circle_with_line(tmp_path: Path) -> None:
    """Маленькая окружность с линией + X1 → terminal."""
    document = pymupdf.open()
    page = document.new_page(width=200, height=200)
    page.draw_circle((150, 150), 3)
    page.draw_line((153, 150), (175, 150))
    page.insert_text((158, 140), "X1")
    pdf = tmp_path / "terminal.pdf"
    document.save(str(pdf))
    document.close()

    components, _ = _detect(pdf)

    terminal = next(item for item in components if item.type == "terminal")
    assert terminal.ref_designation == "X1"
    assert len(terminal.pins) >= 1


def test_capacitor_parallel_lines(tmp_path: Path) -> None:
    """Две параллельные короткие линии + C1 → capacitor."""
    document = pymupdf.open()
    page = document.new_page(width=200, height=200)
    page.draw_line((50, 40), (50, 60))
    page.draw_line((56, 40), (56, 60))
    page.insert_text((60, 50), "C1")
    pdf = tmp_path / "capacitor.pdf"
    document.save(str(pdf))
    document.close()

    components, _ = _detect(pdf)

    capacitor = next(item for item in components if item.type == "capacitor")
    assert capacitor.ref_designation == "C1"


def test_unknown_symbol_creates_issue(tmp_path: Path) -> None:
    """Непонятное скопление линий с обозначением → unknown + Issue."""
    document = pymupdf.open()
    page = document.new_page(width=200, height=200)
    page.draw_line((20, 20), (34, 34))
    page.insert_text((24, 45), "VT1")
    pdf = tmp_path / "unknown.pdf"
    document.save(str(pdf))
    document.close()

    components, issues = _detect(pdf)

    unknown = next(item for item in components if item.type == "unknown")
    assert unknown.ref_designation == "VT1"
    assert unknown.confidence == 0.3
    issue = next(item for item in issues if item.type == "unrecognized_symbol")
    assert issue.required_action == "VLM review"


def test_load_templates_contains_required_symbols() -> None:
    """Словарь шаблонов содержит базовые УГО."""
    names = {template.name for template in load_templates()}

    assert {"resistor", "capacitor", "diode", "terminal", "coil", "contact"} <= names


def test_association_prefers_weighted_score_over_pure_edge_distance() -> None:
    """T-D1: взвешенный score 0.75·edge + 0.25·center разрешает ничью по edge.

    Крупное УГО (0,0,80,60) и компактное (83,0,93,10). Два обозначения:
    «R1» у левой границы крупного УГО и «R2» у компактного. Для крупного
    УГО обе edge-дистанции равны 0 (чистая min-дистанция дала бы ничью и
    выбрала последнее обозначение), но взвешенная оценка «R1» = 10.0
    меньше «R2» = 11.37 — крупное УГО берёт своё обозначение, компактное
    получает «R2».
    """
    from schematic_agent.components import (
        Candidate,
        _attach_ref_designations,
        _association_score,
        _point_bbox_distance,
    )
    from schematic_agent.parser.pdf_extract import TextSpan

    big = Candidate("resistor", 1, (0.0, 0.0, 80.0, 60.0))
    compact = Candidate("resistor", 1, (83.0, 0.0, 93.0, 10.0))
    own = TextSpan("R1", -2.0, 27.0, 2.0, 33.0, page=1)
    neighbor = TextSpan("R2", 76.0, 2.0, 80.0, 8.0, page=1)

    # Чистая edge-дистанция до крупного УГО у обоих обозначений равна 0.
    assert _point_bbox_distance((own.cx, own.cy), big.bbox) == 0.0
    assert _point_bbox_distance((neighbor.cx, neighbor.cy), big.bbox) == 0.0
    # Взвешенная оценка различает их: 10.0 против 11.37.
    assert _association_score(own, big.bbox)[0] == 10.0
    assert abs(_association_score(neighbor, big.bbox)[0] - 11.369) < 0.01

    assignments, unused = _attach_ref_designations(
        [big, compact], [own, neighbor], tolerance=40.0
    )

    assert assignments == {0: own, 1: neighbor}
    assert unused == []
