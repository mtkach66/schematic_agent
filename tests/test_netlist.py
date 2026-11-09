"""Тесты этапа 6: граф цепей, выводы компонентов и сквозные метки."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from schematic_agent.components import detect_components
from schematic_agent.geometry import build_geometry
from schematic_agent.netlist import build_netlist
from schematic_agent.parser.pdf_extract import extract_pdf


def _pipeline(path: Path):
    """Прогоняет этапы 2–4 и возвращает extraction, geometry, components."""
    extraction = extract_pdf(str(path))
    geometry = build_geometry(extraction)
    components, _ = detect_components(geometry, extraction)
    return extraction, geometry, components


def test_two_segments_and_resistor_form_one_labeled_net(tmp_path: Path) -> None:
    """Два провода и резистор → одна цепь с двумя выводами и именем L1."""
    document = pymupdf.open()
    page = document.new_page(width=300, height=300)
    page.draw_rect(pymupdf.Rect(80, 95, 120, 115))
    page.draw_line((80, 105), (80, 150))
    page.draw_line((80, 150), (120, 105))
    page.insert_text((130, 108), "R1")
    page.insert_text((70, 158), "L1")
    pdf = tmp_path / "net.pdf"
    document.save(str(pdf))
    document.close()

    extraction, geometry, components = _pipeline(pdf)
    nets, issues = build_netlist(geometry, components, extraction)

    assert issues == []
    assert len(nets) == 1
    net = nets[0]
    assert net.name == "L1"
    assert net.text_labels == ["L1"]
    assert len(net.component_pins) >= 2
    assert all(pair[0] == components[0].id for pair in net.component_pins)
    assert net.pages == [1]
    # Внутренняя геометрия корпуса резистора не создаёт отдельных цепей.
    assert len(net.nodes) == 3


def test_cross_page_same_label_merges_nets(tmp_path: Path) -> None:
    """Одинаковая метка L1 на двух страницах объединяет цепи (Т8)."""
    document = pymupdf.open()
    for _ in range(2):
        page = document.new_page(width=200, height=150)
        page.draw_line((20, 50), (100, 50))
        page.insert_text((18, 53), "L1")
    pdf = tmp_path / "two_pages.pdf"
    document.save(str(pdf))
    document.close()

    extraction, geometry, components = _pipeline(pdf)
    nets, _ = build_netlist(geometry, components, extraction)

    assert len(nets) == 1
    net = nets[0]
    assert net.name == "L1"
    assert net.pages == [1, 2]
    assert {node for node in net.nodes} == {"n001", "n002", "n003", "n004"}


def test_crossing_without_node_is_not_connected(tmp_path: Path) -> None:
    """X-образное пересечение проводов без узла не склеивает цепи."""
    document = pymupdf.open()
    page = document.new_page(width=200, height=200)
    page.draw_line((20, 20), (120, 120))
    page.draw_line((20, 120), (120, 20))
    pdf = tmp_path / "crossing.pdf"
    document.save(str(pdf))
    document.close()

    extraction, geometry, components = _pipeline(pdf)
    nets, _ = build_netlist(geometry, components, extraction)

    assert len(geometry.crossings) == 1
    assert len(nets) == 2
