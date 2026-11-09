"""Регрессии smoke-теста пакета 2: З1 (метки цепей) и З2 (привязка R1)."""

from __future__ import annotations

import math
from pathlib import Path

import pymupdf

from schematic_agent.components import detect_components
from schematic_agent.geometry import build_geometry
from schematic_agent.netlist import build_netlist
from schematic_agent.parser.pdf_extract import extract_pdf


def _make_smoke_pdf(path: Path) -> Path:
    """Схема smoke: резистор, два провода, R1 далеко от центра, метка L1."""
    document = pymupdf.open()
    page = document.new_page(width=300, height=300)
    page.draw_rect(pymupdf.Rect(80, 100, 120, 110))
    page.draw_line((20, 105), (80, 105))
    page.draw_line((120, 105), (200, 105))
    page.insert_text((150, 108), "R1")  # >40 pt от центра bbox
    page.insert_text((70, 108), "L1")  # метка цепи у узла (80, 105)
    document.save(str(path))
    document.close()
    return path


def test_r1_attached_by_pin_distance(tmp_path: Path) -> None:
    """З2: R1 привязан по расстоянию до пин-узла, а не центра bbox."""
    pdf = _make_smoke_pdf(tmp_path / "smoke.pdf")
    extraction = extract_pdf(str(pdf))
    geometry = build_geometry(extraction)

    components, issues = detect_components(geometry, extraction)

    resistor = next(item for item in components if item.type == "resistor")
    assert resistor.ref_designation == "R1"
    assert issues == []
    # Проверяем, что старый критерий (центр bbox) не сработал бы.
    center_distance = math.hypot(150 - resistor.x, 108 - resistor.y)
    assert center_distance > 40


def test_net_label_is_not_a_component(tmp_path: Path) -> None:
    """З1: L1 — метка цепи, а не компонент и не позиционное обозначение."""
    pdf = _make_smoke_pdf(tmp_path / "smoke.pdf")
    extraction = extract_pdf(str(pdf))
    geometry = build_geometry(extraction)

    components, _ = detect_components(geometry, extraction)

    assert all(item.ref_designation != "L1" for item in components)
    assert all(item.type != "unknown" for item in components)

    nets, _ = build_netlist(geometry, components, extraction)
    assert any(net.name == "L1" for net in nets)
