"""Тесты M1: декларативная библиотека УГО ``symbols.yaml``."""

from __future__ import annotations

import time
from pathlib import Path

import pymupdf
import pytest

from schematic_agent.components import (
    detect_components,
    enrich_components_with_symbol_library,
    load_symbols_config,
)
from schematic_agent.config import SYMBOLS_FILE_ENV
from schematic_agent.contract import Component
from schematic_agent.geometry import build_geometry
from schematic_agent.parser.pdf_extract import extract_pdf
from schematic_agent.symbols_library import (
    default_symbols_library_path,
    detect_symbols_format,
    load_symbols_library,
)


def _detect(path: Path):
    """Прогоняет этапы 2–4 для PDF."""
    extraction = extract_pdf(str(path))
    geometry = build_geometry(extraction)
    return detect_components(geometry, extraction)


def _rect_with_ref(tmp_path: Path, ref: str, width: float, height: float) -> Path:
    """PDF из одного прямоугольника и обозначения рядом."""
    document = pymupdf.open()
    page = document.new_page(width=220, height=220)
    page.draw_rect(pymupdf.Rect(40, 40, 40 + width, 40 + height))
    page.insert_text((45, 35), ref)
    pdf = tmp_path / f"{ref}.pdf"
    document.save(str(pdf))
    document.close()
    return pdf


def test_default_yaml_contains_industrial_symbols() -> None:
    """Пакетный symbols.yaml содержит 5 промышленных шаблонов."""
    config = load_symbols_config()

    names = {template.name for template in config.templates}
    assert {"contactor", "breaker", "transformer", "vfd", "terminal_block"} <= names
    assert config.issues == []
    for template in config.templates:
        assert 0.4 <= template.confidence <= 0.6


def test_yaml_contactor_detected_on_synthetic_rect(tmp_path: Path) -> None:
    """Вытянутый прямоугольник 30×60 + «KM1» → contactor из YAML."""
    pdf = _rect_with_ref(tmp_path, "KM1", 30.0, 60.0)

    components, issues = _detect(pdf)

    contactor = next(item for item in components if item.type == "contactor")
    assert contactor.ref_designation == "KM1"
    assert 0.4 <= contactor.confidence <= 0.6
    assert not any(issue.type == "symbols_config_error" for issue in issues)


def test_yaml_requires_matching_ref_prefix(tmp_path: Path) -> None:
    """Тот же прямоугольник с R1 остаётся встроенным резистором."""
    pdf = _rect_with_ref(tmp_path, "R1", 30.0, 60.0)

    components, _ = _detect(pdf)

    assert any(item.type == "resistor" for item in components)
    assert not any(item.type == "contactor" for item in components)


def test_broken_yaml_reports_issue_and_keeps_builtin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Битый внешний YAML → symbols_config_error и рабочие builtin-детекторы."""
    broken = tmp_path / "broken.yaml"
    broken.write_text("symbols: [", encoding="utf-8")
    monkeypatch.setenv(SYMBOLS_FILE_ENV, str(broken))
    pdf = _rect_with_ref(tmp_path, "R1", 30.0, 60.0)

    components, issues = _detect(pdf)

    assert any(item.type == "resistor" for item in components)
    error = next(issue for issue in issues if issue.type == "symbols_config_error")
    assert "broken.yaml" in error.description


def test_external_yaml_overrides_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Внешний файл переопределяет одноимённый шаблон и добавляет новый."""
    external = tmp_path / "symbols.yaml"
    external.write_text(
        """
symbols:
  - name: contactor
    category: contactor
    ref_prefixes: [KM]
    confidence: 0.42
    geometry:
      kind: rect
      min_aspect: 1.5
      max_aspect: 4.0
      min_short: 12.0
      max_short: 70.0
      min_long: 20.0
      max_long: 160.0
  - name: soft_starter
    category: soft_starter
    ref_prefixes: [QS]
    confidence: 0.5
    geometry:
      kind: rect
      min_aspect: 1.0
      max_aspect: 4.0
net_aliases:
  - [L1, U]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(SYMBOLS_FILE_ENV, str(external))

    config = load_symbols_config()
    templates = {template.name: template for template in config.templates}

    assert templates["contactor"].confidence == pytest.approx(0.42)
    assert "soft_starter" in templates
    assert config.net_aliases["U"] == "L1"
    assert config.issues == []


# --- Rich-библиотека УГО (symbols_library.yaml) -------------------------


def _write_rich_library(path: Path, pins: int = 3) -> Path:
    """Пишет минимальную rich-библиотеку с УГО ``S``."""
    points = "\n".join(
        f"""  - designation: ''
    index: {index}
    direction: {'Left' if index % 2 == 0 else 'Right'}
    x: {float(index)}
    y: 0.0
    x_rel: 0.0
    y_rel: 0.0"""
        for index in range(pins)
    )
    path.write_text(
        f"""symbols:
- name: S
  variant: '0'
  heuristic: switch
  bounding_box:
    width: 4.0
    height: 8.0
  connection_points:
{points}
""",
        encoding="utf-8",
    )
    return path


def test_s1_rich_library_loads_and_is_cached() -> None:
    """S1: 7997 записей, категории из heuristic, кэш повторного вызова."""
    library = load_symbols_library()

    assert library.issues == []
    assert len(library.symbols) == 898
    assert library.entry_count == 7997
    assert library.symbols["S"].categories == frozenset({"switch"})
    assert library.symbols["S"].variant_count == 9

    started = time.perf_counter()
    cached = load_symbols_library()
    elapsed = time.perf_counter() - started

    assert cached is library
    assert elapsed < 0.1


def test_s2_broken_rich_yaml_reports_issue_and_pipeline_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2: битый rich-YAML → Issue + пустая библиотека, пайплайн жив."""
    broken = tmp_path / "broken_rich.yaml"
    broken.write_text("symbols: [", encoding="utf-8")
    monkeypatch.setenv(SYMBOLS_FILE_ENV, str(broken))

    library = load_symbols_library()

    assert library.symbols == {}
    assert library.entry_count == 0
    assert any(issue.type == "symbols_config_error" for issue in library.issues)

    pdf = _rect_with_ref(tmp_path, "R1", 30.0, 60.0)
    components, issues = _detect(pdf)

    assert any(item.type == "resistor" for item in components)
    assert any(issue.type == "symbols_config_error" for issue in issues)


def test_s3_pin_count_mismatch_on_synthetic_component(tmp_path: Path) -> None:
    """S3: ref S1 с 2 pins при ожидании 3 по rich-УГО → Issue."""
    rich = _write_rich_library(tmp_path / "rich.yaml", pins=3)
    library = load_symbols_library(rich)
    component = Component(
        id="c001",
        type="switch",
        ref_designation="S1",
        x=10.0,
        y=10.0,
        pins=[{"id": "p001"}, {"id": "p002"}],
        attributes={"bbox": [0.0, 0.0, 8.0, 4.0], "page": 2},
        confidence=0.6,
    )

    issues = enrich_components_with_symbol_library([component], library)

    assert component.attributes["symbol_name"] == "S"
    assert component.attributes["expected_pin_count"] == 3
    assert component.attributes["heuristic"] == "switch"
    assert len(issues) == 1
    issue = issues[0]
    assert issue.type == "pin_count_mismatch"
    assert issue.page == 2
    assert issue.confidence == 0.5
    assert issue.required_action == "VLM review"
    assert "S1" in issue.description
    assert "3" in issue.description and "2" in issue.description


def test_s4_symbols_format_autodetect(tmp_path: Path) -> None:
    """S4: rich и M1-форматы различаются по ключам записей."""
    rich = _write_rich_library(tmp_path / "rich.yaml")
    m1 = tmp_path / "m1.yaml"
    m1.write_text(
        """symbols:
  - name: contactor
    category: contactor
    ref_prefixes: [KM]
    geometry:
      kind: rect
""",
        encoding="utf-8",
    )

    assert detect_symbols_format(default_symbols_library_path()) == "rich"
    assert detect_symbols_format(rich) == "rich"
    assert detect_symbols_format(m1) == "template"
    # M1-файл для rich-загрузчика — пустая библиотека без ошибок.
    assert load_symbols_library(m1).symbols == {}
    assert load_symbols_library(m1).issues == []


def test_enrichment_keeps_builtin_detection() -> None:
    """Rich-обогащение не меняет тип/уверенность builtin-компонента."""
    library = load_symbols_library()
    component = Component(
        id="c001",
        type="switch",
        ref_designation="S1",
        x=10.0,
        y=10.0,
        pins=[{"id": "p001"}, {"id": "p002"}],
        attributes={"bbox": [0.0, 0.0, 8.0, 4.0], "page": 1},
        confidence=0.42,
    )

    enrich_components_with_symbol_library([component], library)

    assert component.type == "switch"
    assert component.confidence == pytest.approx(0.42)
    assert component.attributes["symbol_name"] == "S"
    assert component.attributes["symbol_variant"] in {"0", "1", "2", "3"}
