"""Тесты M1: декларативная библиотека УГО ``symbols.yaml``."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from schematic_agent.components import (
    detect_components,
    load_symbols_config,
)
from schematic_agent.config import SYMBOLS_FILE_ENV
from schematic_agent.geometry import build_geometry
from schematic_agent.parser.pdf_extract import extract_pdf


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
