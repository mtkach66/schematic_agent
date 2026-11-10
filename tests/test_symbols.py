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


# --- M6: маппинг «ref-префикс → УГО EPLAN» ------------------------------


def _rich_record(
    name: str,
    variant: str,
    heuristic: str,
    width: float,
    height: float,
    pins: int = 2,
) -> str:
    """Фрагмент rich-библиотеки: одно имя + один вариант."""
    points = "\n".join(
        f"""  - designation: ''
    index: {index}
    direction: {'Up' if index % 2 == 0 else 'Down'}
    x: 0.0
    y: 0.0
    x_rel: 0.0
    y_rel: 0.0"""
        for index in range(pins)
    )
    return f"""- name: {name}
  variant: '{variant}'
  heuristic: {heuristic}
  bounding_box:
    width: {width}
    height: {height}
  connection_points:
{points}
"""


def _write_rich_with_map(
    path: Path, records: list[str], ref_prefix_map: str
) -> Path:
    """Пишет rich-библиотеку с заданной секцией ref_prefix_map."""
    path.write_text(
        "ref_prefix_map:\n" + ref_prefix_map + "\nsymbols:\n" + "".join(records),
        encoding="utf-8",
    )
    return path


def _symbol_component(ref: str, bbox: tuple[float, ...], pins: int = 2) -> Component:
    """Компонент с обозначением, bbox и выводами."""
    return Component(
        id="c001",
        type="unknown",
        ref_designation=ref,
        x=10.0,
        y=10.0,
        pins=[{"id": f"p{index:03d}"} for index in range(pins)],
        attributes={"bbox": list(bbox), "page": 1},
        confidence=0.5,
    )


def test_m6_qf_breaker_category_enriches_component(tmp_path: Path) -> None:
    """M1 плана: QF1 + категория breaker → УГО и вариант по аспекту bbox."""
    rich = _write_rich_with_map(
        tmp_path / "rich_qf.yaml",
        [
            _rich_record("QFB", "0", "breaker", 4.0, 16.0, pins=2),
            _rich_record("QFB", "1", "breaker", 8.0, 4.0, pins=2),
            _rich_record("QA", "0", "breaker", 4.0, 4.0, pins=2),
        ],
        "  QF: breaker\n",
    )
    library = load_symbols_library(rich)
    assert library.issues == []
    assert library.ref_prefix_map == {"QF": "breaker"}
    component = _symbol_component("QF1", (0.0, 0.0, 8.0, 4.0))

    issues = enrich_components_with_symbol_library([component], library)

    attributes = component.attributes
    assert attributes["symbol_name"] == "QFB"
    assert attributes["symbol_variant"] == "1"  # аспект 2.0 как у bbox
    assert attributes["expected_pin_count"] == 2
    assert attributes["heuristic"] == "breaker"
    assert attributes["symbol_match"] == "category"
    assert issues == []


def test_m6_name_list_has_priority_over_category(tmp_path: Path) -> None:
    """M2 плана: KM: [SCHB] приоритетнее категории switch."""
    rich = _write_rich_with_map(
        tmp_path / "rich_km.yaml",
        [
            _rich_record("SCHB", "0", "switch", 4.0, 16.0, pins=2),
            # SCHX геометрически ближе к bbox, но не входит в список имён.
            _rich_record("SCHX", "0", "switch", 8.0, 4.0, pins=2),
        ],
        "  KM: [SCHB]\n",
    )
    library = load_symbols_library(rich)
    assert library.ref_prefix_map == {"KM": ("SCHB",)}
    component = _symbol_component("KM1", (0.0, 0.0, 8.0, 4.0))

    enrich_components_with_symbol_library([component], library)

    assert component.attributes["symbol_name"] == "SCHB"
    assert component.attributes["symbol_match"] == "names"


def test_m6_exact_name_match_still_wins(tmp_path: Path) -> None:
    """M3 плана: точное совпадение S1→S сохраняет поведение M5."""
    rich = _write_rich_with_map(
        tmp_path / "rich_s.yaml",
        [
            _rich_record("S", "0", "switch", 4.0, 16.0, pins=2),
            _rich_record("OTHER", "0", "switch", 8.0, 4.0, pins=2),
        ],
        "  S: [OTHER]\n",
    )
    library = load_symbols_library(rich)
    component = _symbol_component("S1", (0.0, 0.0, 8.0, 4.0))

    enrich_components_with_symbol_library([component], library)

    assert component.attributes["symbol_name"] == "S"
    assert component.attributes["symbol_match"] == "exact"


def test_m6_unknown_prefix_is_not_enriched(tmp_path: Path) -> None:
    """M4 плана: префикс вне таблицы и библиотеки → без обогащения и issue."""
    rich = _write_rich_with_map(
        tmp_path / "rich_other.yaml",
        [_rich_record("QFB", "0", "breaker", 4.0, 16.0)],
        "  QF: breaker\n",
    )
    library = load_symbols_library(rich)
    component = _symbol_component("QQ1", (0.0, 0.0, 8.0, 4.0))

    issues = enrich_components_with_symbol_library([component], library)

    assert "symbol_name" not in component.attributes
    assert issues == []


def test_m6_table_available_from_m1_config_without_rich(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Таблица из символов M1-конфига доступна без rich-библиотеки."""
    external = tmp_path / "symbols.yaml"
    external.write_text(
        """ref_prefix_map:
  QF: breaker
  KM: [SCHB, SCHL]
symbols: []
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(SYMBOLS_FILE_ENV, str(external))

    config = load_symbols_config()
    library = load_symbols_library()

    # Пакетная таблица дополняется внешней (внешняя переопределяет ключи).
    assert config.ref_prefix_map["QF"] == "breaker"
    assert config.ref_prefix_map["KM"] == ("SCHB", "SCHL")
    assert config.ref_prefix_map["XT"] == "terminal"
    assert config.issues == []
    assert library.symbols == {}


def test_m6_broken_table_reports_issue_and_empty_map(tmp_path: Path) -> None:
    """Битый ref_prefix_map → symbols_config_error и пустая таблица."""
    rich = _write_rich_with_map(
        tmp_path / "rich_broken_map.yaml",
        [_rich_record("QFB", "0", "breaker", 4.0, 16.0)],
        "  QF:\n",
    )
    library = load_symbols_library(rich)

    assert library.ref_prefix_map == {}
    assert library.symbols  # записи УГО при этом загружены
    assert any(issue.type == "symbols_config_error" for issue in library.issues)


def test_m6_pipeline_enriches_qf_from_default_table(tmp_path: Path) -> None:
    """M6 в конвейере: QF1 обогащается категорией breaker из пакетной таблицы."""
    pdf = _rect_with_ref(tmp_path, "QF1", 30.0, 60.0)

    components, _ = _detect(pdf)

    qf = next(item for item in components if item.ref_designation == "QF1")
    assert qf.attributes["heuristic"] == "breaker"
    assert qf.attributes["symbol_match"] == "category"
