"""Тесты этапа 5: извлечение BOM из таблиц и сверка со схемой."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from schematic_agent.bom import (
    BOM_SOURCE_FALLBACK,
    FALLBACK_PLACEHOLDER,
    build_bom,
    build_bom_with_source,
    components_to_bom,
)
from schematic_agent.contract import Component
from schematic_agent.parser.pdf_extract import extract_pdf

#: Шрифт с кириллицей для синтетической таблицы.
CYRILLIC_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def _make_bom_pdf(path: Path) -> Path:
    """Создаёт PDF с реальной таблицей 3×2 (R1/K1) линиями и текстом."""
    document = pymupdf.open()
    page = document.new_page(width=500, height=300)
    page.insert_font(fontname="DejaVu", fontfile=str(CYRILLIC_FONT))
    page.draw_rect(pymupdf.Rect(40, 40, 460, 160))
    page.draw_line((140, 40), (140, 160))
    page.draw_line((300, 40), (300, 160))
    page.draw_line((40, 100), (460, 100))
    cells = [
        (50, 73, "R1"),
        (150, 73, "Резистор 10 кОм"),
        (310, 73, "1"),
        (50, 133, "K1"),
        (150, 133, "Реле 24В"),
        (310, 133, "1"),
    ]
    for x, y, text in cells:
        page.insert_text((x, y), text, fontname="DejaVu")
    document.save(str(path))
    document.close()
    return path


def _component(component_id: str, ref: str, component_type: str = "unknown") -> Component:
    """Компонент схемы для сверки BOM."""
    return Component(
        id=component_id,
        type=component_type,
        ref_designation=ref,
        x=0.0,
        y=0.0,
    )


@pytest.mark.skipif(not CYRILLIC_FONT.is_file(), reason="нет шрифта DejaVu")
def test_bom_table_extracted_and_normalized(tmp_path: Path) -> None:
    """Таблица 3×2 → строки BOM с категорией и параметрами."""
    pdf = _make_bom_pdf(tmp_path / "bom.pdf")

    extraction = extract_pdf(str(pdf))
    assert extraction.all_tables(), "pdfplumber должен найти таблицу"

    items, issues = build_bom(extraction, [])

    assert [item.ref_designation for item in items] == ["R1", "K1"]
    resistor, relay = items
    assert resistor.name == "Резистор 10 кОм"
    assert resistor.quantity == 1
    assert resistor.page == 1
    assert resistor.category == "resistor"
    assert resistor.normalized_name == "Резистор"
    assert "10" in resistor.parameters and "кОм" in resistor.parameters
    assert relay.category == "relay"
    assert relay.normalized_name == "Реле"
    assert "24" in relay.parameters and "В" in relay.parameters

    # Без компонентов обе строки отсутствуют на схеме.
    assert not resistor.found_on_schematic
    assert {issue.type for issue in issues} == {"bom_item_not_on_schematic"}


@pytest.mark.skipif(not CYRILLIC_FONT.is_file(), reason="нет шрифта DejaVu")
def test_bom_reconcile_marks_found_and_reports_extra_component(
    tmp_path: Path,
) -> None:
    """Сверка: R1 найден, K1 отсутствует, C5 на схеме лишний."""
    pdf = _make_bom_pdf(tmp_path / "bom.pdf")
    extraction = extract_pdf(str(pdf))
    components = [
        _component("c001", "R1", "resistor"),
        _component("c002", "C5", "capacitor"),
    ]

    items, issues = build_bom(extraction, components)

    by_ref = {item.ref_designation: item for item in items}
    assert by_ref["R1"].found_on_schematic is True
    assert by_ref["K1"].found_on_schematic is False

    issue_types = [issue.type for issue in issues]
    assert issue_types.count("bom_item_not_on_schematic") == 1
    assert issue_types.count("component_not_in_bom") == 1
    missing = next(
        issue for issue in issues if issue.type == "bom_item_not_on_schematic"
    )
    assert "K1" in missing.description


def test_bom_missing_table_reports_issue(tmp_path: Path) -> None:
    """PDF без таблицы → bom_table_missing с действием manual BOM entry."""
    document = pymupdf.open()
    page = document.new_page(width=200, height=200)
    page.draw_rect(pymupdf.Rect(20, 20, 60, 40))
    page.insert_text((28, 55), "R1")
    pdf = tmp_path / "no_table.pdf"
    document.save(str(pdf))
    document.close()

    extraction = extract_pdf(str(pdf))
    items, issues = build_bom(extraction, [])

    assert items == []
    issue = next(issue for issue in issues if issue.type == "bom_table_missing")
    assert issue.required_action == "manual BOM entry"


# --- П1: алиасы колонки количества (без PDF-рендера) -----------------------


@pytest.mark.parametrize(
    "alias", ["Кол.", "Количество", "Кол-во", "QTY", "шт.", "Quan"]
)
def test_quantity_column_aliases_detected(alias: str) -> None:
    """П1: заголовки-алиасы количества распознаются как quantity."""
    from schematic_agent.bom import detect_columns

    rows = {
        0: {0: "Поз.", 1: "Наименование", 2: alias, 3: "Примечание"},
        1: {0: "R1", 1: "Резистор 10 кОм", 2: "2", 3: "smd"},
    }

    header_row, mapping = detect_columns(rows)

    assert header_row == 0
    assert mapping["quantity"] == 2
    assert mapping["note"] == 3


def test_quantity_does_not_leak_into_note() -> None:
    """П1: при заголовке «Кол.» количество не попадает в note."""
    from schematic_agent.bom import extract_bom
    from schematic_agent.parser.pdf_extract import TableCell, TableData

    cells = [
        TableCell("Поз.", 0, 0, 10, 10, 0, 0),
        TableCell("Наименование", 0, 0, 10, 10, 0, 1),
        TableCell("Кол.", 0, 0, 10, 10, 0, 2),
        TableCell("R1", 0, 0, 10, 10, 1, 0),
        TableCell("Резистор 10 кОм", 0, 0, 10, 10, 1, 1),
        TableCell("2", 0, 0, 10, 10, 1, 2),
    ]
    table = TableData(page=1, x0=0, y0=0, x1=100, y1=100, cells=cells)

    items = extract_bom([table])

    assert len(items) == 1
    assert items[0].quantity == 2
    assert items[0].note == ""


def test_quantity_fallback_numeric_column_without_header_alias() -> None:
    """П1: без алиаса количества числовая колонка определяется fallback-ом."""
    from schematic_agent.bom import detect_columns

    rows = {
        0: {0: "Обозначение", 1: "Описание", 2: "Кол-во, ед."},
        1: {0: "C1", 1: "Конденсатор 10 мкФ", 2: "3"},
    }

    header_row, mapping = detect_columns(rows)

    assert header_row == 0
    assert mapping["quantity"] == 2
    assert mapping.get("note", -1) != 2


# --- Р2: дефис/пробелы в заголовке количества ------------------------------


def test_quantity_header_with_hyphen_normalized_to_quantity() -> None:
    """Р2: «Кол-во» → «колво» → распознаётся как quantity, не уходит в note."""
    from schematic_agent.bom import detect_columns

    rows = {
        0: {0: "Поз.", 1: "Наименование", 2: "Кол-во", 3: "Примечание"},
        1: {0: "R1", 1: "Резистор 10 кОм", 2: "2", 3: "smd"},
    }

    header_row, mapping = detect_columns(rows)

    assert header_row == 0
    assert mapping["quantity"] == 2
    assert mapping["note"] == 3


def test_kol_kont_header_is_not_quantity() -> None:
    """Р2: «Кол.конт.» → «колконт» — точный алиас не совпадает, это note."""
    from schematic_agent.bom import detect_columns

    rows = {
        0: {
            0: "Поз.",
            1: "Наименование",
            2: "Кол.конт.",
            3: "Примечание",
        },
        # Значения в «Кол.конт.» не числовые, чтобы fallback не сделал их qty.
        1: {0: "K1", 1: "Реле 24В", 2: "2 з.", 3: "smd"},
    }

    header_row, mapping = detect_columns(rows)

    assert header_row == 0
    assert mapping.get("quantity", -1) != 2
    assert mapping.get("note") == 3


# --- Задача 5: BOM-фолбэк по распознанным компонентам -----------------------


def test_components_fallback_aggregates_duplicate_ref() -> None:
    """Фолбэк: два компонента R1 → одна строка BOM с quantity=2."""
    first = _component("c001", "R1", "resistor").model_copy(
        update={"confidence": 0.8, "attributes": {"page": 1}}
    )
    second = _component("c002", "R1", "resistor").model_copy(
        update={"confidence": 0.6, "attributes": {"page": 2}}
    )

    items = components_to_bom([first, second])

    assert len(items) == 1
    item = items[0]
    assert item.ref_designation == "R1"
    assert item.quantity == 2
    assert item.category == "resistor"
    assert item.page == 1
    assert item.found_on_schematic is True
    # Честно пониженная уверенность: min(0.8, 0.6) * 0.9.
    assert item.confidence == pytest.approx(0.54)


def test_components_fallback_unknown_type_and_missing_ref() -> None:
    """Фолбэк: unknown → плейсхолдер; компонент без ref не попадает в BOM."""
    unknown = _component("c001", "U1", "unknown")
    without_ref = _component("c002", "", "resistor")

    items = components_to_bom([unknown, without_ref])

    assert len(items) == 1
    assert items[0].ref_designation == "U1"
    assert items[0].name == FALLBACK_PLACEHOLDER
    assert items[0].category == "unknown"


def test_build_bom_without_table_uses_components_fallback(
    tmp_path: Path,
) -> None:
    """Нет таблицы + есть компоненты → непустой BOM, Issue сохраняется."""
    document = pymupdf.open()
    page = document.new_page(width=200, height=200)
    page.draw_rect(pymupdf.Rect(20, 20, 60, 40))
    page.insert_text((28, 55), "R1")
    pdf = tmp_path / "no_table.pdf"
    document.save(str(pdf))
    document.close()

    extraction = extract_pdf(str(pdf))
    components = [_component("c001", "R1", "resistor")]
    items, issues, source = build_bom_with_source(extraction, components)

    assert source == BOM_SOURCE_FALLBACK
    assert [item.ref_designation for item in items] == ["R1"]
    assert all(item.found_on_schematic for item in items)
    issue_types = {issue.type for issue in issues}
    assert "bom_table_missing" in issue_types
    assert "component_not_in_bom" not in issue_types
    assert "bom_item_not_on_schematic" not in issue_types


# --- T-C2: нормализация обозначений при сверке ------------------------------


def _bom_item(ref: str) -> object:
    """Строка BOM для юнит-тестов сверки."""
    from schematic_agent.contract import BomItem

    return BomItem(ref_designation=ref, name="Резистор", quantity=1)


def test_normalize_ref_uppercases_and_strips_non_alnum() -> None:
    """T-C2: нормализация ключа сопоставления."""
    from schematic_agent.bom import normalize_ref

    assert normalize_ref("c101 ") == "C101"
    assert normalize_ref("C-101") == "C101"
    assert normalize_ref(" r. 12 ") == "R12"


def test_reconcile_matches_normalized_ref() -> None:
    """T-C2: "c101 " на схеме ↔ "C-101" в BOM — совпадение."""
    from schematic_agent.bom import reconcile_bom

    item = _bom_item("C-101")
    component = _component("c001", "c101 ", "capacitor")

    issues: list = []
    result = reconcile_bom([item], [component], issues)  # type: ignore[arg-type]

    assert result[0].found_on_schematic is True
    assert issues == []


def test_reconcile_does_not_match_different_refs() -> None:
    """T-C2: "C101" и "C102" — НЕ совпадение (вложенность не внедряется)."""
    from schematic_agent.bom import reconcile_bom

    item = _bom_item("C101")
    component = _component("c001", "C102", "capacitor")

    issues: list = []
    result = reconcile_bom([item], [component], issues)  # type: ignore[arg-type]

    assert result[0].found_on_schematic is False
    types = [issue.type for issue in issues]
    assert types.count("bom_item_not_on_schematic") == 1
    assert types.count("component_not_in_bom") == 1
