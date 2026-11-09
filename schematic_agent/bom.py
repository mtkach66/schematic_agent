"""Извлечение перечня элементов (BOM) и сверка со схемой.

Этап 5 конвейера (ТЗ §2, §10). Источник BOM — таблицы pdfplumber
(:class:`~schematic_agent.parser.pdf_extract.TableData`). Колонки
распознаются по заголовкам, при их отсутствии — по содержимому:

* позиционное обозначение — ``^[A-ZА-Я]{1,3}[0-9]{1,4}$``;
* наименование — самый длинный текстовый столбец;
* количество — целочисленный столбец;
* примечание — оставшийся текстовый столбец.

Нормализация выделяет категорию по словарю :data:`CATEGORIES` и параметры
(сопротивление, ёмкость, напряжение) из наименования. Сверка со схемой
заполняет :attr:`~schematic_agent.contract.BomItem.found_on_schematic` и
порождает проблемы ``bom_item_not_on_schematic`` /
``component_not_in_bom``. Если ни одной таблицы BOM не найдено —
``Issue(type="bom_table_missing")``.
"""

from __future__ import annotations

import csv
import re
from typing import Any, Iterable

from .contract import BomItem, Issue

#: Позиционное обозначение: R1, VD12, KM3, XT1 (1–3 буквы + 1–4 цифры).
REF_RE = re.compile(r"^[A-ZА-Я]{1,3}[0-9]{1,4}$")

#: Всё, кроме латинских букв и цифр: удаляется при нормализации ключей
#: сопоставления обозначений (пробелы, дефисы, точки и т.п.).
_REF_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")

#: Источник BOM: таблица перечня со схемы.
BOM_SOURCE_TABLE = "table"

#: Источник BOM: фолбэк по распознанным компонентам (таблицы перечня нет).
BOM_SOURCE_FALLBACK = "components_fallback"

#: Наименование строки BOM-фолбэка при неопознанном типе компонента.
FALLBACK_PLACEHOLDER = "[NOT IN BOM]"

#: Колонки BOM в порядке CSV (ТЗ §10).
BOM_CSV_FIELDS: tuple[str, ...] = (
    "ref_designation",
    "name",
    "quantity",
    "note",
    "page",
    "normalized_name",
    "category",
    "parameters",
    "found_on_schematic",
    "confidence",
)

#: Словари ключевых слов заголовков таблицы.
_HEADER_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ref": ("поз", "обозн", "design", "ref"),
    "name": ("наимен", "name", "описан", "элемент", "description"),
    "quantity": (
        "кол", "количество", "кол-во", "колво", "qty", "шт",
        "quan", "quantity", "count",
    ),
    "note": ("примеч", "note", "прим"),
}

#: Нормализация заголовков: нижний регистр, ё→е, удаление точек, двоеточий,
#: знака «№», пробелов и дефисов. «Кол-во» → «колво» (Р2), «Кол. конт.» →
#: «колконт» (не количество).
_HEADER_NORMALIZE_RE = re.compile(
    r"[\s.:№\-–—_\u2010\u2011\u2212\u200b]+"
)

#: Роли, для которых алиас сравнивается ТОЧНО (после нормализации).
#: Для количества это принципиально: иначе «Кол.конт.» содержит «кол» и
#: ошибочно распознаётся как количество (Р2).
_HEADER_EXACT_ROLES: frozenset[str] = frozenset({"quantity"})

#: Категории: (ключевые слова, слаг, каноническое русское имя).
#: Порядок важен: «термореле» проверяется раньше «реле».
CATEGORIES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("термореле", "термостат", "thermal"), "thermal", "Термореле"),
    (("стабилитрон", "zener"), "zener", "Стабилитрон"),
    (("контактор", "contactor"), "contactor", "Контактор"),
    (("конденсатор", "емкост", "ёмкост", "capacitor"), "capacitor", "Конденсатор"),
    (("резистор", "сопротивлен", "resistor"), "resistor", "Резистор"),
    (("реле", "relay"), "relay", "Реле"),
    (("кнопка", "button"), "button", "Кнопка"),
    (("переключатель", "выключатель", "switch"), "switch", "Выключатель"),
    (("лампа", "светодиод", "lamp", "led"), "lamp", "Лампа"),
    (("клапан", "valve"), "valve", "Клапан"),
    (("датчик", "sensor"), "sensor", "Датчик"),
    (("диод", "diode"), "diode", "Диод"),
    (("предохранитель", "fuse"), "fuse", "Предохранитель"),
    (("клемм", "terminal"), "terminal", "Клемма"),
)

#: Параметры из наименования.
_RESISTANCE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(к|k|м|m)?\s*(?:Ом|ohm)", re.IGNORECASE
)
_CAPACITANCE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(мкФ|нФ|пФ|µF|uF|nF|pF)", re.IGNORECASE
)
_VOLTAGE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:В|V)\b", re.IGNORECASE)


def _cell_rows(table: Any) -> dict[int, dict[int, str]]:
    """Собирает ячейки таблицы в отображение ``row → col → text``."""
    rows: dict[int, dict[int, str]] = {}
    for cell in table.cells:
        text = str(cell.text).strip()
        if not text:
            continue
        rows.setdefault(int(cell.row), {})[int(cell.col)] = text
    return rows


def _normalize_header(text: str) -> str:
    """Нормализует заголовок колонки для сравнения с алиасами."""
    lowered = str(text).lower().replace("ё", "е")
    return _HEADER_NORMALIZE_RE.sub("", lowered).strip()


def _header_columns(
    rows: dict[int, dict[int, str]]
) -> tuple[int | None, dict[str, int]]:
    """Ищет строку заголовков и отображает роли колонок.

    Заголовки распознаются по алиасам без учёта регистра, точек и лишних
    знаков: «Кол.», «Количество», «Кол-во», «QTY», «шт.», «Quan».
    Пробелы и дефисы удаляются: «Кол-во» → «колво». Для роли количества
    алиас сравнивается точно (после нормализации), поэтому «Кол.конт.» →
    «колконт» количеством НЕ считается (Р2).

    Returns:
        Индекс строки заголовка (``None``, если заголовков нет) и словарь
        «роль → индекс колонки».
    """
    for row_index, row in sorted(rows.items()):
        mapping: dict[str, int] = {}
        for role, keywords in _HEADER_KEYWORDS.items():
            for col, text in row.items():
                lowered = _normalize_header(text)
                if role in _HEADER_EXACT_ROLES:
                    matched = lowered in keywords
                else:
                    matched = any(keyword in lowered for keyword in keywords)
                if matched:
                    mapping.setdefault(role, col)
        if "ref" in mapping and "name" in mapping:
            return row_index, mapping
    return None, {}


def detect_columns(
    rows: dict[int, dict[int, str]]
) -> tuple[int | None, dict[str, int]]:
    """Определяет роли колонок таблицы BOM (публичная точка для тестов).

    Сначала ищет строку заголовков по алиасам, затем при необходимости
    дополняет роли по содержимому (количество — числовая колонка).

    Args:
        rows: Отображение ``row → col → text``.

    Returns:
        Кортеж ``(индекс строки заголовка, роль → индекс колонки)``.
    """
    header_row, mapping = _header_columns(rows)
    if "ref" in mapping:
        return header_row, _complete_mapping(rows, header_row, mapping)
    inferred = _infer_columns(rows, header_row)
    return header_row, inferred


def _column_values(
    rows: dict[int, dict[int, str]], col: int, skip: int | None
) -> list[str]:
    """Непустые значения колонки, кроме строки заголовка."""
    return [
        cell
        for row_index, row in sorted(rows.items())
        if row_index != skip and (cell := row.get(col, "")).strip()
    ]


def _infer_columns(
    rows: dict[int, dict[int, str]], header_row: int | None
) -> dict[str, int]:
    """Определяет роли колонок по содержимому таблицы."""
    if not rows:
        return {}
    columns = sorted({col for row in rows.values() for col in row})
    mapping: dict[str, int] = {}

    def score_ref(col: int) -> float:
        values = _column_values(rows, col, header_row)
        if not values:
            return 0.0
        matched = sum(1 for value in values if REF_RE.match(value))
        return matched / len(values)

    ref_scores = {col: score_ref(col) for col in columns}
    ref_col = max(ref_scores, key=lambda col: ref_scores[col])
    if ref_scores[ref_col] < 0.5:
        return {}
    mapping["ref"] = ref_col

    remaining = [col for col in columns if col != ref_col]

    def average_length(col: int) -> float:
        values = _column_values(rows, col, header_row)
        return sum(len(value) for value in values) / len(values) if values else 0.0

    def numeric_ratio(col: int) -> float:
        values = _column_values(rows, col, header_row)
        if not values:
            return 0.0
        numeric = sum(1 for value in values if value.strip().isdigit())
        return numeric / len(values)

    quantity_candidates = [col for col in remaining if numeric_ratio(col) >= 0.8]
    if quantity_candidates:
        qty_col = max(quantity_candidates)
        mapping["quantity"] = qty_col
        remaining.remove(qty_col)

    if remaining:
        name_col = max(remaining, key=average_length)
        mapping["name"] = name_col
        remaining.remove(name_col)
    if remaining:
        mapping["note"] = remaining[0]
    return mapping


def _complete_mapping(
    rows: dict[int, dict[int, str]],
    header_row: int | None,
    mapping: dict[str, int],
) -> dict[str, int]:
    """Дополняет отображение колонок по содержимому таблицы.

    Если заголовок количества не распознан (или распознан неверно), роль
    ``quantity`` получает числовая колонка. Колонка количества никогда не
    используется как ``note`` — иначе «1»/«2» утекают в примечание (П1).
    """
    result = dict(mapping)
    columns = sorted({col for row in rows.values() for col in row})
    used = {col for col in result.values()}

    def numeric_ratio(col: int) -> float:
        values = _column_values(rows, col, header_row)
        if not values:
            return 0.0
        return sum(1 for value in values if value.strip().isdigit()) / len(values)

    if "quantity" not in result:
        candidates = [
            col for col in columns
            if col not in used and numeric_ratio(col) >= 0.8
        ]
        if candidates:
            result["quantity"] = max(candidates)

    if "note" in result and result.get("note") == result.get("quantity"):
        result.pop("note", None)

    if "note" not in result:
        remaining = [
            col for col in columns
            if col not in result.values()
        ]
        if remaining:
            result["note"] = remaining[0]

    return result


def _parse_quantity(text: str) -> int:
    """Извлекает целое количество из строки (по умолчанию 1)."""
    match = re.search(r"\d+", text or "")
    if not match:
        return 1
    value = int(match.group(0))
    return value if value > 0 else 1


def _rows_to_items(
    rows: dict[int, dict[int, str]],
    header_row: int | None,
    mapping: dict[str, int],
    page: int,
) -> list[BomItem]:
    """Преобразует строки таблицы в элементы BOM."""
    items: list[BomItem] = []
    ref_col = mapping.get("ref")
    if ref_col is None:
        return items
    for row_index, row in sorted(rows.items()):
        if row_index == header_row:
            continue
        ref = row.get(ref_col, "").strip()
        if not REF_RE.match(ref):
            continue
        name = row.get(mapping.get("name", -1), "").strip()
        quantity_text = row.get(mapping.get("quantity", -1), "")
        note_col = mapping.get("note", -1)
        if note_col == mapping.get("quantity"):
            note_col = -1
        note = row.get(note_col, "").strip()
        items.append(
            BomItem(
                ref_designation=ref,
                name=name,
                quantity=_parse_quantity(quantity_text),
                note=note,
                page=page,
                confidence=0.9,
            )
        )
    return items


def _table_items(table: Any) -> list[BomItem]:
    """Извлекает BOM-строки одной таблицы."""
    rows = _cell_rows(table)
    if not rows:
        return []
    header_row, header_mapping = _header_columns(rows)
    if "ref" in header_mapping:
        mapping = _complete_mapping(rows, header_row, header_mapping)
    else:
        mapping = _infer_columns(rows, header_row)
    if "ref" not in mapping:
        return []
    return _rows_to_items(rows, header_row, mapping, int(getattr(table, "page", 0)))


def _component_parameters(members: list[Any]) -> str:
    """Извлекает параметры компонента из его атрибутов (первое значение)."""
    for component in members:
        attributes = getattr(component, "attributes", {}) or {}
        value = str(attributes.get("parameters", "") or "").strip()
        if value:
            return value
    return ""


def components_to_bom(components: Iterable[Any]) -> list[BomItem]:
    """Собирает BOM-фолбэк из распознанных компонентов.

    Используется, когда таблица перечня элементов не найдена. Компоненты
    агрегируются по паре ``(ref_designation, type)``: ``quantity`` — число
    компонентов с одинаковым обозначением и типом, ``name`` — тип
    (или :data:`FALLBACK_PLACEHOLDER` для ``unknown``), ``category`` — тип.
    Компоненты без обозначения не попадают в фолбэк.

    Уверенность честно понижается: ``min(confidence) * 0.9``.

    Args:
        components: Распознанные компоненты (этап 4).

    Returns:
        Строки BOM-фолбэка.
    """
    groups: dict[tuple[str, str], list[Any]] = {}
    order: list[tuple[str, str]] = []
    for component in components:
        ref = str(getattr(component, "ref_designation", "") or "").strip()
        if not ref:
            continue
        component_type = (
            str(getattr(component, "type", "") or "unknown").strip() or "unknown"
        )
        key = (ref, component_type)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(component)

    items: list[BomItem] = []
    for key in order:
        ref, component_type = key
        members = groups[key]
        first_page = int(
            (getattr(members[0], "attributes", {}) or {}).get("page", 0)
        )
        name = (
            FALLBACK_PLACEHOLDER
            if component_type == "unknown"
            else component_type
        )
        confidence = min(
            float(getattr(member, "confidence", 0.0)) for member in members
        ) * 0.9
        items.append(
            BomItem(
                ref_designation=ref,
                name=name,
                quantity=len(members),
                page=first_page,
                normalized_name="",
                category=component_type,
                parameters=_component_parameters(members),
                found_on_schematic=True,
                confidence=round(min(1.0, confidence), 3),
            )
        )
    return items


def _build_bom(
    extraction: Any, components: Iterable[Any]
) -> tuple[list[BomItem], list[Issue], str]:
    """Строит BOM и возвращает источник (``table`` / ``components_fallback``)."""
    component_list = list(components)
    items: list[BomItem] = []
    for table in extraction.all_tables():
        items.extend(_table_items(table))
    items = normalize_bom(items)
    issues: list[Issue] = []
    source = BOM_SOURCE_TABLE
    if not items:
        pages = [page.page for page in extraction.pages]
        issues.append(
            Issue(
                type="bom_table_missing",
                page=pages[0] if pages else 0,
                description=(
                    "Таблица перечня элементов не найдена или не распознана "
                    "ни на одной странице"
                ),
                confidence=0.6,
                required_action="manual BOM entry",
            )
        )
        fallback = normalize_bom(components_to_bom(component_list))
        if fallback:
            items = fallback
            source = BOM_SOURCE_FALLBACK
    items = reconcile_bom(
        items,
        component_list,
        issues,
        fallback=(source == BOM_SOURCE_FALLBACK),
    )
    return items, issues, source


def build_bom(
    extraction: Any, components: Iterable[Any]
) -> tuple[list[BomItem], list[Issue]]:
    """Строит перечень элементов и сверяет его со схемой.

    Если таблицы перечня нет, BOM собирается из распознанных компонентов
    (фолбэк), при этом Issue ``bom_table_missing`` сохраняется.

    Args:
        extraction: Результат этапа 2 (:class:`PdfExtraction`).
        components: Распознанные компоненты (этап 4).

    Returns:
        Кортеж ``(items, issues)``: строки BOM и проблемы извлечения/сверки.
    """
    items, issues, _source = _build_bom(extraction, components)
    return items, issues


def build_bom_with_source(
    extraction: Any, components: Iterable[Any]
) -> tuple[list[BomItem], list[Issue], str]:
    """Строит BOM и сообщает источник данных.

    Returns:
        Кортеж ``(items, issues, source)``, где ``source`` — одно из
        :data:`BOM_SOURCE_TABLE` / :data:`BOM_SOURCE_FALLBACK`.
    """
    return _build_bom(extraction, components)


def normalize_bom(items: list[BomItem]) -> list[BomItem]:
    """Нормализует наименования, категории и параметры строк BOM.

    Args:
        items: Исходные строки BOM.

    Returns:
        Строки с заполненными ``normalized_name``, ``category`` и
        ``parameters``.
    """
    for item in items:
        lowered = item.name.lower().replace("ё", "е")
        for keywords, category, canonical in CATEGORIES:
            if any(keyword.replace("ё", "е") in lowered for keyword in keywords):
                item.category = category
                item.normalized_name = canonical
                break
        else:
            item.normalized_name = " ".join(item.name.split())
        item.parameters = _extract_parameters(item.name)
    return items


def _extract_parameters(name: str) -> str:
    """Извлекает сопротивление, ёмкость и напряжение из наименования."""
    parameters: list[str] = []
    resistance = _RESISTANCE_RE.search(name or "")
    if resistance:
        value = resistance.group(1).replace(",", ".")
        prefix = (resistance.group(2) or "").lower()
        unit = {"к": "кОм", "k": "кОм", "м": "МОм", "m": "МОм"}.get(prefix, "Ом")
        parameters.append(f"R={value} {unit}")
    capacitance = _CAPACITANCE_RE.search(name or "")
    if capacitance:
        value = capacitance.group(1).replace(",", ".")
        unit = capacitance.group(2).replace("uF", "мкФ").replace("µF", "мкФ")
        unit = unit.replace("nF", "нФ").replace("pF", "пФ")
        parameters.append(f"C={value} {unit}")
    voltage = _VOLTAGE_RE.search(name or "")
    if voltage:
        parameters.append(f"U={voltage.group(1).replace(',', '.')} В")
    return "; ".join(parameters)


def normalize_ref(ref: str) -> str:
    """Нормализует позиционное обозначение для сопоставления.

    Приводит к верхнему регистру и удаляет всё, кроме ``[A-Z0-9]``:
    ``"c101 "`` → ``"C101"``, ``"C-101"`` → ``"C101"``. Фактическое значение
    ``ref_designation`` не меняется — нормализуется только ключ сравнения.

    Args:
        ref: Исходное обозначение.

    Returns:
        Нормализованный ключ сопоставления (может быть пустым).
    """
    return _REF_NON_ALNUM_RE.sub("", str(ref or "").upper())


def reconcile_bom(
    items: list[BomItem],
    components: list[Any],
    issues: list[Issue] | None = None,
    *,
    fallback: bool = False,
) -> list[BomItem]:
    """Сверяет BOM с компонентами, найденными на схеме.

    Args:
        items: Строки BOM.
        components: Распознанные компоненты.
        issues: Список проблем, пополняемый расхождениями.
        fallback: ``True`` для BOM, собранного из компонентов: все строки по
            определению найдены на схеме, расхождения не порождаются.

    Returns:
        Сверенные строки BOM.
    """
    issues = issues if issues is not None else []
    if fallback:
        for item in items:
            item.found_on_schematic = True
        return items

    # Ключи сопоставления нормализуются (верхний регистр, только [A-Z0-9]):
    # "c101 " и "C-101" дают один ключ "C101". Фактические значения
    # ref_designation не изменяются. Вложенность (C101 ⊂ C101HUD) не
    # поддерживается — только точное совпадение нормализованных ключей.
    component_refs = {
        normalize_ref(component.ref_designation)
        for component in components
        if getattr(component, "ref_designation", "")
    }
    component_refs.discard("")
    bom_refs: set[str] = set()

    for item in items:
        item_key = normalize_ref(item.ref_designation)
        bom_refs.add(item_key)
        item.found_on_schematic = bool(item_key) and item_key in component_refs
        item.confidence = round(
            min(1.0, item.confidence + (0.05 if item.found_on_schematic else 0.0)),
            3,
        )
        if not item.found_on_schematic:
            issues.append(
                Issue(
                    type="bom_item_not_on_schematic",
                    page=item.page,
                    description=(
                        f"Строка BOM {item.ref_designation} не найдена "
                        "на схеме"
                    ),
                    confidence=0.7,
                    required_action="Проверить схему и перечень элементов",
                )
            )

    for component in components:
        ref = getattr(component, "ref_designation", "")
        if not ref or normalize_ref(ref) in bom_refs:
            continue
        attributes = getattr(component, "attributes", {}) or {}
        issues.append(
            Issue(
                type="component_not_in_bom",
                page=int(attributes.get("page", 0)),
                x=float(getattr(component, "x", 0.0)),
                y=float(getattr(component, "y", 0.0)),
                description=f"Обозначение {ref} на схеме отсутствует в BOM",
                confidence=0.7,
                required_action="Дополнить перечень элементов",
            )
        )
    return items


def write_bom_csv(items: list[BomItem], path: str) -> None:
    """Записывает BOM в CSV-файл (колонки ТЗ §10).

    Args:
        items: Строки BOM.
        path: Путь к CSV-файлу.
    """
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BOM_CSV_FIELDS))
        writer.writeheader()
        for item in items:
            row = item.model_dump()
            row["found_on_schematic"] = str(bool(row["found_on_schematic"]))
            writer.writerow({field: row[field] for field in BOM_CSV_FIELDS})


def extract_bom(tables: list[Any], pages: list[int] | None = None) -> list[BomItem]:
    """Извлекает строки BOM из готовых таблиц (совместимый API).

    Args:
        tables: Таблицы :class:`TableData`.
        pages: Номера страниц (не используется: страница берётся из таблицы).

    Returns:
        Ненормализованные строки BOM.
    """
    items: list[BomItem] = []
    for table in tables:
        items.extend(_table_items(table))
    return items


__all__ = [
    "REF_RE",
    "BOM_CSV_FIELDS",
    "CATEGORIES",
    "BOM_SOURCE_TABLE",
    "BOM_SOURCE_FALLBACK",
    "FALLBACK_PLACEHOLDER",
    "build_bom",
    "build_bom_with_source",
    "components_to_bom",
    "detect_columns",
    "normalize_bom",
    "normalize_ref",
    "reconcile_bom",
    "extract_bom",
    "write_bom_csv",
]
