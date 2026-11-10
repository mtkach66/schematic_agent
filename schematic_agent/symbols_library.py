"""Загрузчик rich-библиотеки УГО (``symbols_library.yaml``).

Библиотека описана в формате экспорта EPlan/Owen Logic::

    symbols:
      - name: S
        variant: '0'
        heuristic: switch
        bounding_box: {width: 4.0, height: 8.0}
        connection_points:
          - {designation: '', index: 0, direction: Up, x: 0.0, y: 4.0,
             x_rel: 0.0, y_rel: 0.5}

Одно имя (``name``) имеет несколько вариантов (``variant``), различающихся
габаритами и точками подключения. Загрузчик агрегирует записи по имени:

* :class:`SymbolVariantSpec` — вариант: bbox, ожидаемое число выводов,
  направления точек подключения и эвристическая категория;
* :class:`SymbolSpec` — имя: набор категорий и словарь вариантов.

Секция верхнего уровня ``ref_prefix_map`` (M6) сопоставляет префикс
позиционного обозначения проекта (ГОСТ: ``QF``, ``KM``, ``XT``…) с
категорией ``heuristic`` или со списком имён УГО::

    ref_prefix_map:
      QF: breaker
      KM: [SCHB, SCHL]

Значение-строка — категория (ищутся все записи с этим ``heuristic``),
значение-список — точные имена УГО (приоритет выше категории).

Формат определяется по ключам записей: ``variant``/``heuristic`` — rich,
``category``/``geometry`` — M1-шаблон (см. ``components.SymbolConfig``).
M1-записи этим загрузчиком игнорируются: их обрабатывает
:func:`schematic_agent.components.load_symbols_config`.

Источник:

* ``SCHEMATIC_AGENT_SYMBOLS_FILE`` (если задан) — единственный источник,
  rich или M1 (M1 → пустая rich-библиотека);
* иначе пакетный ``symbols_library.yaml``, если файл существует.

Загрузка ленивая и кэшируется на уровне модуля (``lru_cache`` с ключом
«путь + mtime + размер»). Битый YAML не прерывает конвейер: формируется
:class:`schematic_agent.contract.Issue` ``symbols_config_error`` и пустая
библиотека.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from .config import SYMBOLS_FILE_ENV
from .contract import (
    Issue,
    SymbolBoundingBox,
    SymbolConnectionPoint,
    SymbolLibraryEntry,
)
from .issues import default_action

#: Тип результата определения формата файла.
SymbolsFileFormat = Literal["rich", "template", "unknown"]

#: Значение ``ref_prefix_map``: категория heuristic (строка) или точные
#: имена УГО (кортеж имён).
RefPrefixTarget = str | tuple[str, ...]

#: Обязательные ключи rich-записи.
_RICH_REQUIRED_KEYS = frozenset(
    {"name", "variant", "heuristic", "bounding_box", "connection_points"}
)

#: Обязательные ключи точки подключения.
_CP_REQUIRED_KEYS = frozenset(
    {"designation", "index", "direction", "x", "y", "x_rel", "y_rel"}
)

#: Ключи, значения которых — целые числа.
_INT_KEYS = frozenset({"index"})

#: Ключи, значения которых — числа с плавающей точкой.
_FLOAT_KEYS = frozenset(
    {"x", "y", "x_rel", "y_rel", "width", "height"}
)


class _FastParseError(ValueError):
    """Файл не соответствует ожидаемой rich-структуре."""


@dataclass(frozen=True, slots=True)
class SymbolVariantSpec:
    """Агрегированный вариант rich-УГО.

    Attributes:
        variant: Идентификатор варианта из библиотеки (``'0'``…``'16'``).
        bounding_box: Габариты УГО в единицах библиотеки.
        expected_pin_count: Ожидаемое число выводов (точек подключения).
        directions: Направления точек подключения.
        heuristic: Эвристическая категория записи (``switch``, ``coil``…).
    """

    variant: str
    bounding_box: SymbolBoundingBox
    expected_pin_count: int
    directions: tuple[str, ...]
    heuristic: str = "unknown"

    @property
    def aspect(self) -> float | None:
        """Отношение длинной стороны к короткой или ``None`` для нулей."""
        width = self.bounding_box.width
        height = self.bounding_box.height
        if width <= 0.0 or height <= 0.0:
            return None
        return max(width, height) / min(width, height)

    @property
    def sort_key(self) -> tuple[int, str]:
        """Устойчивый порядок вариантов: числовой, затем лексикографический."""
        try:
            return (int(self.variant), self.variant)
        except ValueError:
            return (10**9, self.variant)


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    """Агрегированное rich-УГО по имени (все варианты).

    Attributes:
        name: Имя УГО (совпадает с префиксом позиционного обозначения,
            например ``S`` для ``S1``).
        categories: Множество эвристических категорий записи.
        variants: Отображение «вариант → :class:`SymbolVariantSpec`».
    """

    name: str
    categories: frozenset[str] = frozenset()
    variants: dict[str, SymbolVariantSpec] = field(default_factory=dict)

    @property
    def variant_count(self) -> int:
        """Число вариантов УГО."""
        return len(self.variants)

    def match_variant(
        self, width: float, height: float
    ) -> SymbolVariantSpec | None:
        """Возвращает вариант, ближайший к габаритам компонента.

        Сравнивается отношение длинной стороны к короткой (ориентация не
        важна). При равенстве выбирается вариант с меньшим числовым
        идентификатором, чтобы результат был детерминированным.

        Args:
            width: Ширина bbox компонента, pt.
            height: Высота bbox компонента, pt.

        Returns:
            Ближайший вариант или ``None``, если вариантов нет.
        """
        if not self.variants:
            return None
        aspect: float | None = None
        if width > 0.0 and height > 0.0:
            aspect = max(width, height) / min(width, height)
        ordered = sorted(self.variants.values(), key=lambda item: item.sort_key)
        if aspect is None:
            return ordered[0]

        def distance(item: SymbolVariantSpec) -> tuple[float, tuple[int, str]]:
            item_aspect = item.aspect
            delta = (
                abs(item_aspect - aspect)
                if item_aspect is not None
                else float("inf")
            )
            return (delta, item.sort_key)

        return min(ordered, key=distance)


@dataclass
class LibrarySpec:
    """Результат загрузки rich-библиотеки УГО.

    Attributes:
        symbols: Отображение «имя → :class:`SymbolSpec`».
        ref_prefix_map: Таблица «ref-префикс → категория или имена УГО»
            (M6); пустая, если секции в файле нет.
        category_index: Индекс «категория heuristic → имена УГО» (M6):
            позволяет не сканировать всю библиотеку на каждый компонент.
        issues: Проблемы загрузки (``symbols_config_error``).
        source: Путь источника (пустая строка, если файла не было).
        exists: Признак существования файла-источника.
    """

    symbols: dict[str, SymbolSpec] = field(default_factory=dict)
    ref_prefix_map: dict[str, RefPrefixTarget] = field(default_factory=dict)
    category_index: dict[str, tuple[str, ...]] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    source: str = ""
    exists: bool = False

    @property
    def entry_count(self) -> int:
        """Суммарное число записей (имя × вариант)."""
        return sum(spec.variant_count for spec in self.symbols.values())

    def get(self, name: str) -> SymbolSpec | None:
        """Возвращает УГО по имени или ``None``."""
        return self.symbols.get(name)


def parse_ref_prefix_map(raw: Any) -> dict[str, RefPrefixTarget]:
    """Разбирает и валидирует секцию ``ref_prefix_map`` (M6).

    Формат: словарь «префикс позиционного обозначения → категория
    ``heuristic`` (строка) | список точных имён УГО». Префиксы
    нормализуются в верхний регистр, категории — в нижний (так они записаны
    в библиотеке). Пустые элементы списка отбрасываются.

    Args:
        raw: Значение секции из YAML (``None``, если секции нет).

    Returns:
        Нормализованная таблица (пустая, если секции нет).

    Raises:
        ValueError: Секция или отдельная запись имеют неверный формат.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("ref_prefix_map должен быть словарём")
    result: dict[str, RefPrefixTarget] = {}
    for key, value in raw.items():
        prefix = str(key).strip().upper()
        if not prefix or not prefix.isalnum():
            raise ValueError(
                f"некорректный ref-префикс в ref_prefix_map: {key!r}"
            )
        if isinstance(value, str):
            category = value.strip().lower()
            if not category:
                raise ValueError(
                    f"пустая категория в ref_prefix_map для {prefix!r}"
                )
            result[prefix] = category
            continue
        if isinstance(value, (list, tuple)):
            names = tuple(
                str(item).strip() for item in value if str(item).strip()
            )
            if not names:
                raise ValueError(
                    f"пустой список имён в ref_prefix_map для {prefix!r}"
                )
            result[prefix] = names
            continue
        raise ValueError(
            f"значение ref_prefix_map для {prefix!r} должно быть строкой "
            f"или списком имён: {value!r}"
        )
    return result


def default_symbols_library_path() -> Path:
    """Путь к пакетной rich-библиотеке ``symbols_library.yaml``."""
    return Path(__file__).with_name("symbols_library.yaml")


def _config_error(path: Path, error: str) -> Issue:
    """Формирует Issue о непригодной rich-библиотеке."""
    return Issue(
        type="symbols_config_error",
        description=(
            f"Не удалось загрузить symbols_library.yaml ({path}): {error}. "
            "Rich-обогащение отключено, встроенные детекторы работают"
        ),
        confidence=0.5,
        required_action=default_action("symbols_config_error"),
    )


def _scalar(raw: str) -> str:
    """Разбирает простой YAML-скаляр (строки в кавычках и без)."""
    value = raw.strip()
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1].replace("''", "'")
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('\\"', '"')
    if value in ("null", "Null", "NULL", "~"):
        return ""
    return value


def _typed_value(key: str, raw: str) -> Any:
    """Приводит значение rich-поля к числу или строке."""
    if key in _INT_KEYS:
        return int(_scalar(raw))
    if key in _FLOAT_KEYS:
        return float(_scalar(raw))
    return _scalar(raw)


def _parse_rich_yaml(text: str) -> dict[str, Any]:
    """Быстрый разбор машиночитаемого rich-YAML без общего парсера.

    PyYAML на 4.2 МБ / 249 тыс. строк занимает десятки секунд, что
    недопустимо для конвейера. Формат строго регулярен, поэтому используется
    линейный разбор по строкам. При любом отклонении от структуры
    поднимается :class:`_FastParseError` и вызывающий код переходит на
    полноценный YAML-парсер.

    Args:
        text: Содержимое rich-YAML.

    Returns:
        Словарь ``{"symbols": [...]}`` в структуре YAML.

    Raises:
        _FastParseError: Файл не соответствует ожидаемой структуре.
    """
    lines = text.splitlines()
    symbols: list[dict[str, Any]] = []
    prefix_map: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    section: str | None = None
    item: dict[str, Any] | None = None
    last_prefix: str | None = None
    started = False

    for raw_line in lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        stripped = raw_line.strip()
        if stripped == "symbols:":
            started = True
            section = None
            current = None
            item = None
            continue
        if stripped == "ref_prefix_map:":
            section = "prefix_map"
            current = None
            item = None
            last_prefix = None
            continue
        if not raw_line[:1].isspace() and ":" in stripped:
            key_name = stripped.split(":", 1)[0].strip()
            if key_name.isidentifier() and not stripped.startswith("- "):
                # Неизвестная корневая секция — переходим на общий парсер.
                raise _FastParseError(
                    f"неожиданная корневая секция: {stripped!r}"
                )
        if section == "prefix_map":
            if stripped.startswith("- "):
                if last_prefix is None:
                    raise _FastParseError(
                        "элемент списка без ключа в ref_prefix_map"
                    )
                prefix_map[last_prefix].append(_scalar(stripped[2:]))
                continue
            if ":" not in stripped:
                raise _FastParseError(
                    f"строка без ':' в ref_prefix_map — {stripped!r}"
                )
            key, value = stripped.split(":", 1)
            last_prefix = _scalar(key)
            inline = value.strip()
            if not inline:
                prefix_map[last_prefix] = []
            elif inline.startswith("[") and inline.endswith("]"):
                prefix_map[last_prefix] = [
                    _scalar(part)
                    for part in inline[1:-1].split(",")
                    if part.strip()
                ]
                last_prefix = None
            else:
                prefix_map[last_prefix] = _scalar(inline)
                last_prefix = None
            continue
        if not started:
            raise _FastParseError("ожидалась корневая секция symbols:")
        if stripped.startswith("- name:"):
            current = {
                "name": _scalar(stripped[len("- name:"):]),
                "connection_points": [],
            }
            symbols.append(current)
            section = None
            item = None
            continue
        if current is None:
            raise _FastParseError("запись без поля name")
        if stripped.startswith("- "):
            if section not in ("cp", "cp_item") or ":" not in stripped:
                raise _FastParseError(f"неожиданный элемент списка: {stripped}")
            key, value = stripped[2:].split(":", 1)
            key = key.strip()
            item = {key: _typed_value(key, value)}
            current["connection_points"].append(item)
            section = "cp_item"
            continue
        if stripped == "bounding_box:":
            current["bounding_box"] = {}
            section = "bbox"
            item = None
            continue
        if stripped == "connection_points:":
            section = "cp"
            item = None
            continue
        if ":" not in stripped:
            raise _FastParseError(f"строка без ':' — {stripped!r}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        if section == "bbox":
            current["bounding_box"][key] = _typed_value(key, value)
        elif section == "cp_item" and item is not None:
            item[key] = _typed_value(key, value)
        else:
            current[key] = _typed_value(key, value)

    if not started:
        raise _FastParseError("секция symbols: не найдена")
    for entry in symbols:
        if not _RICH_REQUIRED_KEYS <= set(entry):
            raise _FastParseError("неполная rich-запись")
        bbox = entry.get("bounding_box")
        if not isinstance(bbox, dict) or not {"width", "height"} <= set(bbox):
            raise _FastParseError("неполный bounding_box")
        for point in entry["connection_points"]:
            if not _CP_REQUIRED_KEYS <= set(point):
                raise _FastParseError("неполная connection_points")
    return {"symbols": symbols, "ref_prefix_map": prefix_map}


def _detect_text_format(text: str) -> SymbolsFileFormat:
    """Определяет формат файла по ключам первой записи.

    Args:
        text: Начало YAML-файла (достаточно первых записей).

    Returns:
        ``"rich"``, ``"template"`` или ``"unknown"``.
    """
    in_symbols = False
    in_prefix_map = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        is_root_key = (
            not raw_line[:1].isspace()
            and ":" in stripped
            and not stripped.startswith("- ")
        )
        if is_root_key:
            # Корневая секция: ref_prefix_map может идти до symbols.
            if stripped == "symbols:":
                in_symbols = True
                in_prefix_map = False
                continue
            if stripped == "ref_prefix_map:":
                in_prefix_map = True
                continue
            if in_symbols:
                # Секция после symbols (например, net_aliases) — записей
                # дальше нет, формат уже определён (или не определится).
                break
            return "unknown"
        if in_prefix_map:
            # Записи таблицы префиксов не участвуют в определении формата.
            continue
        if not in_symbols:
            return "unknown"
        key = stripped
        if key.startswith("- "):
            key = key[2:]
        if ":" not in key:
            continue
        name = key.split(":", 1)[0].strip()
        if name in ("variant", "heuristic"):
            return "rich"
        if name in ("category", "geometry", "ref_prefixes"):
            return "template"
    return "unknown"


def detect_symbols_format(path: str | os.PathLike[str]) -> SymbolsFileFormat:
    """Определяет формат файла библиотеки УГО по первым записям.

    Args:
        path: Путь к YAML-файлу.

    Returns:
        ``"rich"`` для rich-библиотеки, ``"template"`` для M1-шаблонов,
        ``"unknown"`` при ошибке чтения или нераспознанном формате.
    """
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            head = stream.read(65536)
    except OSError:
        return "unknown"
    return _detect_text_format(head)


def _parse_yaml_text(path: Path, text: str) -> dict[str, Any] | None:
    """Разбирает rich-YAML быстрым путём, иначе — общим парсером.

    Returns:
        Корневой словарь или ``None``, если файл непригоден.
    """
    try:
        return _parse_rich_yaml(text)
    except _FastParseError:
        pass
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    try:
        raw = yaml.load(text, Loader=loader)
    except yaml.YAMLError:
        return None
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        return None
    return raw


def _build_ref_prefix_map(
    raw: dict[str, Any], path: Path, issues: list[Issue]
) -> dict[str, RefPrefixTarget]:
    """Разбирает ``ref_prefix_map``, фиксируя битый формат в ``issues``."""
    try:
        return parse_ref_prefix_map(raw.get("ref_prefix_map"))
    except ValueError as error:
        issues.append(_config_error(path, str(error)))
        return {}


def _build_library(
    raw: dict[str, Any], path: Path
) -> tuple[
    dict[str, SymbolSpec],
    dict[str, RefPrefixTarget],
    dict[str, tuple[str, ...]],
    list[Issue],
]:
    """Собирает агрегированную библиотеку из разобранного YAML."""
    issues: list[Issue] = []
    ref_prefix_map = _build_ref_prefix_map(raw, path, issues)
    entries = raw.get("symbols") or []
    if not isinstance(entries, list):
        issues.append(_config_error(path, "symbols должен быть списком"))
        return {}, ref_prefix_map, {}, issues

    categories: dict[str, set[str]] = {}
    variants: dict[str, dict[str, SymbolVariantSpec]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            issues.append(_config_error(path, "запись должна быть словарём"))
            continue
        if _entry_format(entry) != "rich":
            # M1-шаблон: его обрабатывает components.load_symbols_config.
            continue
        try:
            parsed = SymbolLibraryEntry.model_validate(entry)
        except ValidationError as error:
            issues.append(_config_error(path, error.json()))
            continue
        categories.setdefault(parsed.name, set()).add(parsed.heuristic)
        variant = SymbolVariantSpec(
            variant=parsed.variant,
            bounding_box=parsed.bounding_box,
            expected_pin_count=len(parsed.connection_points),
            directions=tuple(
                point.direction for point in parsed.connection_points
            ),
            heuristic=parsed.heuristic,
        )
        variants.setdefault(parsed.name, {})[parsed.variant] = variant

    symbols = {
        name: SymbolSpec(
            name=name,
            categories=frozenset(categories.get(name, set())),
            variants=variants[name],
        )
        for name in variants
    }
    index: dict[str, list[str]] = {}
    for name in sorted(symbols):
        for category in symbols[name].categories:
            index.setdefault(category, []).append(name)
    category_index = {
        category: tuple(names) for category, names in index.items()
    }
    return symbols, ref_prefix_map, category_index, issues


def _entry_format(entry: dict[str, Any]) -> SymbolsFileFormat:
    """Определяет формат одной записи по её ключам."""
    if "variant" in entry or "heuristic" in entry:
        return "rich"
    if "category" in entry or "geometry" in entry or "ref_prefixes" in entry:
        return "template"
    return "unknown"


@lru_cache(maxsize=8)
def _load_cached(
    cache_key: tuple[str, int, int, bool],
    path_text: str,
    explicit: bool,
) -> LibrarySpec:
    """Загружает и кэширует библиотеку по ключу «путь + mtime + размер»."""
    path = Path(path_text)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        issues = [_config_error(path, str(error))] if explicit else []
        return LibrarySpec(issues=issues, source=path_text, exists=False)

    raw = _parse_yaml_text(path, text)
    if raw is None:
        return LibrarySpec(
            issues=[_config_error(path, "некорректный YAML")],
            source=path_text,
            exists=True,
        )
    if _detect_text_format(text) != "rich":
        # Формат M1 (или пустой/неизвестный): rich-записей нет.
        return LibrarySpec(source=path_text, exists=True)
    symbols, ref_prefix_map, category_index, issues = _build_library(raw, path)
    return LibrarySpec(
        symbols=symbols,
        ref_prefix_map=ref_prefix_map,
        category_index=category_index,
        issues=issues,
        source=path_text,
        exists=True,
    )


def load_symbols_library(
    path: str | os.PathLike[str] | None = None,
) -> LibrarySpec:
    """Загружает rich-библиотеку УГО (лениво, с кэшем на процесс).

    Порядок источников:

    1. явный ``path``;
    2. ``SCHEMATIC_AGENT_SYMBOLS_FILE`` — единственный источник (rich или M1);
    3. пакетный ``symbols_library.yaml``, если существует.

    Args:
        path: Необязательный явный путь к YAML-файлу.

    Returns:
        :class:`LibrarySpec` с агрегированными УГО и проблемами загрузки.
    """
    explicit = path is not None or bool(os.environ.get(SYMBOLS_FILE_ENV))
    if path is not None:
        source = Path(path)
    else:
        env_path = os.environ.get(SYMBOLS_FILE_ENV)
        source = (
            Path(env_path)
            if env_path
            else default_symbols_library_path()
        )
    try:
        stat = source.stat()
        mtime_ns, size = stat.st_mtime_ns, stat.st_size
    except OSError:
        mtime_ns, size = 0, 0
    key = (str(source), mtime_ns, size, explicit)
    return _load_cached(key, str(source), explicit)


__all__ = [
    "SymbolsFileFormat",
    "RefPrefixTarget",
    "SymbolVariantSpec",
    "SymbolSpec",
    "LibrarySpec",
    "default_symbols_library_path",
    "detect_symbols_format",
    "load_symbols_library",
    "parse_ref_prefix_map",
]
