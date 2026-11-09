"""Извлечение текста, векторной геометрии и таблиц из PDF.

Модуль отвечает за низкоуровневое чтение страниц (ТЗ §2, §4):

* PyMuPDF — слова с bbox (``page.get_text("words")``) и векторные
  примитивы (``page.get_drawings()``);
* pdfplumber — таблицы перечня элементов (``page.find_tables()``);
* детекция растровых страниц без векторного слоя.

Ошибки открытия/разбора не пробрасываются наружу: повреждённый PDF
возвращает :class:`PdfExtraction` с ``Issue(type="parse_error")`` и
страницами, которые удалось прочитать (ТЗ §11).

ВАЖНО: PyMuPDF распространяется под AGPL-3.0 — см. README.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import pymupdf  # noqa: F401  (PyMuPDF)

from ..contract import Issue

#: Допуск замыкания контура при классификации окружности, pt.
_CIRCLE_CLOSE_TOL_PT = 1.0
#: Допустимое отклонение ширины/высоты bbox окружности (доля).
_CIRCLE_ASPECT_TOL = 0.15
#: Минимальное число кривых Безье для замкнутой окружности.
_CIRCLE_MIN_CURVES = 4


@dataclass
class TextSpan:
    """Фрагмент текста (слово) с координатами.

    Attributes:
        text: Содержимое фрагмента.
        x0: Левая граница bbox, pt.
        y0: Верхняя граница bbox, pt.
        x1: Правая граница bbox, pt.
        y1: Нижняя граница bbox, pt.
        page: Номер страницы (1-based).
    """

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page: int

    @property
    def cx(self) -> float:
        """Центр bbox по X."""
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        """Центр bbox по Y."""
        return (self.y0 + self.y1) / 2.0


@dataclass
class VectorPrimitive:
    """Векторный примитив страницы (линия, дуга, окружность, прямоугольник).

    Attributes:
        kind: Одно из ``line``, ``arc``, ``circle``, ``rect``, ``curve``,
            ``fill``.
        points: Список точек ``(x, y)``, pt. Для ``rect`` — углы, для
            ``fill`` — углы bbox.
        page: Номер страницы (1-based).
        attributes: Дополнительные свойства: ``bbox`` (x0, y0, x1, y1),
            ``width``, ``color``, ``fill``, ``center``, ``radius``.
    """

    kind: str
    points: list[tuple[float, float]] = field(default_factory=list)
    page: int = 1
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Возвращает bbox примитива как ``(x0, y0, x1, y1)``."""
        raw = self.attributes.get("bbox")
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        xs = [p[0] for p in self.points] or [0.0]
        ys = [p[1] for p in self.points] or [0.0]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class TableCell:
    """Ячейка таблицы с текстом и bbox."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    row: int
    col: int


@dataclass
class TableData:
    """Таблица страницы (например, перечень элементов)."""

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    cells: list[TableCell] = field(default_factory=list)


@dataclass
class PageExtraction:
    """Результат извлечения одной страницы."""

    page: int
    width: float
    height: float
    texts: list[TextSpan] = field(default_factory=list)
    primitives: list[VectorPrimitive] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    is_raster: bool = False
    raster_only: bool = False


@dataclass
class PdfExtraction:
    """Результат извлечения всего PDF.

    Attributes:
        pdf_path: Путь к PDF-файлу.
        pages: Успешно прочитанные страницы.
        issues: Проблемы извлечения (``parse_error``, ``raster_only_page``,
            ``unreadable_table`` и т.п.).
        opened: Удалось ли открыть документ. ``False`` для пустого или
            полностью повреждённого файла.
    """

    pdf_path: str
    pages: list[PageExtraction] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    opened: bool = True

    @property
    def page_count(self) -> int:
        """Число прочитанных страниц."""
        return len(self.pages)

    def all_texts(self) -> list[TextSpan]:
        """Все текстовые фрагменты документа."""
        return [text for page in self.pages for text in page.texts]

    def all_primitives(self) -> list[VectorPrimitive]:
        """Все векторные примитивы документа."""
        return [prim for page in self.pages for prim in page.primitives]

    def all_tables(self) -> list[TableData]:
        """Все таблицы документа."""
        return [table for page in self.pages for table in page.tables]


def _rect_bbox(rect: Any) -> tuple[float, float, float, float]:
    """Преобразует ``pymupdf.Rect``/``Quad`` в кортеж bbox."""
    if hasattr(rect, "x0"):
        return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
    points = [rect.ul, rect.ur, rect.ll, rect.lr] if hasattr(rect, "ul") else []
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [float(p.x) for p in points]
    ys = [float(p.y) for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_of_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """bbox по списку точек."""
    xs = [p[0] for p in points] or [0.0]
    ys = [p[1] for p in points] or [0.0]
    return (min(xs), min(ys), max(xs), max(ys))


def _point_near(
    a: tuple[float, float], b: tuple[float, float], tol: float
) -> bool:
    """Расстояние между точками не больше ``tol``."""
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tol


def _classify_curves(
    curves: list[tuple[tuple[float, float], ...]], page: int
) -> VectorPrimitive:
    """Классифицирует набор кривых Безье как окружность, дугу или кривую.

    Args:
        curves: Список сегментов ``(p0, p1, p2, p3)`` одной фигуры.
        page: Номер страницы.

    Returns:
        Векторный примитив ``circle`` / ``arc`` / ``curve``.
    """
    all_points: list[tuple[float, float]] = []
    for curve in curves:
        all_points.extend((float(p[0]), float(p[1])) for p in curve)
    bbox = _bbox_of_points(all_points)
    x0, y0, x1, y1 = bbox
    width, height = x1 - x0, y1 - y0
    attributes: dict[str, Any] = {"bbox": bbox, "width": len(curves)}

    closed = False
    if len(curves) >= _CIRCLE_MIN_CURVES:
        start = (float(curves[0][0][0]), float(curves[0][0][1]))
        end = (float(curves[-1][3][0]), float(curves[-1][3][1]))
        closed = _point_near(start, end, _CIRCLE_CLOSE_TOL_PT)

    if closed and width > 0 and height > 0:
        aspect = abs(width - height) / max(width, height)
        if aspect <= _CIRCLE_ASPECT_TOL:
            center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            radius = (width + height) / 4.0
            attributes.update({"center": center, "radius": radius})
            return VectorPrimitive("circle", [center], page, attributes)

    if len(curves) == 1:
        curve = curves[0]
        attributes["bbox"] = bbox
        return VectorPrimitive(
            "curve",
            [(float(curve[0][0]), float(curve[0][1])),
             (float(curve[3][0]), float(curve[3][1]))],
            page,
            attributes,
        )

    attributes["bbox"] = bbox
    return VectorPrimitive("arc", [(float(curves[0][0][0]), float(curves[0][0][1]))],
                           page, attributes)


def _drawing_to_primitives(drawing: dict[str, Any], page: int) -> list[VectorPrimitive]:
    """Преобразует один элемент ``page.get_drawings()`` в примитивы.

    Одна фигура может содержать несколько элементов ``items``; окружность
    (4+ замкнутые кривые) сворачивается в один примитив ``circle``.
    """
    items = drawing.get("items") or []
    primitives: list[VectorPrimitive] = []
    curve_group: list[tuple[tuple[float, float], ...]] = []

    def flush_curves() -> None:
        """Сбрасывает накопленную группу кривых в примитив."""
        if curve_group:
            primitives.append(_classify_curves(curve_group, page))
            curve_group.clear()

    stroke_width = drawing.get("width")
    color = drawing.get("color")
    fill = drawing.get("fill")

    for item in items:
        op = item[0]
        if op == "l":
            flush_curves()
            p0, p1 = item[1], item[2]
            points = [
                (float(p0.x), float(p0.y)),
                (float(p1.x), float(p1.y)),
            ]
            primitives.append(
                VectorPrimitive(
                    "line",
                    points,
                    page,
                    {"bbox": _bbox_of_points(points), "width": stroke_width},
                )
            )
        elif op in ("re", "qu"):
            flush_curves()
            bbox = _rect_bbox(item[1])
            x0, y0, x1, y1 = bbox
            points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            primitives.append(
                VectorPrimitive(
                    "rect",
                    points,
                    page,
                    {
                        "bbox": bbox,
                        "width": stroke_width,
                        "color": color,
                        "fill": fill,
                        "filled": drawing.get("type") in ("f", "fs"),
                    },
                )
            )
        elif op == "c":
            curve = tuple(
                (float(item[i].x), float(item[i].y)) for i in range(1, 5)
            )
            curve_group.append(curve)
        else:  # неизвестная операция — пропускаем без падения страницы
            flush_curves()
    flush_curves()

    if drawing.get("type") in ("f", "fs") and not primitives:
        rect = drawing.get("rect")
        if rect is not None:
            bbox = _rect_bbox(rect)
            primitives.append(
                VectorPrimitive("fill", [(bbox[0], bbox[1])], page,
                                {"bbox": bbox, "fill": fill})
            )
    return primitives


def _extract_page_words(page: Any, page_number: int) -> list[TextSpan]:
    """Извлекает слова страницы с bbox."""
    texts: list[TextSpan] = []
    for word in page.get_text("words"):
        if len(word) < 5:
            continue
        text = str(word[4]).strip()
        if not text:
            continue
        texts.append(
            TextSpan(
                text=text,
                x0=float(word[0]),
                y0=float(word[1]),
                x1=float(word[2]),
                y1=float(word[3]),
                page=page_number,
            )
        )
    return texts


def _extract_page_primitives(page: Any, page_number: int) -> list[VectorPrimitive]:
    """Извлекает векторные примитивы страницы."""
    primitives: list[VectorPrimitive] = []
    for drawing in page.get_drawings():
        primitives.extend(_drawing_to_primitives(drawing, page_number))
    return primitives


def _extract_tables_from_pdf(
    pdf_path: str, issues: list[Issue]
) -> dict[int, list[TableData]]:
    """Извлекает таблицы pdfplumber, индексируя их по номеру страницы.

    Ошибки чтения таблиц не прерывают обработку: добавляется
    ``Issue(type="unreadable_table")``.
    """
    try:
        import pdfplumber
    except ImportError:  # pragma: no cover - зависимость обязательна
        issues.append(
            Issue(
                type="parse_error",
                description="pdfplumber недоступен: таблицы не извлечены",
                required_action="Установить pdfplumber",
            )
        )
        return {}

    result: dict[int, list[TableData]] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                try:
                    found = page.find_tables()
                except Exception as exc:  # noqa: BLE001 - pdfplumber разнороден
                    issues.append(
                        Issue(
                            type="unreadable_table",
                            page=index,
                            description=f"Не удалось найти таблицы: {exc}",
                            confidence=0.4,
                            required_action="Проверить таблицу вручную/VLM",
                        )
                    )
                    continue
                tables: list[TableData] = []
                for table in found:
                    try:
                        rows = table.extract()
                        data = TableData(
                            page=index,
                            x0=float(table.bbox[0]),
                            y0=float(table.bbox[1]),
                            x1=float(table.bbox[2]),
                            y1=float(table.bbox[3]),
                        )
                        for r_index, row in enumerate(table.rows):
                            for c_index, cell in enumerate(row.cells):
                                if cell is None or r_index >= len(rows):
                                    continue
                                if c_index >= len(rows[r_index]):
                                    continue
                                text = rows[r_index][c_index]
                                if not text:
                                    continue
                                data.cells.append(
                                    TableCell(
                                        text=str(text).strip(),
                                        x0=float(cell[0]),
                                        y0=float(cell[1]),
                                        x1=float(cell[2]),
                                        y1=float(cell[3]),
                                        row=r_index,
                                        col=c_index,
                                    )
                                )
                        tables.append(data)
                    except Exception as exc:  # noqa: BLE001
                        issues.append(
                            Issue(
                                type="unreadable_table",
                                page=index,
                                description=f"Ошибка чтения таблицы: {exc}",
                                confidence=0.4,
                                required_action="Проверить таблицу вручную/VLM",
                            )
                        )
                result[index] = tables
    except Exception as exc:  # noqa: BLE001
        issues.append(
            Issue(
                type="unreadable_table",
                description=f"Таблицы не извлечены: {exc}",
                confidence=0.3,
                required_action="Проверить PDF вручную",
            )
        )
    return result


def _extract_one_page(
    page: Any, page_number: int, tables: list[TableData], issues: list[Issue]
) -> PageExtraction:
    """Извлекает одну открытую страницу PyMuPDF."""
    width = float(page.rect.width)
    height = float(page.rect.height)
    texts = _extract_page_words(page, page_number)
    primitives = _extract_page_primitives(page, page_number)
    has_images = bool(page.get_images(full=True))
    raster_only = not primitives and has_images
    if raster_only:
        issues.append(
            Issue(
                type="raster_only_page",
                page=page_number,
                description="Страница не содержит векторных примитивов, но содержит изображение",
                confidence=1.0,
                required_action="OCR required",
            )
        )
    return PageExtraction(
        page=page_number,
        width=width,
        height=height,
        texts=texts,
        primitives=primitives,
        tables=tables,
        is_raster=raster_only,
        raster_only=raster_only,
    )


def extract_pdf(
    pdf_path: str, doc_timeout_s: float | None = None
) -> PdfExtraction:
    """Извлекает текст, примитивы и таблицы всех страниц PDF.

    Повреждённый PDF не приводит к исключению: возвращается результат с
    ``Issue(type="parse_error")`` и успешно прочитанными страницами.
    Ошибка отдельной страницы изолируется: ``Issue(type="page_failed")``,
    остальные страницы продолжают обрабатываться (ТЗ §11, этап 10).

    Args:
        pdf_path: Путь к PDF-файлу.
        doc_timeout_s: Общий таймаут документа, с. Проверяется между
            страницами (мягко); ``None`` — без ограничения.

    Returns:
        Результат извлечения документа.
    """
    extraction = PdfExtraction(pdf_path=pdf_path)
    try:
        document = pymupdf.open(pdf_path)
    except pymupdf.EmptyFileError:
        # Пустой файл — не PDF; отдельная деградация без parse_error.
        extraction.opened = False
        extraction.issues.append(
            Issue(
                type="empty_file",
                description="PDF-файл пуст",
                confidence=1.0,
                required_action="Передать непустой PDF",
            )
        )
        return extraction
    except Exception as exc:  # noqa: BLE001 - файл может быть любым
        extraction.opened = False
        extraction.issues.append(
            Issue(
                type="parse_error",
                description=f"Не удалось открыть PDF: {exc}",
                confidence=1.0,
                required_action="Проверить целостность PDF",
            )
        )
        return extraction

    tables_by_page = _extract_tables_from_pdf(pdf_path, extraction.issues)
    started = time.monotonic()
    try:
        for index in range(document.page_count):
            page_number = index + 1
            try:
                page = document.load_page(index)
                extraction.pages.append(
                    _extract_one_page(
                        page,
                        page_number,
                        tables_by_page.get(page_number, []),
                        extraction.issues,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - не падать из-за страницы
                extraction.issues.append(
                    Issue(
                        type="page_failed",
                        page=page_number,
                        description=f"Ошибка чтения страницы: {exc}",
                        confidence=1.0,
                        required_action="Проверить страницу вручную",
                    )
                )
            # Мягкий общий таймаут: не прерываем страницу посреди обработки,
            # но не начинаем следующую после истечения бюджета времени.
            if (
                doc_timeout_s is not None
                and time.monotonic() - started > doc_timeout_s
            ):
                extraction.issues.append(
                    Issue(
                        type="doc_timeout",
                        page=page_number,
                        description=(
                            "Превышен общий таймаут документа; оставшиеся "
                            "страницы не обработаны"
                        ),
                        confidence=1.0,
                        required_action="Повторить обработку частично",
                    )
                )
                break
    finally:
        document.close()
    return extraction


def extract_page(pdf_path: str, page: int) -> PageExtraction:
    """Извлекает одну страницу PDF (текст, примитивы, таблицы).

    Args:
        pdf_path: Путь к PDF-файлу.
        page: Номер страницы (1-based).

    Returns:
        Извлечённые данные страницы. При ошибке — пустая страница.
    """
    if page < 1:
        return PageExtraction(page=page, width=0.0, height=0.0)
    try:
        document = pymupdf.open(pdf_path)
    except Exception:  # noqa: BLE001
        return PageExtraction(page=page, width=0.0, height=0.0)
    try:
        if page > document.page_count:
            return PageExtraction(page=page, width=0.0, height=0.0)
        pdf_page = document.load_page(page - 1)
        issues: list[Issue] = []
        tables = _extract_tables_from_pdf(pdf_path, issues).get(page, [])
        return _extract_one_page(pdf_page, page, tables, issues)
    except Exception:  # noqa: BLE001
        return PageExtraction(page=page, width=0.0, height=0.0)
    finally:
        document.close()


def detect_raster_pdf(pdf_path: str) -> bool:
    """Определяет, что все страницы PDF растровые без векторного слоя.

    Args:
        pdf_path: Путь к PDF-файлу.

    Returns:
        ``True``, если документ открыт, содержит страницы и каждая страница
        распознана как растровая.
    """
    extraction = extract_pdf(pdf_path)
    if not extraction.opened or not extraction.pages:
        return False
    return all(page.raster_only for page in extraction.pages)


__all__ = [
    "TextSpan",
    "VectorPrimitive",
    "TableCell",
    "TableData",
    "PageExtraction",
    "PdfExtraction",
    "extract_pdf",
    "extract_page",
    "detect_raster_pdf",
]
