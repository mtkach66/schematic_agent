"""Геометрическая обработка векторных примитивов.

Этап 3 конвейера (ТЗ §4):

* преобразование примитивов в отрезки (прямоугольники → 4 стороны);
* исключение рамок листа и таблиц (большие прямоугольники);
* кластеризация концов отрезков в узлы (union-find, допуск из
  :class:`~schematic_agent.config.Tolerances`);
* фиксация X-образных пересечений без общего узла (развязка);
* привязка текстовых меток цепей к узлам.

Пересечение отрезков без общей точки в пределах допуска НЕ создаёт
электрическую связь (в схемах пересечение без точки — развязка).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .config import Tolerances, get_settings
from .contract import Node
from .parser.pdf_extract import PdfExtraction, TextSpan, VectorPrimitive
from .spatial import (
    build_grid,
    iter_bbox_overlap_index_pairs,
    neighbors_in_tolerance,
)

#: Регулярное выражение метки цепи: L1, N, PE, 24V (без пробелов).
_LABEL_RE = re.compile(
    r"^([A-ZА-Я]{1,4})([0-9]{0,3})$|^[0-9]{1,3}[A-ZА-Я]{1,3}$"
)

#: Префиксы позиционных обозначений, которые НЕ считаются метками цепей.
#: ``L`` намеренно отсутствует: ``L1``/``L2``/``L3`` в схемах — фазы сети,
#: а не индуктивности (З1 smoke-теста).
_COMPONENT_REF_PREFIXES: frozenset[str] = frozenset(
    {
        "R", "C", "K", "KM", "KA", "KT", "VD", "VT", "Q", "F",
        "S", "SA", "SB", "SQ", "B", "H", "HL", "X", "XT", "U", "Y",
    }
)


@dataclass
class Segment:
    """Нормализованный отрезок в координатах страницы.

    Attributes:
        x0: Координата X начала, pt.
        y0: Координата Y начала, pt.
        x1: Координата X конца, pt.
        y1: Координата Y конца, pt.
        page: Номер страницы (1-based).
        primitive_index: Индекс исходного примитива страницы (или ``-1``).
            Нужен этапу Netlist, чтобы исключить внутреннюю геометрию УГО
            из графа электрических цепей.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    page: int
    primitive_index: int = -1

    @property
    def points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Концы отрезка."""
        return ((self.x0, self.y0), (self.x1, self.y1))

    @property
    def length(self) -> float:
        """Длина отрезка."""
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)


@dataclass
class Frame:
    """Рамка листа/таблицы, исключённая из сегментов цепей."""

    x0: float
    y0: float
    x1: float
    y1: float
    page: int


@dataclass
class GeometryResult:
    """Результат геометрического этапа.

    Attributes:
        segments: Нормализованные отрезки.
        nodes: Узлы (точки соединения концов отрезков).
        crossings: X-образные пересечения без электрической связи.
        frames: Исключённые рамки листа/таблиц.
        curves: Окружности/дуги/кривые (кандидаты в УГО на этапе 4).
        segment_nodes: Пары идентификаторов узлов на концах сегментов
            (порядок совпадает с :attr:`segments`).
    """

    segments: list[Segment] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    crossings: list[Node] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)
    curves: list[VectorPrimitive] = field(default_factory=list)
    junctions: list[Node] = field(default_factory=list)
    segment_nodes: list[tuple[str, str]] = field(default_factory=list)


class _UnionFind:
    """Простой union-find для кластеризации концов отрезков."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, item: int) -> int:
        """Возвращает корень множества."""
        parent = self._parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        """Объединяет два множества."""
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self._rank[root_left] < self._rank[root_right]:
            root_left, root_right = root_right, root_left
        self._parent[root_right] = root_left
        if self._rank[root_left] == self._rank[root_right]:
            self._rank[root_left] += 1


def _rect_is_frame(
    bbox: tuple[float, float, float, float], frame_min_pt: float
) -> bool:
    """Прямоугольник считается рамкой, если велик хотя бы по одной оси."""
    return (bbox[2] - bbox[0]) >= frame_min_pt or (bbox[3] - bbox[1]) >= frame_min_pt


def _inside_any(
    bbox: tuple[float, float, float, float],
    regions: list[tuple[float, float, float, float]],
) -> bool:
    """Центр bbox лежит внутри одной из областей (таблиц)."""
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return any(
        x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in regions
    )


def _rect_segments(
    primitive: VectorPrimitive, page: int, primitive_index: int = -1
) -> list[Segment]:
    """Разбивает прямоугольник на 4 стороны."""
    x0, y0, x1, y1 = primitive.bbox
    return [
        Segment(x0, y0, x1, y0, page, primitive_index),
        Segment(x1, y0, x1, y1, page, primitive_index),
        Segment(x1, y1, x0, y1, page, primitive_index),
        Segment(x0, y1, x0, y0, page, primitive_index),
    ]


def merge_segments(
    primitives: list[VectorPrimitive],
    tolerances: Tolerances | None = None,
    skip_indices: set[int] | None = None,
) -> list[Segment]:
    """Преобразует примитивы в отрезки с учётом рамок.

    Args:
        primitives: Векторные примитивы страниц.
        tolerances: Допуски; по умолчанию из настроек.
        skip_indices: Индексы примитивов, исключаемых из отрезков
            (например, геометрия таблиц BOM).

    Returns:
        Нормализованные отрезки (без сторон рамок).
    """
    tol = tolerances or get_settings().tolerances
    skip = skip_indices or set()
    segments: list[Segment] = []
    for primitive_index, primitive in enumerate(primitives):
        if primitive_index in skip:
            continue
        page = primitive.page
        if primitive.kind == "line" and len(primitive.points) >= 2:
            (x0, y0), (x1, y1) = primitive.points[0], primitive.points[1]
            segments.append(
                Segment(
                    float(x0), float(y0), float(x1), float(y1), page,
                    primitive_index,
                )
            )
        elif primitive.kind == "rect":
            if _rect_is_frame(primitive.bbox, tol.frame_min_pt):
                continue
            segments.extend(_rect_segments(primitive, page, primitive_index))
    return segments


def _endpoints(segments: list[Segment]) -> list[tuple[int, int, float, float, int]]:
    """Список концов ``(segment_index, end_index, x, y, page)``."""
    endpoints: list[tuple[int, int, float, float, int]] = []
    for index, segment in enumerate(segments):
        endpoints.append((index, 0, segment.x0, segment.y0, segment.page))
        endpoints.append((index, 1, segment.x1, segment.y1, segment.page))
    return endpoints


def _cluster_endpoints(
    segments: list[Segment], tolerance: float
) -> tuple[list[Node], dict[tuple[int, int], str]]:
    """Кластеризует концы отрезков в узлы union-find.

    Returns:
        Узлы и отображение ``(segment_index, end_index) → node_id``.
    """
    endpoints = _endpoints(segments)
    uf = _UnionFind(len(endpoints))
    # Сетка концов: объединяем только пары в допуске, не перебирая все
    # пары документа. Страницы проверяются явно: точки с разных листов
    # не должны попадать в один узел.
    grid = build_grid(
        [
            (x, y, (index, page))
            for index, (_, _, x, y, page) in enumerate(endpoints)
        ],
        max(tolerance, 1.0),
    )
    for i, (_, _, x_i, y_i, page_i) in enumerate(endpoints):
        for _x, _y, payload in neighbors_in_tolerance(
            (x_i, y_i), grid, tolerance
        ):
            j, page_j = payload
            if j <= i or page_j != page_i:
                continue
            uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for index in range(len(endpoints)):
        clusters.setdefault(uf.find(index), []).append(index)

    nodes: list[Node] = []
    endpoint_map: dict[tuple[int, int], str] = {}
    for number, members in enumerate(
        sorted(clusters.values(), key=lambda group: min(group)), start=1
    ):
        xs = [endpoints[m][2] for m in members]
        ys = [endpoints[m][3] for m in members]
        pages = {endpoints[m][4] for m in members}
        node_id = f"n{number:03d}"
        node = Node(
            id=node_id,
            x=sum(xs) / len(xs),
            y=sum(ys) / len(ys),
            page=min(pages),
            kind="junction",
            attributes={"endpoint_count": len(members)},
        )
        nodes.append(node)
        for member in members:
            endpoint_map[(endpoints[member][0], endpoints[member][1])] = node_id
    return nodes, endpoint_map


def _orientation(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float
) -> float:
    """Знак векторного произведения AB × AC."""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _segment_intersection(
    first: Segment, second: Segment
) -> tuple[float, float] | None:
    """Возвращает точку X-образного пересечения или ``None``.

    Пересечение в концах отрезков не считается развязкой (там может быть
    узел), коллинеарные отрезки также игнорируются.
    """
    ax, ay = first.x0, first.y0
    bx, by = first.x1, first.y1
    cx, cy = second.x0, second.y0
    dx, dy = second.x1, second.y1

    denominator = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
    if abs(denominator) < 1e-9:
        return None

    t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denominator
    u = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / denominator
    if not (1e-6 < t < 1 - 1e-6 and 1e-6 < u < 1 - 1e-6):
        return None
    return (ax + t * (bx - ax), ay + t * (by - ay))


#: Размер ячейки сетки bbox при поиске пересечений, pt.
_CROSSING_GRID_CELL_PT = 50.0


def _find_crossings(
    segments: list[Segment], tolerance: float
) -> list[Node]:
    """Находит X-образные пересечения без общего узла.

    Пары предфильтруются по перекрытию bbox через
    :func:`schematic_agent.spatial.iter_bbox_overlap_index_pairs`; точная
    проверка пересечения и семантика развязки не меняются. Пары
    обрабатываются в порядке ``(i, j)``, как в прежнем двойном цикле,
    поэтому состав и порядок ``crossings`` идентичны.
    """
    crossings: list[Node] = []
    seen: set[tuple[int, int, int, int]] = set()
    bboxes = [
        (
            min(segment.x0, segment.x1),
            min(segment.y0, segment.y1),
            max(segment.x0, segment.x1),
            max(segment.y0, segment.y1),
        )
        for segment in segments
    ]
    items = [(bbox, None) for bbox in bboxes]
    for i, j in sorted(
        iter_bbox_overlap_index_pairs(items, _CROSSING_GRID_CELL_PT)
    ):
        first, second = segments[i], segments[j]
        if first.page != second.page:
            continue
        point = _segment_intersection(first, second)
        if point is None:
            continue
        # Пересечение в пределах допуска от конца — это узел, не развязка.
        if any(
            math.hypot(point[0] - x, point[1] - y) <= tolerance
            for x, y in (*first.points, *second.points)
        ):
            continue
        key = (first.page, round(point[0], 1), round(point[1], 1), 0)
        if key in seen:
            continue
        seen.add(key)
        crossings.append(
            Node(
                id=f"x{len(crossings) + 1:03d}",
                x=point[0],
                y=point[1],
                page=first.page,
                kind="crossing",
                attributes={"connected": False},
            )
        )
    return crossings


def _is_label(text: str) -> bool:
    """Проверяет, похоже ли слово на метку цепи.

    Слова с префиксом известного позиционного обозначения (``R1``, ``C2``,
    ``VT1``...) метками не считаются: иначе они перестали бы привязываться
    к компонентам (З1).
    """
    match = _LABEL_RE.match(text)
    if not match:
        return False
    prefix = match.group(1)
    if prefix is None:
        return True  # 24V, 0V и подобные метки с цифровым началом
    return prefix not in _COMPONENT_REF_PREFIXES


def _attach_labels(
    nodes: list[Node], texts: list[TextSpan], snap_pt: float
) -> None:
    """Привязывает текстовые метки цепей к ближайшим узлам.

    В ``attributes["labels"]`` сохраняются строки меток, в
    ``attributes["label_spans"]`` — координаты исходных слов. Спаны нужны
    этапу 4: слова-метки у узлов нельзя считать позиционными обозначениями
    компонентов (З1 smoke-теста).
    """
    for span in texts:
        if not _is_label(span.text):
            continue
        best: Node | None = None
        best_distance = snap_pt
        for node in nodes:
            if node.page != span.page:
                continue
            distance = math.hypot(node.x - span.cx, node.y - span.cy)
            if distance <= best_distance:
                best = node
                best_distance = distance
        if best is None:
            continue
        labels = best.attributes.setdefault("labels", [])
        if span.text not in labels:
            labels.append(span.text)
        label_spans = best.attributes.setdefault("label_spans", [])
        span_dict = {
            "text": span.text,
            "page": span.page,
            "x0": span.x0,
            "y0": span.y0,
            "x1": span.x1,
            "y1": span.y1,
            "cx": span.cx,
            "cy": span.cy,
        }
        if span_dict not in label_spans:
            label_spans.append(span_dict)


def build_geometry(
    extraction: PdfExtraction, tolerances: Tolerances | None = None
) -> GeometryResult:
    """Строит отрезки, узлы, развязки и метки из результата парсера.

    Args:
        extraction: Результат этапа 2.
        tolerances: Допуски; по умолчанию из настроек.

    Returns:
        Геометрическая модель документа.
    """
    tol = tolerances or get_settings().tolerances
    result = GeometryResult()

    for page in extraction.pages:
        table_regions = [
            (table.x0, table.y0, table.x1, table.y1) for table in page.tables
        ]
        # Геометрия внутри таблиц перечня элементов — не электрические цепи
        # и не УГО.
        skip_indices = {
            index
            for index, primitive in enumerate(page.primitives)
            if table_regions and _inside_any(primitive.bbox, table_regions)
        }
        for index, primitive in enumerate(page.primitives):
            if index in skip_indices:
                continue
            if primitive.kind == "rect" and _rect_is_frame(
                primitive.bbox, tol.frame_min_pt
            ):
                x0, y0, x1, y1 = primitive.bbox
                result.frames.append(Frame(x0, y0, x1, y1, page.page))
            elif primitive.kind in ("circle", "arc", "curve"):
                result.curves.append(primitive)
        result.segments.extend(
            merge_segments(page.primitives, tol, skip_indices)
        )

    nodes, endpoint_map = _cluster_endpoints(
        result.segments, tol.node_merge_pt
    )
    result.nodes = nodes
    result.junctions = [node for node in nodes if node.kind == "junction"]
    result.crossings = _find_crossings(result.segments, tol.node_merge_pt)
    result.segment_nodes = [
        (endpoint_map[(index, 0)], endpoint_map[(index, 1)])
        for index in range(len(result.segments))
    ]
    _attach_labels(result.nodes, extraction.all_texts(), tol.label_snap_pt)
    return result


def build_nodes(
    segments: list[Segment], tolerances: Tolerances | None = None
) -> GeometryResult:
    """Строит узлы и развязки из готовых отрезков.

    Args:
        segments: Нормализованные отрезки.
        tolerances: Допуски; по умолчанию из настроек.

    Returns:
        Геометрическая модель (без привязки меток).
    """
    tol = tolerances or get_settings().tolerances
    nodes, endpoint_map = _cluster_endpoints(segments, tol.node_merge_pt)
    return GeometryResult(
        segments=list(segments),
        nodes=nodes,
        junctions=[node for node in nodes if node.kind == "junction"],
        crossings=_find_crossings(segments, tol.node_merge_pt),
        segment_nodes=[
            (endpoint_map[(index, 0)], endpoint_map[(index, 1)])
            for index in range(len(segments))
        ],
    )


def detect_pins(
    primitives: list[VectorPrimitive],
    geometry: GeometryResult | None = None,
    tolerances: Tolerances | None = None,
) -> list[Node]:
    """Определяет узлы на выводах компонентов.

    Без этапа 4 (компонентов) функция возвращает все узлы, лежащие на
    концах отрезков; окончательная привязка выводов выполняется в
    :mod:`schematic_agent.components`.

    Args:
        primitives: Векторные примитивы страниц.
        geometry: Готовая геометрия; если не задана — строится из примитивов.
        tolerances: Допуски; по умолчанию из настроек.

    Returns:
        Узлы-выводы (все узлы геометрии).
    """
    tol = tolerances or get_settings().tolerances
    if geometry is not None:
        return list(geometry.nodes)
    return list(build_nodes(merge_segments(primitives, tol), tol).nodes)


__all__ = [
    "Segment",
    "Frame",
    "Node",
    "GeometryResult",
    "merge_segments",
    "build_nodes",
    "detect_pins",
    "build_geometry",
]
