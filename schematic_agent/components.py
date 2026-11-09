"""Распознавание компонентов и позиционных обозначений.

Этап 4 конвейера (ТЗ §2, §10). Шаблоны УГО — эвристики по векторным
примитивам ``page.get_drawings()``; каждый шаблон — отдельная функция в
словаре :data:`SYMBOL_TEMPLATES`. Неопознанные скопления линий рядом с
позиционным обозначением превращаются в ``Component(type="unknown")`` и
``Issue(type="unrecognized_symbol")`` — будущие триггеры VLM (этап 8).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Tolerances, get_settings
from .contract import Component, Issue, Pin
from .geometry import GeometryResult, Segment, build_geometry
from .issues import default_action
from .parser.pdf_extract import PageExtraction, PdfExtraction, TextSpan, VectorPrimitive
from .spatial import (
    build_grid,
    iter_bbox_near_index_pairs,
    neighbors_in_tolerance,
)

#: Позиционное обозначение: R1, VD12, KM3, XT1 (1–3 буквы + 1–4 цифры).
_REF_RE = re.compile(r"^([A-ZА-Я]{1,3})([0-9]{1,4})$")

#: Тип компонента по префиксу позиционного обозначения (ГОСТ 2.710-81).
_REF_TYPE_MAP: dict[str, str] = {
    "R": "resistor",
    "C": "capacitor",
    "L": "inductor",
    "K": "relay",
    "KM": "contactor",
    "KA": "relay",
    "KT": "relay",
    "VD": "diode",
    "VT": "transistor",
    "Q": "transistor",
    "F": "fuse",
    "S": "switch",
    "SA": "switch",
    "SB": "button",
    "SQ": "sensor",
    "B": "sensor",
    "H": "lamp",
    "HL": "lamp",
    "Y": "valve",
    "X": "terminal",
    "XT": "terminal",
    "U": "unknown",
}

#: Минимальная площадь треугольника диода, pt².
_TRIANGLE_MIN_AREA = 8.0
#: Допуск замыкания треугольника, pt.
_TRIANGLE_CLOSE_TOL = 1.6

#: Допуски и размеры сеток детекторов, pt.
_CAPACITOR_MIN_LENGTH = 8.0
_CAPACITOR_MAX_LENGTH = 40.0
_CAPACITOR_MIN_GAP = 2.0
_CAPACITOR_MAX_GAP = 15.0
_CAPACITOR_GRID_CELL = 15.0
_TERMINAL_MAX_RADIUS = 5.0
_TERMINAL_ATTACH_TOL = 2.0
_TERMINAL_GRID_CELL = 8.0
_CONTACT_MIN_LENGTH = 5.0
_CONTACT_MAX_LENGTH = 30.0
_CONTACT_GRID_CELL = 2.0
#: Максимум коротких (самозамыкающихся) сегментов для полного перебора
#: троек без соседей. В реальных схемах таких сегментов единицы.
_MAX_SELF_CLOSED_TRIANGLE_SEGMENTS = 256


@dataclass
class SymbolTemplate:
    """Шаблон УГО для сопоставления с геометрией.

    Attributes:
        name: Имя шаблона (например, ``resistor``).
        category: Категория компонента.
        geometry: Описание эталонной геометрии.
    """

    name: str
    category: str = ""
    geometry: dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    """Кандидат в компонент, найденный шаблоном.

    Attributes:
        type: Тип компонента.
        page: Номер страницы.
        bbox: ``(x0, y0, x1, y1)`` УГО.
        primitive_indices: Индексы использованных примитивов.
        confidence: Уверенность шаблона.
    """

    type: str
    page: int
    bbox: tuple[float, float, float, float]
    primitive_indices: list[int] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        """Центр bbox УГО."""
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _bbox_union(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    """Объединяет bbox-ы."""
    return (
        min(item[0] for item in bboxes),
        min(item[1] for item in bboxes),
        max(item[2] for item in bboxes),
        max(item[3] for item in bboxes),
    )


def _angle_deg(segment: Segment) -> float:
    """Угол отрезка в градусах (0–180)."""
    return math.degrees(math.atan2(segment.y1 - segment.y0, segment.x1 - segment.x0)) % 180.0


def _angle_diff(first: float, second: float) -> float:
    """Минимальная разница углов (0–90)."""
    diff = abs(first - second) % 180.0
    return min(diff, 180.0 - diff)


def _point_segment_distance(
    point: tuple[float, float], segment: Segment
) -> float:
    """Расстояние от точки до отрезка."""
    px, py = point
    x0, y0, x1, y1 = segment.x0, segment.y0, segment.x1, segment.y1
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length_sq))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def _point_bbox_distance(
    point: tuple[float, float], bbox: tuple[float, float, float, float]
) -> float:
    """Расстояние от точки до прямоугольника (0 внутри)."""
    px, py = point
    x0, y0, x1, y1 = bbox
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def _association_score(
    span: TextSpan,
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Взвешенная оценка привязки обозначения к УГО.

    ``score = 0.75 * edge_distance + 0.25 * center_distance``, где
    ``edge_distance`` — расстояние от текста до границы bbox (0 внутри),
    ``center_distance`` — расстояние до центра bbox. Вес центра не даёт
    обозначению «прилипнуть» к краю крупного УГО, внутри которого текст
    оказался далеко от центра.

    Args:
        span: Текстовый фрагмент-обозначение.
        bbox: ``(x0, y0, x1, y1)`` УГО.

    Returns:
        Кортеж ``(взвешенная оценка, edge_distance)``. Порог отсечки
        проверяется по ``edge_distance`` (как раньше).
    """
    point = (span.cx, span.cy)
    edge_distance = _point_bbox_distance(point, bbox)
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    center_distance = math.hypot(point[0] - center_x, point[1] - center_y)
    score = 0.75 * edge_distance + 0.25 * center_distance
    return score, edge_distance


def _border_nodes(
    bbox: tuple[float, float, float, float],
    page: int,
    geometry: GeometryResult,
    tolerance: float,
) -> list[Any]:
    """Узлы геометрии на границе bbox (кандидаты в выводы УГО)."""
    x0, y0, x1, y1 = bbox
    border: list[Any] = []
    for node in geometry.nodes:
        if node.page != page:
            continue
        inside = (
            x0 - tolerance <= node.x <= x1 + tolerance
            and y0 - tolerance <= node.y <= y1 + tolerance
        )
        on_border = (
            abs(node.x - x0) <= tolerance
            or abs(node.x - x1) <= tolerance
            or abs(node.y - y0) <= tolerance
            or abs(node.y - y1) <= tolerance
        )
        if inside and on_border:
            border.append(node)
    return border


def _net_label_keys(geometry: GeometryResult) -> set[tuple[int, str, float, float]]:
    """Ключи слов-меток цепей, привязанных к узлам геометрии (З1).

    Компоненты не должны использовать такие слова как позиционные
    обозначения и как центры unknown-кластеров: это метки цепей (L1, N,
    PE, 24V и т.п.), а не ``R1``/``C2``.
    """
    keys: set[tuple[int, str, float, float]] = set()
    for node in geometry.nodes:
        for span in node.attributes.get("label_spans", []) or []:
            keys.add(
                (
                    int(span.get("page", node.page)),
                    str(span.get("text", "")),
                    round(float(span.get("cx", node.x)), 1),
                    round(float(span.get("cy", node.y)), 1),
                )
            )
    return keys


def _span_key(span: TextSpan) -> tuple[int, str, float, float]:
    """Ключ текстового фрагмента для сравнения с метками узлов."""
    return (span.page, span.text, round(span.cx, 1), round(span.cy, 1))


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Убирает кандидатов, использующих уже занятые примитивы."""
    used: set[int] = set()
    result: list[Candidate] = []
    for candidate in candidates:
        indices = set(candidate.primitive_indices)
        if indices & used:
            continue
        used |= indices
        result.append(candidate)
    return result


def _page_of(primitives: list[VectorPrimitive]) -> int:
    """Номер страницы по первому примитиву (0 для пустого списка)."""
    return primitives[0].page if primitives else 0


def _detector_capped_issue(
    detector_name: str,
    page: int,
    count: int,
    limit: int,
    subject: str,
) -> Issue:
    """Формирует Issue о срабатывании предохранителя детектора."""
    return Issue(
        type="detector_capped",
        page=page,
        description=(
            f"Детектор {detector_name} на странице {page}: {subject} "
            f"{count} превышает лимит {limit}; обработаны первые {limit}"
        ),
        confidence=0.5,
        required_action=default_action("detector_capped"),
    )


def _cap_segments(
    entries: list[Any],
    detector_name: str,
    page: int,
    issues: list[Issue] | None,
) -> list[Any]:
    """Усекает входной список сегментов детектора до лимита."""
    limit = get_settings().max_segments_per_detector
    if len(entries) <= limit:
        return entries
    if issues is not None:
        issues.append(
            _detector_capped_issue(
                detector_name, page, len(entries), limit, "сегментов"
            )
        )
    return entries[:limit]


def _candidate_soft_limit() -> int:
    """Внутренний предел генерации: лимит страницы + 1.

    Детектор останавливается после ``max_candidates_per_page + 1``
    кандидатов; централизованная проверка в :func:`_detect_candidates`
    видит превышение и формирует Issue ``detector_capped``.
    """
    return get_settings().max_candidates_per_page + 1


def _detect_resistors(
    primitives: list[VectorPrimitive],
    used: set[int],
    issues: list[Issue] | None = None,
) -> list[Candidate]:
    """Резистор: вытянутый прямоугольник с отношением сторон 2.0–6.0."""
    candidates: list[Candidate] = []
    for index, primitive in enumerate(primitives):
        if index in used or primitive.kind != "rect":
            continue
        x0, y0, x1, y1 = primitive.bbox
        width, height = x1 - x0, y1 - y0
        short, long = sorted((width, height))
        if short <= 0:
            continue
        ratio = long / short
        if 2.0 <= ratio <= 6.0 and 4.0 <= short <= 40.0:
            candidates.append(
                Candidate("resistor", primitive.page, primitive.bbox, [index], 1.0)
            )
    return candidates


def _detect_capacitors(
    primitives: list[VectorPrimitive],
    used: set[int],
    issues: list[Issue] | None = None,
) -> list[Candidate]:
    """Конденсатор: две короткие параллельные линии с малым зазором.

    Пары предфильтруются по зазору bbox (не больше 15 pt): если
    расстояние «точка–отрезок» не больше 15 pt, то и зазор bbox не больше
    15 pt, поэтому истинные пары не теряются.
    """
    candidates: list[Candidate] = []
    lines: list[tuple[int, VectorPrimitive, Segment]] = []
    for index, primitive in enumerate(primitives):
        if index in used or primitive.kind != "line" or len(primitive.points) < 2:
            continue
        (x0, y0), (x1, y1) = primitive.points[0], primitive.points[1]
        segment = Segment(x0, y0, x1, y1, primitive.page)
        if not (_CAPACITOR_MIN_LENGTH <= segment.length <= _CAPACITOR_MAX_LENGTH):
            continue
        lines.append((index, primitive, segment))

    lines = _cap_segments(
        lines, "capacitor", _page_of(primitives), issues
    )
    items = [
        (primitive.bbox, (position, primitive, segment))
        for position, (_, primitive, segment) in enumerate(lines)
    ]
    limit = _candidate_soft_limit()
    for position_a, position_b in sorted(
        iter_bbox_near_index_pairs(
            items, _CAPACITOR_MAX_GAP, _CAPACITOR_GRID_CELL
        )
    ):
        if len(candidates) >= limit:
            break
        index_a, first, seg_a = lines[position_a]
        index_b, second, seg_b = lines[position_b]
        if second.page != first.page:
            continue
        if _angle_diff(_angle_deg(seg_a), _angle_deg(seg_b)) > 5.0:
            continue
        if abs(seg_a.length - seg_b.length) > 0.35 * max(
            seg_a.length, seg_b.length
        ):
            continue
        # Зазор между параллельными линиями.
        distance = min(
            _point_segment_distance(seg_a.points[0], seg_b),
            _point_segment_distance(seg_a.points[1], seg_b),
        )
        if not (_CAPACITOR_MIN_GAP <= distance <= _CAPACITOR_MAX_GAP):
            continue
        bbox = _bbox_union([first.bbox, second.bbox])
        candidates.append(
            Candidate("capacitor", first.page, bbox, [index_a, index_b], 0.8)
        )
    return candidates


def _triangle_candidates(
    segments: list[tuple[int, Segment]]
) -> list[tuple[int, int, int, tuple[float, float, float, float]]]:
    """Находит замкнутые треугольники из трёх отрезков.

    Перебор ведётся по связности концов, а не по всем тройкам:

    1. все концы сегментов кладутся в равномерную сетку;
    2. для каждого сегмента находятся соседи — сегменты, у которых хотя бы
       один конец лежит в ``_TRIANGLE_CLOSE_TOL`` от конца текущего;
    3. для пары соседей ``(A, B)`` третьим кандидатом ``C`` рассматриваются
       только сегменты из объединения окрестностей ``A`` и ``B`` (плюс
       короткие сегменты, концы которых замыкаются сами на себя).

    Полнота: если тройка проходит прежнюю проверку связности, то у неё
    либо есть пара соседей по концам (и она попадает в перебор), либо все
    три сегмента короткие и перебираются отдельно. Финальные проверки
    связности и площади сохранены без изменений, поэтому состав троек
    совпадает с прежним алгоритмом.
    """
    count = len(segments)
    if count < 3:
        return []

    endpoint_entries: list[tuple[float, float, Any]] = []
    for position, (_, segment) in enumerate(segments):
        endpoint_entries.append((segment.x0, segment.y0, (position, 0)))
        endpoint_entries.append((segment.x1, segment.y1, (position, 1)))
    cell_size = max(
        _TRIANGLE_CLOSE_TOL, get_settings().tolerances.point_merge_px
    )
    grid = build_grid(endpoint_entries, cell_size)

    neighbors: list[set[int]] = [set() for _ in range(count)]
    short: set[int] = set()
    for position, (_, segment) in enumerate(segments):
        if segment.length <= _TRIANGLE_CLOSE_TOL:
            short.add(position)
        for point in segment.points:
            for _x, _y, payload in neighbors_in_tolerance(
                point, grid, _TRIANGLE_CLOSE_TOL
            ):
                other, _end = payload
                if other != position:
                    neighbors[position].add(other)

    limit = _candidate_soft_limit()
    triangles: dict[
        tuple[int, int, int],
        tuple[int, int, int, tuple[float, float, float, float]],
    ] = {}

    def consider(a: int, b: int, c: int) -> None:
        """Проверяет тройку финальными фильтрами прежнего алгоритма."""
        if len(triangles) >= limit:
            return
        ordered = tuple(sorted((a, b, c)))
        if ordered in triangles:
            return
        ends: list[tuple[float, float]] = []
        for position in ordered:
            ends.extend(segments[position][1].points)
        # Каждый конец должен иметь пару в пределах допуска (эквивалент
        # прежней проверки ``other is not end``).
        for first_index in range(6):
            end = ends[first_index]
            if not any(
                math.hypot(end[0] - other[0], end[1] - other[1])
                <= _TRIANGLE_CLOSE_TOL
                for other_index, other in enumerate(ends)
                if other_index != first_index
            ):
                return
        xs = [point[0] for point in ends]
        ys = [point[1] for point in ends]
        area = 0.5 * abs(
            (xs[1] - xs[0]) * (ys[2] - ys[0])
            - (xs[2] - xs[0]) * (ys[1] - ys[0])
        )
        if area < _TRIANGLE_MIN_AREA:
            return
        triangles[ordered] = (
            segments[ordered[0]][0],
            segments[ordered[1]][0],
            segments[ordered[2]][0],
            (min(xs), min(ys), max(xs), max(ys)),
        )

    # Тройки, содержащие хотя бы одну пару соседей по концам.
    for a in range(count):
        if len(triangles) >= limit:
            break
        for b in neighbors[a]:
            if len(triangles) >= limit:
                break
            if b <= a:
                continue
            for c in neighbors[a] | neighbors[b] | short:
                if len(triangles) >= limit:
                    break
                if c == a or c == b:
                    continue
                consider(a, b, c)

    # Тройки из коротких сегментов без соседей: их концы замыкаются сами
    # на себя, поэтому прежняя проверка связности проходит без соседей.
    short_list = sorted(short)
    if len(short_list) <= _MAX_SELF_CLOSED_TRIANGLE_SEGMENTS:
        for first in range(len(short_list)):
            if len(triangles) >= limit:
                break
            for second in range(first + 1, len(short_list)):
                if len(triangles) >= limit:
                    break
                for third in range(second + 1, len(short_list)):
                    if len(triangles) >= limit:
                        break
                    consider(
                        short_list[first],
                        short_list[second],
                        short_list[third],
                    )

    return [triangles[key] for key in sorted(triangles)]


def _detect_diodes(
    primitives: list[VectorPrimitive],
    used: set[int],
    issues: list[Issue] | None = None,
) -> list[Candidate]:
    """Диод: треугольник из линий и запирающая линия у основания."""
    candidates: list[Candidate] = []
    segments: list[tuple[int, Segment]] = []
    for index, primitive in enumerate(primitives):
        if index in used or primitive.kind != "line" or len(primitive.points) < 2:
            continue
        (x0, y0), (x1, y1) = primitive.points[0], primitive.points[1]
        segments.append((index, Segment(x0, y0, x1, y1, primitive.page)))

    segments = _cap_segments(segments, "diode", _page_of(primitives), issues)
    for index_a, index_b, index_c, bbox in _triangle_candidates(segments):
        x0, y0, x1, y1 = bbox
        bar: int | None = None
        for index, primitive in enumerate(primitives):
            if index in (index_a, index_b, index_c) or index in used:
                continue
            if primitive.kind != "line":
                continue
            px0, py0, px1, py1 = primitive.bbox
            if px0 >= x0 - 3 and px1 <= x1 + 3 and py0 >= y0 - 3 and py1 <= y1 + 3:
                bar = index
                break
        indices = [index_a, index_b, index_c] + ([bar] if bar is not None else [])
        candidates.append(Candidate("diode", primitives[index_a].page, bbox, indices, 0.7))
    return candidates


def _detect_terminals(
    primitives: list[VectorPrimitive],
    used: set[int],
    issues: list[Issue] | None = None,
) -> list[Candidate]:
    """Клемма: маленькая окружность (радиус ≤ 5 pt) с подходящей линией.

    Все точки линий кладутся в сетку; для каждой окружности запрашиваются
    только точки в радиусе ``radius + 2``. Привязанной считается первая
    по порядку примитивов линия среди найденных.
    """
    candidates: list[Candidate] = []
    lines = [
        (index, primitive)
        for index, primitive in enumerate(primitives)
        if index not in used and primitive.kind == "line"
    ]
    lines = _cap_segments(lines, "terminal", _page_of(primitives), issues)

    line_points: list[tuple[float, float, Any]] = []
    for position, (_, line) in enumerate(lines):
        for point in line.points:
            line_points.append((float(point[0]), float(point[1]), position))
    grid = build_grid(line_points, _TERMINAL_GRID_CELL)

    limit = _candidate_soft_limit()
    for index, primitive in enumerate(primitives):
        if len(candidates) >= limit:
            break
        if index in used or primitive.kind != "circle":
            continue
        center = primitive.attributes.get("center")
        radius = primitive.attributes.get("radius")
        if not center or radius is None or float(radius) > _TERMINAL_MAX_RADIUS:
            continue
        cx, cy = float(center[0]), float(center[1])
        tolerance = float(radius) + _TERMINAL_ATTACH_TOL
        hits = neighbors_in_tolerance((cx, cy), grid, tolerance)
        attached: list[int] = []
        if hits:
            first_position = min(payload for _, _, payload in hits)
            attached = [lines[first_position][0]]
        indices = [index] + attached
        bbox = primitive.bbox
        if attached:
            bbox = _bbox_union([bbox, primitives[attached[0]].bbox])
        candidates.append(Candidate("terminal", primitive.page, bbox, indices, 0.7))
    return candidates


def _detect_coils(
    primitives: list[VectorPrimitive],
    used: set[int],
    issues: list[Issue] | None = None,
) -> list[Candidate]:
    """Катушка/обмотка реле: окружность или дуга среднего радиуса."""
    candidates: list[Candidate] = []
    for index, primitive in enumerate(primitives):
        if index in used:
            continue
        if primitive.kind == "arc":
            candidates.append(
                Candidate("coil", primitive.page, primitive.bbox, [index], 0.6)
            )
            continue
        if primitive.kind != "circle":
            continue
        radius = primitive.attributes.get("radius")
        if radius is None:
            continue
        if 5.0 < float(radius) <= 25.0:
            candidates.append(
                Candidate("coil", primitive.page, primitive.bbox, [index], 0.7)
            )
    return candidates


def _detect_contacts(
    primitives: list[VectorPrimitive],
    used: set[int],
    issues: list[Issue] | None = None,
) -> list[Candidate]:
    """Реле-контакт: два коротких отрезка, сходящихся под углом.

    Пары предфильтруются по зазору bbox (не больше допуска замыкания):
    общий конец в допуске всегда даёт зазор bbox не больше этого допуска.
    """
    candidates: list[Candidate] = []
    lines: list[tuple[int, Segment]] = []
    for index, primitive in enumerate(primitives):
        if index in used or primitive.kind != "line" or len(primitive.points) < 2:
            continue
        (x0, y0), (x1, y1) = primitive.points[0], primitive.points[1]
        segment = Segment(x0, y0, x1, y1, primitive.page)
        if _CONTACT_MIN_LENGTH <= segment.length <= _CONTACT_MAX_LENGTH:
            lines.append((index, segment))

    lines = _cap_segments(lines, "contact", _page_of(primitives), issues)
    items = [
        (primitives[index].bbox, (position, index, segment))
        for position, (index, segment) in enumerate(lines)
    ]
    limit = _candidate_soft_limit()
    seen_pairs: set[tuple[int, int]] = set()
    for position_a, position_b in sorted(
        iter_bbox_near_index_pairs(
            items, _TRIANGLE_CLOSE_TOL, _CONTACT_GRID_CELL
        )
    ):
        if len(candidates) >= limit:
            break
        index_a, first = lines[position_a]
        index_b, second = lines[position_b]
        if first.page != second.page:
            continue
        shared = any(
            math.hypot(a[0] - b[0], a[1] - b[1]) <= _TRIANGLE_CLOSE_TOL
            for a in first.points
            for b in second.points
        )
        if not shared:
            continue
        angle = _angle_diff(_angle_deg(first), _angle_deg(second))
        if not (35.0 <= angle <= 145.0):
            continue
        key = (index_a, index_b)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        bbox = _bbox_union([primitives[index_a].bbox, primitives[index_b].bbox])
        candidates.append(
            Candidate("contact", first.page, bbox, [index_a, index_b], 0.5)
        )
    return candidates


#: Тип функции-детектора: примитивы, занятые индексы, накопитель проблем.
Detector = Callable[
    [list[VectorPrimitive], set[int], list[Issue] | None], list[Candidate]
]

#: Словарь шаблонов УГО: имя → функция-детектор.
SYMBOL_TEMPLATES: dict[str, Detector] = {
    "resistor": _detect_resistors,
    "capacitor": _detect_capacitors,
    "diode": _detect_diodes,
    "terminal": _detect_terminals,
    "coil": _detect_coils,
    "contact": _detect_contacts,
}


def load_templates() -> list[SymbolTemplate]:
    """Возвращает встроенные шаблоны УГО.

    Returns:
        Список шаблонов с именами и категориями.
    """
    return [
        SymbolTemplate(name=name, category=name, geometry={})
        for name in SYMBOL_TEMPLATES
    ]


def _table_text_keys(
    page: PageExtraction,
) -> set[tuple[int, str, float, float]]:
    """Ключи слов внутри таблиц BOM (это данные, а не обозначения УГО)."""
    regions = [(table.x0, table.y0, table.x1, table.y1) for table in page.tables]
    if not regions:
        return set()
    keys: set[tuple[int, str, float, float]] = set()
    for span in page.texts:
        if any(
            rx0 <= span.cx <= rx1 and ry0 <= span.cy <= ry1
            for rx0, ry0, rx1, ry1 in regions
        ):
            keys.add(_span_key(span))
    return keys


def _table_primitive_indices(page: PageExtraction) -> set[int]:
    """Индексы примитивов внутри таблиц перечня элементов."""
    regions = [(table.x0, table.y0, table.x1, table.y1) for table in page.tables]
    if not regions:
        return set()
    skip: set[int] = set()
    for index, primitive in enumerate(page.primitives):
        x0, y0, x1, y1 = primitive.bbox
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if any(rx0 <= cx <= rx1 and ry0 <= cy <= ry1 for rx0, ry0, rx1, ry1 in regions):
            skip.add(index)
    return skip


def _detect_candidates(
    primitives: list[VectorPrimitive],
    skip_indices: set[int] | None = None,
    issues: list[Issue] | None = None,
) -> list[Candidate]:
    """Прогоняет все шаблоны и убирает конфликты примитивов.

    Каждый детектор получает накопитель проблем: он сообщает об усечении
    входного списка сегментов (``detector_capped``). Дополнительно здесь
    проверяется лимит ``max_candidates_per_page`` на результат каждого
    детектора.

    Args:
        primitives: Примитивы страницы.
        skip_indices: Индексы, которые нельзя использовать (таблицы BOM).
        issues: Накопитель проблем этапа 4 (может быть ``None``).
    """
    used: set[int] = set(skip_indices or ())
    result: list[Candidate] = []
    page = _page_of(primitives)
    issue_sink = issues if issues is not None else []
    candidate_limit = get_settings().max_candidates_per_page
    for name, detector in SYMBOL_TEMPLATES.items():
        detected = _dedupe_candidates(detector(primitives, used, issue_sink))
        if len(detected) > candidate_limit:
            issue_sink.append(
                _detector_capped_issue(
                    name,
                    page,
                    len(detected),
                    candidate_limit,
                    "кандидатов",
                )
            )
            detected = detected[:candidate_limit]
        for candidate in detected:
            used.update(candidate.primitive_indices)
        result.extend(detected)
    return result


def _attach_ref_designations(
    candidates: list[Candidate],
    texts: list[TextSpan],
    tolerance: float,
    geometry: GeometryResult | None = None,
    excluded_keys: set[tuple[int, str, float, float]] | None = None,
) -> tuple[dict[int, TextSpan], list[TextSpan]]:
    """Привязывает обозначения к ближайшим кандидатам.

    Ранжирование для bbox — взвешенная оценка
    :func:`_association_score` (0.75·edge + 0.25·center). Пины в оценке не
    участвуют, но сохраняют прежнюю семантику: итоговое расстояние —
    ``min(взвешенная_оценка, расстояние_до_пина)``, а порог отсечки
    проверяется по ``min(edge_distance, расстояние_до_пина)``
    (``ref_attach_pt=40``). Слова-метки цепей, привязанные к узлам
    геометрии, из кандидатов в обозначения исключаются (З1).

    Args:
        candidates: Кандидаты в компоненты.
        texts: Текстовые фрагменты страницы.
        tolerance: Максимальное расстояние «обозначение → УГО», pt.
        geometry: Геометрия для поиска пин-узлов; ``None`` — только bbox.
        excluded_keys: Ключи слов-меток цепей (см. :func:`_net_label_keys`).

    Returns:
        Отображение «индекс кандидата → обозначение» и неиспользованные
        обозначения.
    """
    excluded = excluded_keys or set()
    designations = [
        span
        for span in texts
        if _REF_RE.match(span.text) and _span_key(span) not in excluded
    ]
    assignments: dict[int, TextSpan] = {}
    used_texts: set[int] = set()
    tol = get_settings().tolerances.node_merge_pt

    for index, candidate in enumerate(candidates):
        border = (
            _border_nodes(candidate.bbox, candidate.page, geometry, tol)
            if geometry is not None
            else []
        )
        best: TextSpan | None = None
        best_position: int | None = None
        best_distance = math.inf
        for span_index, span in enumerate(designations):
            if span_index in used_texts or span.page != candidate.page:
                continue
            score, edge_distance = _association_score(span, candidate.bbox)
            pin_distance = math.inf
            for node in border:
                pin_distance = min(
                    pin_distance,
                    math.hypot(span.cx - node.x, span.cy - node.y),
                )
            # Порог — по edge-компоненте (bbox) или по пину, как раньше.
            if min(edge_distance, pin_distance) > tolerance:
                continue
            distance = min(score, pin_distance)
            if distance <= best_distance:
                best = span
                best_position = span_index
                best_distance = distance
        if best is not None and best_position is not None:
            assignments[index] = best
            used_texts.add(best_position)

    unused = [
        span for index, span in enumerate(designations) if index not in used_texts
    ]
    return assignments, unused


def _type_for_ref(ref_designation: str, detected: str) -> str:
    """Уточняет тип компонента по префиксу обозначения."""
    match = _REF_RE.match(ref_designation)
    if not match:
        return detected
    prefix = match.group(1)
    mapped = _REF_TYPE_MAP.get(prefix)
    if mapped is None:
        return detected
    if detected in ("unknown", ""):
        return mapped
    return detected


def _pins_for_component(
    component_id: str,
    bbox: tuple[float, float, float, float],
    page: int,
    geometry: GeometryResult,
    start: int,
    tolerance: float,
) -> tuple[list[dict[str, Any]], int]:
    """Собирает выводы компонента из узлов на границе УГО.

    Returns:
        Список ``Pin.model_dump()`` и следующий свободный номер вывода.
    """
    pins: list[dict[str, Any]] = []
    number = start
    for node in _border_nodes(bbox, page, geometry, tolerance):
        pin = Pin(
            id=f"p{number:03d}",
            node_id=node.id,
            component_id=component_id,
            x=node.x,
            y=node.y,
        )
        pins.append(pin.model_dump())
        number += 1
    return pins, number


def detect_components(
    geometry: GeometryResult,
    extraction: PdfExtraction,
    tolerances: Tolerances | None = None,
) -> tuple[list[Component], list[Issue]]:
    """Распознаёт компоненты и проблемы извлечения.

    Args:
        geometry: Результат этапа 3.
        extraction: Результат этапа 2 (тексты и примитивы).
        tolerances: Допуски; по умолчанию из настроек.

    Returns:
        Кортеж ``(components, issues)``.
    """
    tol = tolerances or get_settings().tolerances
    components: list[Component] = []
    issues: list[Issue] = []
    pin_number = 1
    net_label_keys = _net_label_keys(geometry)

    for page in extraction.pages:
        skip_indices = _table_primitive_indices(page)
        candidates = _detect_candidates(page.primitives, skip_indices, issues)
        excluded_keys = net_label_keys | _table_text_keys(page)
        assignments, unused_refs = _attach_ref_designations(
            candidates, page.texts, tol.ref_attach_pt,
            geometry=geometry, excluded_keys=excluded_keys,
        )

        for index, candidate in enumerate(candidates):
            ref_span = assignments.get(index)
            ref_designation = ref_span.text if ref_span else ""
            component_id = f"c{len(components) + 1:03d}"
            pins, pin_number = _pins_for_component(
                component_id, candidate.bbox, candidate.page, geometry,
                pin_number, tol.node_merge_pt,
            )
            component_type = (
                _type_for_ref(ref_designation, candidate.type)
                if ref_designation
                else candidate.type
            )
            confidence = candidate.confidence if ref_designation else 0.5
            cx, cy = candidate.center
            components.append(
                Component(
                    id=component_id,
                    type=component_type,
                    ref_designation=ref_designation,
                    x=cx,
                    y=cy,
                    pins=pins,
                    attributes={
                        "bbox": list(candidate.bbox),
                        "page": candidate.page,
                        "template": candidate.type,
                        "primitive_count": len(candidate.primitive_indices),
                        "primitive_indices": list(candidate.primitive_indices),
                    },
                    confidence=confidence,
                )
            )

        # Неопознанные обозначения → unknown-компонент + issue для VLM.
        used_primitives: set[int] = set(skip_indices)
        for candidate in candidates:
            used_primitives.update(candidate.primitive_indices)
        for span in unused_refs:
            nearby = [
                (index, primitive)
                for index, primitive in enumerate(page.primitives)
                if index not in used_primitives
                and math.hypot(
                    (primitive.bbox[0] + primitive.bbox[2]) / 2.0 - span.cx,
                    (primitive.bbox[1] + primitive.bbox[3]) / 2.0 - span.cy,
                )
                <= tol.unknown_cluster_pt
            ]
            if not nearby:
                continue
            bbox = _bbox_union(
                [primitive.bbox for _, primitive in nearby]
            )
            component_id = f"c{len(components) + 1:03d}"
            pins, pin_number = _pins_for_component(
                component_id, bbox, span.page, geometry, pin_number,
                tol.node_merge_pt,
            )
            cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            components.append(
                Component(
                    id=component_id,
                    type="unknown",
                    ref_designation=span.text,
                    x=cx,
                    y=cy,
                    pins=pins,
                    attributes={
                        "bbox": list(bbox),
                        "page": span.page,
                        "template": "unknown",
                        "primitive_count": len(nearby),
                        "primitive_indices": [index for index, _ in nearby],
                    },
                    confidence=0.3,
                )
            )
            issues.append(
                Issue(
                    type="unrecognized_symbol",
                    page=span.page,
                    x=cx,
                    y=cy,
                    description=(
                        f"Неопознанное УГО рядом с обозначением {span.text}"
                    ),
                    confidence=0.3,
                    required_action="VLM review",
                )
            )

    return components, issues


def attach_ref_designations(
    components: list[Component],
    texts: list[Any],
    tolerance: float | None = None,
) -> list[Component]:
    """Привязывает позиционные обозначения к готовым компонентам.

    Args:
        components: Распознанные компоненты.
        texts: Текстовые фрагменты с bbox (``TextSpan``).
        tolerance: Максимальное расстояние привязки, pt.

    Returns:
        Компоненты с обновлёнными ``ref_designation``.
    """
    tol = tolerance if tolerance is not None else get_settings().tolerances.ref_attach_pt
    designations = [
        span for span in texts
        if isinstance(span, TextSpan) and _REF_RE.match(span.text)
    ]
    used: set[int] = set()
    for component in components:
        if component.ref_designation:
            continue
        best: TextSpan | None = None
        best_position: int | None = None
        best_distance = tol
        bbox = component.attributes.get("bbox")
        bbox_tuple: tuple[float, float, float, float] | None = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox_tuple = (
                float(bbox[0]), float(bbox[1]),
                float(bbox[2]), float(bbox[3]),
            )
        for index, span in enumerate(designations):
            if index in used:
                continue
            distance = math.hypot(span.cx - component.x, span.cy - component.y)
            if bbox_tuple is not None:
                distance = min(
                    distance, _point_bbox_distance((span.cx, span.cy), bbox_tuple)
                )
            for pin in component.pins:
                px = pin.get("x") if isinstance(pin, dict) else getattr(pin, "x", None)
                py = pin.get("y") if isinstance(pin, dict) else getattr(pin, "y", None)
                if px is None or py is None:
                    continue
                distance = min(
                    distance, math.hypot(span.cx - float(px), span.cy - float(py))
                )
            if distance <= best_distance:
                best = span
                best_distance = distance
                best_position = index
        if best is not None and best_position is not None:
            component.ref_designation = best.text
            component.confidence = min(component.confidence, 0.8)
            used.add(best_position)
    return components


def recognize_components(
    primitives: list[VectorPrimitive],
    texts: list[TextSpan],
    page: int = 1,
) -> list[Component]:
    """Совместимый API: распознаёт компоненты одной страницы.

    Args:
        primitives: Векторные примитивы.
        texts: Текстовые фрагменты с bbox.
        page: Номер страницы (1-based).

    Returns:
        Распознанные компоненты.
    """
    page_extraction = PageExtraction(
        page=page, width=0.0, height=0.0, texts=list(texts),
        primitives=list(primitives),
    )
    extraction = PdfExtraction(
        pdf_path="", pages=[page_extraction], opened=True
    )
    geometry = build_geometry(extraction)
    components, _ = detect_components(geometry, extraction)
    return components


__all__ = [
    "SymbolTemplate",
    "Candidate",
    "SYMBOL_TEMPLATES",
    "load_templates",
    "detect_components",
    "recognize_components",
    "attach_ref_designations",
]
