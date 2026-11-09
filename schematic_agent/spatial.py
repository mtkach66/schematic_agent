"""Пространственные индексы для геометрических выборок.

Модуль предоставляет равномерную сетку (uniform grid) для двух задач:

* поиск точек в пределах допуска (концы отрезков при поиске треугольников,
  концы линий при детекции клемм);
* поиск пар прямоугольников с перекрытием или зазором не больше заданного
  (предфильтрация пар для пересечений и квадратичных детекторов).

Использование сетки снижает сложность переборов с O(n²)/O(n³) до
~O(n·k), где ``k`` — число соседей в локальной окрестности.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

#: Ключ ячейки сетки.
Cell = tuple[int, int]
#: Точка в координатах страницы.
Point = tuple[float, float]
#: Прямоугольник ``(x0, y0, x1, y1)``.
BBox = tuple[float, float, float, float]
#: Элемент точечной сетки: ``(x, y, payload)``.
PointEntry = tuple[float, float, Any]

#: Размер ячейки сетки bbox по умолчанию, pt.
DEFAULT_BBOX_CELL_PT = 50.0


class SpatialGrid(dict[Cell, list[Any]]):
    """Словарь ``ячейка → элементы`` с запомненным размером ячейки.

    Наследуется от :class:`dict`, поэтому совместим с контрактом
    ``dict[cell, list[...]]``; атрибут :attr:`cell_size` нужен функциям
    поиска соседей, которым не передают размер ячейки явно.
    """

    def __init__(self, cell_size: float) -> None:
        super().__init__()
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.cell_size = float(cell_size)


def _cell_key(x: float, y: float, cell_size: float) -> Cell:
    """Возвращает ключ ячейки для точки."""
    return (math.floor(x / cell_size), math.floor(y / cell_size))


def _normalize_bbox(bbox: BBox) -> BBox:
    """Приводит bbox к виду ``(min_x, min_y, max_x, max_y)``."""
    x0, y0, x1, y1 = bbox
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def build_grid(points: Iterable[PointEntry], cell_size: float) -> SpatialGrid:
    """Строит равномерную сетку точек.

    Args:
        points: Итерируемое ``(x, y, payload)``.
        cell_size: Размер ячейки, pt (> 0).

    Returns:
        Сетка ``ячейка → [(x, y, payload), ...]``.
    """
    grid = SpatialGrid(cell_size)
    for x, y, payload in points:
        key = _cell_key(float(x), float(y), grid.cell_size)
        grid.setdefault(key, []).append((float(x), float(y), payload))
    return grid


def neighbors_in_tolerance(
    point: Point,
    grid: SpatialGrid,
    tol: float,
    cell_size: float | None = None,
) -> list[PointEntry]:
    """Возвращает точки сетки в пределах допуска ``tol``.

    Перебираются только ячейки в радиусе ``ceil(tol / cell_size) + 1``;
    точное расстояние проверяется через :func:`math.hypot`.

    Args:
        point: Точка запроса ``(x, y)``.
        grid: Сетка из :func:`build_grid`.
        tol: Радиус поиска, pt.
        cell_size: Размер ячейки; по умолчанию берётся из сетки.

    Returns:
        Список ``(x, y, payload)`` точек в допуске (включая саму точку,
        если она присутствует в сетке).
    """
    size = cell_size if cell_size is not None else grid.cell_size
    if size <= 0:
        raise ValueError("cell_size must be positive")
    px, py = float(point[0]), float(point[1])
    radius = int(math.ceil(tol / size)) + 1
    cx, cy = _cell_key(px, py, size)
    found: list[PointEntry] = []
    for ix in range(cx - radius, cx + radius + 1):
        for iy in range(cy - radius, cy + radius + 1):
            for x, y, payload in grid.get((ix, iy), ()):
                if math.hypot(x - px, y - py) <= tol:
                    found.append((x, y, payload))
    return found


def bbox_overlaps(first: BBox, second: BBox) -> bool:
    """Проверяет пересечение (включая касание) двух bbox."""
    a = _normalize_bbox(first)
    b = _normalize_bbox(second)
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def bbox_gap(first: BBox, second: BBox) -> float:
    """Евклидов зазор между bbox (0, если они пересекаются)."""
    a = _normalize_bbox(first)
    b = _normalize_bbox(second)
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def _bbox_cells(bbox: BBox, cell_size: float) -> Iterator[Cell]:
    """Перебирает ячейки, покрытые bbox."""
    x0, y0, x1, y1 = _normalize_bbox(bbox)
    ix0 = math.floor(x0 / cell_size)
    ix1 = math.floor(x1 / cell_size)
    iy0 = math.floor(y0 / cell_size)
    iy1 = math.floor(y1 / cell_size)
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            yield (ix, iy)


def build_bbox_grid(
    items: Sequence[tuple[BBox, Any]], cell_size: float
) -> SpatialGrid:
    """Строит сетку bbox: каждый элемент попадает во все покрытые ячейки.

    Args:
        items: Последовательность ``(bbox, payload)``; порядок задаёт
            индексы, возвращаемые :func:`iter_bbox_overlap_index_pairs`.
        cell_size: Размер ячейки, pt (> 0).

    Returns:
        Сетка ``ячейка → [(order, bbox, payload), ...]``.
    """
    grid = SpatialGrid(cell_size)
    for order, (bbox, payload) in enumerate(items):
        normalized = _normalize_bbox(bbox)
        for cell in _bbox_cells(normalized, grid.cell_size):
            grid.setdefault(cell, []).append((order, normalized, payload))
    return grid


def iter_bbox_overlap_index_pairs(
    items: Sequence[tuple[BBox, Any]],
    cell_size: float = DEFAULT_BBOX_CELL_PT,
) -> Iterator[tuple[int, int]]:
    """Перебирает пары индексов с перекрывающимися bbox.

    Пара возвращается один раз в каноническом порядке ``(min, max)``.
    Предфильтрация по ячейкам не заменяет точную проверку
    :func:`bbox_overlaps` — она выполняется перед выдачей пары.

    Args:
        items: Последовательность ``(bbox, payload)``.
        cell_size: Размер ячейки, pt.

    Yields:
        Пары индексов ``(i, j)``, ``i < j``.
    """
    grid = build_bbox_grid(items, cell_size)
    seen: set[tuple[int, int]] = set()
    for entries in grid.values():
        count = len(entries)
        for left in range(count):
            order_a, bbox_a, _ = entries[left]
            for right in range(left + 1, count):
                order_b, bbox_b, _ = entries[right]
                key = (
                    (order_a, order_b)
                    if order_a < order_b
                    else (order_b, order_a)
                )
                if key in seen:
                    continue
                seen.add(key)
                if bbox_overlaps(bbox_a, bbox_b):
                    yield key


def iter_bbox_overlap_pairs(
    items: Sequence[tuple[BBox, Any]],
    cell_size: float = DEFAULT_BBOX_CELL_PT,
) -> Iterator[tuple[Any, Any]]:
    """Перебирает пары payload с перекрывающимися bbox."""
    for first, second in iter_bbox_overlap_index_pairs(items, cell_size):
        yield items[first][1], items[second][1]


def iter_bbox_near_index_pairs(
    items: Sequence[tuple[BBox, Any]],
    max_gap: float,
    cell_size: float | None = None,
) -> Iterator[tuple[int, int]]:
    """Перебирает пары индексов с зазором bbox не больше ``max_gap``.

    Для предфильтрации bbox расширяются на ``max_gap / 2`` с каждой
    стороны: если исходный зазор по оси больше ``max_gap``, расширенные
    bbox не пересекутся. После предфильтрации выполняется точная проверка
    евклидова зазора :func:`bbox_gap`.

    Args:
        items: Последовательность ``(bbox, payload)``.
        max_gap: Максимальный зазор, pt (>= 0).
        cell_size: Размер ячейки; по умолчанию ``max(max_gap, 1.0)``.

    Yields:
        Пары индексов ``(i, j)``, ``i < j``.
    """
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    size = cell_size if cell_size is not None else max(max_gap, 1.0)
    half = max_gap / 2.0
    expanded: list[tuple[BBox, Any]] = []
    for bbox, payload in items:
        x0, y0, x1, y1 = _normalize_bbox(bbox)
        expanded.append(((x0 - half, y0 - half, x1 + half, y1 + half), payload))
    for first, second in iter_bbox_overlap_index_pairs(expanded, size):
        if bbox_gap(items[first][0], items[second][0]) <= max_gap:
            yield first, second


def iter_bbox_near_pairs(
    items: Sequence[tuple[BBox, Any]],
    max_gap: float,
    cell_size: float | None = None,
) -> Iterator[tuple[Any, Any]]:
    """Перебирает пары payload с зазором bbox не больше ``max_gap``."""
    for first, second in iter_bbox_near_index_pairs(items, max_gap, cell_size):
        yield items[first][1], items[second][1]


__all__ = [
    "Cell",
    "Point",
    "BBox",
    "PointEntry",
    "DEFAULT_BBOX_CELL_PT",
    "SpatialGrid",
    "build_grid",
    "neighbors_in_tolerance",
    "bbox_overlaps",
    "bbox_gap",
    "build_bbox_grid",
    "iter_bbox_overlap_index_pairs",
    "iter_bbox_overlap_pairs",
    "iter_bbox_near_index_pairs",
    "iter_bbox_near_pairs",
]
