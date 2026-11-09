"""Unit-тесты пространственного индекса ``spatial.py``."""

from __future__ import annotations

import math

import pytest

from schematic_agent.spatial import (
    SpatialGrid,
    bbox_gap,
    bbox_overlaps,
    build_bbox_grid,
    build_grid,
    iter_bbox_near_index_pairs,
    iter_bbox_overlap_index_pairs,
    neighbors_in_tolerance,
)


def test_build_grid_places_points_into_cells() -> None:
    """Точки попадают в ячейки по floor(x / cell_size)."""
    grid = build_grid(
        [(1.0, 1.0, "a"), (11.0, 1.0, "b"), (1.0, 21.0, "c")],
        cell_size=10.0,
    )

    assert isinstance(grid, SpatialGrid)
    assert grid.cell_size == 10.0
    assert [payload for _, _, payload in grid[(0, 0)]] == ["a"]
    assert [payload for _, _, payload in grid[(1, 0)]] == ["b"]
    assert [payload for _, _, payload in grid[(0, 2)]] == ["c"]


def test_neighbors_in_tolerance_finds_only_close_points() -> None:
    """Возвращаются точки в допуске, дальние точки отбрасываются."""
    grid = build_grid(
        [
            (0.0, 0.0, "origin"),
            (1.6, 0.0, "close"),
            (5.0, 5.0, "far"),
        ],
        cell_size=2.0,
    )

    found = neighbors_in_tolerance((0.0, 0.0), grid, tol=1.6)

    payloads = {payload for _, _, payload in found}
    assert payloads == {"origin", "close"}
    for x, y, _ in found:
        assert math.hypot(x, y) <= 1.6


def test_neighbors_in_tolerance_uses_explicit_cell_size() -> None:
    """Явный размер ячейки переопределяет размер из сетки."""
    grid = build_grid([(0.0, 0.0, "a")], cell_size=100.0)

    found = neighbors_in_tolerance((0.0, 0.0), grid, tol=0.1, cell_size=1.0)

    assert [payload for _, _, payload in found] == ["a"]


def test_bbox_overlap_matches_brute_force() -> None:
    """Пары пересекающихся bbox совпадают с полным перебором."""
    boxes = [
        (0.0, 0.0, 10.0, 10.0),
        (5.0, 5.0, 15.0, 15.0),
        (20.0, 20.0, 30.0, 30.0),
        (10.0, 0.0, 20.0, 10.0),  # касается первого по x
        (100.0, 100.0, 130.0, 140.0),
    ]
    items = [(bbox, index) for index, bbox in enumerate(boxes)]

    expected: set[tuple[int, int]] = set()
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            if bbox_overlaps(boxes[first], boxes[second]):
                expected.add((first, second))

    assert set(iter_bbox_overlap_index_pairs(items, cell_size=7.0)) == expected


def test_bbox_near_matches_brute_force() -> None:
    """Пары с зазором bbox ≤ max_gap совпадают с полным перебором."""
    boxes = [
        (0.0, 0.0, 10.0, 10.0),
        (12.0, 0.0, 20.0, 10.0),   # зазор 2
        (30.0, 0.0, 40.0, 10.0),   # зазор 10
        (0.0, 30.0, 10.0, 40.0),   # диагональный зазор
        (0.0, 24.0, 10.0, 34.0),   # перекрывается с предыдущим
    ]
    items = [(bbox, index) for index, bbox in enumerate(boxes)]
    max_gap = 5.0

    expected: set[tuple[int, int]] = set()
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            if bbox_gap(boxes[first], boxes[second]) <= max_gap:
                expected.add((first, second))

    assert set(
        iter_bbox_near_index_pairs(items, max_gap, cell_size=3.0)
    ) == expected


def test_build_bbox_grid_skips_empty_space() -> None:
    """bbox занимает только пересечённые ячейки."""
    items = [((0.0, 0.0, 1.0, 1.0), "small")]

    grid = build_bbox_grid(items, cell_size=10.0)

    assert list(grid) == [(0, 0)]


def test_invalid_cell_size_rejected() -> None:
    """Нулевой размер ячейки — ошибка конфигурации."""
    with pytest.raises(ValueError):
        SpatialGrid(0)
