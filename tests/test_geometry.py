"""Unit-тесты этапа 3: отрезки, узлы, развязки, рамки, метки."""

from __future__ import annotations

from schematic_agent.config import Tolerances
from schematic_agent.parser.pdf_extract import (
    PageExtraction,
    PdfExtraction,
    TextSpan,
    VectorPrimitive,
)
from schematic_agent.geometry import (
    Segment,
    build_geometry,
    build_nodes,
    merge_segments,
)


def _shared_node(segments: list[Segment], tolerance: float) -> tuple[bool, int]:
    """Проверяет, что два отрезка имеют общий узел; возвращает (есть, число)."""
    result = build_nodes(segments, Tolerances(node_merge_pt=tolerance))
    shared = [
        node for node in result.nodes if node.attributes.get("endpoint_count", 0) >= 2
    ]
    return bool(shared), len(result.nodes)


def test_two_segments_share_endpoint() -> None:
    """Отрезки [0,0]-[10,0] и [10,0]-[20,0] дают общий узел."""
    segments = [
        Segment(0, 0, 10, 0, 1),
        Segment(10, 0, 20, 0, 1),
    ]

    has_shared, total = _shared_node(segments, 1.5)

    assert has_shared is True
    assert total == 3


def test_near_endpoints_merged_with_tolerance() -> None:
    """Зазор 0.5 pt при допуске 1.5 pt объединяет концы в один узел."""
    segments = [
        Segment(0, 0, 10, 0, 1),
        Segment(10, 0.5, 20, 0, 1),
    ]

    has_shared, total = _shared_node(segments, 1.5)

    assert has_shared is True
    assert total == 3


def test_x_crossing_without_node() -> None:
    """X-образное пересечение не создаёт связь: 4 конца, отдельная развязка."""
    segments = [
        Segment(0, 0, 10, 10, 1),
        Segment(0, 10, 10, 0, 1),
    ]

    result = build_nodes(segments, Tolerances(node_merge_pt=1.5))

    # Концы попарно не совпадают: 4 независимых узла, ни одного общего.
    assert len(result.nodes) == 4
    assert len(result.crossings) == 1
    crossing = result.crossings[0]
    assert crossing.x == 5.0 and crossing.y == 5.0
    assert crossing.attributes["connected"] is False
    # Точка пересечения не входит в узлы.
    assert all((node.x, node.y) != (5.0, 5.0) for node in result.nodes)


def test_rect_becomes_four_segments_and_frame_excluded() -> None:
    """Малый прямоугольник → 4 стороны; большой → рамка без сегментов."""
    small = VectorPrimitive(
        "rect",
        [(0, 0), (20, 0), (20, 10), (0, 10)],
        1,
        {"bbox": (0.0, 0.0, 20.0, 10.0)},
    )
    big = VectorPrimitive(
        "rect",
        [(0, 0), (400, 0), (400, 300), (0, 300)],
        1,
        {"bbox": (0.0, 0.0, 400.0, 300.0)},
    )
    extraction = PdfExtraction(
        pdf_path="",
        pages=[
            PageExtraction(
                page=1, width=400, height=300,
                primitives=[small, big],
            )
        ],
    )

    tolerances = Tolerances(frame_min_pt=200.0)
    segments = merge_segments([small, big], tolerances)
    geometry = build_geometry(extraction, tolerances)

    assert len(segments) == 4
    assert len(geometry.segments) == 4
    assert len(geometry.frames) == 1
    assert geometry.frames[0].x1 == 400.0


def test_text_label_attached_to_node() -> None:
    """Слово-метка цепи рядом с узлом попадает в attributes.labels."""
    primitive = VectorPrimitive(
        "line", [(0.0, 0.0), (50.0, 0.0)], 1,
        {"bbox": (0.0, 0.0, 50.0, 0.0)},
    )
    text = TextSpan("L1", 48.0, -4.0, 54.0, 4.0, 1)
    extraction = PdfExtraction(
        pdf_path="",
        pages=[
            PageExtraction(
                page=1, width=100, height=100,
                texts=[text], primitives=[primitive],
            )
        ],
    )

    geometry = build_geometry(extraction, Tolerances(label_snap_pt=12.0))

    labelled = [node for node in geometry.nodes if node.attributes.get("labels")]
    assert labelled
    assert "L1" in labelled[0].attributes["labels"]
