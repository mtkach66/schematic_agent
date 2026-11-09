"""Тесты производительности и предохранителей (T1–T10, T12 плана).

Integration tests on an external reference PDF are skipped unless the
``SCHEMATIC_AGENT_TEST_PDF`` environment variable points to an existing file.
The document is not bundled with the repository.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import resource
import signal
import time
from pathlib import Path
from typing import Iterator

import pymupdf
import pytest

import schematic_agent.components as components
from schematic_agent import run
from schematic_agent.components import (
    _TRIANGLE_CLOSE_TOL,
    _TRIANGLE_MIN_AREA,
    _detect_capacitors,
    _detect_coils,
    _detect_contacts,
    _detect_diodes,
    _detect_terminals,
    _triangle_candidates,
    detect_components,
)
from schematic_agent.config import OUTPUT_DIR_ENV, get_settings
from schematic_agent.contract import Task, TaskOptions, TaskPayload
from schematic_agent.geometry import (
    Segment,
    _find_crossings,
    _segment_intersection,
    build_geometry,
)
from schematic_agent.parser.pdf_extract import (
    PageExtraction,
    PdfExtraction,
    VectorPrimitive,
    extract_pdf,
)
from schematic_agent.pipeline import ARTIFACT_FILES

#: Environment variable pointing to an external reference PDF.
REFERENCE_PDF_ENV = "SCHEMATIC_AGENT_TEST_PDF"

#: External reference document; not bundled with the repository.
_reference_path = os.environ.get(REFERENCE_PDF_ENV)
REFERENCE_PDF = Path(_reference_path) if _reference_path else None

requires_eplan = pytest.mark.skipif(
    REFERENCE_PDF is None or not REFERENCE_PDF.is_file(),
    reason=(
        f"external reference PDF not configured ({REFERENCE_PDF_ENV}) "
        "or file is missing"
    ),
)


# --- Общие помощники ---------------------------------------------------------


def _task(pdf: Path, document_id: str) -> Task:
    """Задание по контракту ТЗ §7 (VLM отключена: в pytest только моки)."""
    return Task(
        task_id=f"task-{document_id}",
        payload=TaskPayload(
            pdf_path=str(pdf),
            document_id=document_id,
            options=TaskOptions(use_vlm=False),
        ),
    )


@pytest.fixture
def output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Перенаправляет артефакты во временный каталог."""
    target = tmp_path / "out"
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(target))
    return target


@pytest.fixture(scope="module")
def eplan_extraction() -> PdfExtraction:
    """Результат извлечения внешнего PDF (один раз на модуль)."""
    if REFERENCE_PDF is None or not REFERENCE_PDF.is_file():
        pytest.skip("external reference PDF is not configured or missing")
    return extract_pdf(str(REFERENCE_PDF))


@pytest.fixture(scope="module")
def eplan_geometry(eplan_extraction: PdfExtraction):
    """Геометрия внешнего PDF (один раз на модуль)."""
    return build_geometry(eplan_extraction)


def _line_segments(
    primitives: list[VectorPrimitive],
) -> list[tuple[int, Segment]]:
    """Превращает линии страницы в ``(primitive_index, Segment)``."""
    segments: list[tuple[int, Segment]] = []
    for index, primitive in enumerate(primitives):
        if primitive.kind != "line" or len(primitive.points) < 2:
            continue
        (x0, y0), (x1, y1) = primitive.points[0], primitive.points[1]
        segments.append((index, Segment(x0, y0, x1, y1, primitive.page)))
    return segments


@contextlib.contextmanager
def _timebox(seconds: int) -> Iterator[None]:
    """Ограничивает время выполнения блока (POSIX ``SIGALRM``)."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum: int, frame: object) -> None:
        del signum, frame
        raise TimeoutError(f"превышен таймбокс {seconds} с")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _old_triangle_candidates(
    segments: list[tuple[int, Segment]],
) -> list[tuple[int, int, int, tuple[float, float, float, float]]]:
    """Эталонный прежний O(n³)-алгоритм (для T3)."""
    triangles: list[
        tuple[int, int, int, tuple[float, float, float, float]]
    ] = []
    for i in range(len(segments)):
        index_a, first = segments[i]
        for j in range(i + 1, len(segments)):
            index_b, second = segments[j]
            for k in range(j + 1, len(segments)):
                index_c, third = segments[k]
                points = [first.points, second.points, third.points]
                ends = [point for pair in points for point in pair]
                connected = all(
                    any(
                        math.hypot(end[0] - other[0], end[1] - other[1])
                        <= _TRIANGLE_CLOSE_TOL
                        for other in ends
                        if other is not end
                    )
                    for end in ends
                )
                if not connected:
                    continue
                xs = [point[0] for point in ends]
                ys = [point[1] for point in ends]
                area = 0.5 * abs(
                    (xs[1] - xs[0]) * (ys[2] - ys[0])
                    - (xs[2] - xs[0]) * (ys[1] - ys[0])
                )
                if area < _TRIANGLE_MIN_AREA:
                    continue
                triangles.append(
                    (
                        index_a,
                        index_b,
                        index_c,
                        (min(xs), min(ys), max(xs), max(ys)),
                    )
                )
    return triangles


def _triangle_signature(
    triangles: list[tuple[int, int, int, tuple[float, float, float, float]]],
) -> set[tuple[int, int, int, tuple[float, float, float, float]]]:
    """Множество троек с округлённым bbox."""
    return {
        (a, b, c, tuple(round(value, 9) for value in bbox))
        for a, b, c, bbox in triangles
    }


def _old_find_crossings(
    segments: list[Segment], tolerance: float
) -> list[tuple[int, float, float]]:
    """Эталонный прежний O(n²)-поиск пересечений (для T9)."""
    result: list[tuple[int, float, float]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            first, second = segments[i], segments[j]
            if first.page != second.page:
                continue
            point = _segment_intersection(first, second)
            if point is None:
                continue
            if any(
                math.hypot(point[0] - x, point[1] - y) <= tolerance
                for x, y in (*first.points, *second.points)
            ):
                continue
            key = (first.page, round(point[0], 1), round(point[1], 1), 0)
            if key in seen:
                continue
            seen.add(key)
            result.append((first.page, point[0], point[1]))
    return result


def _candidate_signature(candidates: list) -> list[tuple]:
    """Устойчивая подпись кандидатов для регрессионных сравнений."""
    return [
        (
            candidate.type,
            candidate.page,
            tuple(round(value, 9) for value in candidate.bbox),
            tuple(candidate.primitive_indices),
            round(candidate.confidence, 9),
        )
        for candidate in candidates
    ]


# --- Эталонные (старые) реализации квадратичных детекторов (T10) ------------


def _old_detect_capacitors(
    primitives: list[VectorPrimitive], used: set[int]
) -> list:
    """Прежняя реализация детектора конденсаторов."""
    candidates: list = []
    lines = [
        (index, primitive)
        for index, primitive in enumerate(primitives)
        if index not in used
        and primitive.kind == "line"
        and len(primitive.points) >= 2
    ]
    for i in range(len(lines)):
        index_a, first = lines[i]
        seg_a = Segment(
            first.points[0][0],
            first.points[0][1],
            first.points[1][0],
            first.points[1][1],
            first.page,
        )
        if not (8.0 <= seg_a.length <= 40.0):
            continue
        angle_a = components._angle_deg(seg_a)
        for j in range(i + 1, len(lines)):
            index_b, second = lines[j]
            if second.page != first.page:
                continue
            seg_b = Segment(
                second.points[0][0],
                second.points[0][1],
                second.points[1][0],
                second.points[1][1],
                second.page,
            )
            if not (8.0 <= seg_b.length <= 40.0):
                continue
            if components._angle_diff(angle_a, components._angle_deg(seg_b)) > 5.0:
                continue
            if abs(seg_a.length - seg_b.length) > 0.35 * max(
                seg_a.length, seg_b.length
            ):
                continue
            distance = min(
                components._point_segment_distance(seg_a.points[0], seg_b),
                components._point_segment_distance(seg_a.points[1], seg_b),
            )
            if not (2.0 <= distance <= 15.0):
                continue
            bbox = components._bbox_union([first.bbox, second.bbox])
            candidates.append(
                components.Candidate(
                    "capacitor", first.page, bbox, [index_a, index_b], 0.8
                )
            )
    return candidates


def _old_detect_terminals(
    primitives: list[VectorPrimitive], used: set[int]
) -> list:
    """Прежняя реализация детектора клемм."""
    candidates: list = []
    lines = [
        (index, primitive)
        for index, primitive in enumerate(primitives)
        if index not in used and primitive.kind == "line"
    ]
    for index, primitive in enumerate(primitives):
        if index in used or primitive.kind != "circle":
            continue
        center = primitive.attributes.get("center")
        radius = primitive.attributes.get("radius")
        if not center or radius is None or float(radius) > 5.0:
            continue
        cx, cy = float(center[0]), float(center[1])
        attached = [
            line_index
            for line_index, line in lines
            if any(
                math.hypot(point[0] - cx, point[1] - cy)
                <= float(radius) + 2.0
                for point in line.points
            )
        ]
        indices = [index] + attached[:1]
        bbox = primitive.bbox
        if attached:
            bbox = components._bbox_union(
                [bbox, primitives[attached[0]].bbox]
            )
        candidates.append(
            components.Candidate(
                "terminal", primitive.page, bbox, indices, 0.7
            )
        )
    return candidates


def _old_detect_coils(
    primitives: list[VectorPrimitive], used: set[int]
) -> list:
    """Прежняя реализация детектора катушек."""
    candidates: list = []
    for index, primitive in enumerate(primitives):
        if index in used:
            continue
        if primitive.kind == "arc":
            candidates.append(
                components.Candidate(
                    "coil", primitive.page, primitive.bbox, [index], 0.6
                )
            )
            continue
        if primitive.kind != "circle":
            continue
        radius = primitive.attributes.get("radius")
        if radius is None:
            continue
        if 5.0 < float(radius) <= 25.0:
            candidates.append(
                components.Candidate(
                    "coil", primitive.page, primitive.bbox, [index], 0.7
                )
            )
    return candidates


def _old_detect_contacts(
    primitives: list[VectorPrimitive], used: set[int]
) -> list:
    """Прежняя реализация детектора контактов."""
    candidates: list = []
    lines: list[tuple[int, Segment]] = []
    for index, primitive in enumerate(primitives):
        if index in used or primitive.kind != "line" or len(primitive.points) < 2:
            continue
        (x0, y0), (x1, y1) = primitive.points[0], primitive.points[1]
        segment = Segment(x0, y0, x1, y1, primitive.page)
        if 5.0 <= segment.length <= 30.0:
            lines.append((index, segment))

    seen_pairs: set[tuple[int, int]] = set()
    for i in range(len(lines)):
        index_a, first = lines[i]
        for j in range(i + 1, len(lines)):
            index_b, second = lines[j]
            if first.page != second.page:
                continue
            shared = any(
                math.hypot(a[0] - b[0], a[1] - b[1])
                <= _TRIANGLE_CLOSE_TOL
                for a in first.points
                for b in second.points
            )
            if not shared:
                continue
            angle = components._angle_diff(
                components._angle_deg(first), components._angle_deg(second)
            )
            if not (35.0 <= angle <= 145.0):
                continue
            key = (index_a, index_b)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            bbox = components._bbox_union(
                [primitives[index_a].bbox, primitives[index_b].bbox]
            )
            candidates.append(
                components.Candidate(
                    "contact", first.page, bbox, [index_a, index_b], 0.5
                )
            )
    return candidates


def _pdf_with_resistors(path: Path, count: int) -> Path:
    """Создаёт PDF с заданным числом резисторов (прямоугольник + R<N>)."""
    document = pymupdf.open()
    page = document.new_page(width=300, height=200)
    for index in range(count):
        x = 20.0 + index * 80.0
        page.draw_rect(pymupdf.Rect(x, 20, x + 40, 40))
        page.insert_text((x + 8, 55), f"R{index + 1}")
    document.save(str(path))
    document.close()
    return path


# --- T1–T3: перебор треугольников -------------------------------------------


def test_t1_triangle_with_noise_under_one_second() -> None:
    """T1: один треугольник + 200 далёких шумовых линий, < 1 с."""
    # Направление второго отрезка выбрано так, чтобы площадь по формуле
    # прежнего алгоритма (концы первого отрезка + начало второго) была
    # ненулевой: иначе треугольник отбрасывается проверкой площади.
    segments: list[tuple[int, Segment]] = [
        (0, Segment(0, 0, 40, 0, 1)),
        (1, Segment(20, 30, 40, 0, 1)),
        (2, Segment(20, 30, 0, 0, 1)),
    ]
    for index in range(200):
        x = 300.0 + (index % 20) * 60.0
        y = 300.0 + (index // 20) * 60.0
        segments.append(
            (3 + index, Segment(x, y, x + 10.0, y, 1))
        )

    started = time.perf_counter()
    triangles = _triangle_candidates(segments)
    elapsed = time.perf_counter() - started

    assert len(triangles) == 1
    assert elapsed < 1.0, f"T1: {elapsed:.3f} с"


def test_t2_open_triangle_not_found() -> None:
    """T2: конец вне допуска — ложного треугольника нет."""
    segments = [
        (0, Segment(0, 0, 40, 0, 1)),
        (1, Segment(40, 0, 20, 30, 1)),
        (2, Segment(20, 30, 10, 0, 1)),
    ]

    assert _triangle_candidates(segments) == []


@requires_eplan
def test_t3_old_vs_new_triangle_candidates(
    eplan_extraction: PdfExtraction,
) -> None:
    """T3: старый и новый перебор на срезе ~300 линий стр.1 идентичны."""
    segments = _line_segments(eplan_extraction.pages[0].primitives)[:300]

    with _timebox(60):
        started = time.perf_counter()
        old = _old_triangle_candidates(segments)
        old_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    new = _triangle_candidates(segments)
    new_elapsed = time.perf_counter() - started

    assert old_elapsed < 60.0, f"T3: старый алгоритм {old_elapsed:.1f} с"
    assert _triangle_signature(old) == _triangle_signature(new), (
        f"T3: составы различаются (old={len(old)}, new={len(new)}, "
        f"old={old_elapsed:.3f} с, new={new_elapsed:.3f} с)"
    )


# --- T4–T5: integration measurements on the reference PDF -----------------


@requires_eplan
def test_t4_detect_diodes_under_five_seconds(
    eplan_extraction: PdfExtraction,
) -> None:
    """T4: ``_detect_diodes`` на странице 1 (753 линии) < 5 с."""
    primitives = eplan_extraction.pages[0].primitives

    started = time.perf_counter()
    candidates = _detect_diodes(primitives, set())
    elapsed = time.perf_counter() - started

    assert candidates is not None
    assert elapsed < 5.0, f"T4: _detect_diodes {elapsed:.3f} с"


@requires_eplan
def test_t5_detect_components_all_pages_under_sixty_seconds(
    eplan_extraction: PdfExtraction, eplan_geometry
) -> None:
    """T5: полный ``detect_components`` на 7 страницах < 60 с."""
    started = time.perf_counter()
    detected, issues = detect_components(eplan_geometry, eplan_extraction)
    elapsed = time.perf_counter() - started

    assert detected is not None
    assert issues is not None
    assert elapsed < 60.0, f"T5: detect_components {elapsed:.3f} с"


# --- T7–T8: предохранители и стражные таймауты -----------------------------


def test_t7_candidate_cap_issue_and_partial_status(
    tmp_path: Path, output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T7: заниженный лимит кандидатов → detector_capped + partial."""
    monkeypatch.setenv("SCHEMATIC_AGENT_MAX_CANDIDATES_PER_PAGE", "1")
    assert get_settings().max_candidates_per_page == 1
    pdf = _pdf_with_resistors(tmp_path / "two_resistors.pdf", 2)

    result = run(_task(pdf, "doc_cap"))

    assert result.status == "partial"
    issues = json.loads(
        (output_dir / "doc_cap" / "issues.json").read_text(encoding="utf-8")
    )["issues"]
    capped = [item for item in issues if item["type"] == "detector_capped"]
    assert capped, "нет Issue detector_capped"
    assert "resistor" in capped[0]["description"]
    components_payload = json.loads(
        (output_dir / "doc_cap" / "components.json").read_text(encoding="utf-8")
    )["components"]
    # Второй резистор усечён, но его неиспользованный примитив рядом с R2
    # даёт штатный unknown-компонент; резисторов в результате ровно один.
    resistors = [
        item for item in components_payload if item["type"] == "resistor"
    ]
    assert len(resistors) == 1
    for filename in ARTIFACT_FILES.values():
        assert (output_dir / "doc_cap" / filename).is_file()


def test_t8_stage_timeout_issue_and_artifacts(
    tmp_path: Path, output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T8: заниженный стражный таймаут → stage_timeout + артефакты."""
    monkeypatch.setenv("SCHEMATIC_AGENT_STAGE_TIMEOUT_S", "0.01")
    pdf = _pdf_with_resistors(tmp_path / "one_resistor.pdf", 1)

    import schematic_agent.pipeline as pipeline

    original = pipeline._stage_components

    def slow_stage(*args, **kwargs):
        time.sleep(0.05)
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_stage_components", slow_stage)

    result = run(_task(pdf, "doc_stage_timeout"))

    assert result.status == "partial"
    issues = json.loads(
        (output_dir / "doc_stage_timeout" / "issues.json").read_text(
            encoding="utf-8"
        )
    )["issues"]
    timed_out = [item for item in issues if item["type"] == "stage_timeout"]
    assert timed_out, "нет Issue stage_timeout"
    assert "components" in timed_out[0]["description"]
    for filename in ARTIFACT_FILES.values():
        assert (output_dir / "doc_stage_timeout" / filename).is_file()


# --- T9–T10: регрессии идентичности ----------------------------------------


@requires_eplan
def test_t9_crossings_identical(eplan_geometry) -> None:
    """T9: новый поиск пересечений идентичен прежнему O(n²)."""
    tolerance = get_settings().tolerances.node_merge_pt

    old = _old_find_crossings(eplan_geometry.segments, tolerance)
    new = [
        (node.page, node.x, node.y)
        for node in _find_crossings(eplan_geometry.segments, tolerance)
    ]

    assert len(new) == len(old)
    assert [
        (page, round(x, 9), round(y, 9)) for page, x, y in new
    ] == [
        (page, round(x, 9), round(y, 9)) for page, x, y in old
    ]


@requires_eplan
def test_t10_quadratic_detectors_identical(
    eplan_extraction: PdfExtraction,
) -> None:
    """T10: кандидаты capacitor/terminal/coil/contact стр.1 идентичны."""
    primitives = eplan_extraction.pages[0].primitives
    used: set[int] = set()

    pairs = [
        (
            "capacitor",
            _old_detect_capacitors,
            _detect_capacitors,
        ),
        (
            "terminal",
            _old_detect_terminals,
            _detect_terminals,
        ),
        (
            "coil",
            _old_detect_coils,
            _detect_coils,
        ),
        (
            "contact",
            _old_detect_contacts,
            _detect_contacts,
        ),
    ]
    for name, old_detector, new_detector in pairs:
        old = _candidate_signature(old_detector(primitives, set(used)))
        new = _candidate_signature(new_detector(primitives, set(used)))
        assert new == old, f"T10: кандидаты {name} различаются"


# --- T12: нагрузочный тест --------------------------------------------------


def test_t12_5000_lines_under_120s_and_rss_under_1gb() -> None:
    """T12: 5000 линий → detect_components < 120 с, RSS < 1 ГБ."""
    primitives: list[VectorPrimitive] = []
    for index in range(5000):
        x = (index % 100) * 40.0
        y = (index // 100) * 40.0
        primitives.append(
            VectorPrimitive(
                "line",
                [(x, y), (x + 10.0, y)],
                1,
                {"bbox": (x, y, x + 10.0, y)},
            )
        )
    extraction = PdfExtraction(
        pdf_path="",
        pages=[
            PageExtraction(
                page=1,
                width=4000.0,
                height=2000.0,
                primitives=primitives,
            )
        ],
    )
    geometry = build_geometry(extraction)

    started = time.perf_counter()
    detected, issues = detect_components(geometry, extraction)
    elapsed = time.perf_counter() - started
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assert detected is not None
    assert issues is not None
    assert elapsed < 120.0, f"T12: detect_components {elapsed:.3f} с"
    assert rss_kb < 1024 * 1024, f"T12: RSS {rss_kb / 1024:.1f} МБ"


# --- T11: BOM-фолбэк на внешнем документе -----------------------------------


@requires_eplan
def test_t11_bom_fallback_on_eplan(output_dir: Path) -> None:
    """T11: без таблицы перечня BOM собран из компонентов, Issue сохранён.

    Режим ``no_vlm``: VLM в pytest не вызывается (моки/живой прогон — вне
    тестов), на BOM-фолбэк режим не влияет.
    """
    task = Task(
        task_id="task-t11",
        payload=TaskPayload(
            pdf_path=str(REFERENCE_PDF),
            document_id="doc_t11",
            options=TaskOptions(mode="no_vlm", use_vlm=False),
        ),
    )

    result = run(task)

    doc_dir = output_dir / "doc_t11"
    bom = json.loads((doc_dir / "bom.json").read_text(encoding="utf-8"))
    assert bom["source"] == "components_fallback"
    assert bom["items"], "BOM-фолбэк не должен быть пустым"
    refs = {item["ref_designation"] for item in bom["items"]}
    assert {"B2", "C2", "U1"} <= refs
    assert all(item["found_on_schematic"] for item in bom["items"])

    issues = json.loads((doc_dir / "issues.json").read_text(encoding="utf-8"))
    issue_types = {item["type"] for item in issues["issues"]}
    assert "bom_table_missing" in issue_types
    assert "component_not_in_bom" not in issue_types

    assert result.summary.bom_lines_found == len(bom["items"]) > 0
