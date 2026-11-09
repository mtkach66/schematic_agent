"""Интеграционные приёмочные тесты на эталонных PDF (Т1–Т12).

Тест-кейсы плана приёмки:

* Т1 — полный прогон ``simple.pdf``: 8 артефактов, BOM ≥ 4, nets ≥ 2,
  компоненты ≥ 4, счётчики ``summary`` сходятся с артефактами;
* Т4 — ``problem.pdf``: растровая страница → ``completed_with_issues`` +
  ``raster_only_page``; пересечение без узла НЕ соединено;
* Т8 — ``multipage.pdf``: цепь L1 объединена по страницам (``pages=[1,2]``).

Остальные кейсы покрыты существующими тестами модулей (в отчёте приведены
имена): Т2/Т3 — ``tests/test_bom.py``, ``tests/test_components.py``;
Т5 — ``tests/test_reliability.py::test_bom_table_missing_in_full_mode``;
Т6/Т7 — ``tests/test_netlist.py``; Т9 — ``tests/test_vlm.py``;
Т10/Т11 — ``tests/test_reliability.py`` (partial/doc_timeout);
Т12 — ``tests/test_api.py`` и ``tests/test_pipeline_modes.py``.

VLM в тестах вызывается ТОЛЬКО через мок-транспорт: реальная VLM тянется
отдельно архитектором, не в pytest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from examples.generate_examples import generate_all
from schematic_agent import run
from schematic_agent.components import detect_components
from schematic_agent.config import OUTPUT_DIR_ENV
from schematic_agent.contract import Task, TaskOptions, TaskPayload
from schematic_agent.geometry import build_geometry
from schematic_agent.netlist import build_netlist
from schematic_agent.parser.pdf_extract import extract_pdf
from schematic_agent.pipeline import ARTIFACT_FILES

pytestmark = pytest.mark.acceptance

#: Шрифт кириллицы, которым генерируются эталоны.
CYRILLIC_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

#: Валидный JSON-ответ VLM для мок-транспорта.
MOCK_VLM_RESPONSE = json.dumps(
    {
        "response": (
            '{"symbol": "junction", "confidence": 0.9, '
            '"description": "mock VLM response"}'
        )
    }
).encode("utf-8")


@pytest.fixture
def examples(tmp_path: Path) -> dict[str, Path]:
    """Генерирует эталонные PDF во временный каталог (на лету)."""
    if not CYRILLIC_FONT.is_file():
        pytest.skip("нет шрифта DejaVu для кириллицы")
    return generate_all(tmp_path / "examples")


@pytest.fixture
def output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Перенаправляет артефакты во временный каталог."""
    target = tmp_path / "out"
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(target))
    return target


@pytest.fixture
def mock_vlm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Подменяет HTTP-транспорт VLM валидным JSON (без сети)."""
    import schematic_agent.vlm as vlm_module

    def transport(url: str, payload: dict, timeout_s: float) -> tuple[int, bytes]:
        del url, payload, timeout_s
        return 200, MOCK_VLM_RESPONSE

    monkeypatch.setattr(vlm_module, "_default_transport", transport)


def _task(
    pdf: Path,
    *,
    mode: str = "full",
    document_id: str,
    use_vlm: bool = True,
) -> Task:
    """Задание по контракту ТЗ §7."""
    return Task(
        task_id=f"task-{document_id}",
        payload=TaskPayload(
            pdf_path=str(pdf),
            document_id=document_id,
            options=TaskOptions(mode=mode, use_vlm=use_vlm),
        ),
    )


def _read_json(path: Path) -> dict:
    """Читает JSON-артефакт."""
    return json.loads(path.read_text(encoding="utf-8"))


def test_t1_full_on_simple_pdf(
    examples: dict[str, Path],
    output_dir: Path,
    mock_vlm: None,
) -> None:
    """Т1: полный прогон simple.pdf и сходимость всех счётчиков."""
    result = run(_task(examples["simple.pdf"], document_id="doc_t1"))

    # 1. Все 8 артефактов записаны и перечислены в результате.
    doc_dir = output_dir / "doc_t1"
    created = {path.name for path in doc_dir.iterdir()}
    assert created == set(ARTIFACT_FILES.values())
    assert set(result.artifacts) == set(ARTIFACT_FILES)
    for key, filename in ARTIFACT_FILES.items():
        path = doc_dir / filename
        assert path.is_file()
        assert Path(result.artifacts[key]).resolve() == path.resolve()

    # 2. Минимальные объёмы эталонной схемы.
    bom = _read_json(doc_dir / "bom.json")["items"]
    components = _read_json(doc_dir / "components.json")["components"]
    nets = _read_json(doc_dir / "netlist.json")["nets"]
    issues = _read_json(doc_dir / "issues.json")["issues"]

    assert len(bom) >= 4
    assert len(nets) >= 2
    assert len(components) >= 4
    assert {item["ref_designation"] for item in bom} >= {"F1", "SB1", "K1", "HL1"}
    assert all(item["found_on_schematic"] for item in bom)

    # 3. Счётчики summary совпадают с содержимым артефактов.
    assert result.summary.pages_processed == 1
    assert result.summary.components_found == len(components)
    assert result.summary.nets_found == len(nets)
    assert result.summary.bom_lines_found == len(bom)
    assert result.summary.issues_found == len(issues)
    assert result.summary.vlm_calls == 0  # в simple нет триггеров VLM


def test_t8_multipage_l1_net_merged(
    examples: dict[str, Path], output_dir: Path
) -> None:
    """Т8: метка L1 на двух листах объединяет цепь, pages=[1,2]."""
    result = run(
        _task(
            examples["multipage.pdf"],
            mode="netlist_only",
            document_id="doc_t8",
        )
    )

    nets = _read_json(output_dir / "doc_t8" / "netlist.json")["nets"]
    l1_nets = [net for net in nets if net["name"] == "L1"]
    assert len(l1_nets) == 1, "цепи L1 с двух страниц должны быть объединены"
    assert l1_nets[0]["pages"] == [1, 2]
    assert result.summary.nets_found == len(nets)


def test_t4_problem_pdf_raster_and_crossing(
    examples: dict[str, Path],
    output_dir: Path,
    mock_vlm: None,
) -> None:
    """Т4: растр → completed_with_issues, пересечение без узла не соединено."""
    pdf = examples["problem.pdf"]

    result = run(_task(pdf, document_id="doc_t4"))

    # Р3: растровая страница — деградация, статус with_issues.
    assert result.status == "completed_with_issues"
    issues = _read_json(output_dir / "doc_t4" / "issues.json")["issues"]
    issue_types = {item["type"] for item in issues}
    assert "raster_only_page" in issue_types

    # Пересечение без точки: две отдельные цепи, развязка зафиксирована.
    extraction = extract_pdf(str(pdf))
    geometry = build_geometry(extraction)
    page2_crossings = [
        crossing for crossing in geometry.crossings if crossing.page == 2
    ]
    assert len(page2_crossings) == 1
    assert page2_crossings[0].attributes["connected"] is False

    components, _ = detect_components(geometry, extraction)
    nets, _ = build_netlist(geometry, components, extraction)
    page2_nets = [net for net in nets if 2 in net.pages]
    assert len(page2_nets) == 2, "пересечение без узла не должно склеивать цепи"

    # VLM вызывается только через мок-транспорт (пересечение + пустая BOM).
    assert result.summary.vlm_calls >= 1
    log_lines = [
        line
        for line in (output_dir / "doc_t4" / "vlm_calls.log")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    records = [json.loads(line) for line in log_lines]
    assert any(record.get("status") == "ok" for record in records)


def test_examples_generator_creates_three_pdfs(tmp_path: Path) -> None:
    """Генератор создаёт ровно три эталонных PDF с ожидаемым числом листов."""
    if not CYRILLIC_FONT.is_file():
        pytest.skip("нет шрифта DejaVu для кириллицы")
    created = generate_all(tmp_path / "examples")

    assert set(created) == {"simple.pdf", "multipage.pdf", "problem.pdf"}
    page_counts = {}
    for name, path in created.items():
        assert path.is_file() and path.stat().st_size > 0
        document = pymupdf.open(str(path))
        page_counts[name] = document.page_count
        document.close()
    assert page_counts == {
        "simple.pdf": 1,
        "multipage.pdf": 2,
        "problem.pdf": 3,
    }


def test_examples_cyrillic_is_readable(examples: dict[str, Path]) -> None:
    """Кириллица в эталонах извлекается как текст, а не «····»."""
    extraction = extract_pdf(str(examples["simple.pdf"]))
    words = {span.text for span in extraction.all_texts()}

    assert "Предохранитель" in words
    assert "Лампа" in words
    assert not any("·" in word for word in words)