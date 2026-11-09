"""Тесты публичного API: run/submit/get_result/health_check/describe_capabilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from schematic_agent import (
    describe_capabilities,
    get_result,
    health_check,
    run,
    submit,
)
from schematic_agent.config import OUTPUT_DIR_ENV
from schematic_agent.contract import Summary, Task, TaskOptions, TaskPayload
from schematic_agent.pipeline import ARTIFACT_FILES, MODES


def make_task(pdf_path: Path, *, task_id: str = "task-001") -> Task:
    """Создаёт задание по контракту ТЗ §7."""
    return Task(
        task_id=task_id,
        payload=TaskPayload(
            pdf_path=str(pdf_path),
            document_id="doc_001",
            options=TaskOptions(),
        ),
    )


@pytest.fixture
def output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Перенаправляет артефакты во временный каталог."""
    target = tmp_path / "out"
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(target))
    return target


def test_run_missing_file_returns_failed_without_artifacts(
    tmp_path: Path, output_dir: Path
) -> None:
    """Несуществующий PDF → failed, артефакты не создаются."""
    missing = tmp_path / "missing.pdf"
    task = make_task(missing)

    result = run(task)

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.startswith("file_not_found: ")
    assert result.artifacts == {}
    assert result.summary == Summary()
    assert not (output_dir / "doc_001").exists()


def test_run_existing_pdf_creates_eight_artifacts(
    tmp_path: Path, output_dir: Path
) -> None:
    """Пустой PDF → failed empty_file, но 8 артефактов записаны."""
    pdf = tmp_path / "schema.pdf"
    pdf.write_bytes(b"")

    result = run(make_task(pdf))

    doc_dir = output_dir / "doc_001"
    # Семантика этапа 10: пустой файл — failed с error="empty_file".
    assert result.status == "failed"
    assert result.error == "empty_file"
    assert result.summary.pages_processed == 0
    assert result.summary.components_found == 0
    assert result.summary.issues_found == 1  # empty_file

    created = {path.name for path in doc_dir.iterdir()}
    assert created == set(ARTIFACT_FILES.values())
    assert len(created) == 8

    assert set(result.artifacts) == set(ARTIFACT_FILES)
    for key, filename in ARTIFACT_FILES.items():
        expected = (doc_dir / filename).resolve()
        assert Path(result.artifacts[key]).resolve() == expected
        assert expected.is_file()

    bom = json.loads((doc_dir / "bom.json").read_text(encoding="utf-8"))
    assert bom["items"] == []
    assert bom["source"] == "table"
    assert json.loads((doc_dir / "netlist.json").read_text(encoding="utf-8")) == {
        "nets": []
    }
    assert (doc_dir / "vlm_calls.log").read_text(encoding="utf-8") == ""


def test_submit_then_get_result_returns_same_result(
    tmp_path: Path, output_dir: Path
) -> None:
    """submit → get_result возвращает тот же Result."""
    pdf = tmp_path / "schema.pdf"
    pdf.write_bytes(b"")
    task = make_task(pdf, task_id="task-submit-001")

    task_id = submit(task)
    result = get_result(task_id)

    assert task_id == task.task_id
    assert result.task_id == task.task_id
    assert result.status == "failed"
    assert result.error == "empty_file"
    assert get_result(task_id) is result


def test_get_result_unknown_task_raises() -> None:
    """Неизвестный task_id → KeyError."""
    with pytest.raises(KeyError):
        get_result("task-does-not-exist")


def test_health_check() -> None:
    """health_check возвращает ok."""
    assert health_check() == {"status": "ok"}


def test_describe_capabilities() -> None:
    """describe_capabilities описывает task_type, режимы и VLM."""
    capabilities = describe_capabilities()

    assert capabilities["task_type"] == "parse_electrical_schematic_pdf"
    assert capabilities["modes"] == list(MODES)
    assert set(capabilities["modes"]) == {
        "full",
        "no_vlm",
        "bom_only",
        "netlist_only",
        "issue_review",
    }
    assert capabilities["vlm"] == {
        "model": "qwen3-vl:2b",
        "max_crop": 768,
        "can_disable": True,
    }
