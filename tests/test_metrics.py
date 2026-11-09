"""Тесты M4: метрики качества по ground truth."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.generate_examples import generate_all, ground_truth_path
from schematic_agent.cli import main
from schematic_agent.config import OUTPUT_DIR_ENV
from schematic_agent.contract import Task, TaskOptions, TaskPayload
from schematic_agent.metrics import (
    GroundTruth,
    QualityReport,
    evaluate,
)
from schematic_agent.pipeline import run


@pytest.fixture
def examples(tmp_path: Path) -> dict[str, Path]:
    """Генерирует PDF и эталоны рядом с ними."""
    return generate_all(tmp_path / "examples")


@pytest.fixture
def output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Перенаправляет артефакты во временный каталог."""
    target = tmp_path / "out"
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(target))
    return target


def _task(pdf: Path, document_id: str = "doc_metrics") -> Task:
    """Задание без VLM."""
    return Task(
        task_id="task-metrics",
        payload=TaskPayload(
            pdf_path=str(pdf),
            document_id=document_id,
            options=TaskOptions(mode="no_vlm", use_vlm=False),
        ),
    )


def test_generator_emits_ground_truth(examples: dict[str, Path]) -> None:
    """M4: ground_truth.json создан для simple и multipage, не для problem."""
    for name in ("simple.pdf", "multipage.pdf"):
        path = ground_truth_path(examples[name])
        assert path.is_file()
        GroundTruth.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert not ground_truth_path(examples["problem.pdf"]).exists()


def test_evaluate_simple_precision_recall_one(
    examples: dict[str, Path], output_dir: Path
) -> None:
    """M4/H8: simple.pdf + GT → component/BOM precision и recall = 1.0."""
    gt_path = ground_truth_path(examples["simple.pdf"])
    result = run(_task(examples["simple.pdf"]), ground_truth_path=str(gt_path))

    assert "metrics" in result.artifacts
    metrics_path = Path(result.artifacts["metrics"])
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = QualityReport.model_validate(payload)

    assert report.component_precision == 1.0
    assert report.component_recall == 1.0
    assert report.bom_reconciliation_precision == 1.0
    assert report.bom_reconciliation_recall == 1.0
    assert report.components_gt == 4
    assert report.components_matched == 4
    # Честные значения остальных метрик детерминированы шаблонами.
    assert report.pin_connectivity is not None
    assert 0.0 <= report.pin_connectivity <= 1.0
    assert report.net_connectivity is not None
    assert 0.0 <= report.net_connectivity <= 1.0

    report_md = (output_dir / "doc_metrics" / "report.md").read_text(
        encoding="utf-8"
    )
    assert "## Метрики качества" in report_md


def test_multipage_net_connectivity(
    examples: dict[str, Path], output_dir: Path
) -> None:
    """M4: L1 на двух листах multipage.pdf даёт Jaccard = 1.0."""
    gt_path = ground_truth_path(examples["multipage.pdf"])
    result = run(
        _task(examples["multipage.pdf"], "doc_multi"),
        ground_truth_path=str(gt_path),
    )

    report = QualityReport.model_validate(
        json.loads(Path(result.artifacts["metrics"]).read_text(encoding="utf-8"))
    )
    assert report.component_precision == 1.0
    assert report.component_recall == 1.0
    assert report.net_connectivity == 1.0
    assert report.pin_connectivity == 1.0


def test_cli_parse_with_gt(
    examples: dict[str, Path],
    output_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """M4: CLI ``parse --gt`` пишет metrics.json и печатает Result."""
    gt_path = ground_truth_path(examples["simple.pdf"])
    code = main(
        [
            "parse",
            str(examples["simple.pdf"]),
            "--mode",
            "no_vlm",
            "--output",
            str(output_dir),
            "--document-id",
            "doc_cli",
            "--gt",
            str(gt_path),
        ]
    )

    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert "metrics" in result["artifacts"]
    assert (output_dir / "doc_cli" / "metrics.json").is_file()


def test_evaluate_empty_denominators_are_none() -> None:
    """M4: пустой эталон → метрики не определены (None), а не 1.0."""
    report = evaluate([], [], [], GroundTruth())

    assert report.component_precision is None
    assert report.component_recall is None
    assert report.bom_reconciliation_precision is None
    assert report.bom_reconciliation_recall is None
    assert report.pin_connectivity is None
    assert report.net_connectivity is None
