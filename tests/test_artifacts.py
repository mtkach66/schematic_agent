"""Тесты M3: schema validation JSON-артефактов."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from schematic_agent.artifacts import validate_artifacts, write_artifact
from schematic_agent.cli import main
from schematic_agent.config import OUTPUT_DIR_ENV
from schematic_agent.contract import BomItem, BomOutput
from schematic_agent.pipeline import run
from schematic_agent.contract import Task, TaskOptions, TaskPayload


@pytest.fixture
def artifacts_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Полный прогон simple.pdf с артефактами во временном каталоге."""
    source = Path(__file__).resolve().parent.parent / "examples" / "simple.pdf"
    output = tmp_path / "out"
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(output))
    task = Task(
        task_id="task-artifacts",
        payload=TaskPayload(
            pdf_path=str(source),
            document_id="doc_schema",
            options=TaskOptions(mode="no_vlm", use_vlm=False),
        ),
    )
    run(task)
    return output / "doc_schema"


def test_write_artifact_round_trip(tmp_path: Path) -> None:
    """write_artifact пишет только модель, читаемую обратно той же схемой."""
    path = tmp_path / "bom.json"
    model = BomOutput(
        items=[BomItem(ref_designation="R1", name="Резистор")],
        source="table",
    )

    write_artifact(path, model)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["source"] == "table"
    assert BomOutput.model_validate(payload).items[0].ref_designation == "R1"


def test_full_run_artifacts_validate(artifacts_dir: Path) -> None:
    """M3/H6: артефакты полного прогона проходят pydantic-схемы."""
    report = validate_artifacts(artifacts_dir)

    assert report["ok"] is True, report
    assert set(report["artifacts"]) >= {
        "bom",
        "netlist",
        "components",
        "issues",
    }


def test_cli_validate_ok(artifacts_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """M3/H7: CLI validate печатает отчёт и возвращает 0."""
    code = main(["validate", str(artifacts_dir)])

    printed = json.loads(capsys.readouterr().out)
    assert code == 0
    assert printed["ok"] is True


def test_corrupted_bom_is_reported(
    artifacts_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """M3: исковерканный bom.json ловится и отчитывается, CLI возвращает 1."""
    bom_path = artifacts_dir / "bom.json"
    bom_path.write_text(
        json.dumps({"items": [{"ref_designation": "R1"}], "source": "weird"}),
        encoding="utf-8",
    )

    report = validate_artifacts(artifacts_dir)
    assert report["ok"] is False
    assert any("bom" in error for error in report["errors"])

    code = main(["validate", str(artifacts_dir)])
    printed = json.loads(capsys.readouterr().out)
    assert code == 1
    assert printed["ok"] is False


def test_corrupted_json_is_reported(artifacts_dir: Path) -> None:
    """M3: невалидный JSON фиксируется как json_error."""
    (artifacts_dir / "components.json").write_text("{not json", encoding="utf-8")

    report = validate_artifacts(artifacts_dir)

    assert report["ok"] is False
    assert any("json_error" in error for error in report["errors"])
