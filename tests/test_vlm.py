"""Тесты этапа 8: точечная VLM-проверка на mock-транспорте.

Реальная сеть запрещена: все вызовы идут через injection-параметр
``transport`` или monkeypatch ``_default_transport``.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pymupdf
import pytest

from schematic_agent import run
from schematic_agent.config import OUTPUT_DIR_ENV, VlmSettings
from schematic_agent.contract import Component, Issue, Task, TaskOptions, TaskPayload
from schematic_agent.geometry import build_geometry
from schematic_agent.parser.pdf_extract import PdfExtraction, extract_pdf
from schematic_agent.vlm import run_vlm_review


def _schema_pdf(path: Path, *, refs: tuple[str, ...] = ("VT1",)) -> Path:
    """PDF с несколькими неопознанными УГО (линии + обозначения)."""
    document = pymupdf.open()
    page = document.new_page(width=300, height=200)
    for index, ref in enumerate(refs):
        x = 20 + index * 70
        page.draw_line((x, 20), (x + 14, 34))
        page.insert_text((x + 4, 50), ref)
    document.save(str(path))
    document.close()
    return path


def _inputs(path: Path) -> tuple[PdfExtraction, object]:
    extraction = extract_pdf(str(path))
    return extraction, build_geometry(extraction)


def _components(refs: tuple[str, ...]) -> list[Component]:
    return [
        Component(
            id=f"c{index + 1:03d}",
            type="unknown",
            ref_designation=ref,
            x=27.0 + index * 70,
            y=27.0,
            attributes={
                "bbox": [20.0 + index * 70, 20.0, 34.0 + index * 70, 34.0],
                "page": 1,
            },
            confidence=0.3,
        )
        for index, ref in enumerate(refs)
    ]


def _issues(refs: tuple[str, ...]) -> list[Issue]:
    return [
        Issue(
            type="unrecognized_symbol",
            page=1,
            x=27.0 + index * 70,
            y=27.0,
            description=f"unknown {ref}",
            confidence=0.3,
        )
        for index, ref in enumerate(refs)
    ]


def _ok_body(symbol: str = "resistor", confidence: float = 0.9) -> bytes:
    return json.dumps(
        {
            "response": json.dumps(
                {
                    "symbol": symbol,
                    "confidence": confidence,
                    "description": "mock",
                }
            )
        }
    ).encode()


class FakeTransport:
    """Последовательный mock HTTP-транспорт без реальной сети."""

    def __init__(self, responses: list[bytes | tuple[int, bytes] | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url: str, payload: dict, timeout_s: float):
        self.calls.append(payload)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, tuple):
            return response
        return 200, response


def test_no_vlm_makes_zero_calls_and_clean_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Т6: use_vlm=False/mode=no_vlm → 0 вызовов, лог пуст, артефакты ок."""
    pdf = _schema_pdf(tmp_path / "schema.pdf")
    output = tmp_path / "out"
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(output))
    task = Task(
        task_id="t-no-vlm",
        payload=TaskPayload(
            pdf_path=str(pdf),
            document_id="doc_no_vlm",
            options=TaskOptions(mode="no_vlm", use_vlm=False),
        ),
    )

    result = run(task)

    assert result.summary.vlm_calls == 0
    log = output / "doc_no_vlm" / "vlm_calls.log"
    assert log.is_file()
    assert log.read_text(encoding="utf-8") == ""
    assert result.status in {"completed", "completed_with_issues"}


def test_ok_json_updates_component_and_counts_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock 200 ok-json → тип компонента обновлён, vlm_calls=1."""
    pdf = _schema_pdf(tmp_path / "schema.pdf")
    extraction, geometry = _inputs(pdf)
    components = _components(("VT1",))
    log = tmp_path / "vlm_calls.log"
    transport = FakeTransport([_ok_body("resistor", 0.93)])

    components, issues, calls = run_vlm_review(
        extraction,
        geometry,
        components,
        _issues(("VT1",)),
        VlmSettings(max_calls=5),
        transport=transport,
        log_path=log,
        pdf_path=str(pdf),
    )

    assert calls == 1
    assert transport.calls[0]["stream"] is False
    assert transport.calls[0]["model"] == "qwen3-vl:2b"
    assert components[0].type == "resistor"
    assert components[0].confidence == pytest.approx(0.93)
    assert components[0].attributes["vlm"]["description"] == "mock"
    assert not any(issue.type == "unrecognized_symbol" for issue in issues)


def test_invalid_json_retries_once_and_keeps_unknown(
    tmp_path: Path,
) -> None:
    """Т11: невалидный JSON → ровно 1 повтор → vlm_bad_json, unknown."""
    pdf = _schema_pdf(tmp_path / "schema.pdf")
    extraction, geometry = _inputs(pdf)
    components = _components(("VT1",))
    invalid = json.dumps({"response": "not a json"}).encode()
    transport = FakeTransport([invalid, invalid])

    components, issues, calls = run_vlm_review(
        extraction,
        geometry,
        components,
        _issues(("VT1",)),
        VlmSettings(max_calls=5),
        transport=transport,
        log_path=tmp_path / "vlm_calls.log",
        pdf_path=str(pdf),
    )

    assert calls == 2
    assert len(transport.calls) == 2
    assert components[0].type == "unknown"
    assert any(issue.type == "vlm_bad_json" for issue in issues)


def test_connection_refused_reports_unavailable(
    tmp_path: Path,
) -> None:
    """Т7: connection refused → vlm_unavailable, исключения нет."""
    pdf = _schema_pdf(tmp_path / "schema.pdf")
    extraction, geometry = _inputs(pdf)
    components = _components(("VT1",))
    transport = FakeTransport(
        [urllib.error.URLError(ConnectionRefusedError("refused"))]
    )

    components, issues, calls = run_vlm_review(
        extraction,
        geometry,
        components,
        _issues(("VT1",)),
        VlmSettings(max_calls=5),
        transport=transport,
        log_path=tmp_path / "vlm_calls.log",
        pdf_path=str(pdf),
    )

    assert calls == 1
    assert components[0].type == "unknown"
    assert any(issue.type == "vlm_unavailable" for issue in issues)


def test_budget_exceeded_stops_calls(tmp_path: Path) -> None:
    """max_calls=1 при 3 триггерах → 1 вызов + vlm_budget_exceeded."""
    pdf = _schema_pdf(tmp_path / "schema.pdf", refs=("VT1", "VT2", "VT3"))
    extraction, geometry = _inputs(pdf)
    components = _components(("VT1", "VT2", "VT3"))
    transport = FakeTransport([_ok_body()])

    _, issues, calls = run_vlm_review(
        extraction,
        geometry,
        components,
        _issues(("VT1", "VT2", "VT3")),
        VlmSettings(max_calls=1),
        transport=transport,
        log_path=tmp_path / "vlm_calls.log",
        pdf_path=str(pdf),
    )

    assert calls == 1
    assert len(transport.calls) == 1
    assert any(issue.type == "vlm_budget_exceeded" for issue in issues)


def test_log_contains_attempt_records(tmp_path: Path) -> None:
    """Лог vlm_calls.log содержит JSON-записи каждой попытки."""
    pdf = _schema_pdf(tmp_path / "schema.pdf")
    extraction, geometry = _inputs(pdf)
    log = tmp_path / "vlm_calls.log"
    transport = FakeTransport([_ok_body()])

    run_vlm_review(
        extraction,
        geometry,
        _components(("VT1",)),
        _issues(("VT1",)),
        VlmSettings(max_calls=5),
        transport=transport,
        log_path=log,
        pdf_path=str(pdf),
    )

    records = [
        json.loads(line) for line in log.read_text().splitlines()
    ]
    attempts = [record for record in records if not record.get("summary")]
    assert len(attempts) == 1
    assert records[-1]["summary"] is True
    assert records[-1]["vlm_calls"] == 1
    record = attempts[0]
    assert record["status"] == "ok"
    assert record["trigger"] == "unrecognized_symbol"
    assert record["page"] == 1
    assert record["model"] == "qwen3-vl:2b"
    assert "latency_s" in record and "crop_bbox" in record


def test_model_missing_404_reports_issue(tmp_path: Path) -> None:
    """404 от Ollama (модели нет) → vlm_model_missing, без повторов."""
    pdf = _schema_pdf(tmp_path / "schema.pdf")
    extraction, geometry = _inputs(pdf)
    transport = FakeTransport([(404, b'{"error":"model not found"}')])

    _, issues, calls = run_vlm_review(
        extraction,
        geometry,
        _components(("VT1",)),
        _issues(("VT1",)),
        VlmSettings(max_calls=5),
        transport=transport,
        log_path=tmp_path / "vlm_calls.log",
        pdf_path=str(pdf),
    )

    assert calls == 1
    assert any(issue.type == "vlm_model_missing" for issue in issues)


def test_pipeline_full_updates_component_via_mock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Интеграция: full вызывает этап 8 и обновляет components.json."""
    pdf = _schema_pdf(tmp_path / "schema.pdf")
    output = tmp_path / "out"
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(output))
    calls: list[dict] = []

    def fake_transport(url: str, payload: dict, timeout_s: float):
        calls.append(payload)
        return 200, _ok_body("resistor", 0.88)

    monkeypatch.setattr("schematic_agent.vlm._default_transport", fake_transport)
    task = Task(
        task_id="t-full-vlm",
        payload=TaskPayload(
            pdf_path=str(pdf),
            document_id="doc_full_vlm",
            options=TaskOptions(mode="full"),
        ),
    )

    result = run(task)

    assert result.summary.vlm_calls == 1
    assert len(calls) == 1
    payload = json.loads(
        (output / "doc_full_vlm" / "components.json").read_text(encoding="utf-8")
    )
    component = next(
        item for item in payload["components"]
        if item["ref_designation"] == "VT1"
    )
    assert component["type"] == "resistor"
    issues = json.loads(
        (output / "doc_full_vlm" / "issues.json").read_text(encoding="utf-8")
    )
    assert not any(
        item["type"] == "unrecognized_symbol" for item in issues["issues"]
    )


def test_budget_caps_retry_of_bad_json(tmp_path: Path) -> None:
    """T13: повтор bad JSON не превышает max_calls (жёсткий бюджет)."""
    pdf = _schema_pdf(tmp_path / "schema.pdf")
    extraction, geometry = _inputs(pdf)
    components = _components(("VT1",))
    invalid = json.dumps({"response": "not a json"}).encode()
    transport = FakeTransport([invalid, invalid])

    _, _, calls = run_vlm_review(
        extraction,
        geometry,
        components,
        _issues(("VT1",)),
        VlmSettings(max_calls=1),
        transport=transport,
        log_path=tmp_path / "vlm_calls.log",
        pdf_path=str(pdf),
    )

    assert calls == 1
    assert len(transport.calls) == 1
