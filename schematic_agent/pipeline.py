"""Конвейер субагента: валидация задания, артефакты, результат.

Полный конвейер по ТЗ §3::

    PDF-парсер → геометрия → граф цепей → BOM → VLM-проверка → сверка → отчёт

Реализованы этапы 2–10: извлечение PDF, геометрия, компоненты, BOM,
Netlist, анализ ПЛК, точечная VLM-проверка, единый отчёт и надёжность.
``run()`` поддерживает режимы ТЗ §13 и записывает 8 артефактов в
``<output_dir>/<document_id>/``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from .artifacts import write_artifact
from .bom import (
    BOM_SOURCE_TABLE,
    build_bom_with_source,
    write_bom_csv,
)
from .components import detect_components
from .config import get_settings
from .contract import (
    BomItem,
    BomOutput,
    Component,
    ComponentsOutput,
    Issue,
    IssuesOutput,
    Net,
    NetsOutput,
    PlcAnalysis,
    Result,
    Summary,
    Task,
    TaskOptions,
)
from .geometry import GeometryResult, build_geometry
from .issues import collect_issues, default_action
from .metrics import evaluate, load_ground_truth, write_metrics
from .netlist import build_netlist
from .parser.pdf_extract import PdfExtraction, extract_pdf
from .plc import analyze_plc
from .report import build_report, write_report
from .vlm import run_vlm_review

#: Имена файлов артефактов (ТЗ §9) и ключи в ``Result.artifacts``.
ARTIFACT_FILES: dict[str, str] = {
    "bom": "bom.json",
    "bom_csv": "bom.csv",
    "netlist": "netlist.json",
    "components": "components.json",
    "issues": "issues.json",
    "plc_analysis": "plc_analysis.json",
    "report": "report.md",
    "vlm_log": "vlm_calls.log",
}

#: Режимы работы (ТЗ §13).
MODES: tuple[str, ...] = (
    "full",
    "no_vlm",
    "bom_only",
    "netlist_only",
    "issue_review",
)

#: In-memory хранилище результатов ``submit()``.
_RESULTS: dict[str, Result] = {}


def _write_json(path: Path, payload: Any) -> None:
    """Записывает JSON-артефакт."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


#: Режимы, в которых выполняется этап BOM (ТЗ §13).
_MODES_WITH_BOM: frozenset[str] = frozenset(
    {"full", "no_vlm", "issue_review", "bom_only"}
)

#: Режимы, в которых выполняется этап Netlist (ТЗ §13).
_MODES_WITH_NETLIST: frozenset[str] = frozenset(
    {"full", "no_vlm", "issue_review", "netlist_only"}
)

#: Режимы, в которых выполняется анализ логики ПЛК (ТЗ §13).
_MODES_WITH_PLC: frozenset[str] = frozenset({"full", "no_vlm", "issue_review"})


def _write_artifacts(
    output_dir: Path,
    components: list[Component],
    issues: list[Issue],
    bom_items: list[BomItem],
    nets: list[Net],
    plc: PlcAnalysis | None,
    bom_source: str = BOM_SOURCE_TABLE,
) -> dict[str, str]:
    """Записывает 8 артефактов и возвращает их пути.

    Args:
        output_dir: Каталог артефактов ``<output_dir>/<document_id>``.
        components: Распознанные компоненты (этап 4).
        issues: Проблемы всех выполненных этапов.
        bom_items: Строки BOM (этап 5).
        nets: Электрические цепи (этап 6).
        plc: Анализ ПЛК (этап 7) или ``None`` для режимов без ПЛК.
        bom_source: Источник BOM (``table`` / ``components_fallback``).

    Returns:
        Отображение «ключ артефакта → абсолютный путь».
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: str(output_dir / name) for key, name in ARTIFACT_FILES.items()}

    # Round-trip через корневые pydantic-схемы (M3): контракт проверяется
    # до записи, поэтому на диск попадает только валидная структура.
    write_artifact(
        paths["bom"],
        BomOutput(items=bom_items, source=bom_source),  # type: ignore[arg-type]
    )
    write_bom_csv(bom_items, paths["bom_csv"])
    write_artifact(paths["netlist"], NetsOutput(nets=nets))
    write_artifact(paths["components"], ComponentsOutput(components=components))
    write_artifact(paths["issues"], IssuesOutput(issues=issues))
    analysis = plc or PlcAnalysis()
    _write_json(Path(paths["plc_analysis"]), analysis.model_dump())
    # Журнал VLM — append-only: создаём файл, если его ещё нет, но никогда
    # не перезаписываем записи прошлых попыток.
    Path(paths["vlm_log"]).touch(exist_ok=True)
    return paths


def _options_for(options: TaskOptions) -> tuple[bool, bool, bool]:
    """Определяет, какие этапы выполнять для режима ТЗ §13.

    Returns:
        Кортеж ``(bom, netlist, plc)`` — флаги выполнения этапов 5–7.
    """
    mode = options.mode
    # issue_review пока обрабатывается как full (ТЗ §13, этап 10).
    # VLM-этап 8 не реализован: use_vlm влияет только на будущие вызовы.
    return (
        mode in _MODES_WITH_BOM,
        mode in _MODES_WITH_NETLIST,
        mode in _MODES_WITH_PLC,
    )


def run(task: Task, ground_truth_path: str | None = None) -> Result:
    """Выполняет задание и возвращает единый результат.

    Единый маппинг статусов (этап 10, ТЗ §11):

    * нет файла → ``failed`` (``file_not_found``);
    * повреждённый PDF (``parse_error``) → ``failed``;
    * пустой файл → ``failed`` (``error="empty_file"``);
    * ошибка страницы/общий таймаут → ``partial`` с результатами по
      обработанным страницам;
    * растровый PDF без векторного слоя → ``completed_with_issues`` +
      ``raster_only_page`` (честная отчётность, ТЗ §8/§11);
    * остальное → ``completed_with_issues`` при наличии проблем, иначе
      ``completed``.

    Args:
        task: Валидное задание :class:`~schematic_agent.contract.Task`.
        ground_truth_path: Необязательный путь к ``ground_truth.json`` (M4).
            При задании дополнительно пишется ``metrics.json`` и раздел
            «Метрики качества» в ``report.md``.

    Returns:
        Единый результат обработки.
    """
    pdf_path = Path(task.payload.pdf_path)
    if not pdf_path.is_file():
        return Result(
            task_id=task.task_id,
            status="failed",
            document_id=task.payload.document_id,
            summary=Summary(),
            artifacts={},
            error=f"file_not_found: {task.payload.pdf_path}",
        )

    settings = get_settings()
    output_dir = Path(settings.output_dir) / task.payload.document_id
    stage_issues: list[Issue] = []

    def timed_stage(name: str, func: Callable[..., Any], *args: Any) -> Any:
        """Выполняет этап и фиксирует превышение стражного таймаута.

        Python не прерывает уже выполняющуюся функцию, поэтому страж
        срабатывает после возврата этапа: результат считается частичным,
        формируется Issue ``stage_timeout``, конвейер продолжает работу.
        """
        started = time.monotonic()
        value = func(*args)
        elapsed = time.monotonic() - started
        if elapsed > settings.stage_timeout_s:
            stage_issues.append(
                Issue(
                    type="stage_timeout",
                    page=0,
                    description=(
                        f"Этап {name} занял {elapsed:.3f} с, что превышает "
                        f"стражный таймаут {settings.stage_timeout_s:.3f} с"
                    ),
                    confidence=1.0,
                    required_action=default_action("stage_timeout"),
                )
            )
        return value

    extraction = _stage_extract(
        str(pdf_path), doc_timeout_s=settings.doc_timeout_s
    )
    geometry = _stage_geometry(extraction)
    components, component_issues = timed_stage(
        "components", _stage_components, geometry, extraction
    )
    issues: list[Issue] = collect_issues(extraction.issues, component_issues)
    issues = collect_issues(issues, stage_issues)
    bom_items: list[BomItem] = []
    bom_source = BOM_SOURCE_TABLE
    nets: list[Net] = []
    plc: PlcAnalysis | None = None
    vlm_calls = 0

    file_empty = pdf_path.stat().st_size == 0
    parse_errors = [issue for issue in issues if issue.type == "parse_error"]
    parse_failed = not extraction.opened and bool(parse_errors)
    empty_document = not parse_failed and (
        file_empty or (not extraction.pages and not extraction.issues)
    )

    if empty_document:
        if not any(issue.type == "empty_file" for issue in issues):
            issues = collect_issues(
                issues,
                [
                    Issue(
                        type="empty_file",
                        description="PDF-файл пуст или не содержит страниц",
                        confidence=1.0,
                        required_action="Передать непустой PDF",
                    )
                ],
            )
    elif not parse_failed:
        run_bom, run_netlist, run_plc = _options_for(task.payload.options)

        if run_bom:
            bom_items, bom_issues, bom_source = timed_stage(
                "bom", _stage_bom, extraction, components
            )
            # В1: отсутствие BOM-таблицы — проблема, если BOM запрошен
            # (full/no_vlm/bom_only/issue_review) и extract_bom=True.
            # В netlist_only этап BOM не выполняется и проблема не шумит.
            if extraction.all_tables() or task.payload.options.extract_bom:
                issues = collect_issues(issues, bom_issues)
            issues = collect_issues(issues, stage_issues)
        if run_netlist:
            nets, net_issues = timed_stage(
                "netlist", _stage_netlist, geometry, components, extraction
            )
            issues = collect_issues(issues, net_issues)
            issues = collect_issues(issues, stage_issues)
        if run_plc:
            plc = timed_stage("plc", _stage_plc, nets, components, issues)
            issues = collect_issues(issues, stage_issues)

        # Этап 8: VLM только в full/issue_review, строго последовательно.
        run_vlm = (
            task.payload.options.mode in {"full", "issue_review"}
            and task.payload.options.use_vlm
            and settings.vlm.enabled
        )
        if run_vlm:
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / ARTIFACT_FILES["vlm_log"]
            components, issues, vlm_calls = run_vlm_review(
                extraction,
                geometry,
                components,
                issues,
                settings.vlm,
                log_path=log_path,
                pdf_path=str(pdf_path),
            )

    issues = collect_issues(issues, stage_issues)

    # M4: метрики качества считаются до записи артефактов, чтобы ошибка
    # ground truth попала в issues.json, а не терялась.
    quality_report = None
    if ground_truth_path:
        try:
            truth = load_ground_truth(ground_truth_path)
            quality_report = evaluate(components, nets, bom_items, truth)
        except Exception as error:  # noqa: BLE001 - GT не должен ронять прогон
            issues = collect_issues(
                issues,
                [
                    Issue(
                        type="metrics_error",
                        description=(
                            f"Не удалось вычислить метрики по "
                            f"{ground_truth_path}: {error}"
                        ),
                        confidence=0.5,
                        required_action=default_action("metrics_error"),
                    )
                ],
            )

    status = _result_status(extraction, issues, parse_failed, empty_document)
    error: str | None = None
    if empty_document:
        error = "empty_file"
    elif parse_failed:
        error = f"parse_error: {parse_errors[0].description}"

    summary = Summary(
        pages_processed=extraction.page_count,
        components_found=len(components),
        nets_found=len(nets),
        bom_lines_found=len(bom_items),
        issues_found=len(issues),
        vlm_calls=vlm_calls,
    )
    artifacts = _write_artifacts(
        output_dir, components, issues, bom_items, nets, plc, bom_source
    )
    if quality_report is not None:
        artifacts["metrics"] = write_metrics(
            quality_report, output_dir / "metrics.json"
        )
    result = Result(
        task_id=task.task_id,
        status=status,
        document_id=task.payload.document_id,
        summary=summary,
        artifacts=artifacts,
        error=error,
    )
    write_report(
        build_report(
            result,
            {
                "components": components,
                "bom": bom_items,
                "nets": nets,
                "plc": plc,
                "issues": issues,
                "vlm_log": artifacts.get("vlm_log"),
                "metrics": quality_report,
            },
        ),
        artifacts["report"],
    )
    return result


def _result_status(
    extraction: PdfExtraction,
    issues: list[Issue],
    parse_failed: bool,
    empty_document: bool,
) -> str:
    """Вычисляет статус результата по единой семантике (ТЗ §8, §11)."""
    if empty_document or parse_failed:
        return "failed"
    issue_types = {issue.type for issue in issues}
    if issue_types & {
        "page_failed",
        "doc_timeout",
        "stage_timeout",
        "detector_capped",
    }:
        return "partial"
    # Р3: растровые страницы — это деградация (OCR required), поэтому
    # документ с raster_only_page не может считаться полностью успешным.
    # Отдельная ветка не нужна: raster_only_page входит в issues.
    return "completed_with_issues" if issues else "completed"


def submit(task: Task) -> str:
    """Принимает задание, выполняет его и сохраняет результат.

    Args:
        task: Валидное задание.

    Returns:
        ``task_id`` задания.
    """
    _RESULTS[task.task_id] = run(task)
    return task.task_id


def get_result(task_id: str) -> Result:
    """Возвращает сохранённый результат по ``task_id``.

    Args:
        task_id: Идентификатор задания, ранее переданного в :func:`submit`.

    Returns:
        Результат обработки.

    Raises:
        KeyError: Если результат для ``task_id`` не найден.
    """
    if task_id not in _RESULTS:
        raise KeyError(f"result_not_found: {task_id}")
    return _RESULTS[task_id]


def health_check() -> dict[str, Any]:
    """Возвращает состояние субагента."""
    return {"status": "ok"}


def describe_capabilities() -> dict[str, Any]:
    """Описывает поддерживаемые возможности и режимы."""
    return {
        "task_type": "parse_electrical_schematic_pdf",
        "modes": list(MODES),
        "vlm": {
            "model": "qwen3-vl:2b",
            "max_crop": 768,
            "can_disable": True,
        },
    }


# --- Этапы конвейера --------------------------------------------------------


def _stage_extract(
    pdf_path: str,
    page: int | None = None,
    *,
    doc_timeout_s: float | None = None,
) -> PdfExtraction:
    """Этап 2: извлечение текста, примитивов и таблиц.

    Args:
        pdf_path: Путь к PDF-файлу.
        page: Не используется: извлекается весь документ.
        doc_timeout_s: Мягкий общий таймаут документа, с.

    Returns:
        Результат извлечения PDF.
    """
    return extract_pdf(pdf_path, doc_timeout_s=doc_timeout_s)


def _stage_geometry(extraction: PdfExtraction) -> GeometryResult:
    """Этап 3: геометрическая нормализация и построение узлов."""
    return build_geometry(extraction)


def _stage_components(
    geometry: GeometryResult, extraction: PdfExtraction
) -> tuple[list[Component], list[Issue]]:
    """Этап 4: распознавание компонентов и проблем извлечения."""
    return detect_components(geometry, extraction)


def _stage_bom(
    extraction: PdfExtraction, components: list[Component]
) -> tuple[list[BomItem], list[Issue], str]:
    """Этап 5: извлечение BOM, фолбэк по компонентам и сверка со схемой."""
    return build_bom_with_source(extraction, components)


def _stage_netlist(
    geometry: GeometryResult,
    components: list[Component],
    extraction: PdfExtraction,
) -> tuple[list[Net], list[Issue]]:
    """Этап 6: построение графа цепей."""
    return build_netlist(geometry, components, extraction)


def _stage_plc(
    nets: list[Net], components: list[Component], issues: list[Issue]
) -> PlcAnalysis:
    """Этап 7: анализ логики ПЛК."""
    return analyze_plc(nets, components, issues)


def _stage_vlm(candidates: Any) -> Any:
    """Этап 8: точечная VLM-проверка спорных участков (следующий пакет)."""
    raise NotImplementedError("Этап VLM не реализован (пакет 4)")


def _stage_report(result: Result, details: dict[str, Any]) -> str:
    """Этап 9: формирование ``report.md``."""
    return build_report(result, details)


PIPELINE_STAGES: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("extract", _stage_extract),
    ("geometry", _stage_geometry),
    ("components", _stage_components),
    ("netlist", _stage_netlist),
    ("bom", _stage_bom),
    ("vlm", _stage_vlm),
    ("report", _stage_report),
)


__all__ = [
    "ARTIFACT_FILES",
    "MODES",
    "PIPELINE_STAGES",
    "run",
    "submit",
    "get_result",
    "health_check",
    "describe_capabilities",
]
