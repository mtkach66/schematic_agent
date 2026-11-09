"""Формирование итогового отчёта ``report.md``.

Собирает сводку, счётчики, компоненты по типам, BOM-сверку, Netlist,
анализ ПЛК, таблицу проблем и VLM-сводку (ТЗ §8, §9, §11). Отчёт содержит
только фактически полученные данные; отсутствующие разделы помечаются явно.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contract import Result
from .metrics import QualityReport, metrics_section


def _component_line(item: Any) -> str:
    """Строка таблицы компонента."""
    ref = getattr(item, "ref_designation", "") or "—"
    return (
        f"| {getattr(item, 'id', '')} | {getattr(item, 'type', '')} | {ref} | "
        f"{getattr(item, 'confidence', 0.0):.2f} |"
    )


def _bom_line(item: Any) -> str:
    """Строка таблицы BOM."""
    return (
        f"| {item.ref_designation} | {item.name} | {item.quantity} | "
        f"{item.category} | {item.parameters} | "
        f"{'да' if item.found_on_schematic else 'нет'} |"
    )


def _net_line(item: Any) -> str:
    """Строка таблицы цепей."""
    name = item.name or "—"
    labels = ", ".join(item.text_labels) or "—"
    return (
        f"| {item.id} | {name} | {len(item.nodes)} | "
        f"{len(item.component_pins)} | {labels} | "
        f"{', '.join(str(page) for page in item.pages)} |"
    )


def _type_counts(components: list[Any]) -> dict[str, int]:
    """Счётчик компонентов по типам."""
    counts: dict[str, int] = {}
    for component in components:
        component_type = str(getattr(component, "type", "") or "unknown")
        counts[component_type] = counts.get(component_type, 0) + 1
    return dict(sorted(counts.items()))


def _read_vlm_log(path: str | None) -> list[dict[str, Any]]:
    """Читает JSON Lines журнала VLM; ошибки формата пропускаются."""
    if not path:
        return []
    log_path = Path(path)
    if not log_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def multimodal_consistency_score(channel_a: int, channel_b: int) -> float | None:
    """Считает MULTIMODAL CONSISTENCY SCORE двух независимых каналов.

    Формула: ``(1 − |A−B| / max(A,B)) × 100 %``. При обоих нулях метрика
    не определена.

    Args:
        channel_a: Число объектов канала A (компоненты с обозначением).
        channel_b: Число объектов канала B (строки BOM).

    Returns:
        Скор в процентах или ``None``, если оба канала пусты.
    """
    maximum = max(channel_a, channel_b)
    if maximum <= 0:
        return None
    return (1.0 - abs(channel_a - channel_b) / maximum) * 100.0


def _vlm_confirmation_stats(
    components: list[Any], records: list[dict[str, Any]]
) -> tuple[bool, int, int]:
    """Считает подтверждение unknown-компонентов успешными вызовами VLM.

    Returns:
        Кортеж ``(vlm_used, confirmed, unknown_total)``: признак участия
        VLM, число unknown-компонентов с идентифицированным VLM символом и
        общее число компонентов, бывших unknown (текущие unknown + те, кому
        VLM записала атрибут ``vlm``).
    """
    calls = [record for record in records if not record.get("summary")]
    vlm_used = bool(calls) or any(
        "vlm" in (getattr(component, "attributes", {}) or {})
        for component in components
    )
    confirmed = 0
    unknown_total = 0
    for component in components:
        attributes = getattr(component, "attributes", {}) or {}
        vlm = attributes.get("vlm")
        is_unknown = str(getattr(component, "type", "") or "") == "unknown"
        if not is_unknown and not isinstance(vlm, dict):
            continue
        unknown_total += 1
        symbol = ""
        if isinstance(vlm, dict):
            symbol = str(vlm.get("symbol", "") or "").strip().lower()
        if symbol and symbol not in {"unknown", "unclear", "н/д", "none"}:
            confirmed += 1
    return vlm_used, confirmed, unknown_total


def _channel_audit_lines(
    components: list[Any], bom_items: list[Any], records: list[dict[str, Any]]
) -> list[str]:
    """Формирует раздел «Сверка каналов» с consistency-скором."""
    channel_a = sum(
        1
        for component in components
        if str(getattr(component, "ref_designation", "") or "").strip()
    )
    channel_b = len(bom_items)
    score = multimodal_consistency_score(channel_a, channel_b)
    lines = ["## Сверка каналов", ""]
    lines.append(f"- A — компонентов с обозначением на схеме: {channel_a}")
    lines.append(f"- B — строк BOM: {channel_b}")
    if score is None:
        lines.append(
            "- MULTIMODAL CONSISTENCY SCORE: не определён "
            "(оба канала пусты)"
        )
    else:
        lines.append(
            f"- MULTIMODAL CONSISTENCY SCORE: {score:.2f}% "
            "(формула: (1 − |A−B| / max(A,B)) × 100%)"
        )
    vlm_used, confirmed, unknown_total = _vlm_confirmation_stats(
        components, records
    )
    if vlm_used:
        if unknown_total:
            share = confirmed / unknown_total * 100.0
            lines.append(
                "- VLM: подтверждено unknown-компонентов "
                f"{confirmed} из {unknown_total} ({share:.2f}%)"
            )
        else:
            lines.append(
                "- VLM: unknown-компонентов не было, доля не определена"
            )
    return lines


def _vlm_lines(result: Result, issues: list[Any], log_path: str | None) -> list[str]:
    """Формирует раздел VLM-сводки."""
    records = _read_vlm_log(log_path)
    statuses: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    degradation_types = (
        "vlm_unavailable",
        "vlm_model_missing",
        "vlm_bad_json",
        "vlm_budget_exceeded",
    )
    degradations = [
        issue for issue in issues if getattr(issue, "type", "") in degradation_types
    ]
    lines = [
        "## VLM",
        "",
        f"- Вызовов VLM: {result.summary.vlm_calls}",
        f"- Записей в журнале: {len(records)}",
    ]
    if statuses:
        lines.append(
            "- Статусы попыток: "
            + ", ".join(
                f"{status}={count}" for status, count in sorted(statuses.items())
            )
        )
    else:
        lines.append("- Статусы попыток: — (VLM не вызывалась или отключена)")
    lines.append(f"- Деградаций VLM: {len(degradations)}")
    for issue in degradations:
        lines.append(f"  - {issue.type}: {issue.description}")
    return lines


def build_report(
    result: Result, details: dict[str, Any] | None = None
) -> str:
    """Формирует содержимое ``report.md``.

    Args:
        result: Итоговый результат обработки.
        details: Данные этапов: ``components``, ``bom``, ``nets``,
            ``plc``, ``issues``, ``vlm_log``.

    Returns:
        Markdown-текст отчёта.
    """
    details = details or {}
    components = list(details.get("components") or [])
    bom_items = list(details.get("bom") or [])
    nets = list(details.get("nets") or [])
    issues = list(details.get("issues") or [])
    plc = details.get("plc")
    vlm_log = details.get("vlm_log")
    metrics = details.get("metrics")
    if isinstance(metrics, dict):
        metrics = QualityReport.model_validate(metrics)

    lines: list[str] = [
        "# Отчёт анализа принципиальной схемы",
        "",
        f"- Задание: `{result.task_id}`",
        f"- Документ: `{result.document_id}`",
        f"- Статус: `{result.status}`",
    ]
    if result.error:
        lines.append(f"- Ошибка: `{result.error}`")
    lines.extend(
        [
            "",
            "## Сводка",
            "",
            "| Показатель | Значение |",
            "|---|---|",
            f"| Страниц обработано | {result.summary.pages_processed} |",
            f"| Компонентов найдено | {result.summary.components_found} |",
            f"| Цепей найдено | {result.summary.nets_found} |",
            f"| Строк BOM | {result.summary.bom_lines_found} |",
            f"| Проблем | {result.summary.issues_found} |",
            f"| Вызовов VLM | {result.summary.vlm_calls} |",
            "",
            "## Компоненты по типам",
            "",
        ]
    )
    type_counts = _type_counts(components)
    if type_counts:
        lines.extend(
            ["| Тип | Количество |", "|---|---|"]
            + [f"| {name} | {count} |" for name, count in type_counts.items()]
        )
    else:
        lines.append("Компоненты не распознаны.")

    lines.extend(
        [
            "",
            "## Компоненты",
            "",
        ]
    )
    if components:
        lines.extend(
            ["| ID | Тип | Обозначение | Уверенность |", "|---|---|---|---|"]
            + [_component_line(item) for item in components]
        )
    else:
        lines.append("Компоненты не распознаны.")

    lines.extend(["", "## Перечень элементов (BOM)", ""])
    if bom_items:
        found = sum(1 for item in bom_items if item.found_on_schematic)
        lines.append(
            f"Сверка BOM: найдено на схеме {found} из {len(bom_items)}, "
            f"не найдено {len(bom_items) - found}."
        )
        lines.append("")
        lines.extend(
            [
                "| Обозначение | Наименование | Кол. | Категория | Параметры | На схеме |",
                "|---|---|---|---|---|---|",
            ]
            + [_bom_line(item) for item in bom_items]
        )
    else:
        lines.append("Строки BOM не извлечены.")

    lines.extend([""])
    lines.extend(_channel_audit_lines(components, bom_items, _read_vlm_log(vlm_log)))

    lines.extend(["", "## Цепи (Netlist)", ""])
    if nets:
        lines.extend(
            [
                "| Цепь | Имя | Узлов | Выводов | Метки | Страницы |",
                "|---|---|---|---|---|---|",
            ]
            + [_net_line(item) for item in nets]
        )
    else:
        lines.append("Цепи не построены.")

    lines.extend(["", "## Анализ ПЛК", ""])
    if plc is None:
        lines.append("Анализ ПЛК не выполнялся для выбранного режима.")
    else:
        lines.append(f"- Входы: {', '.join(plc.inputs) or '—'}")
        lines.append(f"- Выходы: {', '.join(plc.outputs) or '—'}")
        lines.append(f"- Катушки: {', '.join(plc.coils) or '—'}")
        lines.append(f"- Контакты: {', '.join(plc.contacts) or '—'}")
        lines.append(f"- Блокировки: {', '.join(plc.interlocks) or '—'}")
        lines.append(f"- Логических блоков: {len(plc.logic_blocks)}")
        if plc.logic_blocks:
            lines.extend(
                [
                    "",
                    "| Блок | Элементы | Уверенность | Описание |",
                    "|---|---|---|---|",
                ]
                + [
                    f"| {block.kind} | {', '.join(block.elements) or '—'} | "
                    f"{block.confidence:.2f} | {block.description or '—'} |"
                    for block in plc.logic_blocks
                ]
            )
        if plc.warnings:
            lines.append("")
            lines.append("Предупреждения:")
            lines.extend(f"- {warning}" for warning in plc.warnings)
        if plc.assumptions:
            lines.append("")
            lines.append("Предположения:")
            lines.extend(f"- {assumption}" for assumption in plc.assumptions)

    # M4: раздел метрик выводится только при заданном ground truth.
    if isinstance(metrics, QualityReport):
        lines.extend(["", *metrics_section(metrics)])

    lines.extend(
        [
            "",
            "## Проблемы",
            "",
        ]
    )
    if issues:
        top_issues = sorted(
            issues,
            key=lambda issue: float(getattr(issue, "confidence", 0.0)),
        )[:20]
        lines.extend(
            [
                "| Тип | Страница | Уверенность | Описание | Требуемое действие |",
                "|---|---|---|---|---|",
            ]
            + [
                f"| {issue.type} | {issue.page} | "
                f"{float(getattr(issue, 'confidence', 0.0)):.2f} | "
                f"{issue.description} | {issue.required_action or '—'} |"
                for issue in top_issues
            ]
        )
    else:
        lines.append("Проблем не обнаружено.")

    lines.extend([""])
    lines.extend(_vlm_lines(result, issues, vlm_log))
    return "\n".join(lines) + "\n"


def write_report(content: str, path: str) -> None:
    """Записывает отчёт на диск.

    Args:
        content: Markdown-текст отчёта.
        path: Путь к ``report.md``.
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


__all__ = ["build_report", "multimodal_consistency_score", "write_report"]
