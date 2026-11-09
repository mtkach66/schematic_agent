"""Метрики качества распознавания по эталону (M4).

Формат эталона ``ground_truth.json``::

    {
      "components": [
        {"ref_designation": "R1", "type": "resistor", "page": 1}
      ],
      "nets": [
        {"name": "L1", "component_refs": ["R1", "R2"]}
      ]
    }

``evaluate`` сравнивает артефакты прогона (компоненты, цепи, BOM) с
эталоном и возвращает :class:`QualityReport`. Все метрики честные:

* компоненты — precision/recall по нормализованным позиционным
  обозначениям (:func:`schematic_agent.bom.normalize_ref`);
* pin connectivity — доля пар «эталонная цепь → компонент», для которых
  предсказанная цепь с тем же именем содержит выводы этого компонента;
* net connectivity — средний Jaccard множеств ``component_refs``
  эталонных и предсказанных цепей с одинаковым именем;
* BOM — precision/recall сверки обозначений строк BOM с эталоном.

Знаменатель 0 даёт значение ``None`` (метрика не определена), а не
искусственные 1.0/0.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .bom import normalize_ref
from .netlist import normalize_net_label


class GroundTruthComponent(BaseModel):
    """Компонент эталона."""

    ref_designation: str
    type: str = ""
    page: int = 0


class GroundTruthNet(BaseModel):
    """Цепь эталона: имя и входящие в неё компоненты."""

    name: str
    component_refs: list[str] = Field(default_factory=list)


class GroundTruth(BaseModel):
    """Корневая схема ``ground_truth.json``."""

    components: list[GroundTruthComponent] = Field(default_factory=list)
    nets: list[GroundTruthNet] = Field(default_factory=list)


class QualityReport(BaseModel):
    """Отчёт метрик качества.

    Все метрики — доли ``[0, 1]``; ``None`` означает неопределённость
    (пустой знаменатель). Счётчики приведены для интерпретации значений.
    """

    component_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    component_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    pin_connectivity: float | None = Field(default=None, ge=0.0, le=1.0)
    net_connectivity: float | None = Field(default=None, ge=0.0, le=1.0)
    bom_reconciliation_precision: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    bom_reconciliation_recall: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    components_predicted: int = 0
    components_gt: int = 0
    components_matched: int = 0
    pins_total: int = 0
    pins_matched: int = 0
    nets_predicted: int = 0
    nets_gt: int = 0
    bom_predicted: int = 0
    bom_gt: int = 0
    bom_matched: int = 0


def load_ground_truth(path: str | Path) -> GroundTruth:
    """Читает и валидирует ``ground_truth.json``.

    Args:
        path: Путь к файлу эталона.

    Returns:
        Валидированный эталон.

    Raises:
        OSError: Файл недоступен.
        json.JSONDecodeError: Некорректный JSON.
        pydantic.ValidationError: Структура не соответствует схеме.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return GroundTruth.model_validate(payload)


def _ratio(numerator: int, denominator: int) -> float | None:
    """Доля или ``None`` при пустом знаменателе."""
    if denominator <= 0:
        return None
    return numerator / denominator


def _ref(value: Any) -> str:
    """Нормализованное обозначение компонента."""
    return normalize_ref(str(value or ""))


def _component_refs(
    components: list[Any],
) -> dict[str, str]:
    """Отображение ``component_id → нормализованное обозначение``."""
    mapping: dict[str, str] = {}
    for component in components:
        identifier = getattr(component, "id", None)
        if identifier is None and isinstance(component, dict):
            identifier = component.get("id")
        if identifier is None:
            continue
        reference = (
            component.get("ref_designation")
            if isinstance(component, dict)
            else getattr(component, "ref_designation", "")
        )
        mapping[str(identifier)] = _ref(reference)
    return mapping


def _net_refs(
    nets: list[Any], component_refs: dict[str, str]
) -> dict[str, set[str]]:
    """Множества обозначений компонентов по нормализованному имени цепи."""
    result: dict[str, set[str]] = {}
    for net in nets:
        name = net.get("name") if isinstance(net, dict) else getattr(net, "name", None)
        label = normalize_net_label(name or "")
        pairs = (
            net.get("component_pins", [])
            if isinstance(net, dict)
            else getattr(net, "component_pins", [])
        )
        refs = result.setdefault(label, set())
        for pair in pairs:
            if isinstance(pair, (list, tuple)) and pair:
                reference = component_refs.get(str(pair[0]), "")
            elif isinstance(pair, dict):
                reference = component_refs.get(str(pair.get("component_id")), "")
            else:
                reference = ""
            if reference:
                refs.add(reference)
    return result


def evaluate(
    components: list[Any],
    nets: list[Any],
    bom: list[Any],
    gt: GroundTruth | dict[str, Any],
) -> QualityReport:
    """Считает метрики качества по эталону.

    Args:
        components: Распознанные компоненты.
        nets: Построенные цепи.
        bom: Строки BOM.
        gt: Эталон :class:`GroundTruth` или его JSON-словарь.

    Returns:
        :class:`QualityReport` с метриками и счётчиками.
    """
    truth = gt if isinstance(gt, GroundTruth) else GroundTruth.model_validate(gt)

    predicted_refs = {
        _ref(
            component.get("ref_designation")
            if isinstance(component, dict)
            else getattr(component, "ref_designation", "")
        )
        for component in components
    }
    predicted_refs.discard("")
    gt_refs = {_ref(item.ref_designation) for item in truth.components}
    gt_refs.discard("")
    matched_refs = predicted_refs & gt_refs

    component_precision = _ratio(len(matched_refs), len(predicted_refs))
    component_recall = _ratio(len(matched_refs), len(gt_refs))

    component_ids = _component_refs(components)
    predicted_by_net = _net_refs(nets, component_ids)

    pins_total = 0
    pins_matched = 0
    jaccards: list[float] = []
    for gt_net in truth.nets:
        label = normalize_net_label(gt_net.name)
        gt_net_refs = {_ref(ref) for ref in gt_net.component_refs}
        gt_net_refs.discard("")
        predicted = predicted_by_net.get(label, set())
        for reference in gt_net_refs:
            pins_total += 1
            if reference in predicted:
                pins_matched += 1
        union = gt_net_refs | predicted
        if union:
            jaccards.append(len(gt_net_refs & predicted) / len(union))
        else:
            jaccards.append(1.0)

    pin_connectivity = _ratio(pins_matched, pins_total)
    net_connectivity = (
        sum(jaccards) / len(jaccards) if jaccards else None
    )

    bom_refs = {
        _ref(
            item.get("ref_designation")
            if isinstance(item, dict)
            else getattr(item, "ref_designation", "")
        )
        for item in bom
    }
    bom_refs.discard("")
    bom_matched = bom_refs & gt_refs
    bom_precision = _ratio(len(bom_matched), len(bom_refs))
    bom_recall = _ratio(len(bom_matched), len(gt_refs))

    return QualityReport(
        component_precision=component_precision,
        component_recall=component_recall,
        pin_connectivity=pin_connectivity,
        net_connectivity=net_connectivity,
        bom_reconciliation_precision=bom_precision,
        bom_reconciliation_recall=bom_recall,
        components_predicted=len(predicted_refs),
        components_gt=len(gt_refs),
        components_matched=len(matched_refs),
        pins_total=pins_total,
        pins_matched=pins_matched,
        nets_predicted=len(nets),
        nets_gt=len(truth.nets),
        bom_predicted=len(bom_refs),
        bom_gt=len(gt_refs),
        bom_matched=len(bom_matched),
    )


def write_metrics(report: QualityReport, path: str | Path) -> str:
    """Записывает метрики в JSON с round-trip-проверкой схемы.

    Args:
        report: Отчёт метрик.
        path: Путь к ``metrics.json``.

    Returns:
        Абсолютный путь записанного файла.
    """
    target = Path(path)
    validated = QualityReport.model_validate(report.model_dump())
    target.write_text(
        json.dumps(validated.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(target)


def _fmt(value: float | None) -> str:
    """Форматирует метрику для Markdown (``—`` при неопределённости)."""
    return "—" if value is None else f"{value:.3f}"


def metrics_section(report: QualityReport) -> list[str]:
    """Формирует раздел ``## Метрики качества`` для ``report.md``."""
    return [
        "## Метрики качества",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| Компоненты: precision | {_fmt(report.component_precision)} |",
        f"| Компоненты: recall | {_fmt(report.component_recall)} |",
        f"| Pin connectivity | {_fmt(report.pin_connectivity)} |",
        f"| Net connectivity (Jaccard) | {_fmt(report.net_connectivity)} |",
        "| BOM: precision | "
        f"{_fmt(report.bom_reconciliation_precision)} |",
        f"| BOM: recall | {_fmt(report.bom_reconciliation_recall)} |",
        "",
        f"Компонентов: эталон {report.components_gt}, распознано "
        f"{report.components_predicted}, совпало {report.components_matched}. "
        f"Пар «цепь-компонент»: {report.pins_matched} из {report.pins_total}. "
        f"Строк BOM: эталон {report.bom_gt}, распознано "
        f"{report.bom_predicted}, совпало {report.bom_matched}.",
    ]


__all__ = [
    "GroundTruthComponent",
    "GroundTruthNet",
    "GroundTruth",
    "QualityReport",
    "load_ground_truth",
    "evaluate",
    "write_metrics",
    "metrics_section",
]
