"""Pydantic-схемы контракта субагента анализа принципиальных схем.

Схемы соответствуют ТЗ §7 (вход), §8 (выход) и §10 (минимальный состав
результата). Версия pydantic — v2.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

#: Режимы работы субагента (ТЗ §13).
WorkMode = Literal["full", "no_vlm", "bom_only", "netlist_only", "issue_review"]

#: Допустимые статусы результата (ТЗ §8, §11).
ResultStatus = Literal["completed", "completed_with_issues", "failed", "partial"]


class TaskOptions(BaseModel):
    """Опции задания (ТЗ §7, ``payload.options``)."""

    mode: WorkMode = "full"
    use_vlm: bool = True
    extract_bom: bool = True
    extract_netlist: bool = True
    analyze_plc_logic: bool = True


class TaskPayload(BaseModel):
    """Полезная нагрузка задания (ТЗ §7)."""

    pdf_path: str
    document_id: str
    options: TaskOptions = Field(default_factory=TaskOptions)


class Task(BaseModel):
    """Входное задание субагента (ТЗ §7)."""

    task_id: str
    task_type: Literal["parse_electrical_schematic_pdf"] = (
        "parse_electrical_schematic_pdf"
    )
    payload: TaskPayload

    @field_validator("task_id")
    @classmethod
    def _task_id_not_empty(cls, value: str) -> str:
        """Отклоняет пустой или пробельный ``task_id``."""
        if not value or not value.strip():
            raise ValueError("task_id must be a non-empty string")
        return value


class Summary(BaseModel):
    """Сводные счётчики результата (ТЗ §8)."""

    pages_processed: int = 0
    components_found: int = 0
    nets_found: int = 0
    bom_lines_found: int = 0
    issues_found: int = 0
    vlm_calls: int = 0


class Artifacts(BaseModel):
    """Пути к артефактам обработки (ТЗ §8, §9).

    Основные поля обязательны, ``bom_csv`` и ``vlm_log`` опциональны.
    """

    bom: str
    netlist: str
    components: str
    issues: str
    plc_analysis: str
    report: str
    bom_csv: str | None = None
    vlm_log: str | None = None


class Result(BaseModel):
    """Единый результат обработки (ТЗ §8)."""

    task_id: str
    status: ResultStatus
    document_id: str
    summary: Summary = Field(default_factory=Summary)
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class BomItem(BaseModel):
    """Строка перечня элементов (ТЗ §10, BOM)."""

    ref_designation: str
    name: str
    quantity: int = 1
    note: str = ""
    page: int = 0
    normalized_name: str = ""
    category: str = ""
    parameters: str = ""
    found_on_schematic: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Node(BaseModel):
    """Геометрический узел цепи (точка соединения сегментов).

    Используется этапами геометрии (3) и Netlist (6). Узлы с одинаковыми
    координатами в пределах допуска объединяются парсером геометрии.
    """

    id: str
    x: float
    y: float
    page: int = 0
    kind: str = "junction"
    attributes: dict[str, Any] = Field(default_factory=dict)


class Pin(BaseModel):
    """Вывод компонента, привязанный к геометрическому узлу.

    Attributes:
        id: Идентификатор вывода (например, ``p001``).
        node_id: Идентификатор узла :class:`Node`, к которому подключён вывод.
        component_id: Идентификатор компонента-владельца.
        x: Координата вывода X, pt.
        y: Координата вывода Y, pt.
    """

    id: str
    node_id: str
    component_id: str
    x: float
    y: float


class Component(BaseModel):
    """Распознанный компонент схемы (ТЗ §10, компоненты)."""

    id: str
    type: str
    ref_designation: str
    x: float
    y: float
    pins: list[Any] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Net(BaseModel):
    """Электрическая цепь (ТЗ §10, Netlist)."""

    id: str
    name: str | None = None
    nodes: list[Any] = Field(default_factory=list)
    connections: list[Any] = Field(default_factory=list)
    component_pins: list[Any] = Field(default_factory=list)
    text_labels: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)


class Issue(BaseModel):
    """Проблема или участок низкой уверенности (ТЗ §10, проблемы)."""

    type: str
    page: int = 0
    x: float = 0.0
    y: float = 0.0
    description: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    required_action: str = ""


class PlcLogicBlock(BaseModel):
    """Предполагаемый логический блок ПЛК (ТЗ §10)."""

    kind: Literal["AND", "OR", "NOT", "LATCH", "UNKNOWN"]
    elements: list[str] = Field(default_factory=list)
    description: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class PlcAnalysis(BaseModel):
    """Данные для анализа логики ПЛК (ТЗ §10)."""

    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    coils: list[str] = Field(default_factory=list)
    contacts: list[str] = Field(default_factory=list)
    interlocks: list[str] = Field(default_factory=list)
    logic_blocks: list[PlcLogicBlock] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class SymbolGeometrySpec(BaseModel):
    """Геометрические параметры декларативного шаблона УГО (M1).

    Шаблон описывает прямоугольное УГО: соотношение сторон
    ``long / short`` и диапазоны размеров короткой и длинной сторон, pt.
    Сложные составные УГО намеренно не декларируются — только честные
    геометрические эвристики.
    """

    kind: Literal["rect"] = "rect"
    min_aspect: float = Field(default=1.0, gt=0.0)
    max_aspect: float = Field(default=10.0, gt=0.0)
    min_short: float = Field(default=2.0, ge=0.0)
    max_short: float = Field(default=300.0, gt=0.0)
    min_long: float = Field(default=2.0, ge=0.0)
    max_long: float = Field(default=600.0, gt=0.0)


class SymbolTemplateSpec(BaseModel):
    """Декларативный шаблон УГО из ``symbols.yaml`` (M1).

    Attributes:
        name: Имя шаблона (``contactor``, ``breaker`` и т.п.).
        category: Категория компонента, не зависящая от SPICE.
        ref_prefixes: Префиксы позиционных обозначений (``KM``, ``QF``).
            Кандидат принимается только при наличии рядом обозначения с
            одним из этих префиксов — это защищает встроенные детекторы от
            ложных перехватов прямоугольников.
        geometry: Геометрические параметры :class:`SymbolGeometrySpec`.
        confidence: Честная уверенность шаблона, 0–1.
    """

    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    ref_prefixes: list[str] = Field(default_factory=list)
    geometry: SymbolGeometrySpec = Field(default_factory=SymbolGeometrySpec)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class BomOutput(BaseModel):
    """Корневая схема ``bom.json`` (M3)."""

    items: list[BomItem] = Field(default_factory=list)
    source: Literal["table", "components_fallback"]


class NetsOutput(BaseModel):
    """Корневая схема ``netlist.json`` (M3)."""

    nets: list[Net] = Field(default_factory=list)


class ComponentsOutput(BaseModel):
    """Корневая схема ``components.json`` (M3)."""

    components: list[Component] = Field(default_factory=list)


class IssuesOutput(BaseModel):
    """Корневая схема ``issues.json`` (M3)."""

    issues: list[Issue] = Field(default_factory=list)


__all__ = [
    "WorkMode",
    "ResultStatus",
    "TaskOptions",
    "TaskPayload",
    "Task",
    "Summary",
    "Artifacts",
    "Result",
    "BomItem",
    "Node",
    "Pin",
    "Component",
    "Net",
    "Issue",
    "PlcLogicBlock",
    "PlcAnalysis",
    "SymbolGeometrySpec",
    "SymbolTemplateSpec",
    "BomOutput",
    "NetsOutput",
    "ComponentsOutput",
    "IssuesOutput",
]
