"""Сбор проблем и участков низкой уверенности.

Типы проблем и их накопление для итогового списка (ТЗ §10, §11).
Конвейер формирует проблемы на каждом этапе; модуль предоставляет единый
способ их объединения без потери порядка.

:data:`ISSUE_TYPES` — единый словарь «тип проблемы → требуемое действие по
умолчанию». Все этапы (2–10) используют только типы из этого словаря.
"""

from __future__ import annotations

from .contract import Issue

#: Единый словарь типов проблем: ``тип → required_action`` по умолчанию.
#: Значение используется, если этап не задал действие явно.
ISSUE_TYPES: dict[str, str] = {
    # Этапы 2–4: извлечение и распознавание.
    "low_confidence": "Проверить участок вручную",
    "unrecognized_symbol": "VLM review",
    "ambiguous_crossing": "VLM review",
    "unreadable_table": "Проверить таблицу вручную/VLM",
    "raster_pdf": "OCR required",
    "raster_only_page": "OCR required",
    "parse_error": "Проверить целостность PDF",
    "unreadable_page": "Проверить страницу вручную",
    # Этапы 3–4: предохранители переборов.
    "detector_capped": "Проверить полноту распознавания (лимит детектора)",
    # M1: конфигурация декларативных шаблонов УГО.
    "symbols_config_error": "Проверить symbols.yaml; используется builtin-набор",
    # Rich-библиотека УГО: сверка числа выводов с ожидаемым.
    "pin_count_mismatch": "VLM review",
    # M2: нечёткое слияние цепей по нормализованным меткам/синонимам.
    "nets_merged_fuzzy": "Проверить склейку одноимённых цепей на разных листах",
    # M3/M4: артефакты и метрики качества.
    "artifact_schema_error": "Проверить схему JSON-артефакта",
    "metrics_error": "Проверить файл ground_truth.json",
    # Этап 5: BOM и сверка.
    "bom_mismatch": "Сверить BOM со схемой",
    "bom_table_missing": "manual BOM entry",
    "bom_item_not_on_schematic": "Проверить схему и перечень элементов",
    "component_not_in_bom": "Дополнить перечень элементов",
    # Этап 8: VLM.
    "vlm_unavailable": "retry without VLM",
    "vlm_model_missing": "retry without VLM",
    "vlm_bad_json": "retry without VLM",
    "vlm_budget_exceeded": "retry without VLM",
    # Этапы 9–10: надёжность и отчётность.
    "page_failed": "Проверить страницу вручную",
    "doc_timeout": "Повторить обработку частично",
    "stage_timeout": "Повторить обработку этапа с большим таймаутом",
    "empty_file": "Передать непустой PDF",
    "timeout": "Повторить обработку",
}

#: Имена типов в порядке объявления (для отчётов и проверок).
ISSUE_TYPE_NAMES: tuple[str, ...] = tuple(ISSUE_TYPES)


def default_action(issue_type: str) -> str:
    """Возвращает действие по умолчанию для типа проблемы.

    Args:
        issue_type: Тип проблемы.

    Returns:
        Действие из :data:`ISSUE_TYPES` или пустая строка для неизвестного
        типа (неизвестные типы не должны теряться).
    """
    return ISSUE_TYPES.get(issue_type, "")


def add_issue(issues: list[Issue], issue: Issue) -> list[Issue]:
    """Добавляет проблему в список (с защитой от дублей).

    Args:
        issues: Текущий список проблем.
        issue: Новая проблема.

    Returns:
        Обновлённый список (тот же объект).
    """
    key = (issue.type, issue.page, round(issue.x, 1), round(issue.y, 1),
           issue.description)
    if not any(
        (existing.type, existing.page, round(existing.x, 1),
         round(existing.y, 1), existing.description) == key
        for existing in issues
    ):
        issues.append(issue)
    return issues


def collect_issues(*sources: list[Issue]) -> list[Issue]:
    """Объединяет проблемы из всех этапов конвейера.

    Args:
        *sources: Списки проблем от этапов конвейера.

    Returns:
        Единый список без дублей в порядке поступления.
    """
    result: list[Issue] = []
    for source in sources:
        for issue in source:
            add_issue(result, issue)
    return result


__all__ = [
    "ISSUE_TYPES",
    "ISSUE_TYPE_NAMES",
    "default_action",
    "add_issue",
    "collect_issues",
]
