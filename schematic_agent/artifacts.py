"""Схемы и валидация JSON-артефактов (M3).

Каждый корневой артефакт конвейера описан pydantic-моделью из
:mod:`schematic_agent.contract`. Запись выполняется только через
:func:`write_artifact` с round-trip-проверкой ``model_dump`` →
``model_validate``: если модель сериализуется в структуру, которую нельзя
прочитать обратно, запись падает до выхода на диск. Это гарантирует
контракт для внешних LLM-агентов.

:func:`validate_artifacts` проверяет уже записанный каталог артефактов и
возвращает отчёт, пригодный для печати в CLI и для автотестов.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from .contract import (
    BomOutput,
    ComponentsOutput,
    IssuesOutput,
    NetsOutput,
)

#: Обязательные артефакты: ключ → (имя файла, корневая модель).
ARTIFACT_SCHEMAS: dict[str, tuple[str, type[BaseModel]]] = {
    "bom": ("bom.json", BomOutput),
    "netlist": ("netlist.json", NetsOutput),
    "components": ("components.json", ComponentsOutput),
    "issues": ("issues.json", IssuesOutput),
}

#: Опциональные артефакты: проверяются, если файл присутствует.
OPTIONAL_ARTIFACT_SCHEMAS: dict[str, tuple[str, type[BaseModel]]] = {}


def write_artifact(path: str | Path, model: BaseModel) -> str:
    """Записывает pydantic-модель в JSON с round-trip-проверкой.

    Args:
        path: Путь к JSON-файлу.
        model: Корневая модель артефакта (:class:`BomOutput` и т.п.).

    Returns:
        Абсолютный путь записанного файла.

    Raises:
        ValidationError: Если ``model_dump()`` нельзя прочитать обратно
            той же схемой (в этом случае файл не записывается).
    """
    target = Path(path)
    schema = type(model)
    payload = model.model_dump()
    validated = schema.model_validate(payload)
    target.write_text(
        json.dumps(validated.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(target)


def _validate_file(
    path: Path, schema: type[BaseModel]
) -> list[str]:
    """Валидирует один файл и возвращает список ошибок (пустой — успех)."""
    if not path.is_file():
        return [f"file_not_found: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"json_error: {error}"]
    try:
        schema.model_validate(payload)
    except ValidationError as error:
        return [f"schema_error: {error.json()}"]
    return []


def validate_artifacts(directory: str | Path) -> dict[str, Any]:
    """Проверяет JSON-артефакты каталога по pydantic-схемам.

    Args:
        directory: Каталог артефактов (``<output_dir>/<document_id>``).

    Returns:
        Отчёт: ``{"ok": bool, "directory": str, "artifacts": {файл:
        {"ok": bool, "errors": [...]}}, "errors": [...]}``. Отсутствие
        обязательного артефакта — ошибка; опциональные проверяются только
        при наличии.
    """
    base = Path(directory)
    report: dict[str, Any] = {
        "ok": True,
        "directory": str(base),
        "artifacts": {},
        "errors": [],
    }
    for key, (filename, schema) in ARTIFACT_SCHEMAS.items():
        errors = _validate_file(base / filename, schema)
        report["artifacts"][key] = {"ok": not errors, "errors": errors}
        if errors:
            report["ok"] = False
            report["errors"].extend(f"{key}: {error}" for error in errors)

    for key, (filename, schema) in OPTIONAL_ARTIFACT_SCHEMAS.items():
        path = base / filename
        if not path.is_file():
            continue
        errors = _validate_file(path, schema)
        report["artifacts"][key] = {"ok": not errors, "errors": errors}
        if errors:
            report["ok"] = False
            report["errors"].extend(f"{key}: {error}" for error in errors)
    return report


__all__ = [
    "ARTIFACT_SCHEMAS",
    "OPTIONAL_ARTIFACT_SCHEMAS",
    "write_artifact",
    "validate_artifacts",
]
