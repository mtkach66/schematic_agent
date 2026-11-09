"""CLI субагента.

Примеры::

    schematic-agent parse schema.pdf --mode full --output /tmp/out
    schematic-agent parse schema.pdf --gt examples/simple_ground_truth.json
    schematic-agent validate /tmp/out/doc_001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from .artifacts import validate_artifacts
from .config import OUTPUT_DIR_ENV
from .contract import Task, TaskOptions, TaskPayload
from .pipeline import MODES, run


def build_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(prog="schematic-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_cmd = subparsers.add_parser(
        "parse", help="Обработать PDF принципиальной схемы"
    )
    parse_cmd.add_argument("pdf", help="Путь к PDF-файлу")
    parse_cmd.add_argument(
        "--mode",
        choices=MODES,
        default="full",
        help="Режим обработки (по умолчанию: full)",
    )
    parse_cmd.add_argument(
        "--no-vlm",
        action="store_true",
        help="Полностью отключить VLM",
    )
    parse_cmd.add_argument(
        "--output",
        default=None,
        help="Каталог артефактов (по умолчанию /output)",
    )
    parse_cmd.add_argument(
        "--task-id",
        default="cli-task",
        help="Идентификатор задания",
    )
    parse_cmd.add_argument(
        "--document-id",
        default=None,
        help="Идентификатор документа (по умолчанию — имя PDF без расширения)",
    )
    parse_cmd.add_argument(
        "--gt",
        default=None,
        help="Путь к ground_truth.json для расчёта метрик качества (M4)",
    )

    validate_cmd = subparsers.add_parser(
        "validate", help="Проверить JSON-артефакты каталога по схемам (M3)"
    )
    validate_cmd.add_argument(
        "directory", help="Каталог артефактов <output>/<document_id>"
    )
    return parser


def _run_validate(directory: str) -> int:
    """Выполняет подкоманду ``validate`` и печатает отчёт."""
    report = validate_artifacts(directory)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI.

    Args:
        argv: Аргументы командной строки (по умолчанию ``sys.argv[1:]``).

    Returns:
        Код возврата процесса.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        return _run_validate(args.directory)

    if args.output:
        os.environ[OUTPUT_DIR_ENV] = args.output

    document_id = args.document_id
    if not document_id:
        document_id = os.path.splitext(os.path.basename(args.pdf))[0]

    options = TaskOptions(
        mode="no_vlm" if args.no_vlm and args.mode == "full" else args.mode,
        use_vlm=not args.no_vlm,
    )
    task = Task(
        task_id=args.task_id,
        payload=TaskPayload(
            pdf_path=args.pdf,
            document_id=document_id,
            options=options,
        ),
    )

    result = run(task, ground_truth_path=args.gt)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0 if result.status != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
