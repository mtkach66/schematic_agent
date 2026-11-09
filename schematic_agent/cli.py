"""CLI субагента.

Пример::

    schematic-agent parse schema.pdf --mode full --output /tmp/out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI.

    Args:
        argv: Аргументы командной строки (по умолчанию ``sys.argv[1:]``).

    Returns:
        Код возврата процесса.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

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

    result = run(task)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0 if result.status != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
