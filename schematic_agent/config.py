"""Конфигурация субагента.

Скелет: значения по умолчанию заданы dataclass-ами. Загрузка YAML-файла пока
не реализована и поднимает :class:`NotImplementedError`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Переменная окружения для переопределения каталога артефактов.
OUTPUT_DIR_ENV = "SCHEMATIC_AGENT_OUTPUT_DIR"

#: Переменная окружения для общего таймаута документа, с.
DOC_TIMEOUT_ENV = "SCHEMATIC_AGENT_DOC_TIMEOUT_S"

#: Переменная окружения для стражного таймаута одного этапа, с.
STAGE_TIMEOUT_ENV = "SCHEMATIC_AGENT_STAGE_TIMEOUT_S"

#: Переменная окружения для лимита кандидатов на страницу.
MAX_CANDIDATES_ENV = "SCHEMATIC_AGENT_MAX_CANDIDATES_PER_PAGE"

#: Переменная окружения для лимита сегментов на один детектор.
MAX_SEGMENTS_ENV = "SCHEMATIC_AGENT_MAX_SEGMENTS_PER_DETECTOR"

#: Общий таймаут документа по умолчанию (ТЗ §11, этап 10).
DEFAULT_DOC_TIMEOUT_S = 600.0

#: Стражный таймаут одного этапа по умолчанию, с (ТЗ §11, этап 10).
DEFAULT_STAGE_TIMEOUT_S = 120.0

#: Лимит кандидатов на страницу (защита от неограниченных переборов).
DEFAULT_MAX_CANDIDATES_PER_PAGE = 10000

#: Лимит сегментов на входе одного детектора.
DEFAULT_MAX_SEGMENTS_PER_DETECTOR = 5000

#: Каталог артефактов по умолчанию (ТЗ §9).
DEFAULT_OUTPUT_DIR = "/output"


@dataclass(frozen=True)
class Tolerances:
    """Допуски геометрической обработки (px, градусы)."""

    point_merge_px: float = 2.0
    line_gap_px: float = 3.0
    parallel_angle_deg: float = 1.0
    endpoint_snap_px: float = 2.5
    #: Допуск объединения концов отрезков в узел, pt (этап 3).
    node_merge_pt: float = 1.5
    #: Привязка текстовой метки к узлу, pt (этап 3).
    label_snap_pt: float = 10.0
    #: Минимальный размер «рамки» (лист/таблица), pt: такие прямоугольники
    #: не превращаются в сегменты цепей (этап 3).
    frame_min_pt: float = 200.0
    #: Максимальное расстояние «обозначение → УГО» при привязке, pt (этап 4).
    ref_attach_pt: float = 40.0
    #: Радиус поиска скопления линий для неопознанного УГО, pt (этап 4).
    unknown_cluster_pt: float = 30.0


@dataclass
class VlmSettings:
    """Настройки точечных вызовов локальной VLM (ТЗ §6)."""

    endpoint: str = "http://127.0.0.1:11434"
    model: str = "qwen3-vl:2b"
    min_crop_px: int = 512
    max_crop_px: int = 768
    max_calls: int = 50
    timeout_s: float = 120.0
    enabled: bool = True


@dataclass
class Settings:
    """Корневые настройки субагента."""

    tolerances: Tolerances = field(default_factory=Tolerances)
    confidence_threshold: float = 0.6
    vlm: VlmSettings = field(default_factory=VlmSettings)
    output_dir: str = DEFAULT_OUTPUT_DIR
    #: Общий таймаут обработки документа, с (ТЗ §11, этап 10).
    doc_timeout_s: float = DEFAULT_DOC_TIMEOUT_S
    #: Стражный таймаут одного этапа конвейера, с (ТЗ §11, этап 10).
    stage_timeout_s: float = DEFAULT_STAGE_TIMEOUT_S
    #: Максимум кандидатов на страницу (защита от переборов).
    max_candidates_per_page: int = DEFAULT_MAX_CANDIDATES_PER_PAGE
    #: Максимум сегментов на входе одного детектора.
    max_segments_per_detector: int = DEFAULT_MAX_SEGMENTS_PER_DETECTOR

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Settings":
        """Загружает настройки из YAML-файла.

        Args:
            path: Путь к YAML-файлу настроек.

        Returns:
            Настройки субагента.

        Raises:
            NotImplementedError: Загрузка YAML на этапе скелета не реализована.
        """
        raise NotImplementedError(
            "Загрузка настроек из YAML не реализована (этап скелета)"
        )


def _env_float(name: str, default: float) -> float:
    """Читает положительное число с плавающей точкой из окружения."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_positive_int(name: str, default: int) -> int:
    """Читает положительное целое из окружения."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def get_settings() -> Settings:
    """Возвращает настройки по умолчанию с учётом переменных окружения.

    Переопределяемые значения (конвенция ``SCHEMATIC_AGENT_*``):

    * ``SCHEMATIC_AGENT_OUTPUT_DIR`` — :attr:`Settings.output_dir`;
    * ``SCHEMATIC_AGENT_DOC_TIMEOUT_S`` — :attr:`Settings.doc_timeout_s`;
    * ``SCHEMATIC_AGENT_STAGE_TIMEOUT_S`` — :attr:`Settings.stage_timeout_s`;
    * ``SCHEMATIC_AGENT_MAX_CANDIDATES_PER_PAGE`` —
      :attr:`Settings.max_candidates_per_page`;
    * ``SCHEMATIC_AGENT_MAX_SEGMENTS_PER_DETECTOR`` —
      :attr:`Settings.max_segments_per_detector`.

    Некорректные или неположительные значения игнорируются и заменяются
    значением по умолчанию.
    """
    output_dir = os.environ.get(OUTPUT_DIR_ENV, DEFAULT_OUTPUT_DIR)
    return Settings(
        output_dir=str(Path(output_dir)),
        doc_timeout_s=_env_float(DOC_TIMEOUT_ENV, DEFAULT_DOC_TIMEOUT_S),
        stage_timeout_s=_env_float(STAGE_TIMEOUT_ENV, DEFAULT_STAGE_TIMEOUT_S),
        max_candidates_per_page=_env_positive_int(
            MAX_CANDIDATES_ENV, DEFAULT_MAX_CANDIDATES_PER_PAGE
        ),
        max_segments_per_detector=_env_positive_int(
            MAX_SEGMENTS_ENV, DEFAULT_MAX_SEGMENTS_PER_DETECTOR
        ),
    )


__all__ = [
    "OUTPUT_DIR_ENV",
    "DOC_TIMEOUT_ENV",
    "STAGE_TIMEOUT_ENV",
    "MAX_CANDIDATES_ENV",
    "MAX_SEGMENTS_ENV",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_DOC_TIMEOUT_S",
    "DEFAULT_STAGE_TIMEOUT_S",
    "DEFAULT_MAX_CANDIDATES_PER_PAGE",
    "DEFAULT_MAX_SEGMENTS_PER_DETECTOR",
    "Tolerances",
    "VlmSettings",
    "Settings",
    "get_settings",
]
