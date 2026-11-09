"""Точечные вызовы локальной VLM.

VLM используется только для спорных участков: неопознанные УГО, сложные
пересечения, реле/микросхемы, нечитаемые таблицы, нормализация наименований
(ТЗ §6). Обращения — к локальному Ollama (``127.0.0.1:11434``), модель
``qwen3-vl:2b``. Кропы 512×512 px, максимум 768×768 px; ответы строго JSON.
Режим полного отключения VLM обязателен.

Гарантии этапа 8:

* ни один сетевой вызов не выполняется при ``enabled=False``,
  ``use_vlm=False`` или ``mode=no_vlm``;
* ошибки VLM/сети не пробрасываются наружу — только ``Issue``;
* вызовы строго последовательные (VRAM-протектор);
* каждая попытка пишется в ``vlm_calls.log`` (JSON Lines, append).
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pymupdf

from .bom import REF_RE, detect_columns
from .config import VlmSettings
from .contract import Component, Issue

#: Промпты для разных типов триггеров (ТЗ §6). Ответ строго JSON.
PROMPTS: dict[str, str] = {
    "unrecognized_symbol": (
        "Ты анализируешь фрагмент электрической принципиальной схемы. "
        "Определи УГО (условное графическое обозначение) на изображении. "
        "Ответь СТРОГО одним JSON-объектом без пояснений и markdown: "
        '{"symbol": "<тип компонента: resistor|capacitor|diode|relay|'
        'contactor|transistor|terminal|coil|contact|switch|button|lamp|'
        'fuse|sensor|valve|unknown>", "confidence": <0..1>, '
        '"description": "<краткое описание>"}'
    ),
    "complex_component": (
        "Ты анализируешь фрагмент электрической схемы со сложным реле или "
        "микросхемой. Определи тип компонента по УГО и обозначению. "
        "Ответь СТРОГО одним JSON-объектом без пояснений: "
        '{"symbol": "<тип компонента>", "confidence": <0..1>, '
        '"description": "<краткое описание>"}'
    ),
    "ambiguous_crossing": (
        "Ты анализируешь пересечение двух проводников на электрической "
        "принципиальной схеме. Определи, есть ли электрическое соединение "
        "(точка) или это развязка без соединения. Ответь СТРОГО одним "
        'JSON-объектом: {"symbol": "junction|crossing|unknown", '
        '"confidence": <0..1>, "description": "<краткое описание>"}'
    ),
    "unreadable_table": (
        "Ты анализируешь фрагмент таблицы перечня элементов электрической "
        "схемы. Прочитай позиционное обозначение и наименование. Ответь "
        'СТРОГО одним JSON-объектом: {"symbol": "<наименование или unknown>", '
        '"confidence": <0..1>, "description": "<прочитанные данные>"}'
    ),
}

#: Уточняющий промпт при невалидном JSON (ровно один повтор).
RETRY_PROMPT = (
    "Предыдущий ответ не был корректным JSON. Верни ТОЛЬКО один JSON-объект "
    'вида {"symbol": "...", "confidence": 0.0, "description": "..."} '
    "без markdown, без текста до и после объекта."
)

#: Ограничение генерации токенов: ответ короткий JSON.
_NUM_PREDICT = 200

#: Типы триггеров (для лога и промптов).
TRIGGER_UNRECOGNIZED = "unrecognized_symbol"
TRIGGER_COMPLEX = "complex_component"
TRIGGER_CROSSING = "ambiguous_crossing"
TRIGGER_TABLE = "unreadable_table"

#: Транспорт: ``(url, payload, timeout_s) -> (status_code, body_bytes)``.
Transport = Callable[[str, dict[str, Any], float], tuple[int, bytes]]


@dataclass
class VlmRequest:
    """Запрос к VLM по кропу страницы."""

    image_png: bytes
    prompt: str
    page: int = 0
    x: float = 0.0
    y: float = 0.0


@dataclass
class VlmResponse:
    """Ответ VLM в виде разобранного JSON."""

    payload: dict[str, Any]
    raw: str = ""
    model: str = ""
    duration_ms: int = 0


@dataclass
class VlmTrigger:
    """Спорный участок, требующий точечной проверки VLM."""

    trigger: str
    page: int
    bbox: tuple[float, float, float, float]
    prompt: str
    component_index: int | None = None
    description: str = ""


def _default_transport(
    url: str, payload: dict[str, Any], timeout_s: float
) -> tuple[int, bytes]:
    """HTTP POST через ``urllib`` (единственный источник сетевых вызовов).

    Args:
        url: Полный URL Ollama.
        payload: JSON-тело запроса.
        timeout_s: Таймаут вызова, с.

    Returns:
        ``(HTTP-статус, тело ответа)``. HTTP-ошибки (4xx/5xx) возвращаются
        как обычный результат, чтобы вызывающий код мог их классифицировать.

    Raises:
        urllib.error.URLError: Сетевая ошибка/недоступность endpoint.
    """
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()


def _extract_json(raw: str) -> dict[str, Any]:
    """Строго разбирает JSON-ответ модели.

    Args:
        raw: Текст ответа Ollama (поле ``response``).

    Returns:
        Разобранный JSON-объект.

    Raises:
        ValueError: Если ответ не является JSON-объектом.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty response")
    # Допускается обрамление в ```json ... ``` — это не меняет строгость
    # самого JSON, поэтому снимаем только внешний fence.
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    return payload


def _response_text(body: bytes) -> str:
    """Извлекает поле ``response`` из ответа Ollama."""
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Ollama response: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("invalid Ollama response: not an object")
    if "response" not in parsed:
        raise ValueError("invalid Ollama response: no 'response' field")
    return str(parsed.get("response") or "")


def _clamp(value: float, low: float, high: float) -> float:
    """Ограничивает значение диапазоном."""
    return max(low, min(high, value))


def _crop_rect(
    bbox: tuple[float, float, float, float],
    page_rect: pymupdf.Rect,
    settings: VlmSettings,
) -> tuple[pymupdf.Rect, float]:
    """Вычисляет квадратный кроп вокруг bbox и масштаб рендера.

    Размер результата — 512–768 px: кроп расширяется до квадрата,
    масштаб подбирается так, чтобы сторона попала в заданный диапазон.

    Returns:
        Прямоугольник кропа в pt и масштаб (px/pt).
    """
    x0, y0, x1, y1 = bbox
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    side_pt = max(x1 - x0, y1 - y0, 1.0) * 1.2

    min_px = max(1, int(settings.min_crop_px))
    max_px = max(min_px, int(settings.max_crop_px))
    zoom = _clamp(min_px / side_pt, 0.05, 12.0)
    if side_pt * zoom > max_px:
        zoom = max_px / side_pt
    if side_pt * zoom < min_px:
        side_pt = min_px / zoom

    half = side_pt / 2.0
    rect = pymupdf.Rect(
        center_x - half,
        center_y - half,
        center_x + half,
        center_y + half,
    )
    rect = rect & page_rect
    if rect.is_empty:
        rect = pymupdf.Rect(
            max(page_rect.x0, center_x - 1.0),
            max(page_rect.y0, center_y - 1.0),
            min(page_rect.x1, center_x + 1.0),
            min(page_rect.y1, center_y + 1.0),
        )
    return rect, zoom


def render_crop(
    pdf_path: str,
    page_number: int,
    bbox: tuple[float, float, float, float],
    settings: VlmSettings,
) -> tuple[bytes, tuple[float, float, float, float]]:
    """Рендерит PNG-кроп участка PDF-страницы.

    Args:
        pdf_path: Путь к PDF.
        page_number: Номер страницы (1-based).
        bbox: Область интереса ``(x0, y0, x1, y1)`` в pt.
        settings: Настройки кропа.

    Returns:
        ``(png_bytes, crop_bbox)``.

    Raises:
        Exception: Ошибки открытия/рендера PDF (обрабатываются вызывающим).
    """
    document = pymupdf.open(pdf_path)
    try:
        page = document.load_page(page_number - 1)
        rect, zoom = _crop_rect(bbox, page.rect, settings)
        matrix = pymupdf.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(clip=rect, matrix=matrix, alpha=False)
        return pixmap.tobytes("png"), (rect.x0, rect.y0, rect.x1, rect.y1)
    finally:
        document.close()


class VlmChecker:
    """Последовательная точечная проверка спорных участков.

    Args:
        settings: Настройки VLM.
        transport: Инъекция HTTP-транспорта для тестов; по умолчанию
            :func:`_default_transport`.
        log_path: Путь к ``vlm_calls.log`` (append). ``None`` — без лога.
        pdf_path: Путь к PDF; по умолчанию берётся из ``extraction.pdf_path``.
    """

    def __init__(
        self,
        settings: VlmSettings,
        *,
        transport: Transport | None = None,
        log_path: str | Path | None = None,
        pdf_path: str | None = None,
    ) -> None:
        self.settings = settings
        self.transport: Transport = transport or _default_transport
        self.log_path = Path(log_path) if log_path else None
        self.pdf_path = pdf_path
        self.calls = 0
        self._stop_reason: str | None = None

    # --- Публичный API ----------------------------------------------------

    def run(
        self,
        extraction: Any,
        geometry: Any,
        components: list[Component],
        issues: list[Issue],
    ) -> tuple[list[Component], list[Issue], int]:
        """Проверяет триггеры и обновляет компоненты/проблемы.

        Returns:
            ``(components, issues, vlm_calls)``. ``vlm_calls`` — число
            фактических HTTP-попыток (включая повторы при bad JSON).
        """
        if not self.settings.enabled:
            return components, issues, 0

        pdf_path = self.pdf_path or str(getattr(extraction, "pdf_path", "") or "")
        triggers = self._collect_triggers(extraction, geometry, components)
        if not triggers or not pdf_path:
            return components, issues, 0

        for trigger in triggers:
            if self._stop_reason is not None:
                break
            if self.calls >= max(0, int(self.settings.max_calls)):
                issues.append(
                    Issue(
                        type="vlm_budget_exceeded",
                        page=trigger.page,
                        x=(trigger.bbox[0] + trigger.bbox[2]) / 2.0,
                        y=(trigger.bbox[1] + trigger.bbox[3]) / 2.0,
                        description=(
                            "Исчерпан лимит вызовов VLM "
                            f"({self.settings.max_calls}); триггер пропущен"
                        ),
                        confidence=0.5,
                        required_action="retry without VLM",
                    )
                )
                break
            self._review_trigger(trigger, pdf_path, components, issues)

        if self.calls:
            self._write_summary()
        return components, issues, self.calls

    # --- Триггеры ---------------------------------------------------------

    def _collect_triggers(
        self, extraction: Any, geometry: Any, components: Sequence[Component]
    ) -> list[VlmTrigger]:
        """Собирает точечные триггеры без обработки всего листа."""
        triggers: list[VlmTrigger] = []
        seen: set[tuple[int, str, int, int]] = set()

        def add(trigger: VlmTrigger) -> None:
            key = (
                trigger.page,
                trigger.trigger,
                round(trigger.bbox[0] / 2.0),
                round(trigger.bbox[1] / 2.0),
            )
            if key in seen:
                return
            seen.add(key)
            triggers.append(trigger)

        # (а) unknown с unrecognized_symbol; (г) сложные реле/МС U*/DA*.
        for index, component in enumerate(components):
            if getattr(component, "type", "") != "unknown":
                continue
            ref = str(getattr(component, "ref_designation", "") or "").upper()
            attributes = getattr(component, "attributes", {}) or {}
            bbox = self._component_bbox(component)
            if ref.startswith("U") or ref.startswith("DA"):
                kind = TRIGGER_COMPLEX
            else:
                kind = TRIGGER_UNRECOGNIZED
            add(
                VlmTrigger(
                    trigger=kind,
                    page=int(attributes.get("page", 0) or 0),
                    bbox=bbox,
                    prompt=PROMPTS[kind],
                    component_index=index,
                    description=f"VLM review: {ref or component.id}",
                )
            )

        # (б) спорные пересечения из геометрии.
        for crossing in getattr(geometry, "crossings", []) or []:
            x = float(getattr(crossing, "x", 0.0))
            y = float(getattr(crossing, "y", 0.0))
            add(
                VlmTrigger(
                    trigger=TRIGGER_CROSSING,
                    page=int(getattr(crossing, "page", 0) or 0),
                    bbox=(x - 8.0, y - 8.0, x + 8.0, y + 8.0),
                    prompt=PROMPTS[TRIGGER_CROSSING],
                    description=f"Спорное пересечение {getattr(crossing, 'id', '')}",
                )
            )

        # (в) BOM-строки с пустым/нечитаемым наименованием.
        for table in getattr(extraction, "all_tables", lambda: [])():
            for bbox, page in self._empty_name_rows(table):
                add(
                    VlmTrigger(
                        trigger=TRIGGER_TABLE,
                        page=page,
                        bbox=bbox,
                        prompt=PROMPTS[TRIGGER_TABLE],
                        description="Пустая/нечитаемая ячейка BOM",
                    )
                )

        return triggers

    @staticmethod
    def _component_bbox(component: Component) -> tuple[float, float, float, float]:
        """bbox компонента или квадрат вокруг его центра."""
        raw = (getattr(component, "attributes", {}) or {}).get("bbox")
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        x = float(getattr(component, "x", 0.0))
        y = float(getattr(component, "y", 0.0))
        return (x - 10.0, y - 10.0, x + 10.0, y + 10.0)

    @staticmethod
    def _empty_name_rows(
        table: Any,
    ) -> Iterable[tuple[tuple[float, float, float, float], int]]:
        """Возвращает bbox строк BOM, где есть обозначение, но нет имени."""
        rows: dict[int, dict[int, str]] = {}
        bboxes: dict[int, tuple[float, float, float, float]] = {}
        for cell in getattr(table, "cells", []) or []:
            text = str(getattr(cell, "text", "")).strip()
            if not text:
                continue
            row = int(getattr(cell, "row", 0))
            rows.setdefault(row, {})[int(getattr(cell, "col", 0))] = text
            cell_bbox = (
                float(getattr(cell, "x0", 0.0)),
                float(getattr(cell, "y0", 0.0)),
                float(getattr(cell, "x1", 0.0)),
                float(getattr(cell, "y1", 0.0)),
            )
            old = bboxes.get(row)
            if old is None:
                bboxes[row] = cell_bbox
            else:
                bboxes[row] = (
                    min(old[0], cell_bbox[0]),
                    min(old[1], cell_bbox[1]),
                    max(old[2], cell_bbox[2]),
                    max(old[3], cell_bbox[3]),
                )
        page = int(getattr(table, "page", 0) or 0)
        table_bbox = (
            float(getattr(table, "x0", 0.0)),
            float(getattr(table, "y0", 0.0)),
            float(getattr(table, "x1", 0.0)),
            float(getattr(table, "y1", 0.0)),
        )
        header_row, mapping = detect_columns(rows)
        ref_col = mapping.get("ref")
        name_col = mapping.get("name")
        for row_index, row in rows.items():
            if row_index == header_row:
                continue
            if ref_col is not None and name_col is not None:
                ref = row.get(ref_col, "").strip()
                name = row.get(name_col, "").strip()
                if REF_RE.match(ref) and not name:
                    yield bboxes.get(row_index, table_bbox), page
                continue
            # Без заголовков: строка только с обозначением — имя отсутствует.
            refs = [text for text in row.values() if REF_RE.match(text)]
            if len(refs) == 1 and len(row) <= 1:
                yield bboxes.get(row_index, table_bbox), page

    # --- Вызовы -----------------------------------------------------------

    def _review_trigger(
        self,
        trigger: VlmTrigger,
        pdf_path: str,
        components: list[Component],
        issues: list[Issue],
    ) -> None:
        """Выполняет вызов VLM для одного триггера (с одним повтором)."""
        try:
            image_png, crop_bbox = render_crop(
                pdf_path, trigger.page, trigger.bbox, self.settings
            )
        except Exception as exc:  # noqa: BLE001 - не падать из-за кропа
            issues.append(
                Issue(
                    type="vlm_unavailable",
                    page=trigger.page,
                    x=(trigger.bbox[0] + trigger.bbox[2]) / 2.0,
                    y=(trigger.bbox[1] + trigger.bbox[3]) / 2.0,
                    description=f"Не удалось подготовить кроп для VLM: {exc}",
                    confidence=0.5,
                    required_action="retry without VLM",
                )
            )
            self._stop_reason = "crop_error"
            return

        prompt = trigger.prompt
        max_calls = max(0, int(self.settings.max_calls))
        for attempt in range(2):
            if self._stop_reason is not None:
                return
            # Жёсткий бюджет: повтор bad JSON не должен превышать max_calls.
            if self.calls >= max_calls:
                return
            call_status, latency, raw, http_status = self._call(image_png, prompt)
            payload: dict[str, Any] | None = None
            if call_status == "ok":
                try:
                    payload = _extract_json(raw)
                    status = "ok"
                except ValueError:
                    status = "bad_json"
            else:
                status = "error"
            self._write_log(
                trigger=trigger,
                crop_bbox=crop_bbox,
                prompt_len=len(prompt),
                status=status,
                latency_s=latency,
            )
            if status == "ok" and payload is not None:
                self._apply_payload(trigger, payload, components, issues)
                return
            if status == "error":
                self._handle_error(trigger, http_status, raw, issues)
                return
            # bad_json: ровно один повтор с уточняющим промптом.
            if attempt == 0:
                prompt = RETRY_PROMPT

        issues.append(
            Issue(
                type="vlm_bad_json",
                page=trigger.page,
                x=(trigger.bbox[0] + trigger.bbox[2]) / 2.0,
                y=(trigger.bbox[1] + trigger.bbox[3]) / 2.0,
                description=(
                    "VLM не вернула валидный JSON после повтора; "
                    "компонент оставлен без изменений"
                ),
                confidence=0.4,
                required_action="retry without VLM",
            )
        )

    def _call(
        self, image_png: bytes, prompt: str
    ) -> tuple[str, float, str, int]:
        """Одна HTTP-попытка; возвращает ``(status, latency, raw, http)``."""
        url = f"{self.settings.endpoint.rstrip('/')}/api/generate"
        payload = {
            "model": self.settings.model,
            "prompt": prompt,
            "images": [base64.b64encode(image_png).decode("ascii")],
            "stream": False,
            "options": {"num_predict": _NUM_PREDICT},
        }
        started = time.monotonic()
        self.calls += 1
        try:
            http_status, body = self.transport(
                url, payload, float(self.settings.timeout_s)
            )
        except Exception:  # noqa: BLE001 - ошибки транспорта не пробрасываем
            return "error", time.monotonic() - started, "", 0
        latency = time.monotonic() - started
        if http_status != 200:
            return "error", latency, body.decode("utf-8", "replace"), http_status
        try:
            return "ok", latency, _response_text(body), http_status
        except ValueError as exc:
            return "error", latency, str(exc), http_status

    def _handle_error(
        self,
        trigger: VlmTrigger,
        http_status: int,
        raw: str,
        issues: list[Issue],
    ) -> None:
        """Преобразует сетевую ошибку в Issue и останавливает VLM-этап."""
        x = (trigger.bbox[0] + trigger.bbox[2]) / 2.0
        y = (trigger.bbox[1] + trigger.bbox[3]) / 2.0
        if http_status == 404:
            issue_type = "vlm_model_missing"
            description = (
                f"Модель VLM '{self.settings.model}' не найдена на endpoint; "
                "дальнейшие вызовы остановлены"
            )
        else:
            issue_type = "vlm_unavailable"
            suffix = f": {raw}" if raw else ""
            description = (
                "VLM недоступна или превышен таймаут"
                f" (HTTP {http_status}){suffix}; детерминированный результат "
                "сохранён"
            )
        issues.append(
            Issue(
                type=issue_type,
                page=trigger.page,
                x=x,
                y=y,
                description=description,
                confidence=0.5,
                required_action="retry without VLM",
            )
        )
        self._stop_reason = issue_type

    # --- Применение ответа ------------------------------------------------

    def _apply_payload(
        self,
        trigger: VlmTrigger,
        payload: dict[str, Any],
        components: list[Component],
        issues: list[Issue],
    ) -> None:
        """Применяет успешный JSON-ответ к компоненту/проблемам."""
        if trigger.component_index is None:
            return
        component = components[trigger.component_index]
        symbol = str(payload.get("symbol", "")).strip().lower()
        confidence = payload.get("confidence", 0.0)
        try:
            confidence_value = _clamp(float(confidence), 0.0, 1.0)
        except (TypeError, ValueError):
            confidence_value = 0.0
        description = str(payload.get("description", "") or "")
        component.attributes["vlm"] = {
            "symbol": symbol,
            "confidence": confidence_value,
            "description": description,
            "trigger": trigger.trigger,
        }
        if symbol and symbol not in {"unknown", "unclear", "н/д", "none"}:
            component.type = symbol
            component.confidence = confidence_value
            # Успешная идентификация снимает исходную проблему unknown.
            issues[:] = [
                issue for issue in issues
                if not (
                    issue.type == "unrecognized_symbol"
                    and issue.page == trigger.page
                    and abs(issue.x - component.x) <= 1.0
                    and abs(issue.y - component.y) <= 1.0
                )
            ]

    # --- Лог --------------------------------------------------------------

    def _write_summary(self) -> None:
        """Дописывает итоговый счётчик вызовов в журнал VLM."""
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "summary": True,
            "vlm_calls": self.calls,
            "model": self.settings.model,
            "stop_reason": self._stop_reason,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_log(
        self,
        *,
        trigger: VlmTrigger,
        crop_bbox: tuple[float, float, float, float],
        prompt_len: int,
        status: str,
        latency_s: float,
    ) -> None:
        """Дописывает строку JSON в ``vlm_calls.log`` (append)."""
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "page": trigger.page,
            "crop_bbox": [round(value, 2) for value in crop_bbox],
            "trigger": trigger.trigger,
            "prompt_len": prompt_len,
            "status": status,
            "latency_s": round(latency_s, 4),
            "model": self.settings.model,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_vlm_review(
    extraction: Any,
    geometry: Any,
    components: list[Component],
    issues: list[Issue],
    settings: VlmSettings,
    *,
    transport: Transport | None = None,
    log_path: str | Path | None = None,
    pdf_path: str | None = None,
) -> tuple[list[Component], list[Issue], int]:
    """Этап 8: точечная VLM-проверка спорных участков.

    Args:
        extraction: Результат этапа 2.
        geometry: Результат этапа 3.
        components: Компоненты этапа 4.
        issues: Накопленные проблемы.
        settings: Настройки VLM.
        transport: Инъекция HTTP-транспорта (тесты).
        log_path: Путь к ``vlm_calls.log``.
        pdf_path: Переопределение пути PDF (по умолчанию из extraction).

    Returns:
        ``(updated_components, updated_issues, vlm_calls)``. Исключения
        наружу не пробрасываются.
    """
    checker = VlmChecker(
        settings,
        transport=transport,
        log_path=log_path,
        pdf_path=pdf_path,
    )
    return checker.run(extraction, geometry, components, issues)


def is_available(endpoint: str, timeout_s: float = 5.0) -> bool:
    """Проверяет доступность локальной VLM.

    Args:
        endpoint: Базовый URL Ollama.
        timeout_s: Таймаут проверки, с.

    Returns:
        ``True``, если сервис отвечает; любые ошибки — ``False``.
    """
    url = f"{endpoint.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return int(response.status) == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def crop_page(image_png: bytes, bbox: tuple[float, float, float, float]) -> bytes:
    """Вырезает кроп участка PNG-изображения страницы.

    Args:
        image_png: PNG-изображение страницы.
        bbox: Область кропа ``(x0, y0, x1, y1)`` в пикселях.

    Returns:
        PNG-изображение кропа.
    """
    document = pymupdf.open(stream=image_png, filetype="png")
    try:
        page = document.load_page(0)
        pixmap = page.get_pixmap(alpha=False)
        x0, y0, x1, y1 = (int(round(value)) for value in bbox)
        x0 = max(0, min(x0, pixmap.width - 1))
        y0 = max(0, min(y0, pixmap.height - 1))
        x1 = max(x0 + 1, min(x1, pixmap.width))
        y1 = max(y0 + 1, min(y1, pixmap.height))
        clip = pymupdf.IRect(x0, y0, x1, y1)
        cropped = pymupdf.Pixmap(pixmap, clip)
        return cropped.tobytes("png")
    finally:
        document.close()


def query(
    request: VlmRequest,
    endpoint: str,
    model: str,
    *,
    settings: VlmSettings | None = None,
    transport: Transport | None = None,
) -> VlmResponse:
    """Отправляет кроп в VLM и разбирает JSON-ответ.

    Args:
        request: Запрос с изображением и промптом.
        endpoint: Базовый URL Ollama (``http://127.0.0.1:11434``).
        model: Имя модели (``qwen3-vl:2b``).
        settings: Настройки (таймаут/лимит генерации); ``None`` — значения
            по умолчанию.
        transport: Инъекция HTTP-транспорта (тесты).

    Returns:
        Разобранный JSON-ответ модели.

    Raises:
        RuntimeError: Сетевая ошибка, неверный HTTP-статус или невалидный
        JSON (вызывающий код этапа 8 превращает это в ``Issue``).
    """
    active = settings or VlmSettings(endpoint=endpoint, model=model)
    checker = VlmChecker(active, transport=transport)
    started = time.monotonic()
    status, _, raw, http_status = checker._call(request.image_png, request.prompt)
    if status != "ok":
        raise RuntimeError(f"VLM HTTP {http_status}: {raw}")
    try:
        payload = _extract_json(raw)
    except ValueError as exc:
        raise RuntimeError(f"VLM bad JSON: {exc}") from exc
    return VlmResponse(
        payload=payload,
        raw=raw,
        model=active.model,
        duration_ms=int((time.monotonic() - started) * 1000),
    )


__all__ = [
    "PROMPTS",
    "RETRY_PROMPT",
    "VlmRequest",
    "VlmResponse",
    "VlmTrigger",
    "VlmChecker",
    "Transport",
    "render_crop",
    "run_vlm_review",
    "is_available",
    "crop_page",
    "query",
]
