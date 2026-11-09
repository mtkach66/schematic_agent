"""Построение графа электрических цепей (Netlist).

Этап 6 конвейера (ТЗ §2, §10). Граф строится по нормализованной геометрии:

* узлы геометрии — вершины :class:`networkx.Graph`;
* сегменты-провода — рёбра;
* внутренняя геометрия распознанных УГО (``primitive_indices`` компонентов)
  из графа исключается: корпус резистора или обкладки конденсатора не
  должны замыкать цепи;
* ``connected_components`` даёт электрические цепи (:class:`Net`);
* выводы компонентов привязываются к цепям через ``Pin.node_id``;
* X-образные пересечения без узла (``crossings``) в граф не входят.

Сквозные метки: цепи с одинаковым именем (меткой узла) на разных страницах
объединяются после построения (:func:`merge_nets`) — тест-кейс Т8.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

import networkx as nx

from .config import get_settings
from .contract import Issue, Net
from .issues import default_action

#: Всё, кроме латинских букв и цифр: удаляется при нормализации меток цепей.
_NET_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")

#: Уверенность слияния по точному имени.
CONFIDENCE_EXACT = 1.0
#: Уверенность слияния по нормализованному имени (``L1`` ↔ ``L1 ``).
CONFIDENCE_NORMALIZED = 0.9
#: Уверенность слияния по синониму (``PE`` ↔ ``PEN``).
CONFIDENCE_ALIAS = 0.7

#: Ключ атрибута компонента со списком индексов его примитивов.
_PRIMITIVES_KEY = "primitive_indices"


def _excluded_primitives(components: Iterable[Any]) -> set[tuple[int, int]]:
    """Возвращает пары ``(page, primitive_index)`` внутренней геометрии УГО."""
    excluded: set[tuple[int, int]] = set()
    for component in components:
        attributes = getattr(component, "attributes", {}) or {}
        page = int(attributes.get("page", 0))
        for index in attributes.get(_PRIMITIVES_KEY, []) or []:
            excluded.add((page, int(index)))
    return excluded


def _component_pin_pairs(component: Any) -> list[tuple[str, str, str]]:
    """Возвращает ``(component_id, pin_id, node_id)`` для выводов."""
    pairs: list[tuple[str, str, str]] = []
    for pin in getattr(component, "pins", []) or []:
        if isinstance(pin, dict):
            pin_id = pin.get("id")
            node_id = pin.get("node_id")
        else:
            pin_id = getattr(pin, "id", None)
            node_id = getattr(pin, "node_id", None)
        if pin_id is None or node_id is None:
            continue
        pairs.append((component.id, str(pin_id), str(node_id)))
    return pairs


def _build_graph(
    geometry: Any, components: Iterable[Any]
) -> nx.Graph:
    """Строит граф узлов и проводов без внутренней геометрии УГО."""
    graph = nx.Graph()
    for node in geometry.nodes:
        graph.add_node(node.id, page=node.page)

    excluded = _excluded_primitives(components)
    segment_nodes = getattr(geometry, "segment_nodes", []) or []
    for segment, (start, end) in zip(geometry.segments, segment_nodes):
        if start == end:
            continue
        if (int(segment.page), int(segment.primitive_index)) in excluded:
            continue
        graph.add_edge(start, end, page=int(segment.page))
    return graph


def _node_labels(nodes: list[Any]) -> list[str]:
    """Собирает метки цепей с узлов, сохраняя порядок и уникальность."""
    labels: list[str] = []
    for node in nodes:
        for label in (node.attributes or {}).get("labels", []) or []:
            if label not in labels:
                labels.append(str(label))
    return labels


def build_netlist(
    geometry: Any,
    components: list[Any],
    extraction: Any | None = None,
) -> tuple[list[Net], list[Any]]:
    """Строит электрические цепи по геометрии и компонентам.

    Args:
        geometry: Результат этапа 3 (:class:`GeometryResult`).
        components: Распознанные компоненты с выводами.
        extraction: Результат этапа 2; используется для контекста страниц.

    Returns:
        Кортеж ``(nets, issues)``. Пересечения без узла в цепи не входят,
        поэтому проблем этап не порождает.
    """
    del extraction  # страницы цепей вычисляются из узлов геометрии
    graph = _build_graph(geometry, components)
    node_by_id = {node.id: node for node in geometry.nodes}

    pins_by_node: dict[str, list[tuple[str, str]]] = {}
    for component in components:
        for component_id, pin_id, node_id in _component_pin_pairs(component):
            pins_by_node.setdefault(node_id, []).append((component_id, pin_id))

    nets: list[Net] = []
    for members in nx.connected_components(graph):
        node_ids = sorted(members)
        nodes = [node_by_id[node_id] for node_id in node_ids]
        component_pins = [
            (component_id, pin_id)
            for node_id in node_ids
            for component_id, pin_id in sorted(
                pins_by_node.get(node_id, []), key=lambda pair: pair[0]
            )
        ]
        connections = [
            {
                "a": start,
                "b": end,
                "page": int(graph.edges[start, end].get("page", 0)),
            }
            for start, end in sorted(
                (tuple(sorted((u, v))) for u, v in graph.subgraph(members).edges())
            )
        ]
        labels = _node_labels(nodes)
        if not component_pins and len(node_ids) <= 1:
            continue
        # Изолированные узлы одного компонента — внутренние углы корпуса УГО,
        # а не электрическая цепь.
        pin_owners = {component_id for component_id, _ in component_pins}
        if not connections and not labels and len(pin_owners) <= 1:
            continue
        pages = sorted({node.page for node in nodes})
        nets.append(
            Net(
                id=f"net{len(nets) + 1:03d}",
                name=labels[0] if labels else None,
                nodes=node_ids,
                connections=connections,
                component_pins=component_pins,
                text_labels=labels,
                pages=pages,
            )
        )

    aliases: dict[str, str] = {}
    try:
        from .components import load_net_aliases

        aliases = load_net_aliases()
    except Exception:  # noqa: BLE001 - синонимы не должны ломать netlist
        aliases = {}
    merged, merge_issues = merge_nets(nets, aliases=aliases)
    for index, net in enumerate(merged, start=1):
        net.id = f"net{index:03d}"
    return merged, merge_issues


def normalize_net_label(label: str) -> str:
    """Нормализует метку цепи для нечёткого сопоставления (M2).

    Верхний регистр и удаление всех символов кроме ``[A-Z0-9]``:
    ``"l-1"`` → ``"L1"``, ``"L1 "`` → ``"L1"``, ``"PEN"`` → ``"PEN"``.
    Исходное имя цепи не изменяется — нормализуется только ключ сравнения.

    Args:
        label: Исходная метка.

    Returns:
        Нормализованный ключ (может быть пустым).
    """
    return _NET_NON_ALNUM_RE.sub("", str(label or "").upper())


def _canonical_net_label(label: str, aliases: dict[str, str] | None) -> str:
    """Возвращает каноническую метку с учётом таблицы синонимов."""
    normalized = normalize_net_label(label)
    if not normalized:
        return ""
    return (aliases or {}).get(normalized, normalized)


def _absorb_net(target: Net, source: Net) -> None:
    """Переносит узлы, связи, выводы и метки одной цепи в другую."""
    for node_id in source.nodes:
        if node_id not in target.nodes:
            target.nodes.append(node_id)
    for connection in source.connections:
        if connection not in target.connections:
            target.connections.append(connection)
    for pair in source.component_pins:
        if pair not in target.component_pins:
            target.component_pins.append(pair)
    for label in source.text_labels:
        if label not in target.text_labels:
            target.text_labels.append(label)
    target.pages = sorted(set(target.pages) | set(source.pages))


def _fuzzy_confidence(
    first: Net, second: Net, aliases: dict[str, str] | None
) -> float | None:
    """Возвращает уверенность нечёткого слияния двух цепей или ``None``."""
    if first.name is None or second.name is None or first.name == second.name:
        return None
    normalized_first = normalize_net_label(first.name)
    normalized_second = normalize_net_label(second.name)
    if normalized_first and normalized_first == normalized_second:
        return CONFIDENCE_NORMALIZED
    canonical_first = _canonical_net_label(first.name, aliases)
    canonical_second = _canonical_net_label(second.name, aliases)
    if canonical_first and canonical_first == canonical_second:
        return CONFIDENCE_ALIAS
    return None


def merge_nets(
    nets: list[Net],
    aliases: dict[str, str] | None = None,
    confidence_threshold: float | None = None,
) -> tuple[list[Net], list[Issue]]:
    """Объединяет цепи: точно, по нормализации и по синонимам (Т8, M2).

    Правила:

    1. Точное совпадение имени — слияние без ограничения по страницам
       (прежняя семантика Т8), уверенность :data:`CONFIDENCE_EXACT`.
    2. Совпадение нормализованных меток — :data:`CONFIDENCE_NORMALIZED`.
    3. Совпадение по синониму ``symbols.yaml`` — :data:`CONFIDENCE_ALIAS`.
    4. Нечёткие слияния (2, 3) выполняются ТОЛЬКО для цепей с разными
       страницами: в пределах одной страницы одинаковые имена остаются
       раздельными (регрессия H5).
    5. Слияние применяется только при уверенности не ниже
       ``net_merge_confidence`` из конфигурации (по умолчанию 0.7).

    Args:
        nets: Исходные цепи.
        aliases: Синонимы «нормализованная метка → каноническая метка».
        confidence_threshold: Порог слияния; по умолчанию из настроек.

    Returns:
        Кортеж ``(объединённые цепи, проблемы)``. Каждое нечёткое слияние
        даёт ``Issue(type="nets_merged_fuzzy")`` с именами, страницами и
        уверенностью.
    """
    threshold = (
        get_settings().net_merge_confidence
        if confidence_threshold is None
        else confidence_threshold
    )
    if aliases is None:
        try:
            from .components import load_net_aliases

            aliases = load_net_aliases()
        except Exception:  # noqa: BLE001 - синонимы не должны ломать netlist
            aliases = {}

    # Шаг 1: точное совпадение имён (как раньше).
    groups: list[Net] = []
    exact: dict[str, Net] = {}
    for net in nets:
        if net.name is None:
            groups.append(net)
            continue
        existing = exact.get(net.name)
        if existing is None:
            exact[net.name] = net
            groups.append(net)
            continue
        _absorb_net(existing, net)

    # Шаг 2: нечёткие cross-page слияния (union-find по индексам групп).
    parent = list(range(len(groups)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    issues: list[Issue] = []
    for first_index in range(len(groups)):
        for second_index in range(first_index + 1, len(groups)):
            first = groups[first_index]
            second = groups[second_index]
            confidence = _fuzzy_confidence(first, second, aliases)
            if confidence is None or confidence < threshold:
                continue
            if set(first.pages) & set(second.pages):
                continue  # одна и та же страница — не сливаем
            parent[find(second_index)] = find(first_index)
            issues.append(
                Issue(
                    type="nets_merged_fuzzy",
                    page=min(first.pages + second.pages)
                    if first.pages or second.pages
                    else 0,
                    description=(
                        f"Цепи {first.name!r} (стр. {first.pages}) и "
                        f"{second.name!r} (стр. {second.pages}) объединены: "
                        f"normalize={normalize_net_label(first.name)!r}, "
                        f"confidence={confidence:.2f}"
                    ),
                    confidence=confidence,
                    required_action=default_action("nets_merged_fuzzy"),
                )
            )

    merged: list[Net] = []
    root_net: dict[int, Net] = {}
    for index, net in enumerate(groups):
        root = find(index)
        target = root_net.get(root)
        if target is None:
            root_net[root] = net
            merged.append(net)
        elif target is not net:
            _absorb_net(target, net)
    return merged, issues


def write_netlist_json(nets: list[Net], path: str) -> None:
    """Записывает Netlist в JSON-файл.

    Args:
        nets: Цепи.
        path: Путь к JSON-файлу.
    """
    payload = {"nets": [net.model_dump() for net in nets]}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


__all__ = [
    "build_netlist",
    "merge_nets",
    "normalize_net_label",
    "write_netlist_json",
]
