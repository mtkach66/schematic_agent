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
from typing import Any, Iterable

import networkx as nx

from .contract import Net

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

    merged = merge_nets(nets)
    for index, net in enumerate(merged, start=1):
        net.id = f"net{index:03d}"
    return merged, []


def merge_nets(nets: list[Net]) -> list[Net]:
    """Объединяет цепи с одинаковым именем (сквозные метки, Т8).

    Args:
        nets: Исходные цепи.

    Returns:
        Цепи, сгруппированные по имени; страницы и элементы объединены.
    """
    result: list[Net] = []
    named: dict[str, Net] = {}
    for net in nets:
        if net.name is None:
            result.append(net)
            continue
        existing = named.get(net.name)
        if existing is None:
            named[net.name] = net
            result.append(net)
            continue
        for node_id in net.nodes:
            if node_id not in existing.nodes:
                existing.nodes.append(node_id)
        for connection in net.connections:
            if connection not in existing.connections:
                existing.connections.append(connection)
        for pair in net.component_pins:
            if pair not in existing.component_pins:
                existing.component_pins.append(pair)
        for label in net.text_labels:
            if label not in existing.text_labels:
                existing.text_labels.append(label)
        existing.pages = sorted(set(existing.pages) | set(net.pages))
    return result


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


__all__ = ["build_netlist", "merge_nets", "write_netlist_json"]
