"""Анализ логики ПЛК по цепям схемы.

Этап 7 конвейера (ТЗ §2, §10). Модуль определяет входы, выходы, катушки,
контакты, блокировки и **предполагаемые** логические функции. Результат
честно помечается уверенностью:

* входы — кнопки ``SB``, выключатели ``SA``, датчики ``B``/``SQ``;
* выходы — лампы ``H``/``HL``, клапаны ``Y``, катушки ``K``/``KM``;
* контакты — компоненты типа ``contact``;
* блокировки — контакт ``Kx`` в цепи другой катушки (в т.ч. одноимённой,
  если контакт вынесен в отдельную цепь);
* логические блоки — последовательные контакты → ``AND``, ветвление цепи →
  ``OR``, нормально-замкнутый контакт → ``NOT``, контакт катушки в её же
  цепи → ``LATCH``; при нехватке топологии → ``UNKNOWN`` + warnings.

Модуль не выдумывает функции: неоднозначные выводы выносятся в
``warnings``/``assumptions`` с низкой уверенностью.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import networkx as nx

from .contract import Issue, PlcAnalysis, PlcLogicBlock

#: Позиционное обозначение (совпадает с этапом 4).
_REF_RE = re.compile(r"^([A-ZА-Я]{1,3})([0-9]{1,4})$")

#: Префиксы катушек/контактов реле и контакторов.
_COIL_PREFIXES = {"K", "KM", "KA", "KT"}
#: Префиксы входов.
_INPUT_PREFIXES = {"SB", "SA", "B", "SQ", "S"}
#: Префиксы выходов.
_OUTPUT_PREFIXES = {"H", "HL", "Y"}


def _ref_prefix(component: Any) -> str:
    """Префикс позиционного обозначения (``K`` для ``K1``)."""
    match = _REF_RE.match(getattr(component, "ref_designation", "") or "")
    return match.group(1) if match else ""


def _ref(component: Any) -> str:
    """Позиционное обозначение или идентификатор компонента."""
    return getattr(component, "ref_designation", "") or getattr(component, "id", "")


def _pair(entry: Any) -> tuple[str, str] | None:
    """Приводит элемент ``component_pins`` к паре ``(component_id, pin_id)``."""
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return str(entry[0]), str(entry[1])
    return None


def _component_nets(netlist: Iterable[Any]) -> dict[str, set[str]]:
    """Отображение ``component_id → множество net_id``."""
    mapping: dict[str, set[str]] = {}
    for net in netlist:
        for entry in getattr(net, "component_pins", []) or []:
            pair = _pair(entry)
            if pair is None:
                continue
            mapping.setdefault(pair[0], set()).add(net.id)
    return mapping


def _net_components(netlist: Iterable[Any]) -> dict[str, set[str]]:
    """Отображение ``net_id → множество component_id``."""
    mapping: dict[str, set[str]] = {}
    for net in netlist:
        for entry in getattr(net, "component_pins", []) or []:
            pair = _pair(entry)
            if pair is None:
                continue
            mapping.setdefault(net.id, set()).add(pair[0])
    return mapping


def _is_coil(component: Any) -> bool:
    """Катушка: тип ``coil`` или обмотка реле/контактора."""
    prefix = _ref_prefix(component)
    if str(getattr(component, "type", "")).lower() == "coil":
        return True
    if str(getattr(component, "type", "")).lower() in ("relay", "contactor"):
        return prefix in _COIL_PREFIXES
    attributes = getattr(component, "attributes", {}) or {}
    return bool(attributes.get("is_coil")) and prefix in _COIL_PREFIXES


def _is_contact(component: Any) -> bool:
    """Контакт реле/контактора."""
    return str(getattr(component, "type", "")).lower() in (
        "contact", "contact_no", "contact_nc",
    )


def _is_normally_closed(component: Any) -> bool:
    """Нормально-замкнутый контакт (атрибут или тип)."""
    if "nc" in str(getattr(component, "type", "")).lower():
        return True
    attributes = getattr(component, "attributes", {}) or {}
    return bool(attributes.get("normally_closed")) or str(
        attributes.get("contact_type", "")
    ).upper() == "NC"


def _net_graph(net: Any) -> nx.Graph:
    """Граф проводов внутри цепи (для поиска ветвлений)."""
    graph = nx.Graph()
    for node in getattr(net, "nodes", []) or []:
        graph.add_node(str(node))
    for connection in getattr(net, "connections", []) or []:
        if isinstance(connection, dict):
            start = connection.get("a")
            end = connection.get("b")
        elif isinstance(connection, (list, tuple)) and len(connection) >= 2:
            start, end = connection[0], connection[1]
        else:
            continue
        if start is not None and end is not None:
            graph.add_edge(str(start), str(end))
    return graph


def _has_parallel_branch(net: Any) -> bool:
    """Есть ли в цепи узел с тремя и более проводными связями."""
    graph = _net_graph(net)
    return any(degree >= 3 for _, degree in graph.degree())


def analyze_plc(
    netlist: list[Any],
    components: list[Any],
    issues: list[Issue] | None = None,
) -> PlcAnalysis:
    """Формирует данные для программирования ПЛК.

    Args:
        netlist: Построенные цепи (:class:`Net`).
        components: Распознанные компоненты.
        issues: Список проблем конвейера (не пополняется: неоднозначности
            выносятся в ``warnings``/``assumptions``).

    Returns:
        Анализ входов/выходов, катушек, контактов и блокировок.
    """
    del issues  # проблемы топологии остаются в warnings/assumptions
    component_nets = _component_nets(netlist)
    net_components = _net_components(netlist)
    by_id = {component.id: component for component in components}

    coils = [component for component in components if _is_coil(component)]
    contacts = [component for component in components if _is_contact(component)]

    inputs: list[str] = []
    outputs: list[str] = []
    coil_refs: list[str] = []
    contact_refs: list[str] = []

    for component in components:
        prefix = _ref_prefix(component)
        component_type = str(getattr(component, "type", "")).lower()
        ref = _ref(component)
        if component in coils:
            if ref not in coil_refs:
                coil_refs.append(ref)
            if ref not in outputs:
                outputs.append(ref)
        if component in contacts:
            if ref not in contact_refs:
                contact_refs.append(ref)
        if prefix in _INPUT_PREFIXES or component_type in (
            "button", "switch", "sensor", "input",
        ):
            if ref not in inputs:
                inputs.append(ref)
        if prefix in _OUTPUT_PREFIXES or component_type in (
            "lamp", "valve", "output",
        ):
            if ref not in outputs:
                outputs.append(ref)

    interlocks, warnings, assumptions, blocks = _infer_topology(
        netlist, by_id, component_nets, net_components, coils, contacts,
    )

    return PlcAnalysis(
        inputs=inputs,
        outputs=outputs,
        coils=coil_refs,
        contacts=contact_refs,
        interlocks=interlocks,
        logic_blocks=blocks,
        warnings=warnings,
        assumptions=assumptions,
    )


def _infer_topology(
    netlist: list[Any],
    by_id: dict[str, Any],
    component_nets: dict[str, set[str]],
    net_components: dict[str, set[str]],
    coils: list[Any],
    contacts: list[Any],
) -> tuple[list[str], list[str], list[str], list[PlcLogicBlock]]:
    """Выводит блокировки и логические блоки по топологии цепей."""
    interlocks: list[str] = []
    warnings: list[str] = []
    assumptions: list[str] = []
    blocks: list[PlcLogicBlock] = []
    coil_ids = {component.id for component in coils}

    def add_interlock(contact_ref: str, coil_ref: str) -> None:
        """Добавляет пару блокировки без дублей."""
        pair = f"{contact_ref}->{coil_ref}"
        if pair not in interlocks:
            interlocks.append(pair)

    for contact in contacts:
        contact_ref = _ref(contact)
        contact_net_ids = component_nets.get(contact.id, set())
        if not contact_net_ids:
            continue
        for coil in coils:
            if coil.id == contact.id:
                continue
            coil_ref = _ref(coil)
            coil_net_ids = component_nets.get(coil.id, set())
            if not coil_net_ids:
                continue
            same_net = bool(contact_net_ids & coil_net_ids)
            if same_net and contact_ref == coil_ref:
                # Контакт в цепи своей катушки — это самоподхват (LATCH).
                continue
            if same_net:
                # Контакт стоит в цепи чужой катушки — блокировка.
                add_interlock(contact_ref, coil_ref)
            elif contact_ref == coil_ref:
                # Одноимённый контакт вынесен в другую цепь — блокировка.
                add_interlock(contact_ref, coil_ref)

    nets_by_id = {net.id: net for net in netlist}

    for coil in coils:
        coil_ref = _ref(coil)
        net_ids = component_nets.get(coil.id, set())
        if not net_ids:
            warnings.append(
                f"Катушка {coil_ref} не привязана к цепи — логика не определена"
            )
            continue
        for net_id in sorted(net_ids):
            net = nets_by_id.get(net_id)
            if net is None:
                continue
            member_ids = net_components.get(net_id, set())
            net_coils = [by_id[cid] for cid in member_ids if cid in coil_ids]
            net_contacts = [
                by_id[cid]
                for cid in member_ids
                if by_id.get(cid) and _is_contact(by_id[cid])
            ]
            if not net_contacts:
                warnings.append(
                    f"Цепь {net_id} катушки {coil_ref} не содержит контактов; "
                    "логическая функция не определена"
                )
                blocks.append(
                    PlcLogicBlock(
                        kind="UNKNOWN",
                        elements=[coil_ref],
                        description=(
                            f"Цепь {net_id}: контакты не распознаны"
                        ),
                        confidence=0.2,
                    )
                )
                continue

            contact_names = [_ref(contact) for contact in net_contacts]
            for contact in net_contacts:
                if _ref(contact) == coil_ref:
                    blocks.append(
                        PlcLogicBlock(
                            kind="LATCH",
                            elements=[coil_ref, _ref(contact)],
                            description=(
                                f"Контакт {_ref(contact)} включён в цепь "
                                f"своей катушки {coil_ref} — самоподхват"
                            ),
                            confidence=0.6,
                        )
                    )
                    assumptions.append(
                        f"Цепь {net_id}: контакт {_ref(contact)} принят за "
                        "самоподхват (LATCH)"
                    )

            if _has_parallel_branch(net):
                blocks.append(
                    PlcLogicBlock(
                        kind="OR",
                        elements=[*contact_names, coil_ref],
                        description=(
                            f"Цепь {net_id}: обнаружено ветвление проводов"
                        ),
                        confidence=0.4,
                    )
                )
                assumptions.append(
                    f"Цепь {net_id}: параллельные ветви приняты за OR"
                )
            else:
                blocks.append(
                    PlcLogicBlock(
                        kind="AND",
                        elements=[*contact_names, coil_ref],
                        description=(
                            f"Цепь {net_id}: последовательные контакты "
                            f"перед катушкой {coil_ref}"
                        ),
                        confidence=0.6,
                    )
                )

            closed = [
                _ref(contact) for contact in net_contacts
                if _is_normally_closed(contact)
            ]
            if closed:
                blocks.append(
                    PlcLogicBlock(
                        kind="NOT",
                        elements=closed,
                        description=(
                            "Нормально-замкнутые контакты: "
                            + ", ".join(closed)
                        ),
                        confidence=0.5,
                    )
                )
                assumptions.append(
                    "Нормально-замкнутые контакты приняты за инверсию (NOT)"
                )

    # Цепи без катушек и без нагрузочных выходов — предупреждение.
    for net in netlist:
        member_ids = net_components.get(net.id, set())
        has_coil = any(cid in coil_ids for cid in member_ids)
        has_output = any(
            _ref_prefix(by_id[cid]) in _OUTPUT_PREFIXES
            for cid in member_ids
            if cid in by_id
        )
        if not has_coil and not has_output:
            warnings.append(
                f"Цепь {net.id} не содержит катушек/нагрузки — "
                "логическая функция не определена"
            )

    return interlocks, warnings, assumptions, blocks


def infer_logic_blocks(analysis: PlcAnalysis) -> list[PlcLogicBlock]:
    """Возвращает выведенные логические блоки анализа.

    Args:
        analysis: Результат :func:`analyze_plc`.

    Returns:
        Список предполагаемых логических блоков.
    """
    return list(analysis.logic_blocks)


__all__ = ["analyze_plc", "infer_logic_blocks"]
