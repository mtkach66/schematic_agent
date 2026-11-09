"""Тесты этапа 7: входы/выходы, катушки, блокировки и логические блоки."""

from __future__ import annotations

from schematic_agent.contract import Component, Net
from schematic_agent.plc import analyze_plc, infer_logic_blocks


def _component(
    component_id: str, component_type: str, ref: str
) -> Component:
    """Компонент с позиционным обозначением."""
    return Component(
        id=component_id,
        type=component_type,
        ref_designation=ref,
        x=0.0,
        y=0.0,
        confidence=1.0,
    )


def _net(
    net_id: str,
    pins: list[tuple[str, str]],
    nodes: list[str],
    connections: list[tuple[str, str]],
) -> Net:
    """Цепь с выводами и проводными связями."""
    return Net(
        id=net_id,
        nodes=nodes,
        connections=[
            {"a": start, "b": end, "page": 1}
            for start, end in connections
        ],
        component_pins=pins,
        pages=[1],
    )


def _fixture() -> tuple[list[Component], list[Net]]:
    """K1 (катушка) + одноимённый контакт в другой цепи; K2 самоподхват."""
    components = [
        _component("c001", "coil", "K1"),
        _component("c002", "contact", "K1"),
        _component("c003", "switch", "SB1"),
        _component("c004", "coil", "K2"),
        _component("c005", "contact", "K2"),
    ]
    nets = [
        _net("net001", [("c001", "p001")], ["n1", "n2"], [("n1", "n2")]),
        _net("net002", [("c002", "p001")], ["n3", "n4"], [("n3", "n4")]),
        _net(
            "net003",
            [("c004", "p001"), ("c005", "p001")],
            ["n5", "n6"],
            [("n5", "n6")],
        ),
    ]
    return components, nets


def test_inputs_and_coils_classified() -> None:
    """Кнопка SB1 → вход; катушки K1/K2 → coils и outputs."""
    components, nets = _fixture()

    analysis = analyze_plc(nets, components)

    assert "SB1" in analysis.inputs
    assert "K1" in analysis.coils
    assert "K2" in analysis.coils
    assert "K1" in analysis.outputs and "K2" in analysis.outputs


def test_contact_in_other_net_creates_interlock() -> None:
    """Контакт K1 вне цепи катушки K1 → блокировка K1->K1."""
    components, nets = _fixture()

    analysis = analyze_plc(nets, components)

    assert "K1->K1" in analysis.interlocks


def test_series_contacts_form_and_block() -> None:
    """Последовательные контакт+катушка в одной цепи → блок AND."""
    components, nets = _fixture()

    analysis = analyze_plc(nets, components)

    and_blocks = [block for block in analysis.logic_blocks if block.kind == "AND"]
    assert and_blocks
    assert "K2" in and_blocks[0].elements
    assert and_blocks[0].confidence > 0
    # Самоподхват K2 в собственной цепи распознан.
    assert any(block.kind == "LATCH" for block in analysis.logic_blocks)
    assert infer_logic_blocks(analysis) == analysis.logic_blocks
