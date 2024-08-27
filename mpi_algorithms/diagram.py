import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from jinja2 import Environment, FileSystemLoader

from mpi_algorithms.network_simulator import NetworkEvent


@dataclass
class _SendArrow:
    source_rank: int
    target_rank: int
    start_y: int
    end_y: int


@dataclass
class _Node:
    rank: int
    y: int
    first_row: bool


def generate_event_diagram(events: List[NetworkEvent], out_file: str):
    """
    Generates a diagram showing the timewise interaction between ranks based on the given events.
    """
    # First, convert network events into a set of distinct layers.
    ranks: Set[int] = set()
    layer_times_set: Set[int] = set()
    for event in events:
        ranks.add(event.source_rank)
        ranks.add(event.target_rank)
        layer_times_set.add(event.source_start_time_microseconds)
        layer_times_set.add(event.target_end_time_microseconds)
    layer_times = list(sorted(layer_times_set))
    # neato uses decreasing y values for down direction (increasing time).
    initial_y = 0
    layer_time_to_y: Dict[int, int] = {}
    for i, layer_time in enumerate(layer_times):
        layer_time_to_y[layer_time] = initial_y - i
    min_y = min(layer_time_to_y.values())
    # Find all processes that started send or finished receive at a specific time.
    layer_to_ranks: Dict[int, Set[int]] = {layer_time: set() for layer_time in layer_times}
    for event in events:
        layer_to_ranks[event.source_start_time_microseconds].add(event.source_rank)
        layer_to_ranks[event.target_end_time_microseconds].add(event.target_rank)
    # Make sure all nodes appear on top row for labels.
    layer_to_ranks[layer_times[0]] = ranks
    nodes: List[_Node] = []
    node_groups = {}
    for layer_time in layer_times:
        ranks_in_layer = layer_to_ranks[layer_time]
        layer_nodes = [
            _Node(rank, layer_time_to_y[layer_time], layer_time == layer_times[0]) for rank in ranks_in_layer
        ]
        nodes.extend(layer_nodes)
        node_groups[layer_time] = layer_nodes

    send_arrows = [
        _SendArrow(
            event.source_rank,
            event.target_rank,
            layer_time_to_y[event.source_start_time_microseconds],
            layer_time_to_y[event.target_end_time_microseconds],
        )
        for event in events
    ]
    if len(nodes) > 1000 or len(send_arrows) > 1000:
        print(
            f"WARNING: diagram is large ({len(nodes)} nodes, {len(send_arrows)} arrows), may take a "
            "while to render.  Try running with --disable-diagrams."
        )
    # Use jinja2 template to render the diagram
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("time-diagram.dot")
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_file = Path(tmpdirname) / "diagram.dot"
        with open(tmp_file, "w") as f:
            f.write(
                template.render(
                    nodes=nodes,
                    node_groups=node_groups,
                    send_arrows=send_arrows,
                    min_y=min_y,
                )
            )
        # Run graphviz.
        subprocess.run(
            ["dot", "-Tpng", tmp_file, "-o", out_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
