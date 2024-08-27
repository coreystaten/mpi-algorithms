import queue
import time
from argparse import ArgumentParser, Namespace
from typing import Any, Callable, Dict, List, Tuple

import torch

from mpi_algorithms.algorithms.allgather import allgather_as_gather_broadcast, allgather_ring
from mpi_algorithms.algorithms.allreduce import allreduce_segmented_ring
from mpi_algorithms.algorithms.barrier import barrier_binary_tree, barrier_butterfly, barrier_linear
from mpi_algorithms.algorithms.broadcast import (
    broadcast_binary_tree,
    broadcast_binomial_tree,
    broadcast_chain_tree,
    broadcast_flat_tree,
    broadcast_k_chain_tree,
)
from mpi_algorithms.algorithms.gather import gather_binary_tree, gather_flat_tree
from mpi_algorithms.communicator import Communicator, build_send_recv_queues
from mpi_algorithms.diagram import generate_event_diagram
from mpi_algorithms.network_simulator import NetworkSimulator
from mpi_algorithms.util import run_threads


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--diagram-world-size", type=int, default=8)
    parser.add_argument("--roundtrip-ms", type=float, default=1)
    parser.add_argument("--throughput-bytes-ms", type=float, default=1)
    # parser.add_argument("--tensor-size", type=int, default=1)
    parser.add_argument("--disable-diagrams", action="store_true")
    return parser.parse_args()


def build_primitives(
    world_size: int, tensor_size: int
) -> List[Tuple[Callable[..., Any], List[Tuple[Any, ...]], Dict[str, Any]]]:
    primitives: List[Tuple[Callable[..., Any], List[Tuple[Any, ...]], Dict[str, Any]]] = []

    # Allgather
    args = [(torch.ones(tensor_size),)] * world_size
    primitives.extend([(allgather_as_gather_broadcast, args, {}), (allgather_ring, args, {})])

    # Allreduce
    args = [(torch.ones(tensor_size),)] * world_size
    primitives.extend([(allreduce_segmented_ring, args, {})])

    # Barrier
    primitives.extend(
        [
            (barrier_linear, [()] * world_size, {}),
            (barrier_binary_tree, [()] * world_size, {}),
            (barrier_butterfly, [()] * world_size, {}),
        ]
    )

    # Broadcast
    args: List[Tuple[Any, ...]] = [(None,)] * world_size
    args[0] = (torch.ones(tensor_size),)
    primitives.extend(
        [
            (broadcast_flat_tree, args, {"source_rank": 0}),
            (broadcast_binary_tree, args, {"source_rank": 0}),
            (broadcast_chain_tree, args, {"source_rank": 0}),
            (broadcast_k_chain_tree, args, {"source_rank": 0, "k": 4}),
            (broadcast_binomial_tree, args, {"source_rank": 0}),
        ]
    )

    # Gather
    args = [(torch.ones(tensor_size),)] * world_size
    primitives.extend(
        [
            (gather_flat_tree, args, {"dest_rank": 0}),
            (gather_binary_tree, args, {"dest_rank": 0}),
        ]
    )

    return primitives


def run_primitive(
    primitive: Callable[..., Any],
    world_size: int,
    roundtrip_ms: int,
    throughput_bytes_ms: int,
    primitive_args: List[Tuple[Any, ...]],
    primitive_kwargs: Dict[str, Any],
) -> NetworkSimulator:
    def thread_func(
        rank: int,
        world_size: int,
        queues: Dict[Tuple[int, int], queue.Queue[torch.Tensor]],
        simulator: NetworkSimulator,
        pargs: Tuple[Any, ...],
        pkwargs: Dict[str, Any],
    ) -> None:
        comm = Communicator(rank, world_size, queues, simulator)
        primitive(comm, *pargs, **pkwargs)

    simulator = NetworkSimulator(
        world_size=world_size, roundtrip_ms=roundtrip_ms, throughput_bytes_ms=throughput_bytes_ms, half_duplex=False
    )
    queues = build_send_recv_queues(world_size)
    run_threads(
        [thread_func] * world_size,
        [[rank, world_size, queues, simulator, primitive_args[rank], primitive_kwargs] for rank in range(world_size)],
    )
    simulator.assert_finished()
    return simulator


def generate_chart(x: List[int], y: List[int], filename: str, xlabel: str, ylabel: str, title: str) -> None:
    import matplotlib.pyplot as plt

    # Add title.
    plt.title(title)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()


def main() -> None:
    args = parse_args()

    print("Generating network event diagrams.")
    for primitive, primitive_args, primitive_kwargs in build_primitives(
        args.diagram_world_size, args.diagram_world_size
    ):
        simulator = run_primitive(
            primitive,
            args.diagram_world_size,
            args.roundtrip_ms,
            args.throughput_bytes_ms,
            primitive_args,
            primitive_kwargs,
        )
        if not args.disable_diagrams:
            generate_event_diagram(simulator.events, f"diagrams/{primitive.__name__}.png")
        print(f"Simulated time for {primitive.__name__}: {simulator.max_time()} microseconds")

    print("Generating world size scaling diagrams.")
    start = time.time()
    times_by_primitive = {}
    WORLD_SIZES = list(range(1, 100))
    for world_size in WORLD_SIZES:
        print(f"World size: {world_size}")
        for primitive, primitive_args, primitive_kwargs in build_primitives(world_size, 1):
            simulator = run_primitive(
                primitive, world_size, args.roundtrip_ms, args.throughput_bytes_ms, primitive_args, primitive_kwargs
            )
            times_by_primitive.setdefault(primitive.__name__, []).append(simulator.max_time())
    for primitive, times in times_by_primitive.items():
        generate_chart(
            WORLD_SIZES,
            times,
            f"charts/{primitive}_world_size_scaling.png",
            "World Size",
            "Simulated Time (microseconds)",
            f"{primitive} scaling by world size",
        )
    print(f"Time taken: {time.time() - start} s")


if __name__ == "__main__":
    main()

# TODO: Charts of time scaling as message size grows.
