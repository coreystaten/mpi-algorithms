import queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pytest
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
from mpi_algorithms.network_simulator import NetworkEvent, NetworkSimulator
from mpi_algorithms.util import run_threads

WORLD_SIZE = 8


def tensor_equal(
    t1: Union[torch.Tensor, List[torch.Tensor], None], t2: Union[torch.Tensor, List[torch.Tensor], None]
) -> bool:
    if t1 is None or t2 is None:
        return t1 is t2
    if isinstance(t1, list) and isinstance(t2, list):
        return all(tensor_equal(t1_elem, t2_elem) for t1_elem, t2_elem in zip(t1, t2))
    assert isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor)
    return torch.equal(t1, t2)


def test_send_recv() -> None:
    simulator = NetworkSimulator(world_size=2, roundtrip_ms=1, throughput_bytes_ms=1000)

    def thread0(queues: Dict[Tuple[int, int], queue.Queue], simulator: NetworkSimulator) -> None:
        comm = Communicator(0, 2, queues, simulator)
        comm.send(1, torch.tensor([1, 2, 3, 4] * 100, dtype=torch.int))
        assert torch.equal(comm.recv(1), torch.tensor([5, 6, 7, 8] * 100))

    def thread1(queues: Dict[Tuple[int, int], queue.Queue], simulator: NetworkSimulator) -> None:
        comm = Communicator(1, 2, queues, simulator)
        assert torch.equal(comm.recv(0), torch.tensor([1, 2, 3, 4] * 100))
        comm.send(0, torch.tensor([5, 6, 7, 8] * 100, dtype=torch.int))

    queues = build_send_recv_queues(2)
    run_threads([thread0, thread1], [[queues, simulator]] * 2)


def test_generate_diagram() -> None:
    events = [NetworkEvent(0, 1, 1, 0, 1, 0, 1), NetworkEvent(1, 0, 1, 0, 1, 0, 1)]
    generate_event_diagram(events, "diagrams/test.png")


def run_prim_threads(prim: Callable, args: List[Tuple], kwargs: Dict) -> List[Any]:
    """
    Helper for primitive tests; runs a primitive in WORLD_SIZE threads and collects results.
    """
    simulator = NetworkSimulator(world_size=WORLD_SIZE, roundtrip_ms=1, throughput_bytes_ms=1000)
    queues = build_send_recv_queues(8)

    def f(
        rank: int, queues: Dict[Tuple[int, int], queue.Queue], simulator: NetworkSimulator, args: List, kwargs: Dict
    ) -> None:
        comm = Communicator(rank, WORLD_SIZE, queues, simulator)
        return prim(comm, *args, **kwargs)

    return run_threads([f] * 8, [[rank, queues, simulator, args[rank], kwargs] for rank in range(8)])


def primitive_cases() -> List[Tuple[Callable, List, Dict, List]]:
    allgather_args = [(i * torch.ones(1),) for i in range(WORLD_SIZE)]
    allgather_results = [[i * torch.ones(1) for i in range(WORLD_SIZE)]] * WORLD_SIZE
    allreduce_args = [(i * torch.ones(WORLD_SIZE),) for i in range(WORLD_SIZE)]
    allreduce_results = [sum(args[0] for args in allreduce_args)] * WORLD_SIZE
    broadcast_args: List[Tuple[Optional[torch.Tensor]]] = [(None,)] * WORLD_SIZE
    broadcast_args[0] = (torch.ones(1),)
    broadcast_results = [torch.ones(1)] * WORLD_SIZE
    gather_args = [(i * torch.ones(1),) for i in range(WORLD_SIZE)]
    gather_results: List[Optional[List[torch.Tensor]]] = [None] * WORLD_SIZE
    gather_results[0] = [i * torch.ones(1) for i in range(WORLD_SIZE)]

    # Format: primitive function, list of args (per rank), kwargs (shared across ranks), list of expected results by rank  aaaaaaaaaaaaaaaaaaaaaaa
    return [
        (allgather_as_gather_broadcast, allgather_args, {}, allgather_results),
        (allgather_ring, allgather_args, {}, allgather_results),
        (allreduce_segmented_ring, allreduce_args, {}, allreduce_results),
        (barrier_binary_tree, [()] * WORLD_SIZE, {}, [None] * WORLD_SIZE),
        (barrier_butterfly, [()] * WORLD_SIZE, {}, [None] * WORLD_SIZE),
        (barrier_linear, [()] * WORLD_SIZE, {}, [None] * WORLD_SIZE),
        (broadcast_flat_tree, broadcast_args, {"source_rank": 0}, broadcast_results),
        (broadcast_binary_tree, broadcast_args, {"source_rank": 0}, broadcast_results),
        (broadcast_chain_tree, broadcast_args, {"source_rank": 0}, broadcast_results),
        (broadcast_k_chain_tree, broadcast_args, {"source_rank": 0, "k": 4}, broadcast_results),
        (broadcast_binomial_tree, broadcast_args, {"source_rank": 0}, broadcast_results),
        (gather_flat_tree, gather_args, {"dest_rank": 0}, gather_results),
        (gather_binary_tree, gather_args, {"dest_rank": 0}, gather_results),
    ]


@pytest.mark.parametrize("prim, args, kwargs, expected_results", primitive_cases())
def test_primitive(prim: Callable, args: List, kwargs: Dict, expected_results: List) -> None:
    results = run_prim_threads(prim, args, kwargs)
    for rank, expected in enumerate(expected_results):
        print(f"Rank {rank} result: {results[rank]}")
        print(f"Expected: {expected}")
        assert tensor_equal(results[rank], expected)
