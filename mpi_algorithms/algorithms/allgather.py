from typing import List

import torch

from mpi_algorithms.algorithms.broadcast import broadcast_chain_tree
from mpi_algorithms.algorithms.gather import gather_binary_tree
from mpi_algorithms.communicator import Communicator
from mpi_algorithms.util import no_nones_list


def allgather_as_gather_broadcast(comm: Communicator, data: torch.Tensor) -> List[torch.Tensor]:
    """
    Allgather data from all ranks to all ranks using gather and broadcast.
    """
    # These were the two fastest implementations.
    gathered = gather_binary_tree(comm, data, 0)
    # We assume all data is equal size for now.
    # We could instead do multiple broadcasts, but this would incur additional latency.
    if comm.rank == 0:
        assert gathered is not None
        gathered = torch.cat(gathered, dim=0)
    assert isinstance(gathered, torch.Tensor) or gathered is None
    broadcast = broadcast_chain_tree(comm, gathered, 0)
    per_rank_size = broadcast.size(0) // comm.world_size
    # Split the broadcasted data back into segments.
    return [broadcast[i * per_rank_size : (i + 1) * per_rank_size] for i in range(comm.world_size)]


def allgather_ring(comm: Communicator, data: torch.Tensor) -> List[torch.Tensor]:
    """
    Allgather data from all ranks to all ranks using a ring algorithm.
    """
    gathered_by_rank: List[None | torch.Tensor] = [None] * comm.world_size
    gathered_by_rank[comm.rank] = data
    to_send = data
    send_rank = (comm.rank + 1) % comm.world_size
    recv_rank = (comm.rank - 1) % comm.world_size
    for i in range(comm.world_size - 1):
        currently_receiving = (comm.rank - 1 - i) % comm.world_size
        if comm.rank % 2 == 0:
            comm.send(send_rank, to_send)
            received = comm.recv(recv_rank)
        else:
            # Reverse order to match send/recvs.
            received = comm.recv(recv_rank)
            comm.send(send_rank, to_send)
        to_send = received
        gathered_by_rank[currently_receiving] = received
    assert no_nones_list(gathered_by_rank)
    return gathered_by_rank
