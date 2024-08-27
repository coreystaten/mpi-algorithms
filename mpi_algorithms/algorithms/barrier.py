import math

import torch

from mpi_algorithms.communicator import Communicator


# http://homepages.math.uic.edu/~jan/mcs572/barriers.pdf
def barrier_linear(comm: Communicator) -> None:
    if comm.rank == 0:
        # Wait for the other ranks to reach the barrier in order.
        for i in range(1, comm.world_size):
            comm.recv(i)
        # Now, send a message to all other ranks to let them know they can proceed.
        for i in range(1, comm.world_size):
            comm.send(i, torch.Tensor())
    else:
        # First, message rank 0 to let them know we've reached barrier.
        comm.send(0, torch.Tensor())
        # Next, wait for rank 0 to let us know everyone's ready to receive.
        comm.recv(0)


def barrier_binary_tree(comm: Communicator) -> None:
    # Example of writing a binary tree algorithm without explicitly constructing the tree.
    # Find k s.t. 2**(k-1) < comm.world_size <= 2**k
    k = math.ceil(math.log2(comm.world_size))
    max_granularity = 2**k
    granularity = 2
    # Tree of sends leading to rank 0.
    while granularity <= max_granularity:
        offset = comm.rank % granularity
        if offset == 0:
            source_rank = comm.rank + granularity // 2
            if source_rank < comm.world_size:
                comm.recv(source_rank)
        elif offset == granularity // 2:
            comm.send(comm.rank - granularity // 2, torch.Tensor())
        granularity *= 2
    # Reverse tree of receives leading from rank 0.
    while granularity >= 2:
        offset = comm.rank % granularity
        if offset == 0:
            target_rank = comm.rank + granularity // 2
            if target_rank < comm.world_size:
                comm.send(target_rank, torch.Tensor())
        elif comm.rank % granularity == granularity // 2:
            comm.recv(comm.rank - granularity // 2)
        granularity //= 2


def barrier_butterfly(comm: Communicator) -> None:
    # Find k s.t. 2**k <= comm.world_size < 2**(k+1)
    k = math.floor(math.log2(comm.world_size))
    max_granularity = 2**k
    granularity = 1
    while granularity <= max_granularity:
        partner_rank = comm.rank ^ granularity
        if partner_rank < comm.world_size:
            comm.send_recv(partner_rank, torch.Tensor())
        granularity *= 2
