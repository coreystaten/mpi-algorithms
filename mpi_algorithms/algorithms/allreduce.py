import torch

from mpi_algorithms.communicator import Communicator

# We use + as the reduce operator throughout this file, since we only care about the communication
# pattern.


def allreduce_segmented_ring(comm: Communicator, data: torch.Tensor) -> torch.Tensor:
    """
    Allreduce data from all ranks to all ranks using a ring algorithm.
    Assumes all data is the same size and is divisible by world_size in dim 0.

    Segments the data into chunks and sends them around the ring, so that:
    - 1. computation can overlap with communication
    - 2. memory usage per worker is reduced
    - 3. with uneven throughput, some sends can start sooner.
    These aren't relevant for our network modeling, but segmentation is included in the implementation anyways to
    show how it works.
    """
    seg_size = data.size(0) / comm.world_size
    num_segments = comm.world_size
    working_copy = data.clone()
    send_rank = (comm.rank + 1) % comm.world_size
    recv_rank = (comm.rank - 1) % comm.world_size

    # Share + reduce phase.
    for i in range(num_segments - 1):
        # Send segment index.  -i causes us to proceed in direction we just recv'ed from.
        s_idx = (comm.rank - i) % num_segments
        # Recv segment index.
        r_idx = (comm.rank - i - 1) % num_segments
        s_segment = working_copy[round(s_idx * seg_size) : round((s_idx + 1) * seg_size)]
        if comm.rank % 2 == 0:
            comm.send(send_rank, s_segment)
            r_segment = comm.recv(recv_rank)
        else:
            # Reverse order to match sends/recvs.
            r_segment = comm.recv(recv_rank)
            comm.send(send_rank, s_segment)
        # Apply received data to working_copy.
        # In principle, this could take place alongside the next send/recv using asynchronous ops.
        working_copy[round(r_idx * seg_size) : round((r_idx + 1) * seg_size)] += r_segment
    # Share phase.
    for i in range(num_segments - 1):
        # +1 compared to share + reduce phase, because we finalized the segment after our starting
        # point.
        s_idx = (comm.rank - i + 1) % num_segments
        r_idx = (comm.rank - i) % num_segments
        s_segment = working_copy[round(s_idx * seg_size) : round((s_idx + 1) * seg_size)]
        if comm.rank % 2 == 0:
            comm.send(send_rank, s_segment)
            r_segment = comm.recv(recv_rank)
        else:
            r_segment = comm.recv(recv_rank)
            comm.send(send_rank, s_segment)
        # Overwrite instead of add; we were sent the final result.
        working_copy[round(r_idx * seg_size) : round((r_idx + 1) * seg_size)] = r_segment
    return working_copy
