from typing import List, Optional

import torch

from mpi_algorithms.communicator import Communicator
from mpi_algorithms.ranktree import RankTree, binary_rank_tree, find_rank_node, preorder_ranks
from mpi_algorithms.util import no_nones_list


def gather_flat_tree(comm: Communicator, data: torch.Tensor, dest_rank: int) -> Optional[List[torch.Tensor]]:
    """
    Gather data from all ranks to the root rank directly.
    """
    if comm.rank == dest_rank:
        gathered_by_rank: List[Optional[torch.Tensor]] = [None] * comm.world_size
        gathered_by_rank[dest_rank] = data
        for rank in range(comm.world_size):
            if rank != dest_rank:
                gathered_by_rank[rank] = comm.recv(rank)
        assert no_nones_list(gathered_by_rank)  # for type-checking
        return gathered_by_rank
    else:
        comm.send(dest_rank, data)
        return None


def gather_binary_tree(comm: Communicator, data: torch.Tensor, dest_rank: int) -> Optional[List[torch.Tensor]]:
    tree = binary_rank_tree(comm.world_size, dest_rank)
    return _gather_tree_algorithm(comm, data, tree)


def _gather_tree_algorithm(
    comm: Communicator,
    data: torch.Tensor,
    tree: RankTree,
) -> Optional[List[torch.Tensor]]:
    if comm.rank == tree.rank:
        # Gather data from children.
        # First, data from entire subtree of first child is gathered in preorder,
        # then the second child and so on.
        gathered_by_rank: List[Optional[torch.Tensor]] = [None for _ in range(comm.world_size)]
        gathered_by_rank[tree.rank] = data
        for child in tree.children:
            rank_order = preorder_ranks(child)
            for recv_rank in rank_order:
                gathered_by_rank[recv_rank] = comm.recv(child.rank)
        assert no_nones_list(gathered_by_rank)  # for type-checking
        return gathered_by_rank
    else:
        node = find_rank_node(tree, comm.rank)
        assert node is not None
        assert node.parent is not None

        comm.send(node.parent.rank, data)
        for child in node.children:
            num_vecs = len(preorder_ranks(child))
            # Forward vectors from subtree.
            for _ in range(num_vecs):
                send_vector = comm.recv(child.rank)
                comm.send(node.parent.rank, send_vector)
        return None
