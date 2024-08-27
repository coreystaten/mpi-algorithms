from typing import Optional

import torch

from mpi_algorithms.communicator import Communicator
from mpi_algorithms.ranktree import (
    RankTree,
    binary_rank_tree,
    binomial_rank_tree,
    chain_rank_tree,
    find_rank_node,
    k_chain_rank_tree,
)

# References:
# - https://doi.org/10.1016/j.jpdc.2022.03.012
# - https://www.tesble.com/10.1109/ipdps.2006.1639364

# Tree-segment broadcast algorithms take as input a segment size and a tree structure on ranks
# starting from the broadcast node as root.
# Algorithm:
# - root/broadcast node segments tensor and sends each segment in turn to its children.
# - each child node receives a segment from its parent and then sends it to each of its children in
#   turn.


def broadcast_flat_tree(comm: Communicator, tensor: Optional[torch.Tensor], source_rank: int) -> torch.Tensor:
    """
    Broadcast entire vector to each rank in turn.
    """
    if comm.rank == source_rank:
        assert tensor is not None
        for rank in range(comm.world_size):
            if rank != source_rank:
                comm.send(rank, tensor)
        return tensor
    else:
        return comm.recv(source_rank)


def broadcast_binary_tree(comm: Communicator, tensor: Optional[torch.Tensor], source_rank: int) -> torch.Tensor:
    tree = binary_rank_tree(comm.world_size, source_rank)
    return _broadcast_tree_algorithm(comm, tensor, tree)


def broadcast_chain_tree(comm: Communicator, tensor: Optional[torch.Tensor], source_rank: int) -> torch.Tensor:
    tree = chain_rank_tree(comm.world_size, source_rank)
    return _broadcast_tree_algorithm(comm, tensor, tree)


def broadcast_k_chain_tree(
    comm: Communicator, tensor: Optional[torch.Tensor], source_rank: int, k: int
) -> torch.Tensor:
    tree = k_chain_rank_tree(comm.world_size, source_rank, k)
    return _broadcast_tree_algorithm(comm, tensor, tree)


def broadcast_binomial_tree(comm: Communicator, tensor: Optional[torch.Tensor], source_rank: int) -> torch.Tensor:
    tree = binomial_rank_tree(comm.world_size, source_rank)
    return _broadcast_tree_algorithm(comm, tensor, tree)


def _broadcast_tree_algorithm(
    comm: Communicator,
    tensor: Optional[torch.Tensor],
    tree: RankTree,
) -> torch.Tensor:
    """
    Broadcast vector forward through tree structure starting at source_rank.
    """
    if comm.rank == tree.rank:
        assert tensor is not None
        for child in tree.children:
            comm.send(child.rank, tensor)
        return tensor
    else:
        node = find_rank_node(tree, comm.rank)
        assert node is not None
        assert node.parent is not None
        tensor = comm.recv(node.parent.rank)
        for child in node.children:
            comm.send(child.rank, tensor)
        return tensor
