from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RankTree:
    rank: int
    children: List["RankTree"]
    parent: Optional["RankTree"]


def binary_rank_tree(world_size: int, root_rank: int) -> RankTree:
    """
    A binary tree of ranks with the given root.
    """
    unused_ranks = list(range(world_size))
    unused_ranks.remove(root_rank)
    root = RankTree(rank=root_rank, children=[], parent=None)
    queue = [root]
    while queue:
        node = queue.pop(0)
        if len(unused_ranks) >= 2:
            left_rank = unused_ranks.pop(0)
            right_rank = unused_ranks.pop(0)
            left = RankTree(rank=left_rank, children=[], parent=node)
            right = RankTree(rank=right_rank, children=[], parent=node)
            node.children = [left, right]
            queue.extend([left, right])
        elif len(unused_ranks) == 1:
            left_rank = unused_ranks.pop(0)
            left = RankTree(rank=left_rank, children=[], parent=node)
            node.children = [left]
            queue.extend([left])
    return root


def chain_rank_tree(world_size: int, root_rank: int) -> RankTree:
    """
    A linear chain tree, e.g. 0->1->2->3->...
    """
    unused_ranks = set(range(world_size))
    unused_ranks.remove(root_rank)
    root = RankTree(rank=root_rank, children=[], parent=None)
    nxt = root
    for rank in unused_ranks:
        child = RankTree(rank=rank, children=[], parent=nxt)
        nxt.children.append(child)
        nxt = child
    return root


def k_chain_rank_tree(world_size: int, root_rank: int, k: int) -> RankTree:
    """
    A tree splitting off from root with k chains of roughly equal length.
    """
    unused_ranks = set(range(world_size))
    unused_ranks.remove(root_rank)
    # Split unused_ranks into k lists.
    chains: List[List[int]] = [[] for _ in range(k)]
    for i, rank in enumerate(unused_ranks):
        chains[i % k].append(rank)
    root = RankTree(rank=root_rank, children=[], parent=None)
    for chain in chains:
        nxt = root
        for rank in chain:
            child = RankTree(rank=rank, children=[], parent=nxt)
            nxt.children.append(child)
            nxt = child
    return root


def binomial_rank_tree(world_size: int, root_rank: int) -> RankTree:
    def _binomial_rank_tree(ranks: List[int]) -> RankTree:
        if len(ranks) == 1:
            return RankTree(rank=ranks[0], children=[], parent=None)
        else:
            left = _binomial_rank_tree(ranks[: len(ranks) // 2])
            right = _binomial_rank_tree(ranks[len(ranks) // 2 :])
            # left will contain root node.
            left.children.append(right)
            right.parent = left
            return left

    ranks = list(range(world_size))
    # Make sure root_rank is root of the tree.
    ranks[0] = root_rank
    ranks[root_rank] = 0
    return _binomial_rank_tree(ranks)


def preorder_ranks(tree: RankTree) -> List[int]:
    """
    Returns the ranks of the nodes in the tree in preorder, e.g.
    [root, ..child[0], ..child[1], ...]
    """
    ranks = [tree.rank]
    for child in tree.children:
        ranks.extend(preorder_ranks(child))
    return ranks


def find_rank_node(tree: RankTree, rank: int) -> Optional[RankTree]:
    """
    Finds the node within a given RankTree assigned the given rank.
    """
    if tree.rank == rank:
        return tree
    for child in tree.children:
        node = find_rank_node(child, rank)
        if node is not None:
            return node
    return None
