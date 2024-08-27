import queue
from typing import Dict, Tuple

import torch

from .network_simulator import NetworkSimulator


def tensor_num_bytes(tensor: torch.Tensor) -> int:
    """
    The number of bytes of memory used by a tensor.
    """
    return tensor.element_size() * tensor.nelement()


def build_send_recv_queues(world_size: int) -> Dict[Tuple[int, int], queue.Queue[torch.Tensor]]:
    """
    Build a dictionary of send/recv queues for each pair of ranks.
    """
    return {(i, j): queue.Queue() for i in range(world_size) for j in range(world_size) if i != j}


class Communicator:
    """
    Handles send/recv between ranks using threaded Queues.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        send_recv_queues: Dict[Tuple[int, int], queue.Queue[torch.Tensor]],
        simulator: NetworkSimulator,
    ):
        self.rank = rank
        self.world_size = world_size
        self.simulator = simulator
        self.send_recv_queues = send_recv_queues

    def send(self, target_rank: int, data: torch.Tensor) -> None:
        """
        Send data to another rank.
        """
        # Add send to the network simulator.
        self.simulator.simulate_send(self.rank, target_rank, tensor_num_bytes(data))
        # Actually perform the send.
        self.send_recv_queues[(self.rank, target_rank)].put(data)

    def recv(self, source_rank: int) -> torch.Tensor:
        """
        Receive data from another rank.
        """
        # Actually perform the receive.
        data = self.send_recv_queues[(source_rank, self.rank)].get()
        # Add recv to the network simulator.
        # We wait until the receive is performed so we can record the final tensor size.
        self.simulator.simulate_recv(source_rank, self.rank, tensor_num_bytes(data))
        return data

    def send_recv(self, partner_rank: int, data: torch.Tensor) -> torch.Tensor:
        """
        Send data to another rank and receive data from that rank.
        """
        # Lowest rank sends first.
        if self.rank < partner_rank:
            self.send(partner_rank, data)
            return self.recv(partner_rank)
        else:
            received_data = self.recv(partner_rank)
            self.send(partner_rank, data)
            return received_data
