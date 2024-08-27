import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional


class _Direction(Enum):
    SEND = auto()
    RECV = auto()


@dataclass
class _Pending:
    direction: _Direction
    source_rank: int
    target_rank: int
    num_bytes: int


@dataclass
class NetworkEvent:
    source_rank: int
    target_rank: int
    num_bytes: int
    source_start_time_microseconds: int
    # For now, source and target end times are always the same.
    source_end_time_microseconds: int
    target_start_time_microseconds: int
    target_end_time_microseconds: int


class NetworkSimulator:
    """
    Simulates a fake network timeline for multiple links using simulated latency and throughput.

    This model simulates full duplex by assuming sends do not block.  A model may send, then recv, and
    they will be allowed to overlap.

    Recv's do block, since they might contain data necessary for the next send.  If the sends/recvs
    can be reordered, this might affect the results compared to fully asynchronous calls.
    """

    def __init__(self, world_size: int, roundtrip_ms: float, throughput_bytes_ms: float, half_duplex: bool = False):
        self.world_size = world_size
        # Could have separate stats for each link if we want.
        self.roundtrip_ms = roundtrip_ms
        self.throughput_bytes_ms = throughput_bytes_ms
        self.half_duplex = half_duplex
        self.send_time_by_rank = {rank: 0 for rank in range(world_size)}
        self.recv_time_by_rank = {rank: 0 for rank in range(world_size)}
        self.bytes_sent_by_rank = {rank: 0 for rank in range(world_size)}
        self.bytes_received_by_rank = {rank: 0 for rank in range(world_size)}
        self.pending_by_rank: Dict[int, List[_Pending]] = {rank: [] for rank in range(world_size)}
        self.last_send_by_rank: Dict[int, Optional[int]] = {rank: None for rank in range(world_size)}
        self.lock = threading.Lock()
        # Events are stored in order of creation.
        self.events: List[NetworkEvent] = []

    def simulate_send(self, source_rank: int, target_rank: int, num_bytes: int) -> None:
        """
        Simulate sending bytes from source_rank to target_rank.
        """
        with self.lock:
            self.pending_by_rank[source_rank].append(_Pending(_Direction.SEND, source_rank, target_rank, num_bytes))
            self._resolve_pending()

    def simulate_recv(self, source_rank: int, target_rank: int, num_bytes: int) -> None:
        """
        Simulate receiving bytes from source_rank on target_rank.
        """
        with self.lock:
            self.pending_by_rank[target_rank].append(_Pending(_Direction.RECV, source_rank, target_rank, num_bytes))
            self._resolve_pending()

    def assert_finished(self) -> None:
        """
        Assert that all sends and recvs have been completed.
        """
        with self.lock:
            for pending in self.pending_by_rank.values():
                assert not pending, "Simulator send/recvs did not match."

    def max_time(self) -> int:
        """
        Get the maximum time in milliseconds.
        """
        with self.lock:
            return max(list(self.send_time_by_rank.values()) + list(self.recv_time_by_rank.values()))

    def _resolve_pending(self):
        """
        Resolve pending send/recv pairs.
        """
        while True:
            # Check the first operation for each rank, and see if it has a matching operation.
            # Break if no progress.
            progress = False
            for rank, pending in self.pending_by_rank.items():
                if not pending:
                    continue
                if pending[0].direction == _Direction.SEND:
                    target_rank = pending[0].target_rank
                    if self.pending_by_rank[target_rank] and self.pending_by_rank[target_rank][0].source_rank == rank:
                        self._complete_send_recv_pair(rank, target_rank, pending[0].num_bytes)
                        progress = True
                else:
                    source_rank = pending[0].source_rank
                    if self.pending_by_rank[source_rank] and self.pending_by_rank[source_rank][0].target_rank == rank:
                        self._complete_send_recv_pair(source_rank, rank, pending[0].num_bytes)
                        progress = True
            if not progress:
                break

    def _complete_send_recv_pair(self, source_rank: int, target_rank: int, num_bytes: int) -> None:
        """
        Complete a send/recv pair, updating the time for both ranks.
        """
        # Need to modify algorithm to allow sends to overlap.
        source_start_time = max(self.send_time_by_rank[source_rank], self.recv_time_by_rank[source_rank])
        target_start_time = self.recv_time_by_rank[target_rank]
        # The send can not begin until both ranks are ready.
        # We're implicitly modeling that every operation uses maximum throughput, so throughput is
        # not available until prior operations on both ends finish.
        sync_time = max(source_start_time, target_start_time)
        # If we're continuing to send to the same source as last send and we're at the last send time,
        # don't need to include latency (we're just continuing to fill the pipe).
        if (
            source_start_time == self.send_time_by_rank[source_rank]
            and target_rank == self.last_send_by_rank[source_rank]
        ):
            latency_ms = 0.0
        else:
            # We don't include the time for the final ACK of send.
            latency_ms = self.roundtrip_ms / 2
        self.last_send_by_rank[source_rank] = target_rank
        # Round to microsecond precision for now to help diagrams be clean.
        end_time = round(sync_time + 1000 * (latency_ms + (num_bytes / self.throughput_bytes_ms)))
        self.send_time_by_rank[source_rank] = end_time
        self.recv_time_by_rank[target_rank] = end_time
        # To limit to half duplex, force send and recv times to agree.
        if self.half_duplex:
            self.send_time_by_rank[target_rank] = end_time
            self.recv_time_by_rank[source_rank] = end_time
        # Inefficient compared to a queue, but we expect only a few pending sends/recvs.
        del self.pending_by_rank[source_rank][0]
        del self.pending_by_rank[target_rank][0]
        self.bytes_sent_by_rank[source_rank] += num_bytes
        self.bytes_received_by_rank[target_rank] += num_bytes
        self.events.append(
            NetworkEvent(
                source_rank=source_rank,
                target_rank=target_rank,
                num_bytes=num_bytes,
                source_start_time_microseconds=source_start_time,
                source_end_time_microseconds=end_time,
                target_start_time_microseconds=target_start_time,
                target_end_time_microseconds=end_time,
            )
        )
