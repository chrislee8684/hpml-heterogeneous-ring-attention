import os
from abc import abstractmethod
from typing import List, Optional, Tuple, Any
import time

import torch
import math
from torch import Tensor, nn
import torch.distributed
import torch.distributed as dist
from torch.distributed import P2POp

from fms.utils import tp_wrapping


if "DISTRIBUTED_STRATEGY_IGNORE_MODULES" in os.environ:
    _distributed_strategy_ignore_modules = os.environ[
        "DISTRIBUTED_STRATEGY_IGNORE_MODULES"
    ].split(",")
else:
    _distributed_strategy_ignore_modules = []


class DistributedStrategy:
    def __init__(self, from_meta=False):
        self.from_meta = from_meta

    def __should_distribute(self, module_name: str) -> bool:
        return module_name not in _distributed_strategy_ignore_modules

    def distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        """
        Optionally a distributed strategy may distribute modules that are not
        numbered layers
        """
        module_name = type(module).__name__
        if self.__should_distribute(module_name):
            return self._distribute_module(module, final_layers)
        else:
            print(f"ignoring module={module_name} when distributing module")
            return module

    def distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        """
        Distribute each layer as-appropriate
        """
        block_name = type(block).__name__
        if self.__should_distribute(block_name):
            return self._distribute_layer(block, layer)
        else:
            print(f"ignoring block={block_name} when distributing layer")
            return block

    @abstractmethod
    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        """
        Distribute modules that are not numbered layers
        """
        pass

    @abstractmethod
    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        """
        Distribute each layer
        """
        pass


class NotDistributed(DistributedStrategy):
    def __init__(self, from_meta=False):
        super().__init__(from_meta)

    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        return module

    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        return block


NoOpStrategy = NotDistributed()


class DeviceMover(nn.Module):
    def __init__(self, module: nn.Module, device):
        super().__init__()
        self.device = device
        # make this wrapper module behave as if it was the wrapped module.
        attr = module.__dict__
        attr["module"] = module.to(device)
        attr["device"] = device
        self.__dict__ = attr

    def forward(self, *args, **kwargs):
        device = self.device
        args = [
            arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args
        ]
        kwargs = {
            k: (
                kwargs[k].to(device)
                if isinstance(kwargs[k], torch.Tensor)
                else kwargs[k]
            )
            for k in kwargs
        }
        return self.module(*args, **kwargs)


class UniformModelParallelStrategy(DistributedStrategy):
    def __init__(self, devices: List[int], num_layers: int, from_meta=False):
        super().__init__(from_meta)
        num_dev = len(devices)
        layers_per_dev = num_layers // num_dev
        remainder = num_layers - (layers_per_dev * num_dev)
        self.layer_to_device = [0] * num_layers
        layer_id = 0
        for dev_idx in range(len(devices)):
            for i in range(layers_per_dev):
                self.layer_to_device[layer_id] = devices[dev_idx]
                layer_id = layer_id + 1
            if remainder > 0:
                self.layer_to_device[layer_id] = devices[dev_idx]
                layer_id = layer_id + 1
                remainder -= 1

    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        device = self.layer_to_device[layer]
        if self.from_meta:
            # https://github.com/pytorch/pytorch/pull/113647
            block.to_empty(device=device)  # type: ignore[arg-type]
        wrapped = DeviceMover(block, device)
        return wrapped

    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        if final_layers:
            device = self.layer_to_device[len(self.layer_to_device) - 1]
        else:
            device = self.layer_to_device[0]
        if self.from_meta:
            return module.to_empty(device=device)  # type: ignore[arg-type]
        wrapped = DeviceMover(module, device)
        return wrapped


class TensorParallelStrategy(DistributedStrategy):
    def __init__(self, group=None, from_meta=False):
        super().__init__(from_meta)
        assert torch.distributed.is_initialized(), "must initialize a process group"
        self.group = group if group is not None else torch.distributed.GroupMember.WORLD

    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        return tp_wrapping.apply_tp(module, self.group)

    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        return tp_wrapping.apply_tp(block, self.group)


class RingAttentionStrategy(DistributedStrategy):
    def __init__(
        self,block_lens: List[int], block_size: Optional[int] = None, group: Optional[dist.ProcessGroup] = None, from_meta: bool = False
    ):
        super().__init__(from_meta)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.group = group
            self.rank = torch.distributed.get_rank(group=self.group)
            self.world_size = torch.distributed.get_world_size(group=self.group)
        else:
            self.group = None
            self.rank = 0
            self.world_size = 1


        # Hetero block lengths
        block_lens = list(block_lens)
        assert len(block_lens) == self.world_size, (
            f"len(block_lens)={len(block_lens)} vs world_size={self.world_size}"
        )

        self.block_lens = block_lens

        # Prefix sums for global starts
        self.block_starts = [0]
        for i in range(self.world_size - 1):
            self.block_starts.append(self.block_starts[-1] + self.block_lens[i])

        # Local valid length
        self._local_valid_len = self.block_lens[self.rank]

        # CRITICAL: a common block_size for padding in ring_shift_start/_pad_to_block_size
        # All ranks will pad up to this.
        self.block_size = max(self.block_lens)
        self._original_seq_len: Optional[int] = None
    
        # Dedicated CUDA stream for async communication overlap
        self._comm_stream = torch.cuda.Stream(priority=-1) if torch.cuda.is_available() else None



    def _pad_to_block_size(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        length = tensor.size(dim)
        if length == self.block_size:
            return tensor
        pad_shape = list(tensor.shape)
        pad_shape[dim] = self.block_size - length
        padding = torch.zeros(*pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=dim)

    def _distribute_module(self, module: nn.Module, final_layers: bool = False) -> nn.Module:
        return module

    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        return block

    def shard_input(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        self._original_seq_len = seq_len

        if self.world_size == 1:
            self._local_valid_len = seq_len
            return x

        if self.block_size is None or seq_len > self.block_size:
            self.block_size = math.ceil(seq_len / self.world_size)
        start = self.rank * self.block_size
        end = min(start + self.block_size, seq_len)
        self._local_valid_len = max(0, end - start)
        if self._local_valid_len > 0:
            return x.narrow(1, start, self._local_valid_len)
        shp = list(x.shape)
        shp[1] = 0
        return torch.empty(*shp, dtype=x.dtype, device=x.device)

    def ring_shift_kv_async(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_len: int,
        enable_timing: bool = False,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.cuda.Event]]:
        """
        Start async KV ring shift. Returns (requests, recv_k, recv_v, recv_len, comm_start_event).
        Communication runs on dedicated stream for overlap with compute.
        If enable_timing=True, returns CUDA event for start time. End event is recorded in ring_shift_kv_wait.
        """
        if self.world_size == 1:
            return None, k, v, torch.tensor([valid_len], device=k.device), None

        send_to = (self.rank + 1) % self.world_size
        recv_from = (self.rank - 1 + self.world_size) % self.world_size
        seq_dim = 2

        # Slice and pad KV to block_size
        if valid_len > 0:
            idx = [slice(None)] * k.ndim
            idx[seq_dim] = slice(0, valid_len)
            send_k = self._pad_to_block_size(k[tuple(idx)], dim=seq_dim).contiguous()
            send_v = self._pad_to_block_size(v[tuple(idx)], dim=seq_dim).contiguous()
        else:
            send_k = self._pad_to_block_size(k.new_zeros(*k.shape[:seq_dim], 0, k.shape[-1]), dim=seq_dim).contiguous()
            send_v = self._pad_to_block_size(v.new_zeros(*v.shape[:seq_dim], 0, v.shape[-1]), dim=seq_dim).contiguous()

        recv_k = torch.empty_like(send_k)
        recv_v = torch.empty_like(send_v)
        send_len = torch.tensor([valid_len], dtype=torch.int32, device=k.device)
        recv_len = torch.empty(1, dtype=torch.int32, device=k.device)

        # Record event so comm stream waits for send buffers to be ready
        ready_event = torch.cuda.Event()
        ready_event.record()

        # Create start timing event if requested
        comm_start_event = torch.cuda.Event(enable_timing=True) if enable_timing else None

        with torch.cuda.stream(self._comm_stream):
            self._comm_stream.wait_event(ready_event)

            # Record start time on comm stream
            if comm_start_event:
                comm_start_event.record()

            ops = [
                P2POp(dist.isend, send_len, send_to),
                P2POp(dist.irecv, recv_len, recv_from),
                P2POp(dist.isend, send_k, send_to),
                P2POp(dist.irecv, recv_k, recv_from),
                P2POp(dist.isend, send_v, send_to),
                P2POp(dist.irecv, recv_v, recv_from),
            ]
            reqs = dist.batch_isend_irecv(ops)

        return reqs, recv_k, recv_v, recv_len, comm_start_event

    def ring_shift_kv_wait(
        self,
        reqs: Any,
        recv_k: torch.Tensor,
        recv_v: torch.Tensor,
        recv_len: torch.Tensor,
        enable_timing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[torch.cuda.Event]]:
        """
        Wait for async KV shift to complete. Returns (k, v, valid_len, comm_end_event).
        If enable_timing=True, records and returns a CUDA event after transfers complete.
        """
        if reqs is None:
            return recv_k, recv_v, recv_len.item(), None

        for req in reqs:
            req.wait()

        # Record end event AFTER transfers complete (on comm stream)
        comm_end_event = None
        if enable_timing:
            comm_end_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self._comm_stream):
                comm_end_event.record()

        self._comm_stream.synchronize()

        new_len = recv_len.item()
        if new_len == 0:
            return recv_k[:, :, :0], recv_v[:, :, :0], 0, comm_end_event
        return recv_k[:, :, :new_len].contiguous(), recv_v[:, :, :new_len].contiguous(), new_len, comm_end_event

    def get_local_valid_len(self) -> int:
        return self._local_valid_len or 0

    def gather_tensor(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        t = tensor.contiguous()
        if t.size(dim) != self.block_size:
            t = self._pad_to_block_size(t, dim)
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered, t, group=self.group)
        result = torch.cat(gathered, dim=dim)
        if dim == 1 and self._original_seq_len is not None:
            result = result.narrow(dim, 0, self._original_seq_len)
        return result
