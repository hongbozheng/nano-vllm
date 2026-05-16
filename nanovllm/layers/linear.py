import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False,
            tp_dim: int | None = None,
    ) -> None:
        super().__init__()
        # set tp_dim, tp_rank, tp_size for tensor parallelism
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        # initialize weight and bias parameters
        self.weight = nn.Parameter(torch.empty(size=(output_size, input_size)))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(data=torch.empty(size=(output_size,)))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False,
    ) -> None:
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            tp_dim=None,
        )

    def weight_loader(
            self,
            param: nn.Parameter,
            load_weight: torch.Tensor,
    ) -> None:
        param.data.copy_(load_weight)


class ColumnParallelLinear(LinearBase):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False,
    ) -> None:
        tp_size = dist.get_world_size()
        super().__init__(
            input_size=input_size,
            output_size=divide(output_size, tp_size),
            bias=bias,
            tp_dim=0,
        )

    def weight_loader(
            self,
            param: nn.Parameter,
            loaded_weight: torch.Tensor,
    ) -> None:
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
            self,
            input_size: int,
            output_sizes: list[int],
            bias: bool = False,
    ) -> None:
        self.output_sizes = output_sizes
        super().__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
        )

    def weight_loader(
            self,
            param: nn.Parameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: int,
    ) -> None:
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)   


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
            self,
            hidden_size: int,
            head_size: int,
            total_num_heads: int,
            total_num_kv_heads: int,
            bias: bool = False,
    ) -> None:
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(
            hidden_size=hidden_size,
            output_size=output_size,
            bias=bias,
        )

    def weight_loader(
            self,
            param: nn.Parameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: str,
    ) -> None:
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
