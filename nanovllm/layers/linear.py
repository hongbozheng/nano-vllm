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

    def weight_loader(
            self,
            param: nn.Parameter,
            loaded_weight: torch.Tensor,
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism over the logical projection matrix.

    The projection is written as ``Y = X W^T + b``, where the stored weight
    ``W`` has shape ``(out_features, in_features)``.

    In tensor-parallel terminology, "column" refers to columns of ``W^T``
    (equivalently, output features). For the stored ``W``, this is a split
    along dimension ``0``.

    Each rank computes a disjoint slice of output features, so no cross-rank
    reduction is needed in ``forward``.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False,
    ) -> None:
        tp_size = dist.get_world_size()
        assert output_size % tp_size == 0, \
            f"output_size {output_size} must be divisible by tp_size {tp_size}"
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


class RowParallelLinear(LinearBase):
    """Linear layer with row parallelism over the logical projection matrix.

    The projection is written as ``Y = X W^T + b``, where the stored weight
    ``W`` has shape ``(out_features, in_features)``.

    In tensor-parallel terminology, "row" refers to rows of ``W^T``
    (equivalently, input features). For the stored ``W``, this is a split
    along dimension ``1``.

    The input is expected to be sharded across its last dimension to match the
    local weight shard. Each rank computes a partial output contribution, and
    ``forward`` sums partial results with ``all_reduce``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        tp_size = dist.get_world_size()
        assert input_size % tp_size == 0, \
            f"input_size {input_size} must be divisible by tp_size {tp_size}"
        super().__init__(
            input_size=divide(input_size, tp_size),
            output_size=output_size,
            bias=bias,
            tp_dim=1,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y


if __name__ == "__main__":
    # Example usage
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )
    layer = ReplicatedLinear(input_size=10, output_size=5)
    x = torch.randn(2, 10)
    y = layer(x)
    print("ReplicatedLinear layer initialized:", layer)
    print("Input shape:", x.shape, "Output shape:", y.shape)
