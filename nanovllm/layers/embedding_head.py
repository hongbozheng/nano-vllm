import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        self.num_embeddings = num_embeddings
        self.num_embeddings_padded = (num_embeddings + self.tp_size - 1) // self.tp_size * self.tp_size
        self.num_embeddings_per_partition = self.num_embeddings_padded // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param_data = param.data

        offset = self.vocab_start_idx
        shard_size = self.num_embeddings_per_partition

        actual_start = min(offset, self.num_embeddings)
        actual_end = min(offset + shard_size, self.num_embeddings)
        actual_size = max(0, actual_end - actual_start)

        if actual_size > 0:
            sharded_weight = loaded_weight.narrow(0, actual_start, actual_size)
            param_data[:actual_size, :].copy_(sharded_weight)

        if actual_size < shard_size:
            param_data[actual_size:].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (
            (x >= self.vocab_start_idx)
            & (x < self.vocab_end_idx)
            & (x < self.num_embeddings)
        )
        local_x = (x - self.vocab_start_idx).masked_fill(~mask, 0)
        output = F.embedding(local_x, self.weight)

        # Ensure out-of-shard tokens contribute zeros before the all-reduce.
        output = output * mask.unsqueeze(-1).to(output.dtype)
        if self.tp_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output
