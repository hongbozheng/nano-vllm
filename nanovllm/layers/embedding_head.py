import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from ..utils.context import get_context


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


# weight tying with embedding layer
class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

    # x: [batch_size, seq_len, hidden_size]
    # weight: [vocab_size_per_partition, hidden_size]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            # cu_seqlens_q = [0, 5, 8, 12]
            # last_indices = [5, 8, 12] - 1 = [4, 7, 11]
            last_token = context.cu_seqlens_q[1:] - 1  # exclude the first element which is 0
            x = x[last_token].contiguous()

        # logits: [batch_size, seq_len, vocab_size_per_partition]
        # F.linear automatically transpose the weight
        logits = torch.nn.functional.linear(x, self.weight)
        if self.tp_size > 1:
            # prepare for all_gather only for GPU 0 which is the main GPU
            all_logits = [torch.empty(logits.size(), device=logits.device) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # dist.gather collects the logits from all GPUs to GPU 0
            dist.gather(logits, gather_list=all_logits, dst=0)
            # concatenate
            if self.tp_rank == 0:
                # [batch_size, seq_len, padded_vocab_size]
                logits = torch.cat(all_logits, dim=-1)
                # trim to original vocab size
                logits = logits[..., :self.num_embeddings]

        return logits
