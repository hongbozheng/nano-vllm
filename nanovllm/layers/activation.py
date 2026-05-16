import torch
import torch.nn as nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(chunks=2, dim=-1)
        return F.silu(x) * y


if __name__ == "__main__":
    import time

    if not torch.cuda.is_available():
        print("CUDA is not available. Please run this code on a GPU.")
        exit(1)

    device = torch.device("cuda")

    layer = SiluAndMul().to(device=device)
    x = torch.randn(64, 1024, 1024).to(device=device)

    for _ in range(10):     # warm-up iterations
        _ = layer(x)

    times = []
    for _ in range(100):    # timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(x)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f}ms")
