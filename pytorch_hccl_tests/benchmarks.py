import torch
from time import time
import torch.distributed as dist


def bench_allreduce(
    vector_size, repeat: int, device, use_int=False, pause=0.05
) -> float:
    time_total = 0  # in us
    for _ in range(repeat):
        if use_int:
            x = torch.randint(10, (vector_size,)).to(device)
        else:
            x = torch.rand(vector_size).to(device)

        start = time.monotonic_ns()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=False)
        end = time.monotonic_ns()
        time_once = (end - start) / 1e3
        time_total += time_once

        time.sleep(pause)

    return time_total / repeat
