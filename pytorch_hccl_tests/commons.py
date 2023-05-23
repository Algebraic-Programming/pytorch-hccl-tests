import torch
import torch.distributed as dist
import time


def dist_init(device: str, rank: int, world_size: int):
    # switch communication backend
    if device == "cpu":
        backend = "gloo"

    elif device == "npu":
        try:
            import torch_npu
        except Exception:
            raise ImportError(
                "You must install PyTorch Ascend Adaptor from https://gitee.com/ascend/pytorch."
            )
        torch.npu.set_device(rank)
        backend = "hccl"

    elif device == "cuda":
        torch.cuda.set_device(rank)
        backend = "nccl"

    else:
        raise ValueError("unknown device")

    dist.init_process_group(backend=backend)
    if rank == 0:
        print(f"using device {device} with {backend} backend")
        print(f"world size is {world_size}")

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
