import logging
import time
from typing import Any, List

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def get_device(rank: int):
    "Returns CUDA device with id rank % n_devices. Falls back to CPU device"
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        device_id = rank % n_devices
        if rank >= n_devices:
            logger.warning(f"Rank *{rank}* assigned CUDA:{device_id}")
        torch.cuda.set_device(f"cuda:{device_id}")
        return torch.device(f"cuda:{device_id}")
    else:
        return torch.device("cpu")


def dist_init(device: str, rank: int, world_size: int):
    backend = None
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
    return backend


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


def wait_all(async_reqs: List[Any]) -> None:
    """Wait for all request handles
    Parameters
    ----------
    async_reqs : List[Any]
        List of torch.distributed communication handles
    """
    n = sum(int(req is not None) for req in async_reqs)
    in_flight_msgs = n
    while in_flight_msgs > 0:
        for req in async_reqs:
            if req is not None:
                req.wait()
                in_flight_msgs = in_flight_msgs - 1
