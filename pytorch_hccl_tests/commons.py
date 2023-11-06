import logging
import platform
import sys
from time import perf_counter_ns as now
from typing import Any, List

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


_TORCH_DTYPES = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "short": torch.int16,
    "int": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
    "float16": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float32,
    "float32": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
}


def get_dtype(dtype: str) -> torch.dtype:
    dtype = _TORCH_DTYPES.get(dtype, "float16")
    return dtype


def is_integral(dtype: str) -> bool:
    tensor = torch.zeros(1, dtype=get_dtype(dtype))
    return not torch.is_floating_point(tensor) and not torch.is_complex(tensor)


def safe_rand(size: int, dtype: str) -> torch.Tensor:
    if is_integral(dtype):
        return torch.randint(low=0, high=100, size=(size,), dtype=get_dtype(dtype))
    return torch.rand(size, dtype=get_dtype(dtype))


def get_device(backend: str, local_rank: int):
    "Returns device"
    if torch.cuda.is_available() and backend in ["nccl", "mpi"]:
        n_devices = torch.cuda.device_count()
        if local_rank >= n_devices:
            raise RuntimeError(
                f"Local rank *{local_rank}* greater than number of CUDA devices {n_devices}"
            )
        return torch.device(f"cuda:{local_rank}")
    elif backend == "hccl":
        return torch.device(f"npu:{local_rank}")
    else:
        return torch.device("cpu")


def get_device_event(backend: str):
    "Returns device Event"
    if torch.cuda.is_available() and backend in ["nccl", "mpi"]:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event
    elif backend == "hccl":
        event = torch.npu.Event(enable_timing=True)
        event.record()
        return event
    else:
        return now()


def sync_device(backend: str):
    "Synchronize device"
    if torch.cuda.is_available() and backend in ["nccl", "mpi"]:
        torch.cuda.synchronize()
    elif backend == "hccl":
        torch.npu.synchronize()


def elaspsed_time_ms(backend: str, start, end):
    if (torch.cuda.is_available() and backend in ["nccl", "mpi"]) or backend == "hccl":
        # See https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution
        return start.elapsed_time(end)
    else:
        return (end - start) / 1000


def dist_init(device: str, local_rank: int):
    logger.info(f"Init distributed env device: {device} / local_rank {local_rank}")
    backend = None
    if device == "cpu":
        backend = "gloo"

    elif device == "npu":
        try:
            import torch_npu  # noqa
        except Exception:
            raise ImportError(
                "You must install PyTorch Ascend Adaptor from https://gitee.com/ascend/pytorch."
            )
        torch.npu.set_device(local_rank)
        backend = "hccl"
    elif device == "cuda":
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        raise ValueError("unknown device")

    dist.init_process_group(backend=backend)
    return backend


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


def print_root(vec_size: int, latency: float, bw: float):
    rank = dist.get_rank()
    if rank == 0:
        print(f"(Rank {rank}) {vec_size * 4}   {latency:.3f}  {bw:.6f}")


def setup_loggers(filename: str) -> List[Any]:
    file_handler = logging.FileHandler(filename=f"{filename}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    return [file_handler, stdout_handler]


def log_env_info(device, backend):
    world_size = dist.get_world_size()
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"PyTorch MPI enabled?: {torch.distributed.is_mpi_available()}")
    logger.info(f"PyTorch CUDA enabled?: {torch.cuda.is_available()}")
    logger.info(f"PyTorch NCCL enabled?: {torch.distributed.is_nccl_available()}")
    logger.info(f"PyTorch Gloo enabled?: {torch.distributed.is_gloo_available()}")
    try:
        import torch_npu  # noqa

        logger.info(f"PyTorch HCCL enabled?: {torch.distributed.is_hccl_available()}")
        logger.info(f"PyTorch Ascend Adapter (NPU) version: {torch_npu.__version__}")
    except Exception:
        logger.warning("*" * 80)
        logger.warning("* PyTorch Ascend (NPU) is NOT installed.")
        logger.warning(
            "* You must install PyTorch Ascend Adaptor from https://gitee.com/ascend/pytorch. *"
        )
        logger.warning("*" * 80)

    logger.info(f"Using device *{device}* with *{backend}* backend")
    logger.info(f"World size: {world_size}")
