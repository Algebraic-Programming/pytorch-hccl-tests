import sys
import torch
from pytorch_hccl_tests.benchmarks import bench_allreduce
import torch.distributed as dist
import os
import argparse


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


def main(args):
    args = parser.parse_args()
    device = args.device
    repeat = args.repeat
    max_power = args.max_power

    rank = int(os.environ["LOCAL_RANK"])  # set by `torchrun`
    world_size = int(os.environ["WORLD_SIZE"])
    dist_init(device, rank, world_size)

    if rank == 0:
        print("=" * 20)
        print("Benchmark using float tensor")
        print("size (Bytes)", "latency (us)", "bandwidth (GB/s)")

    vector_size = 2
    bench_allreduce(vector_size, repeat, device)  # warm-up
    for _ in range(max_power):
        latency = bench_allreduce(vector_size, repeat, device, use_int=False)
        bw = (vector_size * 4) / latency / 1e3  # effective "bandwidth"
        print(f"(rank {rank}) {vector_size * 4}   {latency}  {bw}")
        vector_size *= 2

    if rank == 0:
        print("=" * 20)
        print("Benchmark using int tensor")
        print("size (Bytes)", "latency (us)", "bandwidth (GB/s)")

    vector_size = 2
    bench_allreduce(vector_size, repeat, device)  # warm-up
    for _ in range(max_power):
        latency = bench_allreduce(vector_size, repeat, device, use_int=True)
        bw = (vector_size * 4) / latency / 1e3  # effective "bandwidth"
        print(f"(rank {rank}) {vector_size * 4}   {latency}  {bw}")
        vector_size *= 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "npu", "cuda"],
        type=str,
        help="device type",
    )

    parser.add_argument(
        "--repeat",
        default=20,
        type=int,
        help="repeating allreduce calls and take average",
    )

    parser.add_argument(
        "--max-power",
        default=24,
        type=int,
        help="largest message size in 2's power",
    )
    args = parser.parse_args()

    sys.exit(main(args))  # noqa
