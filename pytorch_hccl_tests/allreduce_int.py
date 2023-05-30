import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

from pytorch_hccl_tests.commons import dist_init, bench_allreduce, print_root

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


def main():
    args = parser.parse_args()
    device = args.device
    repeat = args.repeat
    max_power = args.max_power

    rank = int(os.environ["LOCAL_RANK"])  # set by `torchrun`
    world_size = int(os.environ["WORLD_SIZE"])
    dist_init(device, rank, world_size)

    if rank == 0:
        print("=" * 20)
        print("Benchmark using int tensor")
        print("size (Bytes)", "latency (us)", "bandwidth (GB/s)")

    vector_size = 2
    bench_allreduce(vector_size, repeat, device)  # warm-up
    for _ in range(max_power):
        latency = bench_allreduce(vector_size, repeat, device, use_int=True)
        bw = (vector_size * 4) / latency / 1e3  # effective "bandwidth"
        print_root(vector_size, latency, bw)
        vector_size *= 2


if __name__ == "__main__":
    sys.exit(main())  # noqa
