import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

from pytorch_hccl_tests.commons import dist_init, bench_allreduce

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
        print("Benchmark using float tensor")
        print("size (Bytes)", "latency (us)", "bandwidth (GB/s)")

    vector_size = 2
    bench_allreduce(vector_size, repeat, device)  # warm-up
    for _ in range(max_power):
        latency = bench_allreduce(vector_size, repeat, device, use_int=False)
        bw = (vector_size * 4) / latency / 1e3  # effective "bandwidth"
        print(f"(rank {rank}) {vector_size * 4}   {latency}  {bw}")
        vector_size *= 2


if __name__ == "__main__":
    sys.exit(main())  # noqa
