import os
import sys
from time import perf_counter as now

import torch
import torch.distributed as dist

from pytorch_hccl_tests.commons import dist_init, get_device
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils
from pytorch_hccl_tests.osu.parser import get_parser


def osu_latency(args):
    rank = dist.get_rank()
    numprocs = dist.get_world_size()
    device = get_device(rank)
    pg = None

    options = Options("Latency", args)
    Utils.check_numprocs(numprocs, rank, limit=2)

    # Print header
    Utils.print_header(options.benchmark, rank)

    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large
        iterations = range(options.iterations + options.skip)
        s_msg = torch.rand(size, dtype=torch.float32).to(device)
        r_msg = torch.rand(size, dtype=torch.float32).to(device)

        dist.barrier()
        if rank == 0:
            for i in iterations:
                if i == options.skip:
                    tic = now()
                dist.send(s_msg, 1, pg, 1)
                dist.recv(r_msg, 1, pg, 1)
            toc = now()
        elif rank == 1:
            for i in iterations:
                if i == options.skip:
                    tic = now()
                dist.recv(r_msg, 0, pg, 1)
                dist.send(s_msg, 0, pg, 1)
        toc = now()

        Utils.print_stats(toc, tic, 2 * options.iterations, rank, numprocs, size)


def main():
    args = get_parser().parse_args()
    device = args.device

    # rank and world_size is set by torchrun
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize torch.distributed
    backend = dist_init(device, rank, world_size)
    if rank == 0:
        print(f"using device {device} with {backend} backend")
        print(f"world size is {world_size}")

    osu_latency(args=args)

    # Stop process group
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())  # noqa