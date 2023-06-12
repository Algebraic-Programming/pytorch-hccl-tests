import os
import sys
from time import perf_counter as now
import logging

import torch
import numpy as np
import torch.distributed as dist

from pytorch_hccl_tests.commons import dist_init, get_device, log_env_info
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils
from pytorch_hccl_tests.osu.parser import get_parser


logger = logging.getLogger(__name__)


def osu_reduce_scatter(args):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.float32
    device = get_device(rank)
    pg = None

    options = Options("Reduce_scatter", args)
    Utils.check_numprocs(world_size, rank, limit=3)
    Utils.print_header(options.benchmark, rank)

    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large
        iterations = list(range(options.iterations + options.skip))

        recvcounts = np.zeros(world_size, dtype=np.int32)
        portion = 0
        remainder = 0
        portion = (size / 4) / world_size
        remainder = (size / 4) % world_size
        for i in range(world_size):
            recvcounts[i] = 0
            if (size / 4) < world_size:
                if i < (size / 4):
                    recvcounts[i] = 1
            else:
                if (remainder != 0) and (i < remainder):
                    recvcounts[i] += 1
                recvcounts[i] += portion

        tensor = torch.rand(recvcounts[rank], dtype=dtype).to(device)
        tensor_list = [torch.rand(int(size / 4), dtype=dtype).to(device)] * world_size

        dist.barrier()
        for i in iterations:
            if i == options.skip:
                tic = now()
            dist.reduce_scatter(tensor, tensor_list, dist.ReduceOp.SUM, pg, False)
        toc = now()
        dist.barrier()

        Utils.print_stats(toc, tic, options.iterations, rank, world_size, size)


def main():
    args = get_parser().parse_args()
    device = args.device

    # rank and world_size is set by torchrun
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize torch.distributed
    backend = dist_init(device, rank, world_size)
    if rank == 0:
        log_env_info(device, backend)

    osu_reduce_scatter(args=args)

    # Stop process group
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())  # noqa
