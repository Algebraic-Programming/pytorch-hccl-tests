import os
import sys
from time import perf_counter as now
import logging

import torch.distributed as dist

from pytorch_hccl_tests.commons import dist_init, log_env_info, get_device
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils
from pytorch_hccl_tests.parser import get_parser

logger = logging.getLogger(__name__)


def osu_barrier(args):
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = get_device(backend, rank)

    options = Options("Barrier", args)
    Utils.check_numprocs(world_size, rank, limit=3)
    Utils.print_header(options.benchmark, rank)

    iterations = list(range(options.iterations + options.skip))

    dist.barrier()
    for i in iterations:
        if i == options.skip:
            tic = now()
        dist.barrier()
    toc = now()
    dist.barrier()

    avg_lat = Utils.avg_lat(toc, tic, options.iterations, world_size, device)
    if rank == 0:
        print("%-10d%18.2f" % (0, avg_lat))


def main():
    args = get_parser().parse_args()
    device = args.device

    # rank and world_size is set by torchrun
    rank = int(os.environ["LOCAL_RANK"])

    # Initialize torch.distributed
    backend = dist_init(device, rank)
    if rank == 0:
        log_env_info(device, backend)

    osu_barrier(args=args)

    # Stop process group
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())  # noqa
