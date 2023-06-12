import os
import sys
from time import perf_counter as now
import logging

import torch
import torch.distributed as dist

from pytorch_hccl_tests.commons import dist_init, get_device, log_env_info
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils
from pytorch_hccl_tests.osu.parser import get_parser

logger = logging.getLogger(__name__)


def osu_scatterv(args):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.float32
    device = get_device(rank)
    pg = None

    options = Options("Scatterv", args)
    Utils.check_numprocs(world_size, rank, limit=3)
    Utils.print_header(options.benchmark, rank)

    # TODO: unimplemented
    pass


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

    osu_scatterv(args=args)

    # Stop process group
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())  # noqa
