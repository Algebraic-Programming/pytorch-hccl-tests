import os
import sys

import logging
import torch.distributed as dist

from pytorch_hccl_tests.commons import dist_init, log_env_info
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils
from pytorch_hccl_tests.osu.parser import get_parser


logger = logging.getLogger(__name__)


def osu_gatherv(args):
    # backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # dtype = get_dtype(args.dtype)
    # device = get_device(rank)
    # pg = None

    options = Options("Gatherv", args)
    Utils.check_numprocs(world_size, rank, limit=3)
    Utils.print_header(options.benchmark, rank)

    # TODO: unimplemented


def main():
    args = get_parser().parse_args()
    device = args.device

    # rank and world_size is set by torchrun
    rank = int(os.environ["LOCAL_RANK"])

    # Initialize torch.distributed
    backend = dist_init(device, rank)
    if rank == 0:
        log_env_info(device, backend)

    osu_gatherv(args=args)

    # Stop process group
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())  # noqa
