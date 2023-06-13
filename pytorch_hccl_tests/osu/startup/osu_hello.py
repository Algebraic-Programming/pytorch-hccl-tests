import logging
import os
import sys

import torch.distributed as dist

from pytorch_hccl_tests.commons import dist_init, log_env_info, setup_loggers
from pytorch_hccl_tests.osu.parser import get_parser

logger = logging.getLogger(__name__)


def osu_hello(args):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dist.barrier()
    if rank == 0:
        logger.info("# HCCL Hello test")
        logger.info(f"# This is a test with {world_size} processes")


def main():
    args = get_parser().parse_args()
    device = args.device

    log_handlers = setup_loggers(__name__)
    log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=log_handlers,
    )

    # rank and world_size is set by torchrun
    rank = int(os.environ["LOCAL_RANK"])

    # Initialize torch.distributed
    backend = dist_init(device, rank)
    if rank == 0:
        log_env_info(device, backend)

    osu_hello(args=args)

    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())  # noqa
