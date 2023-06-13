import logging

import torch.distributed as dist

logger = logging.getLogger(__name__)


def osu_hello(args):
    bench = args.benchmark
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dist.barrier()
    if rank == 0:
        logger.info(f"# HCCL Hello test ({bench})")
        logger.info(f"# This is a test with {world_size} processes")
