import logging
import os

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from pytorch_hccl_tests.cli import select_bench
from pytorch_hccl_tests.commons import dist_init, log_env_info, setup_loggers
from pytorch_hccl_tests.parser import get_parser

logger = logging.getLogger(__name__)


THREAD_POOL_NUM = 1
TIMEOUT = 100


def worker_entrypoint(rank, args):
    device = args.device
    dtype = args.dtype
    benchmark = args.benchmark

    log_handlers = setup_loggers(args.benchmark)
    log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=log_handlers,
    )

    # Required by torch distributed
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)

    if rank == 0:
        logger.info("*" * 30)
        logger.info(f"Selected benchmark : {benchmark}")
        logger.info(f"Input device param : {device}")
        logger.info(f"Input dtype param  : {dtype}")
        logger.info("*" * 30)

    # Initialize torch.distributed
    backend = dist_init(device, rank)
    args.backend = backend
    if rank == 0:
        log_env_info(device, backend)

    select_bench(args)

    dist.destroy_process_group()


@pytest.mark.skipif(True, reason="takes too long")
def test_allreduce_size_two():
    args = get_parser().parse_args()
    args.benchmark = "allreduce"
    args.world_size = 2
    mp.spawn(worker_entrypoint, args=(args,), nprocs=args.world_size)
    assert True


@pytest.mark.skipif(True, reason="takes too long")
def test_allgather_size_two():
    args = get_parser().parse_args()
    args.benchmark = "allgather"
    args.world_size = 2
    mp.spawn(worker_entrypoint, args=(args,), nprocs=args.world_size)
    assert True


@pytest.mark.skipif(True, reason="takes too long")
def test_latency_size_two():
    args = get_parser().parse_args()
    args.benchmark = "latency"
    args.world_size = 2
    mp.spawn(worker_entrypoint, args=(args,), nprocs=args.world_size)
    assert True
