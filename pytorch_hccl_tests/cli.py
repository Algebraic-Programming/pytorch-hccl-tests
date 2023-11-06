"""Console script for pytorch-hccl-tests."""
import logging
import os
import sys

import torch.distributed as dist

from pytorch_hccl_tests.commons import dist_init, log_env_info, setup_loggers
from pytorch_hccl_tests.osu.collectives import (
    allgather,
    allreduce,
    alltoall,
    barrier,
    broadcast,
    reducescatter,
)
from pytorch_hccl_tests.osu.p2p import bibw, bw, latency, multi_lat
from pytorch_hccl_tests.osu.startup import hello
from pytorch_hccl_tests.parser import get_parser

logger = logging.getLogger(__name__)


def select_bench(args):
    bench = args.benchmark.lower()
    switcher = {
        "hello": hello,
        "latency": latency,
        "bandwidth": bw,
        "bibw": bibw,
        "multi-latency": multi_lat,
        "allreduce": allreduce,
        "allgather": allgather,
        "broadcast": broadcast,
        "reducescatter": reducescatter,
        "alltoall": alltoall,
        "barrier": barrier,
    }
    if bench in switcher:
        switcher[bench](args)
    else:
        available = "|".join(switcher.keys())
        err_msg = f"Input benchmark *{args.benchmark}* is not supported. Valid benchmarks are {available}"
        logger.error(err_msg)
        raise ValueError(err_msg)


def main():  # noqa
    args = get_parser().parse_args()
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

    # rank and world_size is set by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if rank == 0:
        logger.info("*" * 30)
        logger.info(f"Selected benchmark : {benchmark}")
        logger.info(f"Input device param : {device}")
        logger.info(f"Input dtype param  : {dtype}")
        logger.info(f"Global rank        : {rank}")
        logger.info(f"Local rank         : {local_rank}")
        logger.info("*" * 30)

    # Initialize torch.distributed
    backend = dist_init(device, local_rank)
    args.backend = backend
    args.local_rank = local_rank
    if rank == 0:
        log_env_info(device, backend)

    select_bench(args)

    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())  # noqa
