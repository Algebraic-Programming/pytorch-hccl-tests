"""Console script for pytorch-hccl-tests."""
import logging
import os
import sys

import torch.distributed as dist

from pytorch_hccl_tests.commons import dist_init, log_env_info, setup_loggers
from pytorch_hccl_tests.osu.p2p import osu_bibw, osu_bw, osu_latency, osu_multi_lat
from pytorch_hccl_tests.osu.collectives import (
    osu_allreduce,
    osu_broadcast,
    osu_alltoall,
)
from pytorch_hccl_tests.osu.parser import get_parser
from pytorch_hccl_tests.osu.startup import osu_hello

logger = logging.getLogger(__name__)


def select_bench(args):
    bench = args.benchmark.lower()
    switcher = {
        "hello": osu_hello,
        "latency": osu_latency,
        "bandwidth": osu_bw,
        "bibw": osu_bibw,
        "multi-latency": osu_multi_lat,
        "allreduce": osu_allreduce,
        "broadcast": osu_broadcast,
        "alltoall": osu_alltoall,
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

    log_handlers = setup_loggers(args.benchmark)
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
    args.backend = backend
    if rank == 0:
        log_env_info(device, backend)

    select_bench(args)

    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())  # noqa
