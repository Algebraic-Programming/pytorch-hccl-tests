import logging
import os
import sys
from time import perf_counter as now

import torch
import torch.distributed as dist

from pytorch_hccl_tests.commons import dist_init, get_device, log_env_info, setup_loggers
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils
from pytorch_hccl_tests.osu.parser import get_parser

logger = logging.getLogger(__name__)


def osu_multi_lat(args):
    rank = dist.get_rank()
    numprocs = dist.get_world_size()
    pairs = int(numprocs / 2)
    dtype = torch.float32
    device = get_device(rank)
    pg = None

    options = Options("Multi Latency", args)
    Utils.check_numprocs(numprocs, rank, limit=2)

    Utils.print_header(options.benchmark, rank)

    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large

        iterations = list(range(options.iterations + options.skip))
        s_msg = torch.rand(size, dtype=dtype).to(device)
        r_msg = torch.rand(size, dtype=dtype).to(device)

        dist.barrier()
        if rank < pairs:
            partner = rank + pairs
            for i in iterations:
                if i == options.skip:
                    tic = now()
                dist.send(s_msg, partner, pg, 1)
                dist.recv(r_msg, partner, pg, 1)
            toc = now()
        else:
            partner = rank - pairs
            for i in iterations:
                if i == options.skip:
                    tic = now()
                dist.recv(r_msg, partner, pg, 1)
                dist.send(s_msg, partner, pg, 1)
            toc = now()

        Utils.print_stats(toc, tic, 2 * options.iterations, rank, numprocs, size)


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
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize torch.distributed
    backend = dist_init(device, rank, world_size)
    if rank == 0:
        log_env_info(device, backend)

    osu_multi_lat(args=args)

    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())  # noqa
