import logging
import os
import sys
from time import perf_counter as now

import pandas as pd
import torch
import torch.distributed as dist

from pytorch_hccl_tests.commons import (
    dist_init,
    get_device,
    log_env_info,
    setup_loggers,
)
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils
from pytorch_hccl_tests.osu.parser import get_parser

logger = logging.getLogger(__name__)


def osu_multi_lat(args):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    pairs = int(world_size / 2)
    dtype = torch.float32
    device = get_device(rank)
    pg = None

    options = Options("Multi Latency", args)

    Utils.print_header(options.benchmark, rank)

    df = pd.DataFrame(columns=["size_in_bytes", "avg_latency"])

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

        avg_latency = Utils.avg_lat(toc, tic, 2 * options.iterations, world_size)

        if rank == 0:
            logger.info("%-10d%18.2f" % (size, avg_latency))
            df = df.append(
                {"size_in_bytes": int(size), "avg_latency": avg_latency.item()},
                ignore_index=True,
            )

        # Persist result to CSV file
        df.to_csv(f"osu_multi_latency-{world_size}.csv", index=False)


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

    osu_multi_lat(args=args)

    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())  # noqa
