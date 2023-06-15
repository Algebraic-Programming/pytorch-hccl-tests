import logging
from time import perf_counter as now

import pandas as pd
import torch
import torch.distributed as dist

from pytorch_hccl_tests.commons import (
    get_device,
)
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils

logger = logging.getLogger(__name__)


def osu_latency(args):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = get_device(rank)
    pg = None

    options = Options("Latency", args)
    Utils.check_numprocs(world_size, rank, limit=2)

    # Print header
    Utils.print_header(options.benchmark, rank)

    df = pd.DataFrame(columns=["size_in_bytes", "avg_latency"])

    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large
        iterations = range(options.iterations + options.skip)
        s_msg = torch.rand(size, dtype=torch.float32).to(device)
        r_msg = torch.rand(size, dtype=torch.float32).to(device)

        dist.barrier()
        if rank == 0:
            for i in iterations:
                if i == options.skip:
                    tic = now()
                dist.send(s_msg, 1, pg, 1)
                dist.recv(r_msg, 1, pg, 1)
            toc = now()
        elif rank == 1:
            for i in iterations:
                if i == options.skip:
                    tic = now()
                dist.recv(r_msg, 0, pg, 1)
                dist.send(s_msg, 0, pg, 1)
        toc = now()

        avg_latency = Utils.avg_lat(
            toc, tic, 2 * options.iterations, world_size, device
        )

        if rank == 0:
            logger.info("%-10d%18.2f" % (size, avg_latency))
            df = df.append(
                {"size_in_bytes": int(size), "avg_latency": avg_latency},
                ignore_index=True,
            )

    # Persist result to CSV file
    df.to_csv(f"osu_latency-{device}-{world_size}.csv", index=False)
