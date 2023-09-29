import logging
from time import perf_counter_ns as now

import pandas as pd
import torch
import torch.distributed as dist

from pytorch_hccl_tests.commons import get_device, get_dtype
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils

logger = logging.getLogger(__name__)


def alltoall(args):
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = get_dtype(args.dtype)
    device = get_device(backend, rank)

    options = Options("Alltoall", args)
    Utils.check_numprocs(world_size, rank, limit=3)
    Utils.print_header(options.benchmark, rank)

    df = pd.DataFrame(columns=["size_in_bytes", "avg_latency"])

    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large
        iterations = list(range(options.iterations + options.skip))

        in_tensor = torch.arange(size, dtype=dtype).to(device) + rank * world_size
        out_tensor = torch.zeros(size, dtype=dtype)

        dist.barrier()
        for i in iterations:
            if i == options.skip:
                tic = now()
            dist.all_to_all_single(out_tensor, in_tensor)
        toc = now()
        dist.barrier()

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
    if rank == 0:
        df.to_csv(f"osu_alltoall-{device.type}-{dtype}-{world_size}.csv", index=False)
