from time import perf_counter as now
import logging
import pandas as pd

import torch
import numpy as np
import torch.distributed as dist

from pytorch_hccl_tests.commons import get_device, get_dtype
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils


logger = logging.getLogger(__name__)


def reducescatter(args):
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = get_dtype(args.dtype)
    device = get_device(backend, rank)
    pg = None

    options = Options("Reduce_scatter", args)
    Utils.check_numprocs(world_size, rank, limit=3)
    Utils.print_header(options.benchmark, rank)

    df = pd.DataFrame(columns=["size_in_bytes", "avg_latency"])

    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large
        iterations = list(range(options.iterations + options.skip))

        recvcounts = np.zeros(world_size, dtype=np.int32)
        portion = 0
        remainder = 0
        portion = (size / 4) / world_size
        remainder = (size / 4) % world_size
        for i in range(world_size):
            recvcounts[i] = 0
            if (size / 4) < world_size:
                if i < (size / 4):
                    recvcounts[i] = 1
            else:
                if (remainder != 0) and (i < remainder):
                    recvcounts[i] += 1
                recvcounts[i] += portion

        tensor = torch.rand(recvcounts[rank], dtype=dtype).to(device)
        tensor_list = [torch.rand(int(size / 4), dtype=dtype).to(device)] * world_size

        dist.barrier()
        for i in iterations:
            if i == options.skip:
                tic = now()
            dist.reduce_scatter(tensor, tensor_list, dist.ReduceOp.SUM, pg, False)
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
    df.to_csv(f"osu_reducescatter-{device}-{world_size}.csv", index=False)
