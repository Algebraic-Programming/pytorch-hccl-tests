import logging

import numpy as np
import pandas as pd
import torch.distributed as dist

from pytorch_hccl_tests.commons import (
    elaspsed_time_ms,
    get_device,
    get_device_event,
    safe_rand,
    sync_device,
)
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils

logger = logging.getLogger(__name__)


def reducescatter(args):
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = args.dtype
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
        portion = int((size / 4) // world_size)
        remainder = int((size / 4) % world_size)
        for i in range(world_size):
            recvcounts[i] = 0
            if (size / 4) < world_size:
                if i < (size / 4):
                    recvcounts[i] = 1
            else:
                if (remainder != 0) and (i < remainder):
                    recvcounts[i] += 1
                recvcounts[i] += portion

        # safe_rand is a wrapper of torch.rand for floats and
        # torch.randint for integral types
        tensor = safe_rand(recvcounts[rank], dtype=dtype).to(device)
        tensor_list = [
            safe_rand(portion, dtype=dtype).to(device) for _ in range(world_size)
        ]

        dist.barrier()
        for i in iterations:
            if i == options.skip:
                start_event = get_device_event(backend)
            dist.reduce_scatter(tensor, tensor_list, dist.ReduceOp.SUM, pg, False)
        end_event = get_device_event(backend)
        sync_device(backend)
        dist.barrier()

        total_time_ms = elaspsed_time_ms(backend, start_event, end_event)
        avg_latency = Utils.avg_lat(
            total_time_ms, options.iterations, world_size, device
        )

        if rank == 0:
            logger.info("%-10d%18.2f" % (size, avg_latency))
            df = df.append(
                {"size_in_bytes": int(size), "avg_latency": avg_latency},
                ignore_index=True,
            )

    # Persist result to CSV file
    if rank == 0:
        df.to_csv(
            f"osu_reducescatter-{device.type}-{dtype}-{world_size}.csv", index=False
        )
