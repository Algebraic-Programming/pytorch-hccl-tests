import logging

import pandas as pd
import torch.distributed as dist

from pytorch_hccl_tests.commons import (
    elaspsed_time_ms,
    get_device,
    get_device_event,
    get_nbytes_from_dtype,
    safe_rand,
    sync_device,
)
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils

logger = logging.getLogger(__name__)


def osu_scatter(args):
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = args.dtype
    device = get_device(backend, args.local_rank)
    pg = None

    options = Options("Scatter", args)
    Utils.check_numprocs(world_size, rank, limit=3)
    Utils.print_header(options.benchmark, rank)

    df = pd.DataFrame(columns=["size_in_bytes", "avg_latency_ms"])

    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large

        tensor = safe_rand(size, dtype=dtype).to(device)
        scatter_list = None
        if rank == 0:
            scatter_list = [
                safe_rand(size, dtype=dtype).to(device) for _ in range(world_size)
            ]

        dist.barrier()
        for i in range(options.iterations + options.skip):
            if i == options.skip:
                start_event = get_device_event(backend)
            dist.scatter(tensor, scatter_list, 0, pg, False)
        end_event = get_device_event(backend)
        sync_device(backend)
        dist.barrier()

        total_time_ms = elaspsed_time_ms(backend, start_event, end_event)
        avg_latency_ms = Utils.avg_lat(
            total_time_ms, options.iterations, world_size, device
        )
        if rank == 0:
            logger.info("%-10d%18.2f" % (size, avg_latency_ms))
            size_in_bytes = int(size) * get_nbytes_from_dtype(dtype)
            df = df.append(
                {"size_in_bytes": size_in_bytes, "avg_latency_ms": avg_latency_ms},
                ignore_index=True,
            )
    # Persist result to CSV file
    if rank == 0:
        df.to_csv(f"osu_scatter-{device.type}-{dtype}-{world_size}.csv", index=False)
