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


def latency(args):
    backend = args.backend
    rank = dist.get_rank()
    dtype = args.dtype
    world_size = dist.get_world_size()
    device = get_device(backend, rank)
    pg = None

    options = Options("Latency", args)
    Utils.check_numprocs(world_size, rank, limit=2)

    # Print header
    Utils.print_header(options.benchmark, rank)

    df = pd.DataFrame(columns=["size_in_bytes", "avg_latency_ms"])

    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large
        iterations = range(options.iterations + options.skip)
        s_msg = safe_rand(size, dtype=dtype).to(device)
        r_msg = safe_rand(size, dtype=dtype).to(device)

        dist.barrier()
        if rank == 0:
            for i in iterations:
                if i == options.skip:
                    start_event = get_device_event(backend)
                dist.send(s_msg, 1, pg, 1)
                dist.recv(r_msg, 1, pg, 1)
            end_event = get_device_event(backend)
            sync_device(backend)
        elif rank == 1:
            for i in iterations:
                if i == options.skip:
                    start_event = get_device_event(backend)
                dist.recv(r_msg, 0, pg, 1)
                dist.send(s_msg, 0, pg, 1)
            end_event = get_device_event(backend)
            sync_device(backend)
        dist.barrier()

        total_time_ms = elaspsed_time_ms(backend, start_event, end_event)

        # Divide by 2 since one messsage sent and one message received
        avg_latency_ms = (
            Utils.avg_lat(total_time_ms, options.iterations, world_size, device) / 2
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
        df.to_csv(f"osu_latency-{device.type}-{dtype}-{world_size}.csv", index=False)
