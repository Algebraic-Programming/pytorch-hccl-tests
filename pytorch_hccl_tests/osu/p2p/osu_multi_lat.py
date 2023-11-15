import logging

import pandas as pd
import torch.distributed as dist

from pytorch_hccl_tests.commons import (
    elaspsed_time_ms,
    get_device,
    get_device_event,
    get_nbytes_from_dtype,
    safe_rand,
)
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils

logger = logging.getLogger(__name__)


def multi_lat(args):
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    pairs = int(world_size / 2)
    dtype = args.dtype
    device = get_device(backend, rank)
    pg = None

    options = Options("Multi Latency", args)

    Utils.print_header(options.benchmark, rank)

    df = pd.DataFrame(columns=["size_in_bytes", "avg_latency"])

    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large

        iterations = list(range(options.iterations + options.skip))
        # safe_rand is a wrapper of torch.rand for floats and
        # torch.randint for integral types
        s_msg = safe_rand(size, dtype=dtype).to(device)
        r_msg = safe_rand(size, dtype=dtype).to(device)

        dist.barrier()
        if rank < pairs:
            partner = rank + pairs
            for i in iterations:
                if i == options.skip:
                    start_event = get_device_event(backend)
                dist.send(s_msg, partner, pg, 1)
                dist.recv(r_msg, partner, pg, 1)
            end_event = get_device_event(backend)
        else:
            partner = rank - pairs
            for i in iterations:
                if i == options.skip:
                    start_event = get_device_event(backend)
                dist.recv(r_msg, partner, pg, 1)
                dist.send(s_msg, partner, pg, 1)
            end_event = get_device_event(backend)

        total_time_ms = elaspsed_time_ms(backend, start_event, end_event)

        avg_latency_ms = (
            Utils.avg_lat(total_time_ms, options.iterations, world_size, device) / 2
        )

        if rank == 0:
            size_in_bytes = int(size) * get_nbytes_from_dtype(dtype)
            logger.info("%-10d%18.2f" % (size_in_bytes, avg_latency_ms))
            df = df.append(
                {"size_in_bytes": int(size), "avg_latency_ms": avg_latency_ms},
                ignore_index=True,
            )

    # Persist result to CSV file
    if rank == 0:
        df.to_csv(
            f"osu_multi_latency-{device.type}-{dtype}-{world_size}.csv", index=False
        )
