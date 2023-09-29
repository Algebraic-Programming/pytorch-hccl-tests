import logging
from time import perf_counter_ns as now

import pandas as pd
import torch.distributed as dist

from pytorch_hccl_tests.commons import (
    get_device,
    wait_all,
    safe_rand,
)
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils

logger = logging.getLogger(__name__)


def bibw(args):
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = args.dtype
    device = get_device(backend, rank)
    pg = None

    options = Options("Bi-Directional Bandwidth", args)
    Utils.check_numprocs(world_size, rank, limit=2)

    if rank == 0:
        logger.info("# OMB-Py MPI %s Test" % (options.benchmark))
        logger.info("# %-8s%18s" % ("Size (B)", "Bandwidth (MB/s)"))

    df = pd.DataFrame(columns=["size_in_bytes", "bw_mb_per_sec"])

    window_size = 64
    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large

        iterations = list(range(options.iterations + options.skip))
        window_sizes = list(range(window_size))

        s_msg = safe_rand(size, dtype=dtype).to(device)
        r_msg = safe_rand(size, dtype=dtype).to(device)

        send_requests = [None] * window_size
        recv_requests = [None] * window_size

        dist.barrier()
        if rank == 0:
            for i in iterations:
                if i == options.skip:
                    tic = now()
                for j in window_sizes:
                    recv_requests[j] = dist.irecv(r_msg, 1, pg, 10)
                for j in window_sizes:
                    send_requests[j] = dist.isend(s_msg, 1, pg, 100)

                wait_all(send_requests)
                wait_all(recv_requests)
            toc = now()
        elif rank == 1:
            for i in iterations:
                for j in window_sizes:
                    recv_requests[j] = dist.irecv(r_msg, 0, pg, 100)
                for j in window_sizes:
                    send_requests[j] = dist.isend(s_msg, 0, pg, 10)
                wait_all(recv_requests)
                wait_all(send_requests)

        if rank == 0:
            size_in_bytes = float(
                size / ((options.iterations - options.skip) * window_size)
            )
            time_elapsed_ns = float(toc - tic)
            # 'norm_size' unit is bytes and 'time_elapsed_ns' in nanoseconds, so scaling is 10^9/1024
            scaling = 2 * (1e9 / 1024.0)
            bw = scaling * size_in_bytes / time_elapsed_ns
            logger.info("%-10d%18.2f" % (size, bw))
            df = df.append(
                {"size_in_bytes": int(size), "bw_mb_per_sec": bw}, ignore_index=True
            )

    if rank == 0:
        df.to_csv(f"osu_bibw-{device.type}-{dtype}-{world_size}.csv", index=False)
