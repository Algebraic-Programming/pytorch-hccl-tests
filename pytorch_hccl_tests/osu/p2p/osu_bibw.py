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
    wait_all,
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

        window_sizes = list(range(window_size))

        s_msg = safe_rand(size, dtype=dtype).to(device)
        r_msg = safe_rand(size, dtype=dtype).to(device)

        send_requests = [None] * window_size
        recv_requests = [None] * window_size

        dist.barrier()
        if rank == 0:
            for i in range(options.iterations + options.skip):
                if i == options.skip:
                    start_event = get_device_event(backend)
                for j in window_sizes:
                    recv_requests[j] = dist.irecv(r_msg, 1, pg, 10)
                for j in window_sizes:
                    send_requests[j] = dist.isend(s_msg, 1, pg, 100)

                wait_all(send_requests)
                wait_all(recv_requests)
            end_event = get_device_event(backend)
            sync_device(backend)
        elif rank == 1:
            for i in range(options.iterations + options.skip):
                for j in window_sizes:
                    recv_requests[j] = dist.irecv(r_msg, 0, pg, 100)
                for j in window_sizes:
                    send_requests[j] = dist.isend(s_msg, 0, pg, 10)
                wait_all(recv_requests)
                wait_all(send_requests)

        if rank == 0:
            size_in_bytes = int(size) * get_nbytes_from_dtype(dtype)

            # Number of total_iterations
            total_iterations = options.iterations * window_size

            total_time_ms = elaspsed_time_ms(backend, start_event, end_event)
            total_time_sec_per_iter = total_time_ms / (1000 * total_iterations)

            bw = size_in_bytes / total_time_ms
            logger.info("%-10d%18.2f" % (size_in_bytes, bw))
            df = df.append(
                {"size_in_bytes": int(size), "bw_mb_per_sec": total_time_sec_per_iter},
                ignore_index=True,
            )

    if rank == 0:
        df.to_csv(f"osu_bibw-{device.type}-{dtype}-{world_size}.csv", index=False)
