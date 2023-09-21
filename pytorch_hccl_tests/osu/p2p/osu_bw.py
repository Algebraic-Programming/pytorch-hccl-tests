import logging
from time import perf_counter as now

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


def bw(args):
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = args.dtype
    device = get_device(backend, rank)
    pg = None

    options = Options("Bandwidth", args)
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

        iterations = range(options.iterations + options.skip)
        window_sizes = range(window_size)
        requests = [None] * window_size

        dist.barrier()
        if rank == 0:
            # safe_rand is a wrapper of torch.rand for floats and
            # torch.randint for integral types
            s_msg = [
                safe_rand(size, dtype=dtype).to(device) for _ in range(window_size)
            ]
            r_msg = safe_rand(4, dtype=dtype).to(device)
            for i in iterations:
                if i == options.skip:
                    tic = now()
                for j in window_sizes:
                    requests[j] = dist.isend(s_msg[j], 1, pg, 100)
                wait_all(requests)
                dist.recv(r_msg, 1, pg, 101)
            toc = now()
        elif rank == 1:
            s_msg = safe_rand(4, dtype=dtype).to(device)
            r_msg = [
                safe_rand(size, dtype=dtype).to(device) for _ in range(window_size)
            ]
            for i in iterations:
                for j in window_sizes:
                    requests[j] = dist.irecv(r_msg[j], 0, pg, 100)
                wait_all(requests)
                dist.send(s_msg, 0, pg, 101)

        if rank == 0:
            bw = float(size / 1e6 * options.iterations * window_size)
            time_elapsed = float(toc - tic)
            logger.info("%-10d%18.2f" % (size, bw / time_elapsed))
            df = df.append(
                {"size_in_bytes": int(size), "bw_mb_per_sec": bw}, ignore_index=True
            )

    # Persist result to CSV file
    if rank == 0:
        df.to_csv(f"osu_bandwidth-{device.type}-{dtype}-{world_size}.csv", index=False)
