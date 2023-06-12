import logging
import os
import sys
from time import perf_counter as now

import pandas as pd
import torch
import torch.distributed as dist

from pytorch_hccl_tests.commons import (
    dist_init,
    get_device,
    log_env_info,
    setup_loggers,
    wait_all,
)
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils
from pytorch_hccl_tests.osu.parser import get_parser

logger = logging.getLogger(__name__)


def osu_bibw(args):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.float32
    device = get_device(rank)
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

        s_msg = torch.rand(size, dtype=dtype).to(device)
        r_msg = torch.rand(size, dtype=dtype).to(device)

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
                    wait_all(send_requests)
                    wait_all(recv_requests)

        if rank == 0:
            bw = size / 1e6 * options.iterations * window_size * 2
            time_elapsed = tic - toc
            logger.info("%-10d%18.2f" % (size, bw / time_elapsed))
            df = df.append(
                {"size_in_bytes": int(size), "bw_mb_per_sec": bw}, ignore_index=True
            )

    df.to_csv("osu_bibw-{world_size}.csv", index=False)


def main():
    args = get_parser().parse_args()
    device = args.device

    log_handlers = setup_loggers(__name__)
    log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=log_handlers,
    )

    # rank and world_size is set by torchrun
    rank = int(os.environ["LOCAL_RANK"])

    # Initialize torch.distributed
    backend = dist_init(device, rank)
    if rank == 0:
        log_env_info(device, backend)

    osu_bibw(args=args)

    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())  # noqa
