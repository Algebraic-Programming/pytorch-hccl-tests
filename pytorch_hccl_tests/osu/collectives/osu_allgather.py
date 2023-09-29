from time import perf_counter_ns as now
import logging
import pandas as pd

import torch.distributed as dist

from pytorch_hccl_tests.commons import get_device, safe_rand
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils

logger = logging.getLogger(__name__)


def allgather(args):
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = args.dtype
    device = get_device(backend, rank)
    pg = None

    options = Options("Allgather", args)
    Utils.check_numprocs(world_size, rank, limit=3)
    Utils.print_header(options.benchmark, rank)

    df = pd.DataFrame(columns=["size_in_bytes", "avg_latency"])

    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large
        iterations = list(range(options.iterations + options.skip))

        # safe_rand is a wrapper of torch.rand for floats and
        # torch.randint for integral types
        tensor = safe_rand(size, dtype=dtype).to(device)
        tensor_list = [safe_rand(size, dtype=dtype).to(device)] * world_size

        dist.barrier()
        for i in iterations:
            if i == options.skip:
                tic = now()
            dist.all_gather(tensor_list, tensor, pg, False)
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
        df.to_csv(f"osu_allgather-{device.type}-{dtype}-{world_size}.csv", index=False)
