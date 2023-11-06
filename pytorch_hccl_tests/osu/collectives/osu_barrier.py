import logging
from time import perf_counter_ns as now

import torch.distributed as dist

from pytorch_hccl_tests.commons import get_device
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils

logger = logging.getLogger(__name__)


def barrier(args):
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = get_device(backend, args.local_rank)

    options = Options("Barrier", args)
    Utils.check_numprocs(world_size, rank, limit=3)
    Utils.print_header(options.benchmark, rank)

    iterations = list(range(options.iterations + options.skip))

    dist.barrier()
    for i in iterations:
        if i == options.skip:
            tic = now()
        dist.barrier()
    toc = now()
    dist.barrier()

    avg_lat = Utils.avg_lat(toc - tic, options.iterations, world_size, device)
    if rank == 0:
        print("%-10d%18.2f" % (0, avg_lat))
