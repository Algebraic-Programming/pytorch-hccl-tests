"""
Copyright (C) 2002-2022 the Network-Based Computing Laboratory
(NBCL), The Ohio State University.

Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)

For detailed copyright and licensing information, please refer to the
copyright file COPYRIGHT in the top level OMB directory.
"""


import logging
import math

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class Utils:
    def avg_lat(elapsed_time_ms: int, iterations, num_procs, device: torch.device):
        avg_latency = torch.tensor(
            elapsed_time_ms / float(iterations), dtype=torch.float64
        ).to(device)
        dist.reduce(avg_latency, 0, op=dist.ReduceOp.SUM)
        avg_latency /= float(num_procs)
        return avg_latency.item()

    def print_header(benchmark, rank: int):
        if rank == 0:
            logger.info("# PyTorch Benchmark %s Test" % (benchmark))
            logger.info("# %-8s%18s" % ("Size (B)", "Elapsed Time (ms)"))

    def check_numprocs(numprocs: int, rank: int, limit: int):
        if limit == 2:
            if numprocs != 2:
                if rank == 0:
                    errmsg = "This test requires exactly two processes"
                else:
                    errmsg = None
                raise SystemExit(errmsg)
        else:
            if numprocs < 2:
                if rank == 0:
                    errmsg = "This test requires at least two processes"
                else:
                    errmsg = None
                raise SystemExit(errmsg)

    def nbc_print_stats(
        rank,
        size,
        numprocs,
        loop,
        timer,
        latency,
        test_time,
        tcomp_time,
        wait_time,
        init_time,
    ):
        (
            overall_avg,
            tcomp_avg,
            test_avg,
            avg_comm_time,
            wait_avg,
            init_avg,
        ) = Utils.nbc_calc_stats(
            size,
            numprocs,
            loop,
            timer,
            latency,
            test_time,
            tcomp_time,
            wait_time,
            init_time,
        )
        if rank == 0:
            overlap = max(
                0,
                100 - (((overall_avg - (tcomp_avg - test_avg)) / avg_comm_time) * 100),
            )
            print(
                "%-10d%18.2f%18.2f%18.2f%18.2f%18.2f%18.2f"
                % (
                    size,
                    overall_avg,
                    (tcomp_avg - test_avg),
                    avg_comm_time,
                    overlap,
                    wait_avg,
                    init_avg,
                ),
                flush=True,
            )

    def nbc_print_header(rank):
        if rank == 0:
            print(
                "# Size           Overall(us)       Compute(us)    Pure Comm.(us)        Overlap(%)      Wait avg(us)      Init avg(us)",
                flush=True,
            )

    def nbc_calc_stats(
        size,
        numprocs,
        loop,
        timer,
        latency,
        test_time,
        tcomp_time,
        wait_time,
        init_time,
    ):
        test_total_s = torch.tensor((test_time * 1e6) / loop)
        tcomp_total_s = torch.tensor((tcomp_time * 1e6) / loop)
        overall_time_s = torch.tensor((timer * 1e6) / loop)
        wait_total_s = torch.tensor((wait_time * 1e6) / loop)
        init_total_s = torch.tensor((init_time * 1e6) / loop)
        avg_comm_time_s = torch.tensor(latency)
        min_comm_time_s = torch.tensor(latency)
        max_comm_time_s = torch.tensor(latency)

        test_total = torch.tensor(0.0)
        tcomp_total = torch.tensor(0.0)
        overall_time = torch.tensor(0.0)
        wait_total = torch.tensor(0.0)
        init_total = torch.tensor(0.0)
        avg_comm_time = torch.tensor(0.0)
        min_comm_time = torch.tensor(0.0)
        max_comm_time = torch.tensor(0.0)
        dist.reduce(test_total_s, test_total, op=dist.ReduceOp.SUM)
        dist.reduce(avg_comm_time_s, avg_comm_time, op=dist.ReduceOp.SUM)
        dist.reduce(overall_time_s, overall_time, op=dist.ReduceOp.SUM)
        dist.reduce(tcomp_total_s, tcomp_total, op=dist.ReduceOp.SUM)
        dist.reduce(wait_total_s, wait_total, op=dist.ReduceOp.SUM)
        dist.reduce(init_total_s, init_total, op=dist.ReduceOp.SUM)
        dist.reduce(max_comm_time_s, max_comm_time, op=dist.ReduceOp.SUM)
        dist.reduce(min_comm_time_s, min_comm_time, op=dist.ReduceOp.SUM)
        dist.Barrier()

        overall_time = overall_time / numprocs
        tcomp_total = tcomp_total / numprocs
        test_total = test_total / numprocs
        avg_comm_time = avg_comm_time / numprocs
        wait_total = wait_total / numprocs
        init_total = init_total / numprocs

        return (
            overall_time,
            tcomp_total,
            test_total,
            avg_comm_time,
            wait_total,
            init_total,
        )

    def message_sizes(options):
        max_size = int(math.log(options.max_message_size, 2)) + 1
        if options.min_message_size > 0:
            min_size = int(math.log(options.min_message_size, 2))
        else:
            min_size = 0
        message_sizes = [(1 << i) for i in range(min_size, max_size)]
        if options.min_message_size == 0:
            message_sizes = [0] + message_sizes
        return message_sizes
