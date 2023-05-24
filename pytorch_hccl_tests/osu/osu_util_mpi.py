"""
Copyright (C) 2002-2022 the Network-Based Computing Laboratory
(NBCL), The Ohio State University.

Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)

For detailed copyright and licensing information, please refer to the
copyright file COPYRIGHT in the top level OMB directory.
"""


import math
import time

import torch
import torch.distributed as dist


class Utils:
    def print_stats(t_end, t_start, iterations, rank, numprocs, size):
        avglatency = Utils.avg_lat(t_end, t_start, iterations, numprocs)
        if rank == 0:
            print("%-10d%18.2f" % (size, avglatency), flush=True)

    def avg_lat(t_end, t_start, iterations, num_procs):
        avg_latency = torch.tensor(
            (t_end - t_start) * 1e6 / float(iterations), dtype=torch.float64
        )
        dist.reduce(avg_latency, 0, op=dist.ReduceOp.SUM)
        avg_latency /= float(num_procs)
        return avg_latency

    def print_header(benchmark, rank: int):
        if rank == 0:
            print("# OMB Python MPI %s Test" % (benchmark))
            print("# %-8s%18s" % ("Size (B)", "Latency (us)"))

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

    def dummy_compute(seconds, request, mode):
        test_time = 0
        test_time = Utils.do_compute_and_probe(seconds, request, mode)
        return test_time

    def do_compute_and_probe(seconds, request, mode="cpu"):
        test_time = 0
        if mode == "cpu":
            Utils.do_compute_cpu(seconds)
        elif mode == "gpu":
            Utils.do_compute_gpu(seconds)
        return test_time

    def do_compute_cpu(seconds):
        t1 = 0
        t2 = 0
        time_elapsed = 0
        t1 = time.monotonic_ns()
        while time_elapsed < seconds:
            Utils.compute_on_host()
            t2 = time.monotonic_ns()
            time_elapsed = t2 - t1
