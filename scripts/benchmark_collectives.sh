#!/bin/bash

# Benchmark a communication pattern for various world sizes and dtypes
# Generates multiple CSV/PNG files



export DEVICE="cpu"
DTYPES="int float16 float32"
DTYPES="float16"

BASE_DIR=$(dirname "$0")


COLLECTIVES="broadcast allreduce allgather reducescatter alltoall"
for BENCH in ${COLLECTIVES}
do
    for DTYPE in ${DTYPES}
    do
        export BENCH=${BENCH}
        export HCCL_DTYPE=${DTYPE}
        for SIZE in {2..8}
        do
            echo "${BENCH} (size: ${SIZE} | dtype: ${DTYPE})"
            make "${BENCH} -e WORLD_SIZE=${SIZE} HCCL_DTYPE=${HCCL_DTYPE}"
        done

        python "${BASE_DIR}/plotter_collectives.py"
    done
done
