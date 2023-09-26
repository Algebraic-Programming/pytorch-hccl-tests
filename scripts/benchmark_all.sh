#!/bin/bash

# Benchmark a communication pattern for various world sizes and dtypes
# Generates multiple CSV files

export BENCH="allgather"
export DEVICE="cpu"
DTYPES="int float16 float32"


for SIZE in {2..8}
do
    for DTYPE in ${DTYPES}
    do
        echo "${BENCHMARK} (size: ${SIZE} | dtype: ${DTYPE})"
        make ${BENCH} -e WORLD_SIZE=${SIZE} HCCL_DTYPE=${DTYPE}
    done
done