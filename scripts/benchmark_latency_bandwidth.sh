#!/bin/bash

# Benchmark a communication pattern for various world sizes and dtypes
# Generates multiple CSV files



export DEVICE="cpu"
DTYPES="int float16 float32"
DTYPES="float16"

BASE_DIR=$(dirname "$0")


# Generate results for latency/bandwidth
export P2P_BENCHS="latency bandwidth"
for BENCH in ${P2P_BENCHS}
do
    for DTYPE in ${DTYPES}
    do
        export HCCL_DTYPE=${DTYPE}
        echo "${BENCH} (size: ${SIZE} | dtype: ${DTYPE})"
        make "${BENCH} -e DEVICE=${DEVICE} HCCL_DTYPE=${HCCL_DTYPE}"

        python "${BASE_DIR}/plotter_${BENCH}.py"
    done
done

