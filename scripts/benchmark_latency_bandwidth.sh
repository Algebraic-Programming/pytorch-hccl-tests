#!/bin/bash

# Benchmark a communication pattern for various dtypes.
# Generates multiple CSV/PNG files


DEVICE="npu"
DTYPES="int float16 float32"
DTYPES="float16"

# To surpress a torchrun warning
export OMP_NUM_THREADS=1

BASE_DIR=$(dirname "$0")


# Generate results for latency/bandwidth
P2P_BENCHS="latency bandwidth"
for BENCH in ${P2P_BENCHS}
do
    for DTYPE in ${DTYPES}
    do
        export HCCL_DTYPE=${DTYPE}
        echo "${BENCH} (size: ${SIZE} | dtype: ${DTYPE})"
        torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/cli.py \
                    --benchmark "${BENCH}" \
                    --device ${DEVICE} \
                    --dtype "${HCCL_DTYPE}"

        python "${BASE_DIR}/plotter_${BENCH}.py"
    done
done

