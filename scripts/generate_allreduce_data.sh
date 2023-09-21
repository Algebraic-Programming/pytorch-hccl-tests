#!/bin/bash

# Example to generate allreduce data

DTYPES="int float16 float32"
for SIZE in {2..8}
do
    for DTYPE in ${DTYPES}
    do
        echo "Allreduce (size: ${SIZE} | dtype: ${DTYPE})"
        make allreduce -e WORLD_SIZE=${SIZE} HCCL_DTYPE=${DTYPE}
    done
done