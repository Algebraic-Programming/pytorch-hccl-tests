# pytorch-hccl-tests

HCCL benchmarks based on the PyTorch/Ascend adapter. The benchmarks contain the benchmarks proposed in the article [OMB-Py: Python Micro-Benchmarks for Evaluating Performance of MPI Libraries on HPC Systems](https://arxiv.org/pdf/2110.10659.pdf).


Currently, only the P2P benchmarks are ported and tested. Additionally, `allreduce`, `alltoall` and `broadcast` collectives are supported.

### Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_dev.txt
make install
```


### Benchmark suites

The following benchmarks are available. To view the list of available benchmarks, type `make`


```
> make
...
<development related Make commands>
...
latency              OSU MPI/HCCL latency benchmark
bandwidth            OSU MPI/HCCL bandwidth benchmark
bidirectional-bw     OSU MPI/HCCL bidirectional bandwidth benchmark
allreduce            OSU MPI/HCCL allreduce benchmark
allgather            OSU MPI/HCCL allgather benchmark
alltoall             OSU MPI/HCCL alltoall benchmark
barrier              OSU MPI/HCCL barrier benchmark
broadcast            OSU MPI/HCCL broadcast benchmark
gather               OSU MPI/HCCL Bandwidth benchmark
reduce               OSU MPI/HCCL Bandwidth benchmark
scatter              OSU MPI/HCCL Bandwidth benchmark
collectives          OSU MPI/HCCL collective communications benchmark suite
benchmarks           OSU MPI/HCCL complete benchmark suite
```


#### Example benchmark: Latency

```bash
make latency -e DEVICE=npu (default: cpu)
```

should output

```bash
> export OMP_NUM_THREADS=1
> make latency
torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/p2p/osu_latency.py --device cpu
WARNING:torch.distributed.run:
[2023-06-13 09:16:21,299] {distributed_c10d.py:228} INFO - Added key: store_based_barrier_key:1 to store for rank: 1
[2023-06-13 09:16:21,299] {distributed_c10d.py:228} INFO - Added key: store_based_barrier_key:1 to store for rank: 0
[2023-06-13 09:16:21,299] {distributed_c10d.py:263} INFO - Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 2 nodes.
[2023-06-13 09:16:21,299] {distributed_c10d.py:263} INFO - Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 2 nodes.
[2023-06-13 09:16:21,300] {commons.py:102} INFO - Python version: 3.7.10
[2023-06-13 09:16:21,300] {commons.py:103} INFO - PyTorch version: 1.11.0+cpu
[2023-06-13 09:16:21,300] {commons.py:104} INFO - PyTorch MPI enabled?: False
[2023-06-13 09:16:21,300] {commons.py:105} INFO - PyTorch CUDA enabled?: False
[2023-06-13 09:16:21,301] {commons.py:106} INFO - PyTorch NCCL enabled?: False
[2023-06-13 09:16:21,301] {commons.py:107} INFO - PyTorch Gloo enabled?: True
[2023-06-13 09:16:21,301] {commons.py:108} INFO - Using device *cpu* with *gloo* backend
[2023-06-13 09:16:21,301] {commons.py:109} INFO - World size: 2
[2023-06-13 09:16:21,301] {osu_util_mpi.py:32} INFO - # OMB Python MPI Latency Test
[2023-06-13 09:16:21,302] {osu_util_mpi.py:33} INFO - # Size (B)      Latency (us)
[2023-06-13 09:16:21,773] {osu_latency.py:64} INFO - 1024                   21.39
[2023-06-13 09:16:22,272] {osu_latency.py:64} INFO - 2048                   22.63
[2023-06-13 09:16:22,754] {osu_latency.py:64} INFO - 4096                   21.49
[2023-06-13 09:16:23,274] {osu_latency.py:64} INFO - 8192                   23.53
[2023-06-13 09:16:23,282] {osu_latency.py:64} INFO - 16384                  28.48
[2023-06-13 09:16:23,291] {osu_latency.py:64} INFO - 32768                  38.55
[2023-06-13 09:16:23,306] {osu_latency.py:64} INFO - 65536                  59.47
[2023-06-13 09:16:23,330] {osu_latency.py:64} INFO - 131072                 99.48
[2023-06-13 09:16:23,374] {osu_latency.py:64} INFO - 262144                183.47
[2023-06-13 09:16:23,457] {osu_latency.py:64} INFO - 524288                350.05
[2023-06-13 09:16:23,620] {osu_latency.py:64} INFO - 1048576               682.47
[2023-06-13 09:16:23,943] {osu_latency.py:64} INFO - 2097152              1370.95
[2023-06-13 09:16:24,606] {osu_latency.py:64} INFO - 4194304              2779.32
[2023-06-13 09:16:25,874] {osu_latency.py:64} INFO - 8388608              5424.14
[2023-06-13 09:16:28,245] {osu_latency.py:64} INFO - 16777216            10074.96
[2023-06-13 09:16:33,184] {osu_latency.py:64} INFO - 33554432            21090.99
```

The above data are also persistent into a CSV file named `osu_latency_2.csv`.

### Known issues

* Gloo backend does not support `reduce_scatter`.
