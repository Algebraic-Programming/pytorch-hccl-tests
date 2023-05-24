# pytorch-hccl-tests

HCCL tests based on the PyTorch/Ascend adapter.


```bash
pip install -r requirements_dev.txt
make install
```


### All-reduce benchmark (CPU)

```bash
export OMP_NUM_THREADS=1

# Integer tensors
torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/allreduce_int.py --device cpu

# Float tensors
torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/allreduce_float.py --device cpu
```

or

```bash
make torchrun
```


## Port of OMB-Py MPI benchmarks to Pytorch distributed

The `osu` module contains a list OSU MPI benchmarks that were ported to mpi4py in the article [OMB-Py: Python Micro-Benchmarks for Evaluating
Performance of MPI Libraries on HPC Systems](https://arxiv.org/pdf/2110.10659.pdf).

Currently, only the P2P benchmarks are ported.

#### Latency

```bash
torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/p2p/osu_latency.py --iterations 1000
```

should output

```bash
torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/p2p/osu_latency.py --iterations 1000
using device cpu with gloo backend
world size is 2
# OMB Python MPI Latency Test
# Size (B)      Latency (us)
1024                   21.92
2048                   21.01
4096                   22.18
8192                   24.91
16384                  32.30
32768                  33.46
65536                  50.41
131072                 79.20
262144                142.70
524288                305.87
1048576               840.64
2097152              1438.03
4194304              2668.22
8388608              5067.97
16777216            10204.80
33554432            20518.96
```


#### Bandwidth

```bash
torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/p2p/osu_bw.py --iterations 1000
```
should output

```bash
torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/p2p/osu_bw.py --iterations 1000
using device cpu with gloo backend
world size is 2
# OMB-Py MPI Bandwidth Test
# Size (B)  Bandwidth (MB/s)
1024                   82.20
2048                  162.13
4096                  287.45
8192                  515.59
16384                 689.20
32768                 997.42
65536                1212.27
131072               1349.57
262144               1460.63
524288               1542.24
1048576              1604.63
2097152              1633.82
4194304              1637.85
8388608              1723.22
16777216             1628.28
```


#### Multi-latency


```bash
torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/p2p/osu_multi_lat.py          
```

should output

```bash
torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/osu/p2p/osu_multi_lat.py          
using device cpu with gloo backend
world size is 2
# OMB Python MPI Multi Latency Test
# Size (B)      Latency (us)
1024                   20.62
2048                   21.44
4096                   22.97
8192                   24.88
16384                  31.59
32768                  36.09
65536                  51.34
131072                 83.94
262144                160.86
524288                315.55
1048576               663.96
2097152              1304.58
4194304              2692.07
8388608              5379.78
16777216            10915.15
33554432            20868.44
```

#### Bidirectional bandwidth

This benchmark currently deadlocks on CPU/Gloo backend due a known Gloo bug, see https://github.com/pytorch/pytorch/issues/30723
