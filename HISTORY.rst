=======
History
=======

0.1.15 (2023-XX-YY)
------------------
* TBD

0.1.14 (2023-11-12)
------------------
* Normalize dataframe column name to `avg_latency_ms`
* Fix size_in_bytes to take into account dtype

0.1.13 (2023-11-06)
------------------
* ModelArts multi-node entrypoint script
* Fix local/global rank ambiguity

0.1.12 (2023-10-13)
------------------
* Add device event timing in P2P benchmarks (bug reported by Shanlan li)

0.1.11 (2023-10-08)
------------------
* Fix `alltoall` device issue

0.1.10 (2023-10-07)
------------------
* Fix elapsed time measurement for NPU/CUDA
* Improve plotting scripts

0.1.9 (2023-10-02)
------------------
* Fix alltoall `torch.arange` issue
* Improve further plotting scripts by directly using `torchrun`

0.1.8 (2023-09-29)
------------------
* Measure time in nanoseconds
* Fix bandwidth measurements (MR: 28)

0.1.7 (2023-09-28)
------------------
* Improve scripts functionality
* Introduce plotter scripts for collectives,latency and bandwidth

0.1.6 (2023-09-26)
------------------
* Fix CSV output filename
* Minor plotting improvements

0.1.5 (2023-09-21)
------------------
* Fix bumpversion quotes
* Fixed input tensor size bug on `reduce_scatter` benchmark
* Cover all torch dtypes (int8/int16/uint8/etc.)

0.1.2 (2023-09-20)
------------------
* Fix critical bug on setting up the NPU/HCCL environment
* Introduce `dtype` as an argument parameter
* Port `all_gather` and `reduce_scatter` collectives
* Added basic plotting utility under `scripts/`

0.1.1 (2023-06-15)
------------------
* First release containing P2P OMB-Py benchmarks
* Support allreduce and broadcast collectives
* Add `torch_npu` environment check
