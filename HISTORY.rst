=======
History
=======

0.1.8 (2023-XX-YY)
------------------
* Measure time in nanoseconds


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
