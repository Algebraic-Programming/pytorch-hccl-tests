=======
History
=======

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
