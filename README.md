# pytorch-hccl-tests

HCCL tests based on the PyTorch/Ascend adapter.


```bash
python -m venv venv
source venv/bin/activate
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
