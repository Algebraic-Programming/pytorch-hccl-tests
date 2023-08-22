import torch

from pytorch_hccl_tests.commons import get_device


def test_get_device():
    rank = 0
    backend = "cpu"
    device = get_device(backend, rank)
    assert device == torch.device("cpu")
