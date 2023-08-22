import torch

from pytorch_hccl_tests.commons import get_device


def test_get_device():
    rank = 0
<<<<<<< HEAD
    backend = "cpu"
    device = get_device(backend, rank)
=======
    device = get_device(rank)
>>>>>>> github/master
    assert device == torch.device("cpu")
