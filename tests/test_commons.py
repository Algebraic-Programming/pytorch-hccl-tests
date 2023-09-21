import torch

from pytorch_hccl_tests.commons import get_device, get_dtype, is_integral


def test_get_device():
    rank = 0
    backend = "cpu"
    device = get_device(backend, rank)
    assert device == torch.device("cpu")


def test_get_dtype_int8():
    dtype = get_dtype("int8")
    assert dtype == torch.int8


def test_get_dtype_float():
    dtype = get_dtype("float")
    assert dtype == torch.float


def test_get_dtype_float32():
    dtype = get_dtype("float32")
    assert dtype == torch.float


def test_get_dtype_double():
    dtype = get_dtype("double")
    assert dtype == torch.float64


def test_get_dtype_float64():
    dtype = get_dtype("float64")
    assert dtype == torch.float64


def test_is_integral_int():
    assert is_integral("int")


def test_is_integral_long():
    assert is_integral("long")


def test_is_integral_float():
    assert not is_integral("float")


def test_is_integral_double():
    assert not is_integral("double")
