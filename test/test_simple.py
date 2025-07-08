from utils import run_in_subprocess


def _test_simple_inner():
    import torch
    a = torch.tensor([42], device="cuda")
    b = a * 2
    print(f"{a=} {b=}")
    assert b.item() == 84


def test_simple():
    run_in_subprocess(_test_simple_inner)
