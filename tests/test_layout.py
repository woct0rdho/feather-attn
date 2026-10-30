import pytest
import torch

from feather_attn.layout import to_hnd

DEV = "cuda"
CASES = [(1, 1, 16, 128), (1, 7, 16, 128), (2, 1024, 16, 128), (3, 33, 8, 64), (2, 17, 12, 128), (1, 4096, 16, 128)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", CASES)
def test_to_hnd_matches_transpose_copy(shape, dtype):
    x = torch.randn(*shape, device=DEV, dtype=dtype)
    got = to_hnd(x)
    reference = x.transpose(1, 2).contiguous()
    assert got.shape == reference.shape
    assert got.is_contiguous()
    assert torch.equal(got, reference)


def test_to_hnd_reuses_hnd_input():
    x = torch.randn(1, 16, 4096, 128, device=DEV, dtype=torch.bfloat16)
    bshd = x.transpose(1, 2)
    got = to_hnd(bshd)
    assert got.is_contiguous()
    assert got.data_ptr() == x.data_ptr()


def test_to_hnd_falls_back_for_unexpected_strides():
    x = torch.randn(2, 64, 512, 16, device=DEV, dtype=torch.bfloat16).transpose(1, 2)
    got = to_hnd(x)
    assert got.is_contiguous()
    assert torch.equal(got, x.transpose(1, 2).contiguous())
