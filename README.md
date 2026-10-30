# FeatherAttn

Currently this package provides a fast attn bwd kernel on RDNA3/3.5 . I've tested it on Strix Halo. Compared to AITER FlashAttn Triton bwd kernel (with tuned configs), it can be 4x as fast for head dim 128, and 1.3x as fast for head dim 64.

This package also provides a trainable attn function, which is a drop-in replacement of `flash_attn_func`. It uses AITER fwd kernel and our bwd kernel. When there is any feature not supported by our bwd kernel, it fallbacks to AITER bwd kernel. It supports torch.compile . You can wire it into your training script.

The name 'feather' comes from the idea of fast fp8 matmul on RDNA3/3.5 without native fp8, see https://github.com/woct0rdho/feather-ops . But currently the FeatherAttn bwd kernel does not use fp8 quantization. The use of fp8 quantization in attn fwd and bwd kernels is still an ongoing work.
