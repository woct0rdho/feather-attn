import functools

import triton
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import bwd, fwd_prefill

# (BLOCK_M, BLOCK_N, num_warps, num_stages, waves_per_eu, PRE_LOAD_V)
_FWD_TILES: dict[int, tuple] = {
    64: (128, 32, 4, 1, 1, False),
    128: (128, 32, 8, 1, 6, False),
    256: (128, 32, 8, 1, 2, False),
}
_FWD_FALLBACK_TILE = (128, 64, 8, 1, 1, False)

# (BLOCK_M1, BLOCK_N1, BLOCK_N2, BLK_SLICE_FACTOR, num_warps, num_stages, waves_per_eu)
# BLOCK_M2 equals BLOCK_N1.
_BWD_TILES: dict[int, tuple] = {
    64: (16, 64, 16, 2, 4, 1, 1),
    128: (16, 16, 16, 2, 4, 1, 1),
    256: (16, 16, 16, 2, 4, 1, 1),
}
_BWD_LONG_TILE = (16, 64, 16, 2, 4, 2, 1)
_BWD_LONG_PRESSURE = 200_000
_BWD_CAUSAL_TILES: dict[int, tuple] = {
    64: (16, 64, 16, 2, 4, 1, 1),
    128: (16, 64, 16, 2, 4, 2, 1),
    256: (16, 64, 16, 2, 4, 2, 1),
}
_BWD_FALLBACK_TILE = (32, 32, 32, 2, 4, 1, 1)


def _fwd_tile_to_config(tile: tuple) -> triton.Config:
    block_m, block_n, num_warps, num_stages, waves_per_eu, pre_load_v = tile
    return triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "waves_per_eu": waves_per_eu,
            "PRE_LOAD_V": pre_load_v,
        },
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _fwd_configs() -> list[triton.Config]:
    tiles = [_FWD_TILES[128]]
    tiles += [tile for dim, tile in sorted(_FWD_TILES.items()) if dim != 128]
    tiles.append(_FWD_FALLBACK_TILE)
    return [_fwd_tile_to_config(tile) for tile in tiles]


def _fwd_early_config_prune(configs, named_args, **kwargs):
    tile = _FWD_TILES.get(kwargs.get("BLOCK_DMODEL_QK"), _FWD_FALLBACK_TILE)
    wanted = _fwd_tile_to_config(tile)
    for config in configs:
        if config.kwargs == wanted.kwargs and config.num_warps == wanted.num_warps and config.num_stages == wanted.num_stages:
            return [config]
    return configs


def _bwd_tile_to_config(tile: tuple) -> triton.Config:
    block_m1, block_n1, block_n2, blk_slice_factor, num_warps, num_stages, waves_per_eu = tile
    return triton.Config(
        {
            "BLOCK_M1": block_m1,
            "BLOCK_N1": block_n1,
            "BLOCK_M2": block_n1,
            "BLOCK_N2": block_n2,
            "BLK_SLICE_FACTOR": blk_slice_factor,
            "waves_per_eu": waves_per_eu,
        },
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _bwd_configs(causal: bool) -> list[triton.Config]:
    table = _BWD_CAUSAL_TILES if causal else _BWD_TILES
    tiles = [table[128]]
    tiles += [tile for dim, tile in sorted(table.items()) if dim != 128]
    if not causal:
        tiles.append(_BWD_LONG_TILE)
    tiles.append(_BWD_FALLBACK_TILE)
    unique_tiles = list(dict.fromkeys(tiles))
    return [_bwd_tile_to_config(tile) for tile in unique_tiles]


def _bwd_early_config_prune(configs, named_args, causal: bool, **kwargs):
    table = _BWD_CAUSAL_TILES if causal else _BWD_TILES
    dim = kwargs.get("HEAD_DIM_V", kwargs.get("ACTUAL_HEAD_DIM_V"))
    tile = table.get(dim, _BWD_FALLBACK_TILE)
    if not causal:
        named = named_args or {}
        seqlen_k = named.get("max_seqlen_k")
        if seqlen_k is not None:
            heads_q = named.get("HQ", kwargs.get("HQ"))
            pressure = heads_q * seqlen_k if heads_q is not None else (seqlen_k * 16 if seqlen_k >= 12288 else 0)
            if pressure >= _BWD_LONG_PRESSURE:
                tile = _BWD_LONG_TILE
    wanted = _bwd_tile_to_config(tile)
    for config in configs:
        if config.kwargs == wanted.kwargs and config.num_warps == wanted.num_warps and config.num_stages == wanted.num_stages:
            return [config]
    return configs


def _set_autotuner(autotuner, configs: list[triton.Config], early_config_prune) -> None:
    autotuner.configs = configs
    autotuner.early_config_prune = early_config_prune
    autotuner.cache.clear()


def _apply_aiter_tuned_configs() -> None:
    _set_autotuner(
        fwd_prefill.attn_fwd,
        _fwd_configs(),
        _fwd_early_config_prune,
    )
    _set_autotuner(
        bwd.bwd_kernel_fused_causal,
        _bwd_configs(causal=True),
        functools.partial(_bwd_early_config_prune, causal=True),
    )
    _set_autotuner(
        bwd.bwd_kernel_fused_noncausal,
        _bwd_configs(causal=False),
        functools.partial(_bwd_early_config_prune, causal=False),
    )


def install_aiter_tuned_configs() -> None:
    if fwd_prefill.attn_fwd.early_config_prune is _fwd_early_config_prune:
        return
    _apply_aiter_tuned_configs()
