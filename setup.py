import os
import shlex
import shutil
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils import cpp_extension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
SOURCES = [
    "csrc/extension.cpp",
    "csrc/hip_kernel.cpp",
    "csrc/featherattn_bwd_fused_d64.cu",
    "csrc/featherattn_bwd_fused_d128.cu",
    "csrc/featherattn_fwd_aligned.cu",
    "csrc/featherattn_fwd_query_tail.cu",
    "csrc/featherattn_fwd_key_tail.cu",
    "csrc/featherattn_fwd_query_key_tail.cu",
    "csrc/featherattn_fwd_strided.cu",
]
HEADER_DEPENDENCIES = [path.relative_to(ROOT).as_posix() for pattern in ("*.h", "*.cuh") for path in sorted(CSRC.rglob(pattern))]


def _enable_ccache() -> None:
    ccache = shutil.which("ccache")
    if ccache is None:
        return

    rocm_home = cpp_extension.ROCM_HOME
    hipcc = shutil.which("hipcc")
    if hipcc is None and rocm_home is not None:
        candidate = Path(rocm_home) / "bin" / "hipcc"
        if candidate.is_file():
            hipcc = str(candidate)
    if hipcc is not None:
        os.environ.setdefault(
            "PYTORCH_NVCC",
            f"{shlex.quote(ccache)} {shlex.quote(hipcc)}",
        )

    if "CXX" not in os.environ:
        for candidate in (Path("/usr/lib/ccache/c++"), Path("/usr/lib64/ccache/c++")):
            if candidate.is_file():
                os.environ["CXX"] = str(candidate)
                break


def _ck_include_dir() -> Path:
    root = Path(
        os.environ.get(
            "FEATHER_ATTN_CK_ROOT",
            "~/rocm-libraries/projects/composablekernel",
        )
    ).expanduser()
    include = root / "include"
    required = include / "ck_tile" / "core.hpp"
    if not required.is_file():
        raise RuntimeError(f"Cannot find {required}. Set FEATHER_ATTN_CK_ROOT to the Composable Kernel source tree.")
    return include


def _rocm_sdk_core_include() -> Path | None:
    try:
        import _rocm_sdk_core
    except ImportError:
        return None
    assert _rocm_sdk_core.__file__ is not None
    include = Path(_rocm_sdk_core.__file__).resolve().parent / "include"
    return include if include.is_dir() else None


if not cpp_extension.IS_HIP_EXTENSION:
    raise RuntimeError("Feather Attention requires a ROCm-enabled PyTorch build.")

_enable_ccache()
rocm_arch = os.environ.get("FEATHER_ATTN_ROCM_ARCH")
if rocm_arch:
    os.environ["PYTORCH_ROCM_ARCH"] = rocm_arch
else:
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1151")

include_dirs = [str(CSRC), str(_ck_include_dir())]
rocm_sdk_core_include = _rocm_sdk_core_include()
if rocm_sdk_core_include is not None:
    include_dirs.append(str(rocm_sdk_core_include))

stable_defines = [
    "-DTORCH_TARGET_VERSION=0x020A000000000000",
    "-DTORCH_STABLE_ONLY",
]
common_flags = [
    "-O3",
    "--std=c++20",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-Wno-shift-count-overflow",
    "-Wno-unused-function",
    "-Wno-unused-parameter",
    *stable_defines,
]
hip_flags = [
    *common_flags,
    "-fno-gpu-rdc",
    "-U__HIP_NO_HALF_CONVERSIONS__",
    "-U__HIP_NO_HALF_OPERATORS__",
    "-U__HIP_NO_HALF2_OPERATORS__",
    "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
    "-U__HIP_NO_BFLOAT16_OPERATORS__",
    "-U__HIP_NO_BFLOAT162_OPERATORS__",
    "-DCK_USE_WMMA=1",
    "-DCK_TILE_USE_WMMA=1",
    "-Wno-unknown-warning-option",
    "-Wno-lifetime-safety-intra-tu-suggestions",
    "-Wno-lifetime-safety-lifetimebound-violation",
]


setup(
    packages=find_packages(exclude=("bench", "tests")),
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="feather_attn._C",
            sources=SOURCES,
            include_dirs=include_dirs,
            depends=HEADER_DEPENDENCIES,
            extra_compile_args={"cxx": common_flags, "nvcc": hip_flags},
            py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
