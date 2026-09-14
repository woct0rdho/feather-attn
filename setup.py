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
    "csrc/featherattn_fwd_aligned_bm256.cu",
    "csrc/featherattn_fwd_query_tail.cu",
    "csrc/featherattn_fwd_key_tail.cu",
    "csrc/featherattn_fwd_query_key_tail.cu",
    "csrc/featherattn_fwd_strided.cu",
]
HEADER_DEPENDENCIES = [path.relative_to(ROOT).as_posix()
                        for pattern in ("*.h", "*.cuh")
                        for path in sorted(CSRC.rglob(pattern))]

IS_WINDOWS = os.name == "nt"


def _enable_ccache() -> None:
    if IS_WINDOWS:
        return
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


def _ck_include_dir() -> Path | None:
    """Composable Kernel include path (Linux only; Windows uses the shim)."""
    if IS_WINDOWS:
        return None
    root = Path(
        os.environ.get(
            "FEATHER_ATTN_CK_ROOT",
            "~/rocm-libraries/projects/composablekernel",
        )
    ).expanduser()
    include = root / "include"
    required = include / "ck_tile" / "core.hpp"
    if not required.is_file():
        raise RuntimeError(
            f"Cannot find {required}. "
            "Set FEATHER_ATTN_CK_ROOT to the Composable Kernel source tree."
        )
    return include


def _detect_rocm_arch() -> str:
    """Return a semi-colon-delimited ROCm target arch list.

    Priority: ``FEATHER_ATTN_ROCM_ARCH`` env > detected device >
    ``PYTORCH_ROCM_ARCH`` env > ``gfx1151``.
    """
    explicit = os.environ.get("FEATHER_ATTN_ROCM_ARCH")
    if explicit:
        return explicit
    pytorch_arch = os.environ.get("PYTORCH_ROCM_ARCH")
    if pytorch_arch:
        return pytorch_arch
    try:
        import torch  # noqa: F811 – thin import; only for arch detection
        if torch.cuda.is_available():
            arch_set: set[str] = set()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gcn = getattr(props, "gcnArchName", None)
                if gcn:
                    arch_set.add(gcn.split(":", 1)[0])
            if arch_set:
                return ";".join(sorted(arch_set))
    except Exception:
        pass
    return "gfx1151"


def _rocm_sdk_core_include() -> Path | None:
    try:
        import _rocm_sdk_core  # noqa: F811
    except ImportError:
        return None
    assert _rocm_sdk_core.__file__ is not None
    include = Path(_rocm_sdk_core.__file__).resolve().parent / "include"
    return include if include.is_dir() else None


# --------------------------------------------------------------------------- build
if not cpp_extension.IS_HIP_EXTENSION:
    raise RuntimeError("Feather Attention requires a ROCm-enabled PyTorch build.")

_enable_ccache()
os.environ.setdefault("PYTORCH_ROCM_ARCH", _detect_rocm_arch())

include_dirs = [str(CSRC)]
ck_include = _ck_include_dir()
if ck_include is not None:
    include_dirs.append(str(ck_include))
rocm_sdk_core_include = _rocm_sdk_core_include()
if rocm_sdk_core_include is not None:
    include_dirs.append(str(rocm_sdk_core_include))

stable_defines = [
    "-DTORCH_TARGET_VERSION=0x020A000000000000",
    "-DTORCH_STABLE_ONLY",
]

# hipcc (clang) flags – used for .cu compilation on both platforms.
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
    "-Wno-unused-command-line-argument",
    "-Wno-lifetime-safety-intra-tu-suggestions",
    "-Wno-lifetime-safety-lifetimebound-violation",
]

# Host compiler (.cpp) flags.
if IS_WINDOWS:
    # MSVC: torch's BuildExtension already prepends COMMON_MSVC_FLAGS (which
    # include /EHsc, /MD, /W4 suppressions).  Only add what we specifically
    # need.
    cxx_flags = [
        "/std:c++20",
        *stable_defines,
    ]
else:
    cxx_flags = common_flags

setup(
    packages=find_packages(exclude=("bench", "tests")),
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="feather_attn._C",
            sources=SOURCES,
            include_dirs=include_dirs,
            depends=HEADER_DEPENDENCIES,
            extra_compile_args={"cxx": cxx_flags, "nvcc": hip_flags},
            py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
