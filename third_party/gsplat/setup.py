"""Build the radar CUDA extension `gsplat.csrc`. Modified from gsplat (Apache-2.0) for DyRAD."""
import glob
import os
import os.path as osp
import pathlib

from setuptools import find_packages, setup

__version__ = None
exec(open("gsplat/version.py", "r").read())

if not os.getenv("MAX_JOBS"):
    os.environ["MAX_JOBS"] = "10"
    print(f"Setting MAX_JOBS to {os.environ['MAX_JOBS']}")


def get_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)


def get_extensions():
    from torch.__config__ import parallel_info
    from torch.utils.cpp_extension import CUDAExtension

    extensions_dir = osp.join("gsplat", "cuda")
    sources = glob.glob(osp.join(extensions_dir, "csrc", "*.cu")) + glob.glob(
        osp.join(extensions_dir, "csrc", "*.cpp")
    )
    sources += [osp.join(extensions_dir, "ext.cpp")]

    extra_compile_args = {"cxx": ["-O3", "-Wno-sign-compare"]}
    extra_link_args = ["-s"]

    info = parallel_info()
    if "backend: OpenMP" in info and "OpenMP not found" not in info:
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP", "-fopenmp"]
    else:
        print("Compiling without OpenMP...")

    nvcc_flags = ["-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr"]

    # GLM/Torch has spammy and very annoyingly verbose warnings that this suppresses
    nvcc_flags += ["-diag-suppress", "20012,186"]
    extra_compile_args["nvcc"] = nvcc_flags

    current_dir = pathlib.Path(__file__).parent.resolve()
    glm_path = osp.join(current_dir, "gsplat", "cuda", "csrc", "third_party", "glm")
    include_dirs = [glm_path, osp.join(current_dir, "gsplat", "cuda", "include")]

    extension = CUDAExtension(
        "gsplat.csrc",
        sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    return [extension]


setup(
    name="gsplat",
    version=__version__,
    description="Radar CUDA rasterizer for DyRAD (trimmed fork of gsplat)",
    python_requires=">=3.10",
    install_requires=["ninja", "torch"],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": get_ext()},
    packages=find_packages(),
    include_package_data=True,
)
