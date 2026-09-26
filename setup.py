"""Required native detector kernels. Build with `python setup.py build_ext --inplace`."""
from setuptools import Extension, setup

setup(ext_modules=[Extension(
    "tunnel_guard._native", ["cpp/voxel.cpp", "cpp/kernels.cpp"], language="c++", depends=["cpp/native.h"],
    extra_compile_args=["-O3", "-std=c++17", "-ffp-contract=off"],
)])
