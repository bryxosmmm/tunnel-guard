"""Required C++ detector extension."""
import sys
from setuptools import Extension, setup

setup(ext_modules=[Extension(
    "tunnel_guard._native", ["cpp/voxel.cpp", "cpp/kernels.cpp"], language="c++", depends=["cpp/native.h"],
    extra_compile_args=["-O3", "-std=c++17", "-ffp-contract=off", "-g0"],
    extra_link_args=["-Wl,-reproducible"] if sys.platform == "darwin" else [],
)])
