"""Optional native accelerator; selecting cpp in a recipe requires this module."""
from setuptools import Extension, setup

setup(ext_modules=[Extension(
    "tunnel_guard._native", ["cpp/voxel.cpp"], language="c++",
    extra_compile_args=["-O3", "-std=c++17"], optional=True,
)])
