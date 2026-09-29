"""Optional CUDA backend for the detector's native kernels.

Build with:

    python setup_cuda.py build_ext --inplace

`setup.py` remains the required build: it produces `tunnel_guard._native`, the CPU
implementation the detector runs on. This file adds `tunnel_guard._native_cuda`,
which exposes the same entry points, runs the ported ones on the device and
forwards the rest to `_native`. Nothing in the detector imports it unless the
configuration asks for the CUDA backend and the import succeeds, so an environment
without a GPU keeps working exactly as before.

The compile flags are part of the equivalence claim, not tuning knobs: `--fmad=false`
matches the CPU build's `-ffp-contract=off` so a product-sum is rounded the same
number of times on both sides, and fast math is deliberately absent because it would
change `sqrt`, division and the interpolation the classifier thresholds on.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
ARCHITECTURES = os.environ.get("TG_CUDA_ARCH", "sm_89")


class CudaBuild(build_ext):
    """Compiles every source with nvcc and links the extension here.

    distutils cannot drive nvcc, and its up-to-date check keys off `sources`, so the
    compile and the link are both done explicitly: one object per source, then one
    shared object linked against the CUDA runtime.
    """

    def build_extensions(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            raise SystemExit(
                "nvcc not found. The CUDA backend needs the CUDA toolkit; build the required CPU "
                "extension with `python setup.py build_ext --inplace` instead."
            )
        cuda_root = Path(nvcc).resolve().parents[1]
        library_dirs = [path for path in (cuda_root / "lib64", cuda_root / "lib") if path.is_dir()]
        host_compiler = os.environ.get("CC", "g++")
        include = sysconfig.get_paths()["include"]
        for extension in self.extensions:
            objects = []
            for source in extension.sources:
                source_path = Path(source)
                target = Path(self.build_temp) / (source_path.stem + ".o")
                target.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    nvcc, "-c", str(source_path), "-o", str(target),
                    "-O3", "-std=c++17", "--fmad=false", "-Xcompiler", "-O3,-ffp-contract=off,-fPIC",
                    "-ccbin", host_compiler,
                ]
                for architecture in ARCHITECTURES.split(","):
                    command += ["-gencode", f"arch=compute_{architecture.lstrip('sm_')},code={architecture}"]
                command += [f"-I{include}", f"-I{ROOT}"]
                self.announce(" ".join(command), level=3)
                subprocess.run(command, check=True)
                objects.append(str(target))
            output = Path(self.get_ext_fullpath(extension.name))
            output.parent.mkdir(parents=True, exist_ok=True)
            link = [host_compiler, "-shared", "-fPIC", "-O3", "-o", str(output), *objects,
                    "-lcudart", *(f"-L{path}" for path in library_dirs)]
            link += [*(f"-Wl,-rpath,{path}" for path in library_dirs)]
            self.announce(" ".join(link), level=3)
            subprocess.run(link, check=True)


setup(
    name="tunnel-guard-cuda",
    ext_modules=[Extension("tunnel_guard._native_cuda", ["cpp/cuda/tg_cuda.cu"], language="c++")],
    cmdclass={"build_ext": CudaBuild},
)
