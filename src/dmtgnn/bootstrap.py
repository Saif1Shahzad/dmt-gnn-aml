"""Select one OpenMP runtime on macOS before importing torch or XGBoost.

Console scripts installed beneath a path containing spaces may use a /bin/sh
launcher, which strips DYLD variables. Re-exec Python directly to preserve them.
"""
import importlib.util
import os
from pathlib import Path
import sys


def main():
    if sys.platform=='darwin' and not os.environ.get('DMT_OPENMP_READY'):
        spec = importlib.util.find_spec('torch')
        lib = Path(spec.origin).parent/'lib' if spec and spec.origin else None
        if lib and (lib/'libomp.dylib').exists():
            env = os.environ.copy()
            env['DMT_OPENMP_READY'] = '1'
            env['DYLD_LIBRARY_PATH'] = str(lib)+(':'+env['DYLD_LIBRARY_PATH'] if env.get('DYLD_LIBRARY_PATH') else '')
            os.execve(sys.executable,[sys.executable,'-m','dmtgnn.cli',*sys.argv[1:]],env)
    from .cli import main as run
    run()
