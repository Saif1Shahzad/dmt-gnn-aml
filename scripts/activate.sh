# Source from the repository root: source scripts/activate.sh
source .venv/bin/activate
if [ "$(uname -s)" = Darwin ]; then
  DMT_TORCH_LIB="$(python -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib")')"
  export DYLD_LIBRARY_PATH="$DMT_TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
  unset DMT_TORCH_LIB
fi
export MPLCONFIGDIR="$PWD/runs/.matplotlib"
