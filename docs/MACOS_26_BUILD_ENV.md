# macOS 26 (Tahoe / Darwin 25) — coremltools 9.0 build workaround

The PyPI wheel `coremltools-9.0-*-macosx_11_0_arm64.whl` packages the
C++ extensions as `.so` files but the install_name baked at link time
references `@rpath/libmilstoragepython.dylib` etc. — the matching
`.dylib` files are **NOT** included. On macOS 26, this triggers an
import that loads the module silently without any C++ classes
registered, so `coremltools.libmilstoragepython._BlobStorageWriter`
ends up undefined and Apple Conversion stalls at:

```
RuntimeError: BlobWriter not loaded
```

(Symptom: every conversion script in `conversion/` fails after model
load and trace, before saving the mlpackage.)

This was working on Apr 26, 2026; the symptom appeared after upgrading
to macOS 26. Reproduces in every venv (Python 3.10 / 3.11 / 3.12 /
3.14, coremltools 8.3.0 / 9.0).

## Fix: build coremltools from source into a fresh venv

```bash
# 1. Toolchain
brew install protobuf            # protoc 34.x
xcode-select -p                  # confirm /Applications/Xcode.app/...
which cmake                      # confirm /opt/homebrew/bin/cmake

# 2. Fresh venv (Python 3.10 — the most stable target for coremltools 9.0)
~/.pyenv/versions/3.10.13/bin/python3 -m venv /tmp/ct_build_venv
/tmp/ct_build_venv/bin/pip install --upgrade pip wheel setuptools
/tmp/ct_build_venv/bin/pip install pybind11 numpy

# 3. Source build
cd /tmp
git clone --depth 1 https://github.com/apple/coremltools.git coremltools-src
cd coremltools-src
mkdir -p build && cd build

xcrun --sdk macosx cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=12.3 \
    -DPYTHON_EXECUTABLE:FILEPATH=/tmp/ct_build_venv/bin/python \
    -DPYTHON_INCLUDE_DIR=/Users/$USER/.pyenv/versions/3.10.13/include/python3.10 \
    -DPYTHON_LIBRARY=/Users/$USER/.pyenv/versions/3.10.13/lib/libpython3.10.dylib \
    -DOVERWRITE_PB_SOURCE=0 \
    /tmp/coremltools-src

make -j$(sysctl -n hw.ncpu)
cmake --build . --target dist           # produces build/dist/coremltools-*.whl

# 4. Install the freshly built wheel + copy the dylibs alongside the .so files
/tmp/ct_build_venv/bin/pip install build/dist/coremltools-*.whl
cp build/lib*.dylib \
   /tmp/ct_build_venv/lib/python3.10/site-packages/coremltools/
install_name_tool -add_rpath @loader_path \
   /tmp/ct_build_venv/lib/python3.10/site-packages/coremltools/libmilstoragepython.so
install_name_tool -add_rpath @loader_path \
   /tmp/ct_build_venv/lib/python3.10/site-packages/coremltools/libcoremlpython.so
install_name_tool -add_rpath @loader_path \
   /tmp/ct_build_venv/lib/python3.10/site-packages/coremltools/libmodelpackage.so

# 5. Install conversion deps
/tmp/ct_build_venv/bin/pip install --no-cache-dir \
    torch transformers safetensors huggingface-hub scikit-learn

# 6. Verify
/tmp/ct_build_venv/bin/python -c "
from coremltools.converters.mil.backend.mil.load import BlobWriter
import coremltools as ct
print('ct', ct.__version__, '— BlobWriter:', BlobWriter)
"
# Expected: ct 9.0 — BlobWriter: <class 'coremltools.libmilstoragepython._BlobStorageWriter'>
```

## Use the venv for conversion runs

```bash
PY=/tmp/ct_build_venv/bin/python
$PY conversion/build_gemma4_e2b_stateful_3chunks.py \
    --model gemma4-e4b \
    --hf-dir /path/to/gemma4-e4b/hf_model \
    --output /tmp/gemma4-e4b-stateful-3chunk \
    --linear-projections \
    --prefill-batches "8" \
    --ctx 2048 --nbits 4
```

`/tmp/ct_build_venv` is the pinned env for all `conversion/` scripts on
this machine until coremltools 9.1 (or newer) ships a wheel that bundles
the dylibs alongside the .so files for macOS 26.

## Why the symptom is silent

The Python extension `.so` exports `_PyInit_libmilstoragepython` and
loads cleanly under `dlopen`. PyInit registers the pybind11 module and
attaches `_BlobStorageWriter` / `_BlobStorageReader` only if the
matching `libmilstoragepython.dylib` is found and its C++ symbols
resolve. When the dylib is missing, pybind11 silently skips class
registration; the module loads with `dir(m) == ['__doc__', '__file__',
'__loader__', '__name__', '__package__', '__spec__']` — no error, no
warning, just an empty module.

Confirm with:
```bash
$PY -c "import coremltools.libmilstoragepython as m; print(dir(m))"
```
A working install also lists `_BlobStorageReader` and `_BlobStorageWriter`.
