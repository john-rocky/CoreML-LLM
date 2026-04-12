"""Local workaround for coremltools 9.0 tmp-package leak.

Problem
-------
`coremltools.models.MLModel` constructed from a `Model_pb2` spec allocates
a temp ``.mlpackage`` under ``$TMPDIR`` via ``_create_mlpackage`` and only
cleans it up at interpreter exit (``atexit``). Activation calibration in
``linear_quantize_activations`` builds one such MLModel per op-group batch,
which leaks 300+ MB each iteration and fills a 50 GB disk on a single run
of Gemma 4 chunk2.

This module monkey-patches ``MLModel`` to release the temp package eagerly
in ``__del__``. It is a drop-in equivalent of the upstream fix we submitted
to apple/coremltools; remove this file once that PR lands in a release.

Usage::

    import coremltools_tmp_cleanup  # noqa: F401  (must import before MLModel use)
    # or, explicitly:
    from coremltools_tmp_cleanup import install
    install()
"""
from __future__ import annotations

import os
import shutil

import coremltools as ct
from coremltools.models import model as _mlmodel_module


_PATCHED_FLAG = "_tmp_cleanup_patch_installed"


def _close(self):
    """Release temp mlpackage and compiled mlmodelc held by this MLModel."""
    # Drop the proxy first so CoreML's mmap of the compiled model is released.
    try:
        self.__proxy__ = None
    except Exception:
        pass

    pkg = getattr(self, "package_path", None)
    if getattr(self, "is_temp_package", False) and pkg:
        try:
            if os.path.exists(pkg):
                shutil.rmtree(pkg, ignore_errors=True)
        except Exception:
            pass
        try:
            self.package_path = None
            self.is_temp_package = False
        except Exception:
            pass


def _del(self):
    # Interpreter-shutdown-safe: swallow everything.
    try:
        _close(self)
    except Exception:
        pass


def install() -> None:
    """Monkey-patch ``coremltools.models.MLModel`` if not already patched."""
    MLModel = _mlmodel_module.MLModel
    if getattr(MLModel, _PATCHED_FLAG, False):
        return
    MLModel.close = _close  # type: ignore[attr-defined]
    MLModel.__del__ = _del  # type: ignore[attr-defined]
    setattr(MLModel, _PATCHED_FLAG, True)


# Auto-install on import.
install()


if __name__ == "__main__":
    # Smoke test: confirm the patch is wired up.
    print(f"coremltools {ct.__version__}")
    print(f"MLModel.close installed: "
          f"{hasattr(ct.models.MLModel, 'close')}")
    print(f"MLModel.__del__ installed: "
          f"{getattr(ct.models.MLModel, _PATCHED_FLAG, False)}")
