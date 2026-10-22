"""Training kernels cached on disk, so that benchmarks sharing a data set share the work.

Every benchmark here builds the same five fields of the same molecules at the same handful of
attenuations, and the kernels take minutes per set on the larger collections.  This module keys
a kernel by *what it is a kernel of*: the atom table (coordinates, weights, radii, offsets and
probe coefficients), the complete field set, the model settings that change the kernel
(normalisation, the interpolation grid, the kernel source) and the package version.  The key is
exact, so two scripts that build the same set get the same file, a script that builds it
differently gets another one, and nothing has to be declared or named.

Use in a script, in place of ``model.joint_kernel(model.prepare_training_kernels(table),
table.descriptors)``::

    import kernel_cache as KC
    K = KC.joint_training_kernel(model, table)

After the call the model is in the state :meth:`cmfqsar.model.CMFModel.prepare_training_kernels`
leaves it in (``train``, ``self_diag``, ``K_fields_train``, ``scale`` and the descriptor
statistics), whether the kernels were computed or read back.

The cache lives in ``benchmarks/cache/`` (ignored by git) unless ``$CMF_KERNEL_CACHE`` names
another folder; ``CMF_KERNEL_CACHE=0`` or :func:`disable` turns it off.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from typing import Optional

import numpy as np

import cmfqsar
from cmfqsar.fields import AtomTable
from cmfqsar.model import CMFModel

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "cache")

_enabled = True
_override_dir: Optional[str] = None
#: how many kernels the last calls read back and computed, for a script to report
stats = {"hits": 0, "misses": 0}


def cache_dir() -> Optional[str]:
    """The folder in use, or None when the cache is off."""
    env = os.environ.get("CMF_KERNEL_CACHE")
    if env == "0" or not _enabled:
        return None
    return _override_dir or env or DEFAULT_DIR


def enable(folder: Optional[str] = None) -> None:
    global _enabled, _override_dir
    _enabled, _override_dir = True, folder


def disable() -> None:
    global _enabled
    _enabled = False


def clear() -> int:
    """Delete every cached kernel; returns how many files went."""
    d = cache_dir()
    if not d or not os.path.isdir(d):
        return 0
    n = 0
    for f in os.listdir(d):
        if f.endswith(".npz"):
            os.remove(os.path.join(d, f))
            n += 1
    return n


def key_for(model: CMFModel, table: AtomTable) -> str:
    """A hash of everything the training kernels depend on."""
    h = hashlib.sha256()
    h.update(cmfqsar.__version__.encode())
    settings = {
        "fieldset": dataclasses.asdict(model.fieldset),
        "normalize": bool(model.normalize),
        "alpha_grid": list(model.alpha_grid) if model.alpha_grid else None,
        "kernel_source": model.kernel_source,
        "ac_sigma": float(model.ac_sigma),
        "ac_d_max": float(model.ac_d_max),
    }
    h.update(json.dumps(settings, sort_keys=True, default=str).encode())
    for arr in (table.coords, table.weights, table.radii, table.offsets):
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    for name, coeffs in sorted(table.probe_coeffs.items()):
        h.update(name.encode())
        h.update(np.ascontiguousarray(coeffs).tobytes())
    return h.hexdigest()


def training_kernels(model: CMFModel, table: AtomTable) -> np.ndarray:
    """``model.prepare_training_kernels(table)``, read back from disk when it was done before."""
    d = cache_dir()
    if d is None or not model.uses_fields:
        return model.prepare_training_kernels(table)
    path = os.path.join(d, key_for(model, table) + ".npz")
    if os.path.exists(path):
        with np.load(path) as z:
            K = np.array(z["K"])
            sd = np.array(z["self_diag"])
        model.train = table
        model.self_diag = sd if (model.normalize and sd.size) else None
        model.K_fields_train = K
        model.scale = model._compute_scale(K)
        model.prepare_descriptors(table)
        stats["hits"] += 1
        return K
    K = model.prepare_training_kernels(table)
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:                       # a handle, so savez adds no extension
        np.savez(fh, K=K,
                 self_diag=model.self_diag if model.self_diag is not None else np.zeros(0))
    os.replace(tmp, path)
    stats["misses"] += 1
    return K


def joint_training_kernel(model: CMFModel, table: AtomTable) -> np.ndarray:
    """The joint training kernel, descriptor block included, through the cache."""
    return model.joint_kernel(training_kernels(model, table), table.descriptors)
