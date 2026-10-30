"""
cmfqsar - Continuous Molecular Fields (CMF) 3D-QSAR.

Implementation of the CMF approach described in:

* Baskin I.I., Zhokhova N.I. "The continuous molecular fields approach to
  building 3D-QSAR models", J. Comput.-Aided Mol. Des. 27 (2013) 427-442.
* Sitnikov G.V., Zhokhova N.I., Ustynyuk Y.A., Varnek A., Baskin I.I.
  "Continuous indicator fields: a novel universal type of molecular fields",
  J. Comput.-Aided Mol. Des. 29 (2015) 233-247.
* Baskin I.I., Zhokhova N.I. "Continuous molecular fields and the concept of
  molecular co-fields in structure-activity studies", Future Med. Chem. (2019).

Molecular fields are represented as sums of atom-centred Gaussian functions,
kernels between molecules are overlap integrals of those fields (computed
analytically), and kernel ridge regression (or SVR / GP) is used to fit
3D-QSAR models.  The fields of regression coefficients ("molecular co-fields")
are reconstructed from the dual coefficients and can be visualised.

The stable surface
------------------
The names in ``__all__`` below are the supported interface of this package (plan item G3).  They
are documented in the manual, chapter *Using it as a library*, used by the notebooks under
``docs/tutorials/`` and guarded by ``tests/test_stable_surface.py``, which records the arguments
each one promises.  Within a major version they are not removed or renamed, and an argument that
is recorded there keeps its name and its default; new arguments may be added after them.

Every other module is internal.  It is documented in ``docs/api.md`` and may change in any
release, so a script that imports one should say which version of the package it was written
for.
"""

__version__ = "1.0.0"

from .fields import AtomTable, FieldSpec, FieldSet, MoleculeFields, build_fields  # noqa: F401
from .model import CMFModel, ModelStats  # noqa: F401
from .dataio import Dataset, load_dataset  # noqa: F401

__all__ = [
    "AtomTable",         # the training or test table: the fields of several molecules together
    "CMFModel",          # the model: fit, predict, predict_interval, rank_score, save, load
    "Dataset",           # a data set read from a file: mols, names, activity, the subset indices
    "FieldSet",          # which fields a model uses, and the width of their Gaussians
    "FieldSpec",         # one field of a field set
    "ModelStats",        # what fit() measured: q2, r2, the folds, the search
    "MoleculeFields",    # the fields of one molecule
    "build_fields",      # molecule + field set -> MoleculeFields
    "load_dataset",      # a file -> Dataset
]
