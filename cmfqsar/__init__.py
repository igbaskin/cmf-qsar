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
"""

__version__ = "1.0.0"

from .fields import FieldSpec, FieldSet, MoleculeFields, build_fields  # noqa: F401
from .model import CMFModel, ModelStats  # noqa: F401
from .dataio import Dataset, load_dataset  # noqa: F401
