"""
smurf package
=============

A lightweight package wrapper for the SMuRF MultiModal project.

This module exposes the primary submodules as a package so callers can import
from `smurf` instead of importing from the top-level module files.

Recommended usage examples:
    from smurf import models, datasets
    model = models.Model(args)
    ds = datasets.RadPathDataset(...)

Notes:
- The package intentionally exposes submodules (not individual heavy classes)
  to avoid importing large ML dependencies at package import time when possible.
- If you need only a single class, import it from the submodule directly:
    from smurf.models import Model
"""

# Package version. Update when making releases.
__version__ = "0.1.0"

# Expose submodules for convenience. Consumers can import submodules directly:
#   from smurf import models
# or import symbols:
#   from smurf.models import Model
from . import datasets as datasets
from . import losses as losses
from . import models as models
from . import parameters as parameters
from . import preprocessing as preprocessing
from . import utils as utils

# Public API
__all__ = [
    "models",
    "datasets",
    "utils",
    "losses",
    "parameters",
    "preprocessing",
    "__version__",
]

# Optional: small helper to show available modules (useful in interactive sessions)
def modules():
    """
    Return a list of available submodules exposed by the package.

    Example:
        >>> import smurf
        >>> smurf.modules()
        ['models', 'datasets', 'utils', 'losses', 'parameters', 'preprocessing']
    """
    return list(__all__[:-1])
