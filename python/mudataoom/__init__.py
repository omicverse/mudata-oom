"""mudataoom — Out-of-memory MuData on top of mudata-rs.

Thin Python wrapper that composes:

* :mod:`pymudata` (Rust): low-level ``.h5mu`` reader/writer + HDF5
  ExternalLink workspace.
* :mod:`anndataoom`: per-modality lazy / chunked algorithms.

This package does **not** depend on the upstream Python ``mudata``
package — the joint metadata machinery is in the Rust core.

Quick start
-----------
    >>> import mudataoom as moom
    >>> with moom.read_h5mu("cite.h5mu") as mdata:
    ...     mdata
    MuDataOOM [out-of-memory · backed] n_obs × n_vars = 10000 × 20156, axis=0
      mod (2)
        rna  : AnnDataOOM 10000 × 20000
        prot : AnnDataOOM 10000 × 156

Compatibility
-------------
Each modality is an :class:`anndataoom.AnnDataOOM` — itself a virtual
subclass of :class:`anndata.AnnData` — so per-modality calls into
scanpy / anndataoom chunked ops work unchanged.
"""

from ._core import MuDataOOM, read_h5mu


def is_oom(mdata) -> bool:
    """Return True if ``mdata`` is a :class:`MuDataOOM`."""
    return isinstance(mdata, MuDataOOM) or getattr(mdata, "is_oom", False) is True


__version__ = "0.3.0"

__all__ = [
    "__version__",
    "MuDataOOM",
    "is_oom",
    "read_h5mu",
]
