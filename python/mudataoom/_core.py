"""MuDataOOM — out-of-memory multimodal AnnData on top of mudata-rs.

Mirrors the design of :class:`anndataoom.AnnDataOOM`:

* the low-level on-disk I/O is in Rust (:mod:`pymudata`);
* the Python class composes it into an mudata-like wrapper without
  inheriting from anything in the upstream ``mudata`` Python package
  (this package has **no Python mudata dependency**);
* each modality is opened as an :class:`anndataoom.AnnDataOOM` via the
  thin ``.h5ad`` ExternalLink file that :mod:`pymudata` materialised —
  so per-modality ``X`` stays on disk through the same anndata-rs
  backed reader that powers :mod:`anndataoom`.

API surface intentionally mirrors :class:`mudata.MuData` so existing
mudata-flavoured code (muon-style sketches, plotting helpers, etc.)
keeps reading the way users expect, even though we never import
``mudata``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import anndataoom

# The compiled Rust extension is bundled with this package as
# ``mudataoom._backend`` (built by maturin from src/lib.rs +
# mudata-rs/pymudata). We don't expect users to depend on a separate
# ``pymudata`` PyPI package.
from . import _backend  # type: ignore[attr-defined]

_RsMuData = _backend.MuData
_rs_read_h5mu = _backend.read_h5mu


class _ModView(Mapping[str, "anndataoom.AnnDataOOM"]):
    """Read-only view of ``{name -> AnnDataOOM}`` mirroring ``MuData.mod``.

    The mapping is materialised once when the parent :class:`MuDataOOM`
    is constructed. Each value is the same :class:`AnnDataOOM` object
    that :meth:`MuDataOOM.__getitem__` returns for that name.
    """

    __slots__ = ("_items",)

    def __init__(self, items: dict[str, "anndataoom.AnnDataOOM"]):
        self._items = items

    def __getitem__(self, key: str) -> "anndataoom.AnnDataOOM":
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def items(self):  # type: ignore[override]
        return self._items.items()

    def keys(self):  # type: ignore[override]
        return self._items.keys()

    def values(self):  # type: ignore[override]
        return self._items.values()

    def __repr__(self) -> str:
        names = list(self._items.keys())
        return f"ModView({names})"


class MuDataOOM:
    """Out-of-memory multimodal AnnData (duck-types :class:`mudata.MuData`).

    Constructed via :func:`mudataoom.read_h5mu`. Holds:

    * ``_rs``    — the :class:`pymudata.MuData` that owns the on-disk
      ``.h5mu`` handle and the temp ExternalLink workspace.
    * ``_mods``  — ``{name -> AnnDataOOM}``: one :class:`AnnDataOOM`
      per modality, opened by :func:`anndataoom.read` against the
      corresponding thin ``.h5ad`` from :attr:`pymudata.MuData.thin_paths`.

    Closing the wrapper closes every modality and drops the Rust
    handle, which in turn removes the temp ExternalLink files.
    """

    __slots__ = ("_rs", "_mods", "_mod_view")

    def __init__(
        self,
        rs: "_RsMuData",
        mods: dict[str, "anndataoom.AnnDataOOM"],
    ):
        self._rs = rs
        self._mods = mods
        self._mod_view = _ModView(mods)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def _from_rust(cls, rs: "_RsMuData") -> "MuDataOOM":
        """Build a :class:`MuDataOOM` around a freshly-loaded Rust handle.

        Iterates ``rs.thin_paths`` to open one :class:`AnnDataOOM` per
        modality. The thin files are owned by ``rs``; keeping ``rs``
        alive keeps the backing files alive.
        """
        thin_paths: dict[str, str] = dict(rs.thin_paths)
        mods: dict[str, anndataoom.AnnDataOOM] = {}
        for name in rs.mod_names:
            mods[name] = anndataoom.read(thin_paths[name])
        return cls(rs, mods)

    # ------------------------------------------------------------------
    # Identity / introspection
    # ------------------------------------------------------------------
    @property
    def is_oom(self) -> bool:
        """Always True — sentinel mirroring :func:`anndataoom.is_oom`."""
        return True

    @property
    def source_h5mu(self) -> Path | None:
        """Path to the source ``.h5mu`` if loaded from disk, else ``None``."""
        p = self._rs.source_path
        return Path(p) if p is not None else None

    # ------------------------------------------------------------------
    # mudata.MuData parity surface
    # ------------------------------------------------------------------
    @property
    def mod(self) -> _ModView:
        """``{modality_name: AnnDataOOM}`` (read-only view)."""
        return self._mod_view

    @property
    def n_mod(self) -> int:
        return self._rs.n_mod

    @property
    def n_obs(self) -> int:
        return self._rs.n_obs

    @property
    def n_vars(self) -> int:
        return self._rs.n_vars

    @property
    def shape(self) -> tuple[int, int]:
        return (self._rs.n_obs, self._rs.n_vars)

    @property
    def axis(self) -> int:
        return int(self._rs.axis)

    @property
    def obs_names(self) -> list[str]:
        return list(self._rs.obs_names)

    @property
    def var_names(self) -> list[str]:
        return list(self._rs.var_names)

    @property
    def isbacked(self) -> bool:
        """True — `MuDataOOM` always streams from disk."""
        return True

    @property
    def filename(self) -> Path | None:
        """Alias for :attr:`source_h5mu`."""
        return self.source_h5mu

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> "anndataoom.AnnDataOOM":
        if isinstance(key, str):
            return self._mods[key]
        raise TypeError(
            f"MuDataOOM indexing supports modality names (str); got {type(key).__name__}. "
            "Slice-style view subsetting will land in a follow-up."
        )

    def __contains__(self, key: object) -> bool:
        return key in self._mods

    def __iter__(self) -> Iterator[str]:
        return iter(self._mods)

    def __len__(self) -> int:
        return self.n_obs

    def __repr__(self) -> str:
        head = f"MuDataOOM [out-of-memory · backed] n_obs × n_vars = {self.n_obs} × {self.n_vars}, axis={self.axis}"
        body = [head, f"  mod ({self.n_mod})"]
        for name, m in self._mods.items():
            body.append(f"    {name}: AnnDataOOM {m.shape[0]} × {m.shape[1]}")
        return "\n".join(body)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def write_h5mu(self, path: str | os.PathLike) -> None:
        """Persist this :class:`MuDataOOM` to a new ``.h5mu`` file.

        The Rust layer streams each modality (no full-matrix
        materialisation) into the destination.
        """
        self._rs.write_h5mu(str(Path(path)))

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close every modality. The thin .h5ad workspace is freed when
        the underlying :class:`pymudata.MuData` is garbage-collected."""
        for m in self._mods.values():
            close = getattr(m, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def __enter__(self) -> "MuDataOOM":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _ensure_scratch_tmpdir() -> None:
    """Default `TMPDIR` to a /scratch-side directory if the user hasn't
    set one. Keeps the multi-GB thin .h5ad / temp .h5ad files off the
    small ``/`` partition (and out of `/home/users/...`)."""
    if os.environ.get("TMPDIR"):
        return
    candidate = Path("/scratch/users") / os.environ.get("USER", "tmp") / "tmp"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    os.environ["TMPDIR"] = str(candidate)


def read_h5mu(filename: str | os.PathLike, *, backed: str = "r") -> MuDataOOM:
    """Open a ``.h5mu`` file lazily and return a :class:`MuDataOOM`.

    Each modality's ``X`` stays on disk via anndata-rs's backed reader.
    The Rust layer materialises one thin ``.h5ad`` per modality (HDF5
    ExternalLinks back into the source ``.h5mu``); we then open each
    with :func:`anndataoom.read` so the user gets a real
    :class:`anndataoom.AnnDataOOM` per modality.

    Parameters
    ----------
    filename
        Path to the ``.h5mu`` file.
    backed
        Forwarded to :func:`anndataoom.read` for each modality.
        Default ``'r'`` (read-only).

    Returns
    -------
    MuDataOOM
        Lazy multimodal dataset. Use it as a context manager (``with
        moom.read_h5mu(...) as mdata: ...``) or call
        :meth:`MuDataOOM.close` when done to release file handles.
    """
    _ensure_scratch_tmpdir()
    rs = _rs_read_h5mu(str(Path(filename)))
    return MuDataOOM._from_rust(rs)


__all__ = ["MuDataOOM", "read_h5mu"]
