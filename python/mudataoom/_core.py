"""MuDataOOM — out-of-memory multimodal AnnData on top of mudata-rs.

Mirrors the design of :class:`anndataoom.AnnDataOOM`:

* low-level on-disk I/O is in Rust (:mod:`mudataoom._backend`);
* the Python class composes it into a mudata-like wrapper without
  inheriting from the upstream ``mudata`` Python package;
* each modality is an :class:`anndataoom.AnnDataOOM` opened against
  the thin ExternalLink file that the Rust workspace materialised, so
  per-modality ``X`` stays on disk through anndata-rs's backed reader.

API surface intentionally mirrors :class:`mudata.MuData` so existing
mudata-flavoured code (mofapy, muon, scvi-tools, scanpy) keeps working.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd

import anndataoom

from . import _backend  # type: ignore[attr-defined]
from ._joint import JointState

_RsMuData = _backend.MuData
_rs_read_h5mu = _backend.read_h5mu


class _ModView(Mapping[str, "anndataoom.AnnDataOOM"]):
    """Read-only view of ``{name -> AnnDataOOM}`` mirroring ``MuData.mod``."""

    __slots__ = ("_items",)

    def __init__(self, items: "OrderedDict[str, anndataoom.AnnDataOOM]"):
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
        return f"ModView({list(self._items)})"


class MuDataOOM:
    """Out-of-memory multimodal AnnData container.

    Construction
    ------------

    Three supported paths:

    1. :func:`mudataoom.read_h5mu` — open a ``.h5mu`` lazily.
    2. ``MuDataOOM({'rna': adata1, 'prot': adata2})`` — in-memory
       container around existing :class:`anndataoom.AnnDataOOM`
       (or plain :class:`anndata.AnnData`) modalities. Joint
       ``obs_names`` / ``var_names`` are derived by union of the
       per-modality names (mudata-compatible behaviour, axis=0).
    3. :meth:`MuDataOOM._from_rust` (internal) — wrap a
       :class:`mudataoom._backend.MuData` directly.
    """

    __slots__ = (
        "_rs",
        "_mods",
        "_mod_view",
        "_joint",
        "_in_memory_n_obs",
        "_in_memory_n_vars",
        "_in_memory_obs_names",
        "_in_memory_var_names",
        "_in_memory_axis",
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        data: (
            Mapping[str, "anndataoom.AnnDataOOM"]
            | None
        ) = None,
        *,
        axis: int = 0,
    ):
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise TypeError(
                f"MuDataOOM expects a {{name: AnnData(OOM)}} mapping; "
                f"got {type(data).__name__}"
            )
        mods: "OrderedDict[str, anndataoom.AnnDataOOM]" = OrderedDict()
        for name, ad in data.items():
            mods[str(name)] = ad
        self._mods = mods
        self._mod_view = _ModView(mods)
        self._rs = None
        self._joint = JointState(source_h5mu=None)
        self._in_memory_axis = int(axis)
        self._recompute_in_memory_dims()

    @classmethod
    def _from_rust(cls, rs: "_RsMuData") -> "MuDataOOM":
        """Build a :class:`MuDataOOM` around a freshly-loaded Rust handle."""
        thin_paths: dict[str, str] = dict(rs.thin_paths)
        mods: "OrderedDict[str, anndataoom.AnnDataOOM]" = OrderedDict()
        for name in rs.mod_names:
            mods[name] = anndataoom.read(thin_paths[name])
        new = cls.__new__(cls)
        new._mods = mods
        new._mod_view = _ModView(mods)
        new._rs = rs
        new._joint = JointState(
            source_h5mu=Path(rs.source_path) if rs.source_path else None
        )
        new._in_memory_axis = int(rs.axis)
        new._in_memory_n_obs = 0
        new._in_memory_n_vars = 0
        new._in_memory_obs_names = None
        new._in_memory_var_names = None
        return new

    def _recompute_in_memory_dims(self) -> None:
        """For in-memory MuDataOOMs (no Rust handle): derive joint
        dims by union of per-modality obs_names / var_names."""
        if not self._mods:
            self._in_memory_n_obs = 0
            self._in_memory_n_vars = 0
            self._in_memory_obs_names = pd.Index([])
            self._in_memory_var_names = pd.Index([])
            return
        if self._in_memory_axis == 0:
            obs_union: list[str] = []
            seen: set[str] = set()
            for ad in self._mods.values():
                for n in map(str, ad.obs_names):
                    if n not in seen:
                        seen.add(n)
                        obs_union.append(n)
            var_concat: list[str] = []
            for name, ad in self._mods.items():
                var_concat.extend(f"{name}:{v}" for v in map(str, ad.var_names))
            self._in_memory_obs_names = pd.Index(obs_union, name="obs_names")
            self._in_memory_var_names = pd.Index(var_concat, name="var_names")
        else:
            # Feature-aligned mode (axis=1) is less common; mirror upstream
            # by unioning var_names and concatenating obs.
            var_union: list[str] = []
            seen: set[str] = set()
            for ad in self._mods.values():
                for n in map(str, ad.var_names):
                    if n not in seen:
                        seen.add(n)
                        var_union.append(n)
            obs_concat: list[str] = []
            for name, ad in self._mods.items():
                obs_concat.extend(f"{name}:{v}" for v in map(str, ad.obs_names))
            self._in_memory_var_names = pd.Index(var_union, name="var_names")
            self._in_memory_obs_names = pd.Index(obs_concat, name="obs_names")
        self._in_memory_n_obs = len(self._in_memory_obs_names)
        self._in_memory_n_vars = len(self._in_memory_var_names)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def is_oom(self) -> bool:
        return True

    @property
    def is_view(self) -> bool:
        return False

    @property
    def source_h5mu(self) -> Path | None:
        if self._rs is not None and self._rs.source_path is not None:
            return Path(self._rs.source_path)
        return None

    @property
    def filename(self) -> Path | None:
        return self.source_h5mu

    @property
    def isbacked(self) -> bool:
        return self._rs is not None

    # ------------------------------------------------------------------
    # Modalities
    # ------------------------------------------------------------------
    @property
    def mod(self) -> _ModView:
        return self._mod_view

    @property
    def mod_names(self) -> list[str]:
        return list(self._mods.keys())

    @property
    def n_mod(self) -> int:
        return len(self._mods)

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------
    @property
    def n_obs(self) -> int:
        return self._rs.n_obs if self._rs is not None else self._in_memory_n_obs

    @property
    def n_vars(self) -> int:
        return self._rs.n_vars if self._rs is not None else self._in_memory_n_vars

    @property
    def n_var(self) -> int:
        """Alias for :attr:`n_vars` to match upstream mudata."""
        return self.n_vars

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_obs, self.n_vars)

    @property
    def axis(self) -> int:
        return self._rs.axis if self._rs is not None else self._in_memory_axis

    @property
    def obs_names(self) -> pd.Index:
        if self._rs is not None:
            return pd.Index(list(self._rs.obs_names), name="obs_names")
        return self._in_memory_obs_names

    @property
    def var_names(self) -> pd.Index:
        if self._rs is not None:
            return pd.Index(list(self._rs.var_names), name="var_names")
        return self._in_memory_var_names

    # ------------------------------------------------------------------
    # Joint metadata — read from source .h5mu on first access (with cache)
    # ------------------------------------------------------------------
    @property
    def obs(self) -> pd.DataFrame:
        df = self._joint.obs
        if df.empty and len(self.obs_names) > 0:
            # No /obs in source (or no source) but we know the index:
            # return an empty-column DataFrame indexed by obs_names.
            df = pd.DataFrame(index=self.obs_names)
            self._joint.obs = df
        return df

    @obs.setter
    def obs(self, value: pd.DataFrame) -> None:
        self._joint.obs = value

    @property
    def var(self) -> pd.DataFrame:
        df = self._joint.var
        if df.empty and len(self.var_names) > 0:
            df = pd.DataFrame(index=self.var_names)
            self._joint.var = df
        return df

    @var.setter
    def var(self, value: pd.DataFrame) -> None:
        self._joint.var = value

    @property
    def obsm(self) -> dict:
        return self._joint.obsm

    @obsm.setter
    def obsm(self, value: Mapping[str, Any]) -> None:
        self._joint.obsm = value

    @property
    def varm(self) -> dict:
        return self._joint.varm

    @varm.setter
    def varm(self, value: Mapping[str, Any]) -> None:
        self._joint.varm = value

    @property
    def obsp(self) -> dict:
        return self._joint.obsp

    @obsp.setter
    def obsp(self, value: Mapping[str, Any]) -> None:
        self._joint.obsp = value

    @property
    def varp(self) -> dict:
        return self._joint.varp

    @varp.setter
    def varp(self, value: Mapping[str, Any]) -> None:
        self._joint.varp = value

    @property
    def obsmap(self) -> dict:
        return self._joint.obsmap

    @obsmap.setter
    def obsmap(self, value: Mapping[str, Any]) -> None:
        self._joint.obsmap = value

    @property
    def varmap(self) -> dict:
        return self._joint.varmap

    @varmap.setter
    def varmap(self, value: Mapping[str, Any]) -> None:
        self._joint.varmap = value

    @property
    def uns(self) -> dict:
        return self._joint.uns

    @uns.setter
    def uns(self, value: Mapping[str, Any]) -> None:
        self._joint.uns = value

    # ------------------------------------------------------------------
    # Key accessors (mudata API parity)
    # ------------------------------------------------------------------
    def obs_keys(self) -> list[str]:
        return list(self.obs.columns)

    def var_keys(self) -> list[str]:
        return list(self.var.columns)

    def obsm_keys(self) -> list[str]:
        return list(self.obsm.keys())

    def varm_keys(self) -> list[str]:
        return list(self.varm.keys())

    def uns_keys(self) -> list[str]:
        return list(self.uns.keys())

    # ------------------------------------------------------------------
    # Column vector accessors
    # ------------------------------------------------------------------
    def obs_vector(self, key: str, layer: str | None = None) -> np.ndarray:
        from ._ops import axis_vector_

        return axis_vector_(self, key, "obs", layer)

    def var_vector(self, key: str, layer: str | None = None) -> np.ndarray:
        from ._ops import axis_vector_

        return axis_vector_(self, key, "var", layer)

    # ------------------------------------------------------------------
    # Joint table sync (mudata 0.3-style update + 0.4-style pull/push)
    # ------------------------------------------------------------------
    def update_obs(self) -> None:
        """Rebuild joint ``.obs`` from per-modality ``.obs`` tables."""
        from ._ops import update_obs_

        update_obs_(self)

    def update_var(self) -> None:
        """Rebuild joint ``.var`` from per-modality ``.var`` tables."""
        from ._ops import update_var_

        update_var_(self)

    def update(self) -> None:
        """Rebuild both joint ``.obs`` and ``.var``."""
        from ._ops import update_

        update_(self)

    def pull_obs(
        self,
        columns: list[str] | None = None,
        mods: list[str] | None = None,
        drop: bool = False,
        only_drop: bool = False,
        **_legacy,
    ) -> None:
        """Copy per-modality obs columns up to the joint obs.

        Simplified relative to upstream's ``pull_obs`` (mudata 0.4):
        the ``common`` / ``nonunique`` / ``unique`` categorisation
        knobs are silently ignored — every selected column is lifted
        with a ``"<mod>:"`` prefix. Tracked as a follow-up.
        """
        from ._ops import pull_axis_

        pull_axis_(self, "obs", columns, mods, drop, only_drop)

    def pull_var(
        self,
        columns: list[str] | None = None,
        mods: list[str] | None = None,
        drop: bool = False,
        only_drop: bool = False,
        **_legacy,
    ) -> None:
        from ._ops import pull_axis_

        pull_axis_(self, "var", columns, mods, drop, only_drop)

    def push_obs(
        self,
        columns: list[str] | None = None,
        mods: list[str] | None = None,
        drop: bool = False,
        only_drop: bool = False,
        **_legacy,
    ) -> None:
        """Copy joint obs columns down into per-modality obs.

        ``"<mod>:<key>"`` joint columns route to that modality's
        ``<key>``; unprefixed joint columns broadcast to all target
        modalities (lookup by the modality's own index).
        """
        from ._ops import push_axis_

        push_axis_(self, "obs", columns, mods, drop, only_drop)

    def push_var(
        self,
        columns: list[str] | None = None,
        mods: list[str] | None = None,
        drop: bool = False,
        only_drop: bool = False,
        **_legacy,
    ) -> None:
        from ._ops import push_axis_

        push_axis_(self, "var", columns, mods, drop, only_drop)

    # ------------------------------------------------------------------
    # Name + dtype hygiene
    # ------------------------------------------------------------------
    def obs_names_make_unique(self) -> None:
        """Make per-modality ``obs_names`` unique; rebuild joint axis."""
        for ad in self._mods.values():
            mk = getattr(ad, "obs_names_make_unique", None)
            if callable(mk):
                mk()
        if self._rs is None:
            self._recompute_in_memory_dims()

    def var_names_make_unique(self) -> None:
        for ad in self._mods.values():
            mk = getattr(ad, "var_names_make_unique", None)
            if callable(mk):
                mk()
        if self._rs is None:
            self._recompute_in_memory_dims()

    def strings_to_categoricals(
        self, df: pd.DataFrame | None = None
    ) -> pd.DataFrame | None:
        """Convert object-dtype columns to pandas categoricals.

        With an explicit ``df``: mutate + return that frame.
        Without: apply to joint ``.obs`` / ``.var`` plus every
        modality's ``.obs`` / ``.var``.
        """
        from ._ops import strings_to_categoricals_

        if df is not None:
            return strings_to_categoricals_(df)
        strings_to_categoricals_(self._joint.obs)
        strings_to_categoricals_(self._joint.var)
        for ad in self._mods.values():
            strings_to_categoricals_(getattr(ad, "obs", None))
            strings_to_categoricals_(getattr(ad, "var", None))
        return None

    # ------------------------------------------------------------------
    # Collapse / copy
    # ------------------------------------------------------------------
    def to_anndata(self, **_kwargs) -> Any:
        """Collapse to a single :class:`anndata.AnnData`.

        ``X`` is materialised in memory — not advisable for atlas-scale
        objects. See :func:`mudataoom._ops.to_anndata_` for details
        and current limitations (axis=0 only).
        """
        from ._ops import to_anndata_

        return to_anndata_(self)

    def copy(self, filename: str | os.PathLike | None = None) -> "MuDataOOM":
        """Return a fresh :class:`MuDataOOM` with copied state.

        Per-modality ``X`` is *not* re-written: the backing AnnDataOOM
        objects are reused (they're already file-backed). The joint
        DataFrames/dicts are deep-copied. If ``filename`` is given,
        :meth:`write_h5mu` is called and the result re-opened.
        """
        if filename is not None:
            self.write_h5mu(filename)
            return read_h5mu(filename)
        new = MuDataOOM(dict(self._mods), axis=self.axis)
        if not self._joint.obs.empty:
            new._joint.obs = self._joint.obs.copy()
        if not self._joint.var.empty:
            new._joint.var = self._joint.var.copy()
        for src_name in ("obsm", "varm", "obsp", "varp", "obsmap", "varmap", "uns"):
            if src_name in self._joint._loaded:
                src = getattr(self._joint, src_name)
                if src:
                    setattr(new._joint, src_name, dict(src))
        return new

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_obs

    def __contains__(self, key) -> bool:
        return key in self._mods

    def __iter__(self) -> Iterator[str]:
        return iter(self._mods)

    def __getitem__(self, index):
        """``mdata['rna']`` → AnnDataOOM modality.

        ``mdata[obs_idx, :]`` / ``mdata[obs_idx, var_idx]`` → fresh
        :class:`MuDataOOM` whose modalities and joint state have been
        subset along ``obs_idx`` (and, when applicable, ``var_idx``).
        Slicing is materialising, not a view — anndataoom modalities
        track subsets as O(1) descriptors so this stays cheap.
        """
        if isinstance(index, str):
            return self._mods[index]
        if isinstance(index, tuple) and len(index) == 2:
            return self._subset(index[0], index[1])
        # Single-axis subset is interpreted as obs subset (mudata-compatible).
        return self._subset(index, slice(None))

    def _subset(self, obs_idx, var_idx) -> "MuDataOOM":
        """Build a new MuDataOOM with each modality subset on obs."""
        joint_obs_names = self.obs_names
        joint_obs_index = _resolve_axis_index(obs_idx, joint_obs_names)

        # Per-modality slice: realign each modality's obs to the joint
        # subset via obs_names. Modalities whose obs is exactly the joint
        # axis (CITE-seq-style cell-aligned) take the same positional
        # index; others (rare, but supported by mudata) look up by name.
        new_mods: "OrderedDict[str, anndataoom.AnnDataOOM]" = OrderedDict()
        for name, ad in self._mods.items():
            mod_obs_names = pd.Index(list(map(str, ad.obs_names)))
            if list(mod_obs_names) == list(joint_obs_names):
                # Fast path: positional slice.
                sub = ad[joint_obs_index.tolist(), :]
            else:
                # Generic path: select by name; modalities without all the
                # requested cells get the intersection.
                keep_names = joint_obs_names[joint_obs_index]
                mask = mod_obs_names.isin(keep_names)
                positions = np.flatnonzero(mask)
                sub = ad[positions.tolist(), :]
            new_mods[name] = sub

        new = MuDataOOM(new_mods, axis=self.axis)
        # Carry joint metadata across, sliced on obs.
        if not self._joint.obs.empty:
            new._joint.obs = self._joint.obs.iloc[joint_obs_index].copy()
        for k, v in self._joint.obsm.items():
            arr = np.asarray(v) if not hasattr(v, "shape") else v
            try:
                new._joint.obsm[k] = arr[joint_obs_index]
            except Exception:
                # Keep as-is if the array is opaque (e.g. sparse with
                # selection semantics we don't recognise).
                new._joint.obsm[k] = v
        # var-axis joint state — only subset if a var index was provided.
        if isinstance(var_idx, slice) and var_idx == slice(None):
            pass
        else:
            var_index = _resolve_axis_index(var_idx, self.var_names)
            if not self._joint.var.empty:
                new._joint.var = self._joint.var.iloc[var_index].copy()
        # Preserve uns wholesale.
        if self._joint._loaded and "uns" in self._joint._loaded:
            new._joint.uns = dict(self._joint.uns)
        return new

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def write_h5mu(self, path: str | os.PathLike) -> None:
        """Persist to a new ``.h5mu`` file.

        Backed by the Rust writer when this MuDataOOM was loaded from a
        ``.h5mu``. In-memory MuDataOOMs (built from a dict of
        AnnDataOOMs) fall back to a Python-side writer that uses
        upstream :func:`mudata.write_h5mu` — that path materialises
        per-modality X temporarily.
        """
        path = str(Path(path))
        if self._rs is not None:
            self._rs.write_h5mu(path)
            return
        # In-memory path: use the upstream mudata writer so we don't
        # reimplement the joint encoder. Pay the price of importing
        # mudata only when we don't have a Rust handle to delegate to.
        import mudata as _md

        from anndata import AnnData

        # mudata writes by walking each modality's .X; AnnDataOOM
        # accepts BackedArray X which upstream can't encode. Switch
        # each modality to a fresh in-memory AnnData just for the
        # write call, so the file is well-formed.
        mods_in_memory: dict[str, AnnData] = {}
        for name, ad in self._mods.items():
            X = np.asarray(ad.X[:]) if hasattr(ad, "X") and ad.X is not None else None
            obs = ad.obs.copy() if hasattr(ad, "obs") else pd.DataFrame()
            var = ad.var.copy() if hasattr(ad, "var") else pd.DataFrame()
            mods_in_memory[name] = AnnData(X=X, obs=obs, var=var)
        md = _md.MuData(mods_in_memory)
        if not self._joint.obs.empty:
            md.obs = self._joint.obs
        if not self._joint.var.empty:
            md.var = self._joint.var
        for k, v in self._joint.obsm.items():
            md.obsm[k] = v
        for k, v in self._joint.varm.items():
            md.varm[k] = v
        md.write_h5mu(path)

    write = write_h5mu  # mudata API parity

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close per-modality handles. Temp ExternalLink files are
        released when the underlying Rust handle is gc'd."""
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

    def __repr__(self) -> str:
        head = (
            f"MuDataOOM [out-of-memory · backed] "
            f"n_obs × n_vars = {self.n_obs} × {self.n_vars}, axis={self.axis}"
        )
        body = [head]
        if not self._joint.obs.empty:
            body.append(f"  obs:  {list(self._joint.obs.columns)}")
        if self._joint._loaded and "obsm" in self._joint._loaded and self._joint.obsm:
            body.append(f"  obsm: {list(self._joint.obsm.keys())}")
        body.append(f"  mod ({self.n_mod})")
        for name, m in self._mods.items():
            body.append(f"    {name}: AnnDataOOM {m.shape[0]} × {m.shape[1]}")
        return "\n".join(body)


def _resolve_axis_index(idx, names: pd.Index) -> np.ndarray:
    """Normalise an indexer to a positional int array along ``names``."""
    if isinstance(idx, slice):
        return np.arange(len(names))[idx]
    arr = np.asarray(idx)
    if arr.dtype == bool:
        if len(arr) != len(names):
            raise IndexError(
                f"boolean index length {len(arr)} doesn't match axis length {len(names)}"
            )
        return np.flatnonzero(arr)
    if arr.dtype.kind in {"i", "u"}:
        return arr.astype(np.intp, copy=False)
    # String labels → positional lookup
    return np.array([names.get_loc(x) for x in arr], dtype=np.intp)


def _ensure_scratch_tmpdir() -> None:
    """Default TMPDIR to a /scratch-side directory if the user hasn't
    set one. Keeps the multi-GB thin .h5ad files off the small ``/``
    partition."""
    if os.environ.get("TMPDIR"):
        return
    candidate = Path("/scratch/users") / os.environ.get("USER", "tmp") / "tmp"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    os.environ["TMPDIR"] = str(candidate)


def read_h5mu(filename: str | os.PathLike, *, backed: str = "r") -> MuDataOOM:
    """Open a ``.h5mu`` lazily; returns a :class:`MuDataOOM`."""
    _ensure_scratch_tmpdir()
    rs = _rs_read_h5mu(str(Path(filename)))
    return MuDataOOM._from_rust(rs)


__all__ = ["MuDataOOM", "read_h5mu"]
