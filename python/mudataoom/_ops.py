"""mudata-API parity operations on top of :class:`MuDataOOM`.

Kept separate from `_core.py` so the wrapper class stays focused on
state + property forwarding while the heavier algorithms (joint table
rebuilds, pull/push column sync, modality concatenation) live here.

Semantics follow upstream :class:`mudata.MuData` closely; deviations
are flagged in each function's docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ._core import MuDataOOM


# ---------------------------------------------------------------------
# update_obs / update_var
# ---------------------------------------------------------------------

def _rebuild_joint_table(
    mods: dict[str, Any],
    axis_attr: str,
    joint_index: pd.Index,
    existing: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Rebuild joint ``obs`` (or ``var``) by outer-joining per-modality
    tables onto ``joint_index``. Returns ``(new_df, axis_map)`` where
    ``axis_map[name]`` is the int positions of the modality's rows
    within ``joint_index`` (mirrors mudata's ``obsmap``/``varmap``).

    Per-modality columns are prefixed with ``"<mod>:"`` to mirror
    mudata's behaviour. Existing joint columns that are NOT
    ``<mod>:...`` (i.e. user-added joint columns) are preserved.
    """
    # Preserve user-added joint columns (those without a "mod:" prefix).
    preserved_cols = [c for c in existing.columns if ":" not in str(c)]
    out = existing.loc[joint_index, preserved_cols].copy() if preserved_cols else (
        pd.DataFrame(index=joint_index)
    )

    axis_map: dict[str, np.ndarray] = {}
    for name, ad in mods.items():
        per_mod = getattr(ad, axis_attr).copy()
        per_mod.index = per_mod.index.astype(str)
        per_mod_idx = per_mod.index
        # Position of each modality row within joint_index.
        positions = np.array(
            [joint_index.get_loc(n) if n in joint_index else -1 for n in per_mod_idx],
            dtype=np.int64,
        )
        axis_map[name] = positions
        # Reindex to joint shape; prefix cols.
        per_mod = per_mod.add_prefix(f"{name}:")
        per_mod = per_mod.reindex(joint_index)
        out = out.join(per_mod, how="left")

    return out, axis_map


def update_obs_(m: "MuDataOOM") -> None:
    """Rebuild joint ``.obs`` from per-modality ``.obs`` tables."""
    new_obs, obsmap = _rebuild_joint_table(
        dict(m._mods), "obs", m.obs_names, m.obs
    )
    m._joint.obs = new_obs
    m._joint.obsmap = obsmap


def update_var_(m: "MuDataOOM") -> None:
    """Rebuild joint ``.var`` from per-modality ``.var`` tables."""
    new_var, varmap = _rebuild_joint_table(
        dict(m._mods), "var", m.var_names, m.var
    )
    m._joint.var = new_var
    m._joint.varmap = varmap


def update_(m: "MuDataOOM") -> None:
    """Convenience: run both updates."""
    update_obs_(m)
    update_var_(m)


# ---------------------------------------------------------------------
# pull_obs / pull_var (simplified: explicit column list per modality)
# ---------------------------------------------------------------------

def pull_axis_(
    m: "MuDataOOM",
    axis_attr: str,
    columns: list[str] | None,
    mods: list[str] | None,
    drop: bool,
    only_drop: bool,
) -> None:
    """Copy per-modality columns up to the joint table.

    Simplified semantics relative to upstream ``pull_obs/var``:
    we always prefix joint columns with ``"<mod>:"``. The advanced
    ``common`` / ``nonunique`` / ``unique`` join modes from mudata
    >=0.4 are not yet implemented — that's tracked as a P1 follow-up.
    """
    target_mods = mods or list(m._mods.keys())
    joint_attr = "obs" if axis_attr == "obs" else "var"
    joint_df = getattr(m, joint_attr).copy()
    joint_index = joint_df.index

    for name in target_mods:
        ad = m._mods[name]
        per_mod = getattr(ad, axis_attr)
        if per_mod is None or per_mod.empty:
            continue
        selected = list(per_mod.columns) if columns is None else [
            c for c in columns if c in per_mod.columns
        ]
        if only_drop:
            # Just drop, don't copy.
            for c in selected:
                col = f"{name}:{c}"
                if col in joint_df.columns:
                    joint_df = joint_df.drop(columns=[col])
                if drop and c in per_mod.columns:
                    new_per = per_mod.drop(columns=[c])
                    if axis_attr == "obs":
                        ad.obs = new_per
                    else:
                        ad.var = new_per
            continue
        sub = per_mod[selected].add_prefix(f"{name}:")
        sub.index = sub.index.astype(str)
        sub = sub.reindex(joint_index)
        for c in sub.columns:
            joint_df[c] = sub[c]
        if drop:
            new_per = per_mod.drop(columns=selected, errors="ignore")
            if axis_attr == "obs":
                ad.obs = new_per
            else:
                ad.var = new_per

    if axis_attr == "obs":
        m._joint.obs = joint_df
    else:
        m._joint.var = joint_df


def push_axis_(
    m: "MuDataOOM",
    axis_attr: str,
    columns: list[str] | None,
    mods: list[str] | None,
    drop: bool,
    only_drop: bool,
) -> None:
    """Copy joint columns down into per-modality tables.

    Simplified: ``"<mod>:<col>"`` joint columns are routed to that
    modality's ``<col>``; unprefixed joint columns are routed to ALL
    target modalities (lookup by the modality's own index).
    """
    target_mods = mods or list(m._mods.keys())
    joint_attr = "obs" if axis_attr == "obs" else "var"
    joint_df = getattr(m, joint_attr)
    if joint_df is None or joint_df.empty:
        return

    columns = list(joint_df.columns) if columns is None else columns

    for col in columns:
        if col not in joint_df.columns:
            continue
        if ":" in str(col):
            # Prefixed: <mod>:<key>
            mod_name, _, key = str(col).partition(":")
            if mods is not None and mod_name not in mods:
                continue
            if mod_name not in m._mods:
                continue
            ad = m._mods[mod_name]
            mod_index = pd.Index(map(str, ad.obs_names if axis_attr == "obs" else ad.var_names))
            sub = joint_df[col].reindex(mod_index)
            df = getattr(ad, axis_attr).copy()
            df[key] = sub.values
            if axis_attr == "obs":
                ad.obs = df
            else:
                ad.var = df
            if drop or only_drop:
                joint_df = joint_df.drop(columns=[col])
        else:
            for name in target_mods:
                if not only_drop:
                    ad = m._mods[name]
                    mod_index = pd.Index(map(str, ad.obs_names if axis_attr == "obs" else ad.var_names))
                    sub = joint_df[col].reindex(mod_index)
                    df = getattr(ad, axis_attr).copy()
                    df[col] = sub.values
                    if axis_attr == "obs":
                        ad.obs = df
                    else:
                        ad.var = df
            if drop or only_drop:
                joint_df = joint_df.drop(columns=[col])

    if axis_attr == "obs":
        m._joint.obs = joint_df
    else:
        m._joint.var = joint_df


# ---------------------------------------------------------------------
# strings_to_categoricals
# ---------------------------------------------------------------------

def strings_to_categoricals_(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Convert object-dtype columns in `df` to pandas categoricals.

    Returns the modified frame. If `df` is None, returns None — the
    caller is expected to dispatch over joint+per-mod frames itself.
    """
    if df is None:
        return None
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].astype("category")
            except Exception:
                pass
    return df


# ---------------------------------------------------------------------
# obs_vector / var_vector
# ---------------------------------------------------------------------

def axis_vector_(
    m: "MuDataOOM",
    key: str,
    axis_attr: str,
    layer: str | None = None,
) -> np.ndarray:
    """Return values of `key` along the joint axis.

    `layer` is accepted for upstream API parity but currently ignored
    on the joint level — we only carry joint metadata, not joint X
    layers.
    """
    df = m.obs if axis_attr == "obs" else m.var
    if key in df.columns:
        return df[key].to_numpy()
    # Fall back to per-modality search like upstream mudata.
    for name, ad in m._mods.items():
        ad_df = getattr(ad, axis_attr)
        if key in ad_df.columns:
            raise KeyError(
                f"key {key!r} not in joint .{axis_attr}; found in modality "
                f"{name!r}. Call .update_{axis_attr}() to lift it up, or use "
                f"mdata[{name!r}].{axis_attr}_vector({key!r})."
            )
    raise KeyError(f"key {key!r} not in joint .{axis_attr} or any modality")


# ---------------------------------------------------------------------
# to_anndata
# ---------------------------------------------------------------------

def to_anndata_(m: "MuDataOOM") -> "Any":
    """Collapse a :class:`MuDataOOM` to a single :class:`anndata.AnnData`.

    For axis=0 (cell-aligned) we hstack per-modality ``X`` along the
    var axis and emit a joint AnnData; var carries a ``modality``
    column so the origin of each feature is recoverable. ``X`` is
    materialised into memory in this call — appropriate for export to
    tools that don't speak MuData, not for million-cell atlases.
    """
    from anndata import AnnData
    from scipy.sparse import hstack as _hstack, issparse

    if m.axis != 0:
        raise NotImplementedError(
            "to_anndata() is implemented for axis=0 (cell-aligned) MuData "
            "only; axis=1 will land in a follow-up."
        )

    # Realign each modality to the joint obs_names (rows may differ
    # for partially-overlapping CITE-seq-style data; fill missing with NaN).
    joint_obs_names = m.obs_names
    Xs = []
    var_pieces = []
    for name, ad in m._mods.items():
        mod_obs = pd.Index(map(str, ad.obs_names))
        if list(mod_obs) == list(joint_obs_names):
            X = ad.X[:]
        else:
            # Build a row-aligned matrix by positional reindex.
            X_full = ad.X[:]
            row_pos = [mod_obs.get_loc(n) if n in mod_obs else -1 for n in joint_obs_names]
            # If any are -1 we fall back to dense reindex; the OOM
            # purist case (huge X) deserves a smarter path later.
            X_full = np.asarray(X_full) if not issparse(X_full) else X_full.toarray()
            X = np.full((len(joint_obs_names), X_full.shape[1]), np.nan, dtype=X_full.dtype)
            for j, p in enumerate(row_pos):
                if p >= 0:
                    X[j] = X_full[p]
        Xs.append(X)
        var = ad.var.copy()
        var["modality"] = name
        var.index = [f"{name}:{v}" for v in var.index]
        var_pieces.append(var)

    if all(issparse(x) for x in Xs):
        X = _hstack(Xs).tocsr()
    else:
        X = np.hstack([np.asarray(x.toarray() if issparse(x) else x) for x in Xs])

    obs = m.obs.copy()
    var = pd.concat(var_pieces, axis=0)
    obsm = {k: np.asarray(v) for k, v in m.obsm.items()}
    return AnnData(X=X, obs=obs, var=var, obsm=obsm)
