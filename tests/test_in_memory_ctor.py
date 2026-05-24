"""P0b: in-memory MuDataOOM construction from a dict of AnnData(OOM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import mudataoom as moom


def _make_rna(n=8, d=5):
    rng = np.random.default_rng(0)
    return AnnData(
        X=rng.poisson(2, size=(n, d)).astype(np.float32),
        obs=pd.DataFrame(index=[f"c{i:02d}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{i:02d}" for i in range(d)]),
    )


def _make_prot(n=8, d=3):
    rng = np.random.default_rng(1)
    return AnnData(
        X=rng.normal(5, size=(n, d)).astype(np.float32),
        obs=pd.DataFrame(index=[f"c{i:02d}" for i in range(n)]),
        var=pd.DataFrame(index=[f"p{i:02d}" for i in range(d)]),
    )


def test_construct_from_dict_of_anndata():
    m = moom.MuDataOOM({"rna": _make_rna(), "prot": _make_prot()})
    assert isinstance(m, moom.MuDataOOM)
    assert m.n_obs == 8           # union of per-mod obs (same set)
    assert m.n_vars == 5 + 3      # concatenation
    assert m.mod_names == ["rna", "prot"]


def test_in_memory_isbacked_false():
    m = moom.MuDataOOM({"rna": _make_rna()})
    assert m.isbacked is False
    assert m.source_h5mu is None


def test_set_joint_obs_then_round_trip(tmp_path):
    m = moom.MuDataOOM({"rna": _make_rna(), "prot": _make_prot()})
    m.obs = pd.DataFrame(
        {"cohort": ["young"] * 4 + ["old"] * 4},
        index=m.obs_names,
    )
    out = tmp_path / "out.h5mu"
    m.write_h5mu(out)

    with moom.read_h5mu(out) as reloaded:
        assert "cohort" in reloaded.obs_keys()


def test_empty_constructor():
    m = moom.MuDataOOM()
    assert m.n_obs == 0
    assert m.n_vars == 0
    assert m.n_mod == 0
    assert list(m.mod.keys()) == []


def test_rejects_non_mapping():
    with pytest.raises(TypeError, match="mapping"):
        moom.MuDataOOM(data=[1, 2, 3])  # type: ignore[arg-type]
