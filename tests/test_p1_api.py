"""P1 mudata.MuData API parity: update / pull / push / vectors /
make_unique / strings_to_categoricals / to_anndata / copy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import anndataoom
import mudataoom as moom


def _build():
    """Tiny in-memory MuDataOOM for unit-level checks."""
    rng = np.random.default_rng(0)
    rna = AnnData(
        X=rng.poisson(2, size=(6, 4)).astype(np.float32),
        obs=pd.DataFrame({"sample": ["A"] * 3 + ["B"] * 3, "n_counts": np.arange(6)},
                         index=[f"c{i}" for i in range(6)]),
        var=pd.DataFrame({"gene_type": ["protein_coding"] * 4},
                         index=[f"g{i}" for i in range(4)]),
    )
    prot = AnnData(
        X=rng.normal(5, size=(6, 2)).astype(np.float32),
        obs=pd.DataFrame({"sample": ["A"] * 3 + ["B"] * 3, "ab_count": np.arange(6)},
                         index=[f"c{i}" for i in range(6)]),
        var=pd.DataFrame({"isotype": ["IgG1", "IgG2"]},
                         index=[f"p{i}" for i in range(2)]),
    )
    return moom.MuDataOOM({"rna": rna, "prot": prot})


# -- update_obs / update_var -------------------------------------------

def test_update_obs_lifts_per_mod_cols():
    m = _build()
    m.update_obs()
    assert "rna:sample" in m.obs_keys()
    assert "prot:sample" in m.obs_keys()
    assert "rna:n_counts" in m.obs_keys()
    assert "prot:ab_count" in m.obs_keys()
    # obsmap recorded
    assert set(m.obsmap.keys()) == {"rna", "prot"}
    assert np.array_equal(m.obsmap["rna"], np.arange(6))


def test_update_var_lifts_per_mod_cols():
    m = _build()
    m.update_var()
    assert "rna:gene_type" in m.var_keys()
    assert "prot:isotype" in m.var_keys()


def test_update_runs_both():
    m = _build()
    m.update()
    assert m.obs_keys()
    assert m.var_keys()


# -- pull / push -------------------------------------------------------

def test_pull_obs_explicit_column():
    m = _build()
    m.pull_obs(columns=["n_counts"], mods=["rna"])
    assert "rna:n_counts" in m.obs_keys()
    # Wasn't a column we pulled from prot
    assert "prot:n_counts" not in m.obs_keys()


def test_push_obs_prefixed_routes_to_modality():
    m = _build()
    m.obs["rna:reviewer"] = ["alice"] * m.n_obs
    m.push_obs(columns=["rna:reviewer"])
    assert "reviewer" in m._mods["rna"].obs.columns
    assert "reviewer" not in m._mods["prot"].obs.columns


def test_push_obs_unprefixed_broadcasts():
    m = _build()
    m.obs["cohort"] = ["young"] * 3 + ["old"] * 3
    m.push_obs(columns=["cohort"])
    assert "cohort" in m._mods["rna"].obs.columns
    assert "cohort" in m._mods["prot"].obs.columns


# -- vectors -----------------------------------------------------------

def test_obs_vector_returns_array():
    m = _build()
    m.update_obs()
    v = m.obs_vector("rna:sample")
    assert isinstance(v, np.ndarray)
    assert v.tolist() == ["A", "A", "A", "B", "B", "B"]


def test_obs_vector_unknown_key_raises():
    m = _build()
    with pytest.raises(KeyError):
        m.obs_vector("not_a_column")


# -- name uniquify -----------------------------------------------------

def test_obs_names_make_unique_runs():
    m = _build()
    # Force a duplicate obs name in one modality.
    ad = m._mods["rna"]
    new = ad.obs.copy()
    new.index = ["dup", "dup"] + list(new.index[2:])
    m._mods["rna"].obs = new
    m.obs_names_make_unique()
    assert len(set(m._mods["rna"].obs_names)) == len(m._mods["rna"].obs_names)


# -- strings_to_categoricals ------------------------------------------

def test_strings_to_categoricals_inplace_on_df():
    m = _build()
    df = pd.DataFrame({"x": ["a", "b", "a"], "y": [1, 2, 3]})
    out = m.strings_to_categoricals(df)
    assert isinstance(out["x"].dtype, pd.CategoricalDtype)
    assert out["y"].dtype == np.int64 or out["y"].dtype == np.int_  # untouched


def test_strings_to_categoricals_walks_per_mod():
    m = _build()
    m.update_obs()  # lift cols into joint
    m.strings_to_categoricals()
    assert isinstance(m.obs["rna:sample"].dtype, pd.CategoricalDtype)


# -- to_anndata --------------------------------------------------------

def test_to_anndata_concatenates_modalities():
    m = _build()
    m.update_obs()
    m.update_var()
    adata = m.to_anndata()
    # 6 cells × (4 rna + 2 prot) features
    assert adata.shape == (6, 6)
    # modality origin recoverable
    assert "modality" in adata.var.columns
    assert set(adata.var["modality"]) == {"rna", "prot"}


# -- copy --------------------------------------------------------------

def test_copy_in_memory_returns_new_instance():
    m = _build()
    m.update_obs()
    cp = m.copy()
    assert cp is not m
    assert cp.n_obs == m.n_obs and cp.n_vars == m.n_vars
    # Mutating one doesn't bleed into the other.
    cp.obs["extra"] = 0
    assert "extra" not in m.obs.columns


def test_copy_with_filename_round_trips(tmp_path):
    m = _build()
    out = tmp_path / "snap.h5mu"
    cp = m.copy(filename=out)
    assert out.exists()
    assert isinstance(cp, moom.MuDataOOM)
    assert cp.n_obs == m.n_obs
