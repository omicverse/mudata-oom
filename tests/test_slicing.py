"""P0c: ``mdata[obs_idx, :]`` returns a fresh MuDataOOM subset."""

from __future__ import annotations

import numpy as np
import pandas as pd

import anndataoom
import mudataoom as moom


def test_bool_mask_obs_subset(tiny_mudata):
    h5mu_path, _ = tiny_mudata
    with moom.read_h5mu(h5mu_path) as m:
        mask = np.zeros(m.n_obs, dtype=bool)
        mask[:10] = True
        sub = m[mask, :]
        assert isinstance(sub, moom.MuDataOOM)
        assert sub.n_obs == 10
        for name in m.mod_names:
            assert sub.mod[name].shape[0] == 10
            assert sub.mod[name].shape[1] == m.mod[name].shape[1]


def test_int_array_obs_subset(tiny_mudata):
    h5mu_path, _ = tiny_mudata
    with moom.read_h5mu(h5mu_path) as m:
        sub = m[[0, 5, 10, 15], :]
        assert sub.n_obs == 4
        for name in m.mod_names:
            assert sub.mod[name].shape[0] == 4


def test_slice_obs_subset(tiny_mudata):
    h5mu_path, _ = tiny_mudata
    with moom.read_h5mu(h5mu_path) as m:
        sub = m[0:8, :]
        assert sub.n_obs == 8


def test_subset_preserves_joint_obs(tiny_mudata):
    h5mu_path, _ = tiny_mudata
    with moom.read_h5mu(h5mu_path) as m:
        cohort = m.obs["cohort"]
        idx = np.flatnonzero(cohort == "young")
        sub = m[idx, :]
        assert sub.n_obs == len(idx)
        # Joint obs survived and is consistent
        assert list(sub.obs["cohort"]) == ["young"] * len(idx)


def test_subset_modalities_are_still_anndata_oom(tiny_mudata):
    h5mu_path, _ = tiny_mudata
    with moom.read_h5mu(h5mu_path) as m:
        sub = m[:5, :]
        for name in sub.mod_names:
            assert isinstance(sub.mod[name], anndataoom.AnnDataOOM), name


def test_str_subscript_still_returns_modality(tiny_mudata):
    """Existing 'mdata["rna"]' API must keep working post-slicing patch."""
    h5mu_path, _ = tiny_mudata
    with moom.read_h5mu(h5mu_path) as m:
        rna = m["rna"]
        assert isinstance(rna, anndataoom.AnnDataOOM)
