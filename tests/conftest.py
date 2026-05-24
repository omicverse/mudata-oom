"""Shared fixtures for the mudataoom test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData


@pytest.fixture
def tiny_mudata(tmp_path: Path) -> tuple[Path, MuData]:
    """A small CITE-seq-style MuData (rna + prot) written to ``tmp_path``.

    Returns ``(h5mu_path, mdata_in_memory)`` so tests can compare round-trips
    against the canonical in-memory MuData.
    """
    rng = np.random.default_rng(0)
    n_obs = 32
    obs_names = [f"cell_{i:03d}" for i in range(n_obs)]

    rna = AnnData(
        X=rng.poisson(2.0, size=(n_obs, 40)).astype(np.float32),
        obs=pd.DataFrame({"sample": ["A"] * 20 + ["B"] * 12}, index=obs_names),
        var=pd.DataFrame(index=[f"gene_{j:03d}" for j in range(40)]),
    )
    prot = AnnData(
        X=rng.normal(5, 1, size=(n_obs, 6)).astype(np.float32),
        obs=pd.DataFrame({"sample": ["A"] * 20 + ["B"] * 12}, index=obs_names),
        var=pd.DataFrame(index=[f"prot_{j:02d}" for j in range(6)]),
    )

    mdata = MuData({"rna": rna, "prot": prot})
    mdata.obs["cohort"] = ["young"] * 16 + ["old"] * 16
    mdata.obsm["X_joint"] = rng.normal(size=(n_obs, 4)).astype(np.float32)

    h5mu = tmp_path / "tiny.h5mu"
    mdata.write(str(h5mu))
    return h5mu, mdata
