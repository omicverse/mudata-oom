"""P0a: joint obs / var / obsm / varm / obsp / varp / uns read out as
real pandas / numpy values, mirroring upstream mudata.MuData."""

from __future__ import annotations

import numpy as np
import pandas as pd

import mudataoom as moom


def test_joint_obs_dataframe_round_trips(tiny_mudata):
    h5mu_path, ref = tiny_mudata
    with moom.read_h5mu(h5mu_path) as m:
        # tiny_mudata adds an 'cohort' column on the joint obs
        assert "cohort" in m.obs_keys()
        pd.testing.assert_series_equal(
            m.obs["cohort"].astype(str).reset_index(drop=True),
            ref.obs["cohort"].astype(str).reset_index(drop=True),
            check_names=False,
        )


def test_joint_obsm_round_trips(tiny_mudata):
    h5mu_path, ref = tiny_mudata
    with moom.read_h5mu(h5mu_path) as m:
        assert "X_joint" in m.obsm_keys()
        np.testing.assert_allclose(
            np.asarray(m.obsm["X_joint"]),
            np.asarray(ref.obsm["X_joint"]),
        )


def test_joint_var_empty_default(tiny_mudata):
    h5mu_path, ref = tiny_mudata
    with moom.read_h5mu(h5mu_path) as m:
        # tiny_mudata doesn't add joint var columns — but the DataFrame
        # should still be a real DataFrame indexed by joint var_names.
        assert isinstance(m.var, pd.DataFrame)
        assert len(m.var) == ref.n_vars


def test_joint_uns_round_trips(tmp_path):
    from anndata import AnnData
    from mudata import MuData

    rng = np.random.default_rng(0)
    rna = AnnData(X=rng.poisson(2, size=(8, 4)).astype(np.float32),
                  obs=pd.DataFrame(index=[f"c{i}" for i in range(8)]),
                  var=pd.DataFrame(index=[f"g{i}" for i in range(4)]))
    md = MuData({"rna": rna})
    md.uns["pipeline"] = "demo"
    md.uns["version"] = "0.1.0"
    p = tmp_path / "src.h5mu"
    md.write_h5mu(str(p))

    with moom.read_h5mu(p) as m:
        assert "pipeline" in m.uns_keys()
        assert m.uns["pipeline"] == "demo"
        assert m.uns["version"] == "0.1.0"
