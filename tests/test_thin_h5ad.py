"""mudata-rs ExternalLink workspace: thin .h5ad files must point back
into the source .h5mu without copying X bytes."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

import mudataoom as moom


def test_thin_paths_exposed(tiny_mudata):
    h5mu_path, _ = tiny_mudata
    with moom.read_h5mu(h5mu_path) as mdata:
        thin_paths = dict(mdata._rs.thin_paths)
        assert set(thin_paths.keys()) == {"rna", "prot"}
        for p in thin_paths.values():
            from pathlib import Path

            assert Path(p).exists(), p


def test_thin_files_are_small(tmp_path):
    """Thin .h5ad must be O(KB), not O(MB) — proves X wasn't copied."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 500)).astype(np.float32)  # ~4 MB
    rna = AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(X.shape[1])]),
    )
    h5mu = tmp_path / "src.h5mu"
    MuData({"rna": rna}).write_h5mu(str(h5mu))

    from pathlib import Path

    with moom.read_h5mu(h5mu) as mdata:
        thin = Path(dict(mdata._rs.thin_paths)["rna"])
        assert thin.stat().st_size < h5mu.stat().st_size // 10


def test_thin_x_reads_match_source(tmp_path):
    """Reading X through the thin file must yield identical bytes."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 16)).astype(np.float32)
    rna = AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(64)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(16)]),
    )
    h5mu = tmp_path / "src.h5mu"
    MuData({"rna": rna}).write_h5mu(str(h5mu))

    with moom.read_h5mu(h5mu) as mdata:
        thin_path = dict(mdata._rs.thin_paths)["rna"]
        with h5py.File(thin_path, "r") as f:
            np.testing.assert_array_equal(f["X"][:], X)


def test_missing_modality_raises(tmp_path):
    rna = AnnData(
        X=np.zeros((2, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    h5mu = tmp_path / "src.h5mu"
    MuData({"rna": rna}).write_h5mu(str(h5mu))
    with moom.read_h5mu(h5mu) as mdata:
        assert "rna" in mdata
        assert "atac" not in mdata
        with pytest.raises(KeyError):
            mdata["atac"]
