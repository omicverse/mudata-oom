"""End-to-end I/O parity with upstream ``mudata.MuData`` writers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mudataoom as moom


def test_read_h5mu_returns_mudataoom(tiny_mudata):
    h5mu_path, _ = tiny_mudata
    with moom.read_h5mu(h5mu_path) as mdata:
        assert isinstance(mdata, moom.MuDataOOM)
        assert moom.is_oom(mdata)


def test_modalities_are_anndata_oom(tiny_mudata):
    import anndataoom as aoom

    h5mu_path, _ = tiny_mudata
    with moom.read_h5mu(h5mu_path) as mdata:
        assert set(mdata.mod.keys()) == {"rna", "prot"}
        for name, m in mdata.mod.items():
            assert isinstance(m, aoom.AnnDataOOM), (
                f"modality {name!r} loaded as {type(m).__name__}"
            )


def test_shapes_match_reference(tiny_mudata):
    h5mu_path, mdata_ref = tiny_mudata
    with moom.read_h5mu(h5mu_path) as mdata:
        assert mdata.n_obs == mdata_ref.n_obs
        assert mdata.n_vars == mdata_ref.n_vars
        for name in mdata_ref.mod:
            assert mdata.mod[name].shape == mdata_ref.mod[name].shape


def test_obs_names_match_reference(tiny_mudata):
    h5mu_path, mdata_ref = tiny_mudata
    with moom.read_h5mu(h5mu_path) as mdata:
        assert list(mdata.obs_names) == list(map(str, mdata_ref.obs_names))


def test_write_then_read_back_self(tiny_mudata, tmp_path):
    """mudataoom's own write+read path round-trips cleanly."""
    h5mu_path, mdata_ref = tiny_mudata
    out = tmp_path / "out.h5mu"
    with moom.read_h5mu(h5mu_path) as mdata:
        mdata.write_h5mu(out)
    assert out.exists() and out.stat().st_size > 0

    with moom.read_h5mu(out) as reloaded:
        assert set(reloaded.mod.keys()) == set(mdata_ref.mod.keys())
        for name in mdata_ref.mod:
            assert reloaded.mod[name].shape == mdata_ref.mod[name].shape
        assert reloaded.n_obs == mdata_ref.n_obs
        assert reloaded.n_vars == mdata_ref.n_vars


def test_write_then_read_back_with_upstream_mudata(tiny_mudata, tmp_path):
    """Upstream `mudata.read_h5mu` (Python) can open what we wrote.

    Requires the writer to (a) avoid blosc/zstd filters that stock h5py
    can't decode, and (b) re-emit the joint /obs /var groups."""
    import mudata as md

    h5mu_path, mdata_ref = tiny_mudata
    out = tmp_path / "out.h5mu"
    with moom.read_h5mu(h5mu_path) as mdata:
        mdata.write_h5mu(out)

    reloaded = md.read_h5mu(str(out))
    assert reloaded.shape == mdata_ref.shape
    assert set(reloaded.mod.keys()) == set(mdata_ref.mod.keys())


def test_contains_and_iter(tiny_mudata):
    h5mu_path, _ = tiny_mudata
    with moom.read_h5mu(h5mu_path) as mdata:
        assert "rna" in mdata
        assert "atac" not in mdata
        assert sorted(iter(mdata)) == ["prot", "rna"]
        assert mdata.n_mod == 2
        assert mdata.shape == (32, 46)
        assert mdata.axis == 0
