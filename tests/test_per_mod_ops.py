"""anndataoom chunked operators must keep working on a MuDataOOM modality."""

from __future__ import annotations

import numpy as np

import anndataoom as aoom
import mudataoom as moom


def test_chunked_normalize_then_log1p(tiny_mudata):
    h5mu_path, _ = tiny_mudata
    with moom.read_h5mu(h5mu_path) as mdata:
        rna = mdata["rna"]
        aoom.chunked_normalize_total(rna, target_sum=1.0)
        aoom.chunked_log1p(rna)
        sample = np.asarray(rna.X[:2])
        assert sample.shape == (2, rna.shape[1])
        assert np.all(np.isfinite(sample))


def test_mdata_subscript_returns_anndata_oom(tiny_mudata):
    h5mu_path, _ = tiny_mudata
    with moom.read_h5mu(h5mu_path) as mdata:
        rna = mdata["rna"]
        assert isinstance(rna, aoom.AnnDataOOM)
        prot = mdata["prot"]
        assert isinstance(prot, aoom.AnnDataOOM)
