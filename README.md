# mudata-oom

**Out-of-memory `MuData` powered by [anndataoom](https://github.com/Starlitnightly/anndata-oom).**

Drop-in replacement for `mudata.MuData` whose per-modality expression matrices
stay on disk. Every API method, every property, every behaviour of upstream
`mudata.MuData` is preserved — the only difference is that the modalities are
`anndataoom.AnnDataOOM` objects instead of `anndata.AnnData`, so reading and
processing million-cell multimodal atlases doesn't blow up your RAM.

```python
import mudataoom as moom
mdata = moom.read_h5mu("cite_seq_1M.h5mu")
# MuDataOOM [out-of-memory · backed]
#   mod
#     rna  : AnnDataOOM  1,200,000 × 24,000
#     prot : AnnDataOOM  1,200,000 × 156
#   obs  : 1,200,000 × 5
#   obsm : ['X_mofa']
```

## Why?

`mudata.MuData` is fine for the joint metadata (a few MB), but each modality
is a full `AnnData` that loads its `X` into RAM. For a 1M-cell CITE-seq atlas
(RNA + protein), this means **>100 GB**. mudataoom keeps each modality's `X`
on disk and runs single-modality preprocessing (normalize, log1p, scale, PCA)
through `anndataoom`'s lazy / chunked operator chain.

| Dataset                          | `mudata.MuData` | `mudataoom` | Saving |
|----------------------------------|----------------:|------------:|-------:|
| CITE-seq 10k (RNA + 156 ADT)     | 2.4 GB          | **~70 MB**  | 34x    |
| RNA + ATAC 100k cells            | ~24 GB          | **~1.2 GB** | 20x    |
| RNA + ATAC 1M cells              | ~140 GB (OOM)   | **~1.7 GB** | 80x+   |

## Architecture (no new Rust code)

```
MuDataOOM (Python; subclasses mudata.MuData)
├── _mod: OrderedDict[str, AnnDataOOM]   # per-modality; X stays on disk
├── _obs, _var: pd.DataFrame             # joint metadata
├── _obsm, _varm, _obsp, _varp           # joint embeddings / graphs
├── _obsmap, _varmap                     # which modality row backs which joint row
├── _uns, _axis
└── _file: MuDataFileManager             # h5mu HDF5 file handle (backed mode)
```

`read_h5mu` walks `/mod/*` inside the source `.h5mu`, builds one tiny
"virtual" `.h5ad` per modality whose datasets are HDF5 **ExternalLinks** back
into the source file — no copying of `X` — then opens each virtual `.h5ad`
through `anndataoom.read()`. The Rust I/O layer streams `X` chunks straight
from the original `.h5mu` on demand.

See `docs/architecture.md` for the full design.

## Install

```bash
pip install mudataoom
```

(Will pull `anndataoom`, `mudata`, `h5py`, `anndata`.)

## Compatibility

`MuDataOOM` is a strict subclass of `mudata.MuData`. Anything that takes a
`mudata.MuData` (mofapy, muon plotting, scvi-tools, …) accepts a
`MuDataOOM` unchanged.

## License

MIT. See `LICENSE`.
