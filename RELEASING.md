# Releasing `mudataoom`

## One-time setup — PyPI Trusted Publishing (no API token in repo secrets)

The release workflow uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC), so we don't need to manage an `API_TOKEN` secret. Configure
this once, then every tagged release auto-publishes.

1. Sign in at https://pypi.org as a maintainer of the `mudataoom`
   project (or, on first release, register the name with a manual
   `twine upload` of one wheel — PyPI requires the project to exist
   before adding a trusted publisher).
2. Go to **Manage → Publishing → Add a new publisher → GitHub**.
3. Fill in:

| Field | Value |
|---|---|
| PyPI Project Name | `mudataoom` |
| Owner | `omicverse` |
| Repository name | `mudata-oom` |
| Workflow filename | `release.yml` |
| Environment name | `pypi` |

4. Save. Done — PyPI now trusts GitHub Actions runs from `omicverse/mudata-oom`
   on workflow `release.yml` inside environment `pypi`.

The GitHub side already has `environment: name: pypi` and
`permissions: id-token: write` set in `release.yml`, so no further
GitHub config is needed.

## Cutting a release

```bash
# 1. Bump version (in 3 places that must stay in sync):
vim pyproject.toml                              # [project] version
vim Cargo.toml                                  # [package] version
vim python/mudataoom/__init__.py                # __version__

# 2. Commit + tag
git commit -am "release v0.2.0"
git tag v0.2.0
git push origin main --tags
```

The push of `v*` tag fires `.github/workflows/release.yml`, which:

- Builds wheels for Linux × {x86_64} × py{3.9,…,3.13}
- Builds wheels for macOS × {aarch64, x86_64} × py{3.9,…,3.13}
- Builds wheels for Windows × {x64} × py{3.9,…,3.13}
- Builds the sdist
- Uploads all artefacts to PyPI via OIDC.

The matrix takes ~40 min total; watch progress at
https://github.com/omicverse/mudata-oom/actions.

## What happens to `mudata-rs` (the Rust workspace)?

`mudata-rs` is a **source-only** sibling repo (analogous to `anndata-rs`).
The Python wheel pulls it in at build time via `Cargo.toml` path
dependency, then statically links the compiled extension into
`mudataoom._backend.cpython-...so`. End users never need to install or
build `mudata-rs` themselves.

If you want to publish the `mudata` core crate to crates.io as well
(so other Rust projects can depend on it), there is a corresponding
`release.yml` over in `omicverse/mudata-rs` that triggers on tag push
and requires a `CARGO_REGISTRY_TOKEN` repo secret.

## Sanity-check before pushing the tag

```bash
# Build wheel locally, install into a fresh venv, run smoke tests
maturin build --release --features extension-module --out wheels
python -m venv /tmp/venv && /tmp/venv/bin/pip install \
    --force-reinstall --no-deps wheels/mudataoom-*.whl
/tmp/venv/bin/pip install anndata anndataoom mudata h5py numpy pandas scipy pytest
/tmp/venv/bin/python -m pytest tests/ -q
```

If those pass, the GH Actions matrix is essentially guaranteed to pass
too (modulo platform-specific HDF5 link issues, which the maturin-action
manylinux2_17 baseline tends to absorb).
