//! mudataoom._backend — Python extension module exposing mudata-rs bindings.
//!
//! Mirrors `anndataoom`'s tiny `lib.rs`: a single `#[pymodule]` that
//! delegates registration to a sibling Rust library crate
//! (`pymudata`). The compiled cdylib (`_backend.so`) is bundled into
//! the `mudataoom` Python wheel by maturin, so `pip install mudataoom`
//! gives users the Rust backend without a separate `pymudata` PyPI
//! package.

use pyo3::{prelude::*, pymodule, types::PyModule, PyResult};

#[pymodule]
fn _backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pymudata::register(m)
}
