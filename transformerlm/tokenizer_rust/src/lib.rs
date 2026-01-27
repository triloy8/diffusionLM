use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyIterator;
use pyo3::Bound;

pub mod tokenizer;

#[cfg(test)]
mod tests;

#[pyclass(name = "Tokenizer", module = "transformerlm.tokenizer_rust")]
pub struct PyTokenizer {
    inner: tokenizer::Tokenizer,
}

impl PyTokenizer {
    fn new(inner: tokenizer::Tokenizer) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTokenizer {
    #[staticmethod]
    #[pyo3(signature = (vocab_filepath, merges_filepath, special_tokens_path))]
    pub fn from_files(
        vocab_filepath: &str,
        merges_filepath: &str,
        special_tokens_path: &str,
    ) -> PyResult<Self> {
        let inner = tokenizer::Tokenizer::from_files(vocab_filepath, merges_filepath, special_tokens_path)?;
        Ok(Self::new(inner))
    }

    pub fn encode(&self, text: &str) -> PyResult<Vec<usize>> {
        self.inner
            .encode(text.to_string())
            .map_err(PyErr::from)
    }

    pub fn encode_iterable(&self, iterable: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
        let iterator: Bound<'_, PyIterator> = iterable.try_iter()?;
        let mut acc: Vec<usize> = Vec::new();
        for item in iterator {
            let line: String = item?.extract()?;
            let ids = self.inner.encode(line).map_err(PyErr::from)?;
            acc.extend(ids);
        }
        Ok(acc)
    }

    pub fn decode(&self, ids: Vec<usize>) -> PyResult<String> {
        self.inner.decode(ids).map_err(PyErr::from)
    }
}

impl From<tokenizer::TokenizerError> for PyErr {
    fn from(value: tokenizer::TokenizerError) -> Self {
        PyValueError::new_err(value.to_string())
    }
}

#[pymodule]
fn tokenizer_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    Ok(())
}
