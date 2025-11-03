use pyo3::prelude::*;

#[pyclass]
pub struct Tokenizer;

#[pymethods]
impl Tokenizer {
    #[new]
    fn new() -> Self {
        Tokenizer
    }

    #[staticmethod]
    pub fn from_files() -> Self{
        todo!()
    }

    pub fn encode(&self) {
        todo!()
    }

    pub fn encode_iteralble(&self) {
        todo!()
    }

    pub fn decode(&self) {
        todo!()
    }
}