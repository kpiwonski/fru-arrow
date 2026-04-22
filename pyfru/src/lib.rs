use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_rust")]
mod pyfru {
    use minarrow::{Array, CategoricalArray, FloatArray, NumericArray, TextArray};
    use minarrow_pyo3::{PyArray, PyRecordBatch};
    use pyo3::{prelude::*, types::PyBytes};

    #[pyclass]
    struct RandomForest(fru::RandomForest);

    #[pymethods]
    impl RandomForest {
        #[new]
        #[pyo3(signature = (df, y, trees, tries, save_forest, importance, oob, seed, classifier, threads=None))]
        fn fit(
            df: PyRecordBatch,
            y: PyArray,
            trees: usize,
            tries: Option<usize>,
            save_forest: bool,
            importance: bool,
            oob: bool,
            seed: u64,
            classifier: bool,
            threads: Option<usize>,
        ) -> Self {
            let df = df.into_inner();
            let tries = tries.unwrap_or((df.n_cols() as f64).sqrt().round() as usize);
            if classifier {
                if let Array::TextArray(arr) = y.into_inner().array.clone() {
                    let y_array = match arr {
                        TextArray::Categorical8(x) => Into::<CategoricalArray<u64>>::into(&*x),
                        TextArray::Categorical16(x) => Into::<CategoricalArray<u64>>::into(&*x),
                        TextArray::Categorical32(x) => Into::<CategoricalArray<u64>>::into(&*x),
                        TextArray::Categorical64(x) => (*x).clone(),
                        _ => unreachable!("Decision is not categorical"),
                    };

                    return RandomForest(fru::RandomForest::fit(
                        df,
                        y_array.into(),
                        trees,
                        tries,
                        save_forest,
                        importance,
                        oob,
                        seed,
                        threads,
                    ));
                } else {
                    unreachable!("Decision is not categorical")
                }
            } else {
                if let Array::NumericArray(arr) = y.into_inner().array.clone() {
                    let y_array = match arr {
                        NumericArray::Int8(x) => Into::<FloatArray<f64>>::into(&*x),
                        NumericArray::Int16(x) => Into::<FloatArray<f64>>::into(&*x),
                        NumericArray::Int32(x) => Into::<FloatArray<f64>>::into(&*x),
                        NumericArray::Int64(x) => Into::<FloatArray<f64>>::into(&*x),
                        NumericArray::UInt8(x) => Into::<FloatArray<f64>>::into(&*x),
                        NumericArray::UInt16(x) => Into::<FloatArray<f64>>::into(&*x),
                        NumericArray::UInt32(x) => Into::<FloatArray<f64>>::into(&*x),
                        NumericArray::UInt64(x) => Into::<FloatArray<f64>>::into(&*x),
                        NumericArray::Float32(x) => Into::<FloatArray<f64>>::into(&*x),
                        NumericArray::Float64(x) => (*x).clone(),
                        _ => unreachable!("Decision is not numeric"),
                    };

                    RandomForest(fru::RandomForest::fit(
                        df,
                        y_array.into(),
                        trees,
                        tries,
                        save_forest,
                        importance,
                        oob,
                        seed,
                        threads,
                    ))
                } else {
                    unreachable!("Decision is not categorical")
                }
            }
        }

        fn importance(&self, normalised: bool) -> PyRecordBatch {
            self.0.importance(normalised).into()
        }

        #[pyo3(signature = (x, seed, validate_input, threads=None))]
        fn predict(
            &self,
            x: PyRecordBatch,
            seed: u64,
            validate_input: bool,
            threads: Option<usize>,
        ) -> PyArray {
            self.0
                .predict(x.into_inner(), seed, validate_input, threads)
                .into()
        }

        fn oob(&self, seed: u64) -> PyArray {
            self.0.oob(seed).into()
        }

        #[pyo3(signature = (x, validate_input, threads=None))]
        fn predict_votes(
            &self,
            x: PyRecordBatch,
            validate_input: bool,
            threads: Option<usize>,
        ) -> PyRecordBatch {
            self.0
                .predict_votes(x.into(), validate_input, threads)
                .into()
        }

        fn oob_votes(&self) -> PyRecordBatch {
            self.0.oob_votes().into()
        }

        fn to_bytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
            Ok(PyBytes::new(
                py,
                &self
                    .0
                    .to_bytes()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?,
            )
            .into())
        }

        #[staticmethod]
        fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
            Ok(Self(fru::RandomForest::from_bytes(bytes).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}"))
            })?))
        }
    }
}
