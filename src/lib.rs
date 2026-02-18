use minarrow::{
    Array, CategoricalArray, FieldArray, FloatArray, NumericArray, Table, TextArray, Vec64,
};
use xrf::{Forest, RfRng};

mod attribute;

mod classification;
use classification::DataFrame as DataFrameClassification;
mod regression;

use crate::{classification::ClsDecisionBasicType, regression::DataFrameRegression};

pub struct RandomForestRegressor {
    forest: Forest<DataFrameRegression>,
}

pub struct RandomForestClassifier {
    forest: Forest<DataFrameClassification>,
    decision_unique_values: Vec64<String>,
}

pub enum RandomForest {
    Classifier(RandomForestClassifier),
    Regressor(RandomForestRegressor),
}

impl RandomForest {
    pub fn importance(&self) -> Vec<f64> {
        // TODO check if has importance
        match self {
            RandomForest::Classifier(rf) => {
                let mut imp: Vec<_> = rf.forest.importance().collect();
                imp.sort_unstable_by_key(|(k, _)| *k);
                imp.into_iter().map(|(_, x)| x).collect()
            }
            RandomForest::Regressor(rf) => {
                let mut imp: Vec<_> = rf.forest.importance().collect();
                imp.sort_unstable_by_key(|(k, _)| *k);
                imp.into_iter().map(|(_, x)| x).collect()
            }
        }
    }

    pub fn importance_normalised(&self) -> Vec<f64> {
        match self {
            RandomForest::Classifier(rf) => {
                let mut imp: Vec<_> = rf.forest.importance_normalised().collect();
                imp.sort_unstable_by_key(|(k, _)| *k);
                imp.into_iter().map(|(_, x)| x).collect()
            }
            RandomForest::Regressor(rf) => {
                let mut imp: Vec<_> = rf.forest.importance_normalised().collect();
                imp.sort_unstable_by_key(|(k, _)| *k);
                imp.into_iter().map(|(_, x)| x).collect()
            }
        }
    }

    pub fn oob(&self, seed: u64) -> FieldArray {
        match self {
            RandomForest::Classifier(rf) => {
                let mut rng = RfRng::from_seed(seed, 1);
                let mut oob: Vec<_> = rf
                    .forest
                    .oob()
                    .map(|(e, v)| (e, v.collapse_empty_random(&mut rng)))
                    .collect();

                oob.sort_unstable_by_key(|(k, _)| *k);
                let res: Vec<_> = oob.into_iter().map(|(_, x)| x).collect();
                FieldArray::from_arr(
                    "oob_prediction",
                    Array::from_categorical64(CategoricalArray::from_slices(
                        &res,
                        &rf.decision_unique_values,
                    )),
                )
            }
            RandomForest::Regressor(rf) => {
                let mut oob: Vec<_> = rf.forest.oob().map(|(e, v)| (e, v.collapse())).collect();

                oob.sort_unstable_by_key(|(k, _)| *k);
                let res: Vec<_> = oob.into_iter().map(|(_, x)| x).collect();
                FieldArray::from_arr(
                    "oob_prediction",
                    Array::from_float64(FloatArray::from_slice(&res)),
                )
            }
        }
    }

    pub fn oob_votes(&self) -> Vec<Vec<usize>> {
        match self {
            RandomForest::Classifier(rf) => {
                let mut oob: Vec<_> = rf.forest.oob().map(|(e, v)| (e, v.0.clone())).collect();
                oob.sort_unstable_by_key(|(k, _)| *k);
                oob.into_iter().map(|(_, x)| x).collect()
            }
            RandomForest::Regressor(_) => {
                unreachable!("Votes for regression are not supported")
            }
        }
    }

    pub fn fit(
        x: Table,
        y: FieldArray,
        trees: usize,
        tries: usize,
        save_forest: bool,
        importance: bool,
        oob: bool,
        seed: u64,
        threads: Option<usize>,
    ) -> Self {
        match y.array {
            Array::NumericArray(num) => match num {
                NumericArray::Float64(y) => {
                    let df = DataFrameRegression::new(x, (*y).clone());

                    let forest = Forest::new_parallel(
                        &df,
                        trees,
                        tries,
                        save_forest,
                        importance,
                        oob,
                        seed,
                        RandomForest::get_threads(threads),
                    );
                    RandomForest::Regressor(RandomForestRegressor { forest })
                }
                _ => panic!("Unsupported decision type"),
            },
            Array::TextArray(arr) => match arr {
                TextArray::Categorical64(y) => {
                    let decision_unique_values = y.unique_values.clone();
                    let df = DataFrameClassification::new(
                        x,
                        (*y).clone(),
                        decision_unique_values.len() as ClsDecisionBasicType,
                    );
                    let forest = Forest::new_parallel(
                        &df,
                        trees,
                        tries,
                        save_forest,
                        importance,
                        oob,
                        seed,
                        RandomForest::get_threads(threads),
                    );
                    RandomForest::Classifier(RandomForestClassifier {
                        forest,
                        decision_unique_values,
                    })
                }
                _ => panic!("Unsupported decision type"),
            },
            _ => panic!("Unsupported decision type"),
        }
    }

    pub fn predict(&self, x: Table, seed: u64, threads: Option<usize>) -> FieldArray {
        match self {
            RandomForest::Classifier(rf) => {
                let mut rng = RfRng::from_seed(seed, 1);
                let df = DataFrameClassification::new(
                    x,
                    CategoricalArray::default(),
                    rf.decision_unique_values.len() as ClsDecisionBasicType,
                );
                let pred = rf
                    .forest
                    .predict_parallel(&df, RandomForest::get_threads(threads));
                let mut pred: Vec<_> = pred
                    .predictions()
                    .map(|(e, v)| (e, v.collapse_empty_random(&mut rng)))
                    .collect();

                pred.sort_unstable_by_key(|(k, _)| *k);
                let res: Vec<_> = pred.into_iter().map(|(_, x)| x).collect();
                FieldArray::from_arr(
                    "prediction",
                    Array::from_categorical64(CategoricalArray::from_slices(
                        &res,
                        &rf.decision_unique_values,
                    )),
                )
            }
            RandomForest::Regressor(rf) => {
                let df = DataFrameRegression::new(x, FloatArray::default());
                let pred = rf
                    .forest
                    .predict_parallel(&df, RandomForest::get_threads(threads));
                let mut pred: Vec<_> = pred.predictions().map(|(e, v)| (e, v.collapse())).collect();

                pred.sort_unstable_by_key(|(k, _)| *k);
                let res: Vec<_> = pred.into_iter().map(|(_, x)| x).collect();
                FieldArray::from_arr(
                    "prediction",
                    Array::from_float64(FloatArray::from_slice(&res)),
                )
            }
        }
    }

    pub fn predict_votes(&self, x: Table, threads: Option<usize>) -> Vec<Vec<usize>> {
        match self {
            RandomForest::Classifier(rf) => {
                let df = DataFrameClassification::new(
                    x,
                    CategoricalArray::default(),
                    rf.decision_unique_values.len() as ClsDecisionBasicType,
                );
                let mut pred: Vec<_> = rf
                    .forest
                    .predict_parallel(&df, RandomForest::get_threads(threads))
                    .predictions()
                    .map(|(e, v)| (e, v.0.clone()))
                    .collect();

                pred.sort_unstable_by_key(|(k, _)| *k);
                pred.into_iter().map(|(_, x)| x).collect()
            }
            RandomForest::Regressor(_) => {
                unreachable!("Votes for regression are not supported")
            }
        }
    }

    fn get_threads(threads: Option<usize>) -> usize {
        threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
    }
}
