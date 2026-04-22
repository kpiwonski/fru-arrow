use minarrow::{
    Array, CategoricalArray, FieldArray, FloatArray, IntegerArray, NumericArray, StringArray,
    Table, TextArray,
};
use thiserror::Error;
use xrf::{Forest, ImportanceAggregator, Prediction, RfRng, Walk};

mod attribute;

mod classification;
mod serialize;
use classification::DataFrame as DataFrameClassification;
use regression::DataFrame as DataFrameRegression;
mod regression;
pub mod tools;

use classification::ClsDecisionBasicType;

use crate::serialize::{
    ArrayDType, CategoricalUniqueValues, SerializedForest, SerializedForestClassificationV1,
    SerializedForestRegressionV1, WalkClassification, WalkRegression,
    get_categorical_features_unique_values,
};

pub struct RandomForestRegressor {
    forest: Forest<DataFrameRegression>,
    ncol: usize,
    train_nrow: usize,
    col_names: Vec<String>,
    col_dtypes: Vec<ArrayDType>,
    categorical_unique_values: CategoricalUniqueValues,
}

pub struct RandomForestClassifier {
    forest: Forest<DataFrameClassification>,
    decision_unique_values: Vec<String>,
    ncol: usize,
    train_nrow: usize,
    col_names: Vec<String>,
    col_dtypes: Vec<ArrayDType>,
    categorical_unique_values: CategoricalUniqueValues,
}

pub enum RandomForest {
    Classifier(RandomForestClassifier),
    Regressor(RandomForestRegressor),
}

macro_rules! map_either {
    ($value:expr, $pattern:pat => $result:expr) => {
        match $value {
            RandomForest::Classifier($pattern) => $result,
            RandomForest::Regressor($pattern) => $result,
        }
    };
}

impl RandomForest {
    pub fn importance_raw(&self, normalised: bool) -> Vec<(usize, f64)> {
        map_either!(self, rf => {
            if !rf.forest.has_importance() {
                panic!("Importance was not calculated for the forest")
            }
        });

        let trees = self.trees();
        let f = |(feature, value): (usize, &ImportanceAggregator)| {
            if normalised {
                value.value_normalised(trees).map(|val| (feature, val))
            } else {
                Some((feature, value.value(trees)))
            }
        };

        map_either!(self, rf => rf.forest.raw_importance().filter_map(f).collect())
    }

    pub fn importance(&self, normalised: bool) -> Table {
        let raw_imp = self.importance_raw(normalised);
        let mut imp = vec![f64::NAN; self.ncol()];

        for (i, val) in raw_imp {
            imp[i] = val;
        }

        let col_names = map_either!(
            self, rf => FieldArray::from_arr(
                "column",
                Array::from_string64(
                    StringArray::from_vec(rf.col_names.iter().map(|x| x.as_str()).collect::<Vec<_>>(), None)
                )
            )
        );
        let imp_arr = FieldArray::from_arr(
            "importance",
            Array::from_float64(FloatArray::from_slice(&imp)),
        );
        Table::new("importance".into(), vec![col_names, imp_arr].into())
    }

    pub fn oob(&self, seed: u64) -> Array {
        map_either!(self, rf => {
            if !rf.forest.has_oob() {
                panic!("OOB was not calculated for the forest")
            }
        });

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
                Array::from_categorical64(CategoricalArray::from_slices(
                    &res,
                    &rf.decision_unique_values,
                ))
            }
            RandomForest::Regressor(rf) => {
                let mut oob: Vec<_> = rf.forest.oob().map(|(e, v)| (e, v.collapse())).collect();

                oob.sort_unstable_by_key(|(k, _)| *k);
                let res: Vec<_> = oob.into_iter().map(|(_, x)| x).collect();
                Array::from_float64(FloatArray::from_slice(&res))
            }
        }
    }

    pub fn oob_votes_raw(&self) -> impl Iterator<Item = (usize, usize, usize)> {
        match self {
            RandomForest::Classifier(rf) => {
                if !rf.forest.has_oob() {
                    panic!("OOB was not calculated for the forest")
                }

                rf.forest.oob().flat_map(|(e, v)| {
                    v.0.clone()
                        .into_iter()
                        .enumerate()
                        .map(move |(i, x)| (e, i, x))
                })
            }
            RandomForest::Regressor(_) => unreachable!("Votes for regression are not supported"),
        }
    }

    pub fn oob_votes(&self) -> Table {
        let votes: Vec<_> = self.oob_votes_raw().collect();

        let max_row = votes.iter().map(|(r, _, _)| *r).max().unwrap_or(0);
        let mut col_arrays = vec![vec![0; max_row + 1]; self.ncat()];

        for (r, c, val) in votes {
            col_arrays[c][r] = val as u64;
        }
        let decision_unique_values = self.decision_unique_values();

        Table::new(
            "oob_votes".into(),
            col_arrays
                .into_iter()
                .enumerate()
                .map(|(i, x)| {
                    FieldArray::from_arr(
                        decision_unique_values[i].clone(),
                        Array::from_uint64(IntegerArray::<u64>::from_slice(&x)),
                    )
                })
                .collect::<Vec<_>>()
                .into(),
        )
    }

    pub fn fit(
        x: Table,
        y: Array,
        trees: usize,
        tries: usize,
        save_forest: bool,
        importance: bool,
        oob: bool,
        seed: u64,
        threads: Option<usize>,
    ) -> Self {
        if trees == 0 {
            panic!("Trees parameter must be greater than 0")
        }

        if tries == 0 {
            panic!("Tries parameter must be greater than 0")
        }

        if tries > x.n_cols() {
            panic!("Tries cannot be greater than ncol")
        }

        if x.n_rows() == 0 {
            panic!("Data frame cannot be empty")
        }

        if x.n_rows() != y.len() {
            panic!("Decision length must be equal number of data frame rows")
        }

        let categorical_unique_values = get_categorical_features_unique_values(&x);
        let ncol = x.n_cols();
        let train_nrow = x.n_rows();
        let col_names: Vec<_> = x.col_names().iter().map(|x| x.to_string()).collect();
        let col_dtypes: Vec<_> = x
            .cols
            .iter()
            .map(|col| ArrayDType::from_field_array(col))
            .collect();
        if let Array::NumericArray(NumericArray::Float64(y)) = y {
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
            return RandomForest::Regressor(RandomForestRegressor {
                forest,
                ncol,
                train_nrow,
                col_names,
                col_dtypes,
                categorical_unique_values,
            });
        }
        if let Array::TextArray(TextArray::Categorical64(y)) = y {
            let decision_unique_values = y.unique_values.to_vec();
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
            return RandomForest::Classifier(RandomForestClassifier {
                forest,
                decision_unique_values,
                ncol,
                train_nrow,
                col_names,
                col_dtypes,
                categorical_unique_values,
            });
        }

        panic!("Unsupported decision type");
    }

    fn validate_x(&self, x: Table) -> Table {
        map_either!(self, rf => {
            x.schema()
                .iter()
                .zip(rf.col_names.iter())
                .for_each(|(field, reference_name)| {
                    if &field.name != reference_name {
                        panic!("Column names do not match")
                    }
                });
            x.cols
                .iter()
                .zip(rf.col_dtypes.iter())
                .for_each(|(field, reference_dtype)| {
                    if &ArrayDType::from_field_array(field) != reference_dtype {
                        panic!("Column types do not match")
                    }
                });
            let categorical_unique_values = get_categorical_features_unique_values(&x);
            categorical_unique_values.iter()
                .zip(rf.categorical_unique_values.iter())
                .for_each(|(val, ref_val)| {
                    if val != ref_val {
                        panic!("Categorical features unique values do not match")
                    }
                }
            );
            x
        })
    }

    pub fn predict(
        &self,
        x: Table,
        seed: u64,
        validate_input: bool,
        threads: Option<usize>,
    ) -> Array {
        let x = if validate_input {
            self.validate_x(x)
        } else {
            x
        };

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
                Array::from_categorical64(CategoricalArray::from_slices(
                    &res,
                    &rf.decision_unique_values,
                ))
            }
            RandomForest::Regressor(rf) => {
                let df = DataFrameRegression::new(x, FloatArray::default());
                let pred = rf
                    .forest
                    .predict_parallel(&df, RandomForest::get_threads(threads));
                let mut pred: Vec<_> = pred.predictions().map(|(e, v)| (e, v.collapse())).collect();

                pred.sort_unstable_by_key(|(k, _)| *k);
                let res: Vec<_> = pred.into_iter().map(|(_, x)| x).collect();
                Array::from_float64(FloatArray::from_slice(&res))
            }
        }
    }

    pub fn predict_votes_raw(
        &self,
        x: Table,
        validate_input: bool,
        threads: Option<usize>,
    ) -> Prediction<DataFrameClassification> {
        let x = if validate_input {
            self.validate_x(x)
        } else {
            x
        };

        match self {
            RandomForest::Classifier(rf) => {
                let df = DataFrameClassification::new(
                    x,
                    CategoricalArray::default(),
                    rf.decision_unique_values.len() as ClsDecisionBasicType,
                );
                rf.forest
                    .predict_parallel(&df, RandomForest::get_threads(threads))
            }
            RandomForest::Regressor(_) => unreachable!("Votes for regression are not supported"),
        }
    }

    pub fn predict_votes(&self, x: Table, validate_input: bool, threads: Option<usize>) -> Table {
        let nrows = x.n_rows();
        let votes: Vec<_> = self
            .predict_votes_raw(x, validate_input, threads)
            .predictions()
            .flat_map(|(e, v)| {
                v.0.clone()
                    .into_iter()
                    .enumerate()
                    .map(move |(i, x)| (e, i, x))
            })
            .collect();

        let mut col_arrays = vec![vec![0; nrows]; self.ncat()];

        for (r, c, val) in votes {
            col_arrays[c][r] = val as u64;
        }
        let decision_unique_values = self.decision_unique_values();

        Table::new(
            "predict_votes".into(),
            col_arrays
                .into_iter()
                .enumerate()
                .map(|(i, x)| {
                    FieldArray::from_arr(
                        decision_unique_values[i].clone(),
                        Array::from_uint64(IntegerArray::<u64>::from_slice(&x)),
                    )
                })
                .collect::<Vec<_>>()
                .into(),
        )
    }

    pub fn serialize(&self) -> Result<SerializedForest, FruError> {
        match self {
            RandomForest::Classifier(rfc) => {
                if !rfc.forest.has_trees() {
                    Err(FruError::NoForestSerializeEror)?
                }
                let nodes = rfc
                    .forest
                    .walk()
                    .map(|walk| Into::<WalkClassification>::into(walk))
                    .collect();
                let oob = rfc.forest.oob().map(|(_, v)| v.clone()).collect();
                let importance: Vec<_> = rfc
                    .forest
                    .raw_importance()
                    .map(|(id, agg)| (id, agg.into_raw()))
                    .collect();
                Ok(SerializedForest::V1Classification(
                    SerializedForestClassificationV1::new(
                        nodes,
                        rfc.decision_unique_values.clone(),
                        rfc.ncol,
                        rfc.train_nrow,
                        oob,
                        importance,
                        rfc.col_names.clone(),
                        rfc.col_dtypes.clone(),
                        rfc.categorical_unique_values.clone(),
                    ),
                ))
            }
            RandomForest::Regressor(rfr) => {
                if !rfr.forest.has_trees() {
                    Err(FruError::NoForestSerializeEror)?
                }

                let nodes = rfr
                    .forest
                    .walk()
                    .map(|walk| Into::<WalkRegression>::into(walk))
                    .collect();
                let oob = rfr.forest.oob().map(|(_, v)| v.clone()).collect();
                let importance: Vec<_> = rfr
                    .forest
                    .raw_importance()
                    .map(|(id, agg)| (id, agg.into_raw()))
                    .collect();

                Ok(SerializedForest::V1Regression(
                    SerializedForestRegressionV1::new(
                        nodes,
                        rfr.ncol,
                        rfr.train_nrow.clone(),
                        oob,
                        importance,
                        rfr.col_names.clone(),
                        rfr.col_dtypes.clone(),
                        rfr.categorical_unique_values.clone(),
                    ),
                ))
            }
        }
    }

    pub fn deserialize(serialized_forest: SerializedForest) -> Self {
        match serialized_forest {
            SerializedForest::V1Classification(sfc) => {
                let nodes = sfc
                    .nodes
                    .iter()
                    .map(|node| Into::<Walk<DataFrameClassification>>::into(node));
                let mut forest = Forest::from_walk(nodes).unwrap();
                forest.replace_oob(sfc.oob.into_iter());
                let importance = sfc
                    .importance_raw
                    .iter()
                    .map(|(id, imp_raw)| (*id, ImportanceAggregator::from_raw(imp_raw)));
                forest.replace_importance(importance);

                RandomForest::Classifier(RandomForestClassifier {
                    forest: forest,
                    decision_unique_values: sfc.decision_unique_values,
                    ncol: sfc.ncol,
                    train_nrow: sfc.train_nrow,
                    col_names: sfc.col_names,
                    col_dtypes: sfc.col_dtypes,
                    categorical_unique_values: sfc.categorical_unique_values,
                })
            }
            SerializedForest::V1Regression(sfr) => {
                let nodes = sfr
                    .nodes
                    .iter()
                    .map(|node| Into::<Walk<DataFrameRegression>>::into(node));
                let mut forest = Forest::from_walk(nodes).unwrap();
                forest.replace_oob(sfr.oob.into_iter());

                let importance = sfr
                    .importance_raw
                    .iter()
                    .map(|(id, imp_raw)| (*id, ImportanceAggregator::from_raw(imp_raw)));
                forest.replace_importance(importance);

                RandomForest::Regressor(RandomForestRegressor {
                    forest: forest,
                    ncol: sfr.ncol,
                    train_nrow: sfr.train_nrow,
                    col_names: sfr.col_names,
                    col_dtypes: sfr.col_dtypes,
                    categorical_unique_values: sfr.categorical_unique_values,
                })
            }
        }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, FruError> {
        Ok(postcard::to_allocvec(&self.serialize()?)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FruError> {
        let serialized_forest: SerializedForest = postcard::from_bytes(bytes)?;
        Ok(Self::deserialize(serialized_forest))
    }

    pub fn trees(&self) -> usize {
        map_either!(self, rf =>  rf.forest.trees())
    }

    pub fn ncol(&self) -> usize {
        map_either!(self, rf =>  rf.ncol)
    }

    pub fn train_nrow(&self) -> usize {
        map_either!(self, rf =>  rf.train_nrow)
    }

    pub fn ncat(&self) -> usize {
        match self {
            RandomForest::Classifier(rf) => rf.decision_unique_values.len(),
            RandomForest::Regressor(_) => unreachable!("No categorical values for regression"),
        }
    }

    fn decision_unique_values(&self) -> Vec<String> {
        match self {
            RandomForest::Classifier(rf) => rf.decision_unique_values.clone(),
            RandomForest::Regressor(_) => unreachable!("No categorical values for regression"),
        }
    }

    fn get_threads(threads: Option<usize>) -> usize {
        if let Some(val) = threads {
            if val == 0 {
                panic!("Number of threads have to greater than 0")
            }
            val
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        }
    }
}

#[derive(Error, Debug)]
pub enum FruError {
    #[error("Cannot serialize model, when forest is not saved")]
    NoForestSerializeEror,
    #[error("Postcard error while converting model to/from bytes: {0}")]
    PostcardError(#[from] postcard::Error),
}
