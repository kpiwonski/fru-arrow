//! Fru is a highly performant implementation of the **Random Forest** model.
//! It provides permutation importance using a novel, optimized algorithm.
//! It supports both **classification** and **regression**, as well as out-of-bag predictions.

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
#[doc(hidden)]
pub mod tools;

use classification::ClsDecisionBasicType;

use crate::serialize::{
    ArrayDType, CategoricalUniqueValues, SerializedForest, SerializedForestClassificationV1,
    SerializedForestRegressionV1, WalkClassification, WalkRegression,
    get_categorical_features_unique_values,
};

#[doc(hidden)]
pub struct RandomForestRegressor {
    forest: Forest<DataFrameRegression>,
    ncol: usize,
    train_nrow: usize,
    col_names: Vec<String>,
    col_dtypes: Vec<ArrayDType>,
    categorical_unique_values: CategoricalUniqueValues,
}

#[doc(hidden)]
pub struct RandomForestClassifier {
    forest: Forest<DataFrameClassification>,
    decision_unique_values: Vec<String>,
    ncol: usize,
    train_nrow: usize,
    col_names: Vec<String>,
    col_dtypes: Vec<ArrayDType>,
    categorical_unique_values: CategoricalUniqueValues,
}

/// The main structure for the Random Forest model.
///
/// The Random Forest Classifier selects the best feature at each split using
/// Gini impurity. The Random Forest Regressor selects the best feature at each split using
/// variance reduction. For numerical features, the threshold is optimized by an
/// exhaustive scan and chosen as the midpoint between adjacent values. In case
/// of ties, the smaller threshold is selected.
///
/// Ordered categorical variables are currently treated as categorical. Variables
/// with six or more levels are treated as integers. Variables with five or fewer
/// levels are split by exhaustively evaluating all possible partitions into two
/// subsets using the Gini criterion.       
///
/// The maximum tree depth is fixed at 512. A leaf is created when the sample size
/// reaches one. Leaves may also be formed earlier if no valid split is found; in
/// such cases, ties are broken randomly.
///
/// Pyfru uses its own PRNG (pcg32, by Melissa E. O'Neill) to provide reproducible
/// results in parallel settings. For a given input and seed, the same trees are
/// built regardless of the number of threads, although their order may differ.
/// OOB predictions and importance scores are therefore consistent up to
/// numerical precision.
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
    /// Returns the raw permutation importance.
    ///
    /// # Arguments
    /// * `normalised` - If true, values are scaled by their standard deviation
    ///   across the ensemble.
    ///
    /// # Returns
    /// A vector of tuples `(usize, f64)` where:
    /// * The first element is the column index.
    /// * The second element is the permutation importance (mean decrease in accuracy).
    ///
    /// # Notes
    /// Other packages often scale importance by its standard error estimate,
    /// resulting in values larger by a factor of the square root of the number
    /// of trees compared to Fru.
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

    /// Returns permutation importance as a table.
    ///
    /// # Arguments
    /// * `normalised` - If true, values are scaled by their standard deviation
    ///   across the ensemble.
    ///
    /// # Returns
    /// A table with two columns:
    /// * `column` - the column name for which importance was calculated.
    /// * `importance` - the permutation importance (mean decrease in accuracy).
    ///
    /// # Notes
    /// Other packages often scale importance by its standard error estimate,
    /// resulting in values larger by a factor of the square root of the number
    /// of trees compared to Fru.
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

    /// Returns out-of-bag predictions for the model.
    ///
    /// # Arguments
    /// * `seed` - Seed for the PRNG (pcg32) used to break ties, if they occur.
    ///
    /// # Returns
    /// A categorical array containing out-of-bag predictions.
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

    /// Returns raw out-of-bag votes for the model.
    /// Votes for regression are not supported.
    ///
    /// # Returns
    /// An iterator over tuples `(usize, usize, usize)` where:
    /// * The first element is the row index in the training data.
    /// * The second element is the class index.
    /// * The third element is the number of votes.
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

    /// Returns out-of-bag votes for the model.
    /// Votes for regression are not supported.
    ///
    /// # Returns
    /// A table with the same number of rows as the training data frame
    /// and a number of columns equal to the number of target categories.
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

    /// Builds a forest from training data.
    ///
    /// # Arguments
    /// * `x` - minarrow `Table`. Must not contain `NA` values.
    /// * `y` - minarrow `Array`. Must not contain `NA` values.
    /// * `trees` - Number of trees in the forest.
    /// * `tries` - Number of features to try at each split (often called `mtry`).
    ///   Must be greater than zero and less than or equal to the number of features.
    ///   A common default is the square root of the number of columns.
    /// * `save_forest` - If `true`, the fitted forest is stored and can be used
    ///   for prediction or serialization. Set to `false` when only importance
    ///   or OOB results are needed.
    /// * `importance` - If `true`, feature importance is calculated (adds overhead).
    /// * `oob` - If `true`, out-of-bag predictions are calculated.
    /// * `seed` - Random seed used by the algorithm.
    /// * `threads` - Number of threads to use. Must be greater than zero.
    ///   If `None`, all available CPU cores are used.
    #[allow(clippy::too_many_arguments)]
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
        let col_dtypes: Vec<_> = x.cols.iter().map(ArrayDType::from_field_array).collect();
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

    /// Predicts for `x`. For classification, the predicted class is the one with
    /// the highest number of votes across the trees. For regression, predictions
    /// are the mean across the ensemble.
    ///
    /// # Arguments
    /// * `x` - A `Table`.
    /// * `seed` - Seed for the PRNG used to break ties in classification.
    /// * `validate_input` - If `true`, column data types and names in `x` are
    ///   checked against the data provided to `fit`. For categorical columns,
    ///   category names and their order must match.
    /// * `threads` - Number of threads to use. Must be greater than zero.
    ///   If `None`, all available CPU cores are used.
    ///
    /// # Returns
    /// An array with predictions.
    ///
    /// # Notes
    /// Voting in classification may result in ties. In such cases, a PRNG is used
    /// to break ties, making classification deterministic for a given seed.
    /// Regression uses leaf averages and is deterministic, aside from minor
    /// numerical differences due to multithreading and tree construction order.
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

    /// Predicts for `x`, returning per-tree votes instead of final predictions.
    ///
    /// # Arguments
    /// * `x` - A `Table`.
    /// * `validate_input` - If `true`, column data types and names in `x` are
    ///   checked against the data provided to `fit`. For categorical columns,
    ///   category names and their order must match.
    /// * `threads` - Number of threads to use. Must be greater than zero.
    ///   If `None`, all available CPU cores are used.
    ///
    /// # Returns
    /// A `Prediction` structure containing all votes.
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

    /// Predicts for `x`. Instead of final predictions, returns votes from each tree.
    /// # Arguments
    /// * `x` - A `Table`.
    /// * `validate_input` - If `true`, column data types and names in `x` are
    ///   checked against the data provided to `fit`. For categorical columns,
    ///   category names and their order must match.
    /// * `threads` - Number of threads to use. Must be greater than zero.
    ///   If `None`, all available CPU cores are used.
    ///
    /// # Returns
    /// A table with the same number of rows as the input data frame
    /// and a number of columns equal to the number of target categories.
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

    fn serialize(&self) -> Result<SerializedForest, FruError> {
        match self {
            RandomForest::Classifier(rfc) => {
                if !rfc.forest.has_trees() {
                    Err(FruError::NoForestSerializeEror)?
                }
                let nodes = rfc
                    .forest
                    .walk()
                    .map(Into::<WalkClassification>::into)
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
                    .map(Into::<WalkRegression>::into)
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
                        rfr.train_nrow,
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

    fn deserialize(serialized_forest: SerializedForest) -> Self {
        match serialized_forest {
            SerializedForest::V1Classification(sfc) => {
                let nodes = sfc
                    .nodes
                    .iter()
                    .map(Into::<Walk<DataFrameClassification>>::into);
                let mut forest = Forest::from_walk(nodes).unwrap();
                forest.replace_oob(sfc.oob.into_iter());
                let importance = sfc
                    .importance_raw
                    .iter()
                    .map(|(id, imp_raw)| (*id, ImportanceAggregator::from_raw(imp_raw)));
                forest.replace_importance(importance);

                RandomForest::Classifier(RandomForestClassifier {
                    forest,
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
                    .map(Into::<Walk<DataFrameRegression>>::into);
                let mut forest = Forest::from_walk(nodes).unwrap();
                forest.replace_oob(sfr.oob.into_iter());

                let importance = sfr
                    .importance_raw
                    .iter()
                    .map(|(id, imp_raw)| (*id, ImportanceAggregator::from_raw(imp_raw)));
                forest.replace_importance(importance);

                RandomForest::Regressor(RandomForestRegressor {
                    forest,
                    ncol: sfr.ncol,
                    train_nrow: sfr.train_nrow,
                    col_names: sfr.col_names,
                    col_dtypes: sfr.col_dtypes,
                    categorical_unique_values: sfr.categorical_unique_values,
                })
            }
        }
    }

    /// Serializes the forest into a `Vec<u8>` byte stream.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FruError> {
        Ok(postcard::to_allocvec(&self.serialize()?)?)
    }

    /// Deserializes a forest from a `u8` byte stream.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FruError> {
        let serialized_forest: SerializedForest = postcard::from_bytes(bytes)?;
        Ok(Self::deserialize(serialized_forest))
    }

    fn trees(&self) -> usize {
        map_either!(self, rf =>  rf.forest.trees())
    }

    fn ncol(&self) -> usize {
        map_either!(self, rf =>  rf.ncol)
    }

    fn ncat(&self) -> usize {
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
