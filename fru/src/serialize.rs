use minarrow::{Array, FieldArray, NumericArray, Table, TextArray};
use serde::{Deserialize, Serialize};
use xrf::{RfInput, Walk};

use crate::classification::DataFrame as DataFrameClassification;
use crate::regression::DataFrame as DataFrameRegression;

type ImportanceRaw = (usize, [u8; 24]);

#[derive(Serialize, Deserialize)]
pub enum SerializedForest {
    V1Classification(SerializedForestClassificationV1),
    V1Regression(SerializedForestRegressionV1),
}

#[derive(Serialize, Deserialize)]
pub struct SerializedForestClassificationV1 {
    pub nodes: Vec<WalkClassification>,
    pub decision_unique_values: Vec<String>,
    pub ncol: usize,
    pub train_nrow: usize,
    pub oob: Vec<<DataFrameClassification as RfInput>::VoteAggregator>,
    pub importance_raw: Vec<ImportanceRaw>,
    pub col_names: Vec<String>,
    pub col_dtypes: Vec<ArrayDType>,
    pub categorical_unique_values: CategoricalUniqueValues,
}

#[derive(Serialize, Deserialize)]
pub struct SerializedForestRegressionV1 {
    pub nodes: Vec<WalkRegression>,
    pub ncol: usize,
    pub train_nrow: usize,
    pub oob: Vec<<DataFrameRegression as RfInput>::VoteAggregator>,
    pub importance_raw: Vec<ImportanceRaw>,
    pub col_names: Vec<String>,
    pub col_dtypes: Vec<ArrayDType>,
    pub categorical_unique_values: CategoricalUniqueValues,
}

impl SerializedForestClassificationV1 {
    pub fn new(
        nodes: Vec<WalkClassification>,
        decision_unique_values: Vec<String>,
        ncol: usize,
        train_nrow: usize,
        oob: Vec<<DataFrameClassification as RfInput>::VoteAggregator>,
        importance_raw: Vec<ImportanceRaw>,
        col_names: Vec<String>,
        col_dtypes: Vec<ArrayDType>,
        categorical_unique_values: CategoricalUniqueValues,
    ) -> Self {
        SerializedForestClassificationV1 {
            nodes,
            decision_unique_values,
            ncol,
            train_nrow,
            oob,
            importance_raw,
            col_names,
            col_dtypes,
            categorical_unique_values,
        }
    }
}

impl SerializedForestRegressionV1 {
    pub fn new(
        nodes: Vec<WalkRegression>,
        ncol: usize,
        train_nrow: usize,
        oob: Vec<<DataFrameRegression as RfInput>::VoteAggregator>,
        importance_raw: Vec<ImportanceRaw>,
        col_names: Vec<String>,
        col_dtypes: Vec<ArrayDType>,
        categorical_unique_values: CategoricalUniqueValues,
    ) -> Self {
        SerializedForestRegressionV1 {
            nodes,
            ncol,
            train_nrow,
            oob,
            importance_raw,
            col_names,
            col_dtypes,
            categorical_unique_values,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub enum WalkClassification {
    /// The currently visited vertex is a leaf in a decision tree
    VisitLeaf(<DataFrameClassification as RfInput>::Vote),
    /// The currently visited vertex is a branch in a decision tree
    VisitBranch(
        <DataFrameClassification as RfInput>::FeatureId,
        <DataFrameClassification as RfInput>::Pivot,
    ),
}

#[derive(Serialize, Deserialize)]
pub enum WalkRegression {
    /// The currently visited vertex is a leaf in a decision tree
    VisitLeaf(<DataFrameRegression as RfInput>::Vote),
    /// The currently visited vertex is a branch in a decision tree
    VisitBranch(
        <DataFrameRegression as RfInput>::FeatureId,
        <DataFrameRegression as RfInput>::Pivot,
    ),
}

impl From<Walk<DataFrameClassification>> for WalkClassification {
    fn from(value: Walk<DataFrameClassification>) -> Self {
        match value {
            Walk::VisitLeaf(vote) => WalkClassification::VisitLeaf(vote),
            Walk::VisitBranch(feature_id, pivot) => {
                WalkClassification::VisitBranch(feature_id, pivot)
            }
        }
    }
}

impl From<&WalkClassification> for Walk<DataFrameClassification> {
    fn from(value: &WalkClassification) -> Walk<DataFrameClassification> {
        match value {
            WalkClassification::VisitLeaf(vote) => Walk::VisitLeaf(*vote),
            WalkClassification::VisitBranch(feature_id, pivot) => {
                Walk::VisitBranch(*feature_id, pivot.clone())
            }
        }
    }
}

impl From<Walk<DataFrameRegression>> for WalkRegression {
    fn from(value: Walk<DataFrameRegression>) -> Self {
        match value {
            Walk::VisitLeaf(vote) => WalkRegression::VisitLeaf(vote),
            Walk::VisitBranch(feature_id, pivot) => WalkRegression::VisitBranch(feature_id, pivot),
        }
    }
}

impl From<&WalkRegression> for Walk<DataFrameRegression> {
    fn from(value: &WalkRegression) -> Walk<DataFrameRegression> {
        match value {
            WalkRegression::VisitLeaf(vote) => Walk::VisitLeaf(*vote),
            WalkRegression::VisitBranch(feature_id, pivot) => {
                Walk::VisitBranch(*feature_id, pivot.clone())
            }
        }
    }
}

pub type CategoricalUniqueValues = Vec<Option<Vec<String>>>;

pub fn get_categorical_features_unique_values(x: &Table) -> CategoricalUniqueValues {
    let mut res = vec![];
    for feature in x.cols() {
        let new_res = match &feature.array {
            Array::TextArray(arr) => match arr {
                TextArray::Categorical8(x) => Some(x.unique_values.to_vec()),
                TextArray::Categorical16(x) => Some(x.unique_values.to_vec()),
                TextArray::Categorical32(x) => Some(x.unique_values.to_vec()),
                TextArray::Categorical64(x) => Some(x.unique_values.to_vec()),
                _ => None,
            },
            _ => None,
        };
        res.push(new_res);
    }
    res
}

#[derive(Serialize, Deserialize, Clone, Eq, PartialEq)]
pub enum ArrayDType {
    Boolean,
    Float32,
    Float64,
    Integer8,
    Integer16,
    Integer32,
    Integer64,
    UInteger8,
    UInteger16,
    UInteger32,
    UInteger64,
    Categorical8,
    Categorical16,
    Categorical32,
    Categorical64,
}

impl ArrayDType {
    pub fn from_field_array(field: &FieldArray) -> Self {
        match &field.array {
            Array::NumericArray(num) => match num {
                NumericArray::Float32(_) => Self::Float32,
                NumericArray::Float64(_) => Self::Float64,
                NumericArray::UInt8(_) => Self::UInteger8,
                NumericArray::UInt16(_) => Self::UInteger16,
                NumericArray::UInt32(_) => Self::UInteger32,
                NumericArray::UInt64(_) => Self::UInteger64,
                NumericArray::Int8(_) => Self::Integer8,
                NumericArray::Int16(_) => Self::Integer16,
                NumericArray::Int32(_) => Self::Integer32,
                NumericArray::Int64(_) => Self::Integer64,
                _ => panic!("Unsupported data type!"),
            },
            Array::TextArray(arr) => match arr {
                TextArray::Categorical8(_) => Self::Categorical8,
                TextArray::Categorical16(_) => Self::Categorical16,
                TextArray::Categorical32(_) => Self::Categorical32,
                TextArray::Categorical64(_) => Self::Categorical64,
                _ => panic!("Unsupported data type!"),
            },
            Array::BooleanArray(_) => Self::Boolean,
            _ => panic!("Unsupported data type!"),
        }
    }
}
