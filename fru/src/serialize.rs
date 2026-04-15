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
}

#[derive(Serialize, Deserialize)]
pub struct SerializedForestRegressionV1 {
    pub nodes: Vec<WalkRegression>,
    pub ncol: usize,
    pub train_nrow: usize,
    pub oob: Vec<<DataFrameRegression as RfInput>::VoteAggregator>,
    pub importance_raw: Vec<ImportanceRaw>,
}

impl SerializedForestClassificationV1 {
    pub fn new(
        nodes: Vec<WalkClassification>,
        decision_unique_values: Vec<String>,
        ncol: usize,
        train_nrow: usize,
        oob: Vec<<DataFrameClassification as RfInput>::VoteAggregator>,
        importance_raw: Vec<ImportanceRaw>,
    ) -> Self {
        SerializedForestClassificationV1 {
            nodes,
            decision_unique_values,
            ncol,
            train_nrow,
            oob,
            importance_raw,
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
    ) -> Self {
        SerializedForestRegressionV1 {
            nodes,
            ncol,
            train_nrow,
            oob,
            importance_raw,
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
