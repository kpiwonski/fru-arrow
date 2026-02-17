use crate::attribute::{DfPivot, FYSampler, SplittingIterator};
use minarrow::{Array, CategoricalArray, NumericArray, Table, TextArray};
use xrf::{Mask, RfInput, RfRng, VoteAggregator};

mod da;
mod impurity;
mod votes;
pub use votes::Votes;

pub type DecisionBasicType = u64;

pub struct DataFrame {
    features: Table,
    decision: CategoricalArray<DecisionBasicType>,
    ncat: DecisionBasicType,
}

impl RfInput for DataFrame {
    type FeatureId = u32;
    type FeatureSampler = FYSampler<Self>;
    type DecisionSlice = DecisionSlice;
    type Pivot = DfPivot;
    type Vote = DecisionBasicType;
    type VoteAggregator = Votes;
    type AccuracyDecreaseAggregator = da::ClsDaAggregator;
    fn observation_count(&self) -> usize {
        self.features.n_rows()
    }
    fn feature_count(&self) -> usize {
        self.features.n_cols()
    }
    fn feature_sampler(&self) -> Self::FeatureSampler {
        super::attribute::FYSampler::new(self)
    }

    fn decision_slice(&self, mask: &Mask) -> Self::DecisionSlice {
        DecisionSlice::new(mask, &self.decision, self.ncat)
    }
    fn new_split(
        &self,
        on: &Mask,
        using: Self::FeatureId,
        y: &Self::DecisionSlice,
        rng: &mut RfRng,
    ) -> Option<(Self::Pivot, f64)> {
        let feature = &self.features.cols[using as usize];
        match &feature.array {
            Array::NumericArray(num) => match num {
                NumericArray::Float32(x) => impurity::scan_float(x, y, on),
                NumericArray::Float64(x) => impurity::scan_float(x, y, on),
                NumericArray::UInt8(x) => impurity::scan_integer(x, y, on),
                NumericArray::UInt16(x) => impurity::scan_integer(x, y, on),
                NumericArray::UInt32(x) => impurity::scan_integer(x, y, on),
                NumericArray::UInt64(x) => impurity::scan_integer(x, y, on),
                NumericArray::Int8(x) => impurity::scan_integer(x, y, on),
                NumericArray::Int16(x) => impurity::scan_integer(x, y, on),
                NumericArray::Int32(x) => impurity::scan_integer(x, y, on),
                NumericArray::Int64(x) => impurity::scan_integer(x, y, on),
                _ => panic!("Unsupported data type!"),
            },
            Array::TextArray(arr) => match arr {
                TextArray::Categorical8(x) => {
                    impurity::scan_categorical(&x, x.unique_values.len(), y, on, rng)
                }
                TextArray::Categorical16(x) => {
                    impurity::scan_categorical(&x, x.unique_values.len(), y, on, rng)
                }
                TextArray::Categorical32(x) => {
                    impurity::scan_categorical(&x, x.unique_values.len(), y, on, rng)
                }
                TextArray::Categorical64(x) => {
                    impurity::scan_categorical(&x, x.unique_values.len(), y, on, rng)
                }
                _ => panic!("Unsupported data type!"),
            },
            Array::BooleanArray(x) => impurity::scan_bin(x, y, on),
            _ => panic!("Unsupported data type!"),
        }
    }
    fn split_iter(
        &self,
        on: &Mask,
        using: Self::FeatureId,
        by: &Self::Pivot,
    ) -> impl Iterator<Item = bool> {
        let feature = &self.features.cols[using as usize];
        SplittingIterator::new(&feature.array, by, on.iter())
    }
}

pub struct DecisionSlice {
    values: Vec<DecisionBasicType>,
    ncat: DecisionBasicType,
    summary: Votes,
}
impl DecisionSlice {
    fn new(
        mask: &Mask,
        values: &CategoricalArray<DecisionBasicType>,
        ncat: DecisionBasicType,
    ) -> Self {
        let mut summary = Votes::new(ncat);
        let values = mask
            .iter()
            .map(|&e| values[e])
            .inspect(|&x| summary.ingest_vote(x))
            .collect();
        DecisionSlice {
            values,
            ncat,
            summary,
        }
    }
}

impl xrf::DecisionSlice<DecisionBasicType> for DecisionSlice {
    fn is_pure(&self) -> bool {
        self.summary.is_pure()
    }
    fn condense(&self, rng: &mut RfRng) -> DecisionBasicType {
        self.summary.collapse_empty_random(rng)
    }
}

impl DataFrame {
    pub fn new(
        features: Table,
        decision: CategoricalArray<DecisionBasicType>,
        ncat: DecisionBasicType,
    ) -> Self {
        Self {
            features,
            decision,
            ncat,
        }
    }
}
