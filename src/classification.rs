use crate::attribute::{DfPivot, FYSampler, SplittingIterator};
use minarrow::{Array, IntegerArray, NumericArray, Table};
use xrf::{Mask, RfInput, RfRng, VoteAggregator};

mod da;
mod impurity;
mod votes;
pub use votes::Votes;

pub struct DataFrame {
    features: Table,
    decision: IntegerArray<u32>,
    ncat: u32,
}

impl RfInput for DataFrame {
    type FeatureId = u32;
    type FeatureSampler = FYSampler<Self>;
    type DecisionSlice = DecisionSlice;
    type Pivot = DfPivot;
    type Vote = u32;
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
                NumericArray::Float64(x) => impurity::scan_f64(x, y, on),
                NumericArray::Int64(x) => impurity::scan_i64(x, y, on),
                _ => panic!("Unsupported data type!"),
            }, // TODO support categories
            Array::TextArray(arr) => match arr {
                minarrow::TextArray::Categorical8(x) => impurity::scan_factor(
                    &x.into_iter().map(|&xx| xx as i64).collect::<Vec<_>>(),
                    x.unique_values.len() as u32,
                    y,
                    on,
                    rng,
                ),
                _ => panic!("###"),
            },
            // Array::CategoricalArray(x) => impurity::scan_factor(x, xc, ys, mask, rng),
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
    values: Vec<u32>,
    ncat: u32,
    summary: Votes,
}
impl DecisionSlice {
    fn new(mask: &Mask, values: &IntegerArray<u32>, ncat: u32) -> Self {
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

impl xrf::DecisionSlice<u32> for DecisionSlice {
    fn is_pure(&self) -> bool {
        self.summary.is_pure()
    }
    fn condense(&self, rng: &mut RfRng) -> u32 {
        self.summary.collapse()
    }
}

impl DataFrame {
    //TOOD: Better order of arguments, maybe?
    pub fn new(features: Table, decision: IntegerArray<u32>, ncat: u32) -> Self {
        Self {
            features,
            decision,
            ncat,
        }
    }
}
