use crate::attribute::{DfPivot, SplittingIterator};
use minarrow::{Array, IntegerArray, NumericArray};
use xrf::{Mask, RfInput, RfRng, VoteAggregator};

mod da;
mod impurity;
mod votes;
use votes::Votes;

pub struct DataFrame {
    features: Vec<Array>,
    decision: IntegerArray<u32>,
    ncat: u32,
    ncol: usize,
    nrow: usize,
}

impl RfInput for DataFrame {
    type FeatureId = u32;
    type DecisionSlice = DecisionSlice;
    type Pivot = DfPivot;
    type Vote = u32;
    type VoteAggregator = Votes;
    type AccuracyDecreaseAggregator = da::ClsDaAggregator;
    fn observation_count(&self) -> usize {
        self.nrow
    }
    fn feature_count(&self) -> usize {
        self.ncol
    }
    fn random_feature(&self, rng: &mut RfRng) -> Self::FeatureId {
        rng.upto(self.ncol) as Self::FeatureId
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
        let feature = &self.features[using as usize];
        match feature {
            Array::NumericArray(num) => match num {
                NumericArray::Float64(x) => impurity::scan_f64(x, y, on),
                NumericArray::Int64(x) => impurity::scan_i64(x, y, on),
                _ => panic!("Unsupported data type!"),
            }, // TODO support categories
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
        let feature = &self.features[using as usize];
        SplittingIterator::new(feature, by, on.iter())
    }
    fn score_accuracy<I: Iterator<Item = Self::Vote>>(&self, mask: &Mask, other: I) -> f64 {
        let ok: usize = mask
            .iter()
            .map(|&e| self.decision[e])
            .zip(other)
            .map(|(a, b)| if a.eq(&b) { 1 } else { 0 })
            .sum();
        //TODO Maybe this should normalise by mask len?
        (ok as f64) / (self.observation_count() as f64)
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
    fn condense(&self) -> u32 {
        self.summary.collapse()
    }
}

impl DataFrame {
    //TOOD: Better order of arguments, maybe?
    pub fn new(
        features: Vec<Array>,
        decision: IntegerArray<u32>,
        ncat: u32,
        ncol: usize,
        nrow: usize,
    ) -> Self {
        Self {
            features,
            decision,
            ncat,
            ncol,
            nrow,
        }
    }
}
