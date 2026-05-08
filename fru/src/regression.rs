use crate::attribute::{DfPivot, FYSampler, SplittingIterator};
use minarrow::{Array, FloatArray, NumericArray, Table, TextArray};
use xrf::{Mask, RfInput, RfRng};

mod da;
mod impurity;
mod votes;
use votes::Votes;

pub type RegDecisionBasicType = f64;

pub struct DataFrame {
    features: Table,
    decision: FloatArray<RegDecisionBasicType>,
}

impl RfInput for DataFrame {
    type FeatureId = usize;
    type FeatureSampler = FYSampler<Self>;
    type DecisionSlice = DecisionSlice;
    type Pivot = DfPivot;
    type Vote = RegDecisionBasicType;
    type VoteAggregator = Votes;
    type AccuracyDecreaseAggregator = da::DaAggregator;
    fn observation_count(&self) -> usize {
        self.features.n_rows()
    }
    fn feature_count(&self) -> usize {
        self.features.n_cols()
    }
    fn decision_slice(&self, mask: &Mask) -> Self::DecisionSlice {
        DecisionSlice::new(mask, &self.decision)
    }
    fn new_split(
        &self,
        on: &Mask,
        using: Self::FeatureId,
        y: &Self::DecisionSlice,
        rng: &mut RfRng,
    ) -> Option<(Self::Pivot, RegDecisionBasicType)> {
        // TODO
        let feature = &self.features.cols[using];
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
                    impurity::scan_categorical(x, x.unique_values.len(), y, on, rng)
                }
                TextArray::Categorical16(x) => {
                    impurity::scan_categorical(x, x.unique_values.len(), y, on, rng)
                }
                TextArray::Categorical32(x) => {
                    impurity::scan_categorical(x, x.unique_values.len(), y, on, rng)
                }
                TextArray::Categorical64(x) => {
                    impurity::scan_categorical(x, x.unique_values.len(), y, on, rng)
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
        let feature = &self.features.cols[using];
        SplittingIterator::new(&feature.array, by, on.iter())
    }

    fn feature_sampler(&self) -> Self::FeatureSampler {
        super::attribute::FYSampler::new(self)
    }
}

pub struct DecisionSlice {
    values: Vec<RegDecisionBasicType>,
    summary: VarAggregator,
}

impl DecisionSlice {
    fn new(mask: &Mask, values: &[RegDecisionBasicType]) -> Self {
        let mut summary = VarAggregator::new();
        let values = mask
            .iter()
            .map(|&e| values[e])
            .inspect(|&e| summary.ingest(e))
            .collect();
        DecisionSlice { values, summary }
    }
}

impl xrf::DecisionSlice<RegDecisionBasicType> for DecisionSlice {
    fn is_pure(&self) -> bool {
        self.values.len() < 5
    }
    fn condense(&self, _rng: &mut RfRng) -> RegDecisionBasicType {
        self.summary.ave()
    }
}

impl DataFrame {
    pub fn new(features: Table, decision: FloatArray<RegDecisionBasicType>) -> Self {
        Self { features, decision }
    }
}

#[derive(Clone)]
struct VarAggregator {
    sum: RegDecisionBasicType,
    sum_sq: RegDecisionBasicType,
    n: usize,
}

impl VarAggregator {
    fn new() -> Self {
        Self {
            sum: 0.,
            sum_sq: 0.,
            n: 0,
        }
    }
    fn ingest(&mut self, x: RegDecisionBasicType) {
        self.sum += x;
        self.sum_sq += x * x;
        self.n += 1;
    }
    fn degest(&mut self, x: RegDecisionBasicType) {
        self.sum -= x;
        self.sum_sq -= x * x;
        self.n -= 1;
    }
    fn ave(&self) -> RegDecisionBasicType {
        self.sum / (self.n as RegDecisionBasicType)
    }
    fn var_n(&self) -> RegDecisionBasicType {
        self.sum_sq - self.sum * self.sum / (self.n as RegDecisionBasicType)
    }
    fn merge(&mut self, other: &Self) {
        self.sum += other.sum;
        self.sum_sq += other.sum_sq;
        self.n += other.n;
    }
}
