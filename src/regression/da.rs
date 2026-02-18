use crate::regression::RegDecisionBasicType;

use super::DataFrameRegression;
use minarrow::FloatArray;
use std::collections::HashMap;
use xrf::{AccuracyDecreaseAggregator, Mask, RfInput};

pub struct DaAggregator {
    direct: Vec<Option<RegDecisionBasicType>>,
    drops: HashMap<usize, f64>,
    n: usize,
    true_decision: FloatArray<RegDecisionBasicType>,
}
impl AccuracyDecreaseAggregator<DataFrameRegression> for DaAggregator {
    fn new(input: &DataFrameRegression, on: &Mask, n: usize) -> Self {
        Self {
            direct: vec![None; n],
            drops: HashMap::new(),
            n: on.len(),
            //TODO: Reference here
            true_decision: input.decision.clone(),
        }
    }
    fn ingest(&mut self, permutted: Option<usize>, mask: &Mask, vote: &RegDecisionBasicType) {
        if let Some(permutted) = permutted {
            let diff: f64 = mask
                .iter()
                .map(|&e| {
                    let oob_vote = self.direct.get(e).unwrap().unwrap();
                    let truth = self.true_decision[e];
                    (truth - vote) * (truth - vote) - (truth - oob_vote) * (truth - oob_vote)
                })
                .sum();
            *self.drops.entry(permutted).or_insert(0.) += diff;
        } else {
            for &e in mask.iter() {
                self.direct[e] = Some(*vote);
            }
        }
    }
    fn get_direct_vote(&self, e: usize) -> RegDecisionBasicType {
        self.direct.get(e).unwrap().unwrap()
    }
    fn mda_iter(&self) -> impl Iterator<Item = (<DataFrameRegression as RfInput>::FeatureId, f64)> {
        self.drops
            .iter()
            .map(|(a, b)| (*a, (*b as f64) / (self.n as f64)))
    }
}
