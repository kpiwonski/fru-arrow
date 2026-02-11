use crate::classification::DecisionBasicType;

use super::DataFrame;
use minarrow::CategoricalArray;
use std::collections::HashMap;
use xrf::{AccuracyDecreaseAggregator, Mask, RfInput};

pub struct ClsDaAggregator {
    direct: Vec<Option<DecisionBasicType>>,
    drops: HashMap<u32, isize>,
    n: usize,
    true_decision: CategoricalArray<DecisionBasicType>,
}
impl AccuracyDecreaseAggregator<DataFrame> for ClsDaAggregator {
    fn new(input: &DataFrame, on: &Mask, n: usize) -> Self {
        Self {
            direct: vec![None; n],
            drops: HashMap::new(),
            n: on.len(),
            //TODO: Reference here
            true_decision: input.decision.clone(),
        }
    }
    fn ingest(&mut self, permutted: Option<u32>, mask: &Mask, vote: &DecisionBasicType) {
        if let Some(permutted) = permutted {
            let diff: isize = mask
                .iter()
                .map(|&e| {
                    let oob_vote = self.direct.get(e).unwrap().unwrap();
                    if !oob_vote.eq(vote) {
                        let tru = self.true_decision[e];
                        match (tru.eq(&vote), tru.eq(&oob_vote)) {
                            (true, true) => unreachable!("Logic error"),
                            (true, false) => -1,
                            (false, true) => 1,

                            (false, false) => 0,
                        }
                    } else {
                        0
                    }
                })
                .sum();
            *self.drops.entry(permutted).or_insert(0) += diff;
        } else {
            for &e in mask.iter() {
                self.direct[e] = Some(*vote);
            }
        }
    }
    fn get_direct_vote(&self, e: usize) -> DecisionBasicType {
        self.direct.get(e).unwrap().unwrap()
    }
    fn mda_iter(&self) -> impl Iterator<Item = (<DataFrame as RfInput>::FeatureId, f64)> {
        self.drops
            .iter()
            .map(|(a, b)| (*a, (*b as f64) / (self.n as f64)))
    }
}
