use crate::classification::DecisionBasicType;

use super::DataFrame;
use xrf::{FairBest, RfRng, VoteAggregator};

#[derive(Clone)]
pub struct Votes(pub Vec<usize>); //TODO: Fix impurity to make it private

impl Votes {
    pub fn is_pure(&self) -> bool {
        self.0.iter().filter(|&&x| x > 0).count() <= 1
    }
    pub fn new(ncat: DecisionBasicType) -> Self {
        Self(std::iter::repeat_n(0, ncat as usize).collect())
    }

    pub fn collapse_empty_random(&self, rng: &mut RfRng) -> DecisionBasicType {
        self.0
            .iter()
            .enumerate()
            .fold(FairBest::new(), |mut best, (cls, count)| {
                best.ingest(count, cls, rng);
                best
            })
            .consume()
            .map(|(_score, class)| class as DecisionBasicType)
            .unwrap()
    }
}

impl VoteAggregator<DataFrame> for Votes {
    fn new(input: &DataFrame) -> Self {
        Votes::new(input.ncat)
    }
    fn ingest_vote(&mut self, v: DecisionBasicType) {
        self.0[v as usize] += 1;
    }
    fn merge(&mut self, other: &Self) {
        self.0
            .iter_mut()
            .zip(other.0.iter())
            .for_each(|(t, s)| *t += s);
    }
}
