use crate::regression::RegDecisionBasicType;

use super::DataFrameRegression;
use xrf::VoteAggregator;

#[derive(Clone)]
pub struct Votes {
    sum: f64,
    n: usize,
}

impl Votes {
    pub fn new() -> Self {
        Votes { sum: 0., n: 0 }
    }

    pub fn collapse(&self) -> RegDecisionBasicType {
        if self.n > 0 {
            self.sum / (self.n as f64)
        } else {
            f64::NAN
        }
    }
}

impl VoteAggregator<DataFrameRegression> for Votes {
    fn new(_input: &DataFrameRegression) -> Self {
        Votes::new()
    }
    fn ingest_vote(&mut self, v: RegDecisionBasicType) {
        self.sum += v;
        self.n += 1;
    }
    fn merge(&mut self, other: &Self) {
        self.sum += other.sum;
        self.n += other.n;
    }
}
