use super::DataFrame;
use xrf::VoteAggregator;

#[derive(Clone)]
pub struct Votes(pub Vec<u32>); //TODO: Fix impurity to make it private

impl Votes {
    pub fn is_pure(&self) -> bool {
        self.0.iter().filter(|&&x| x > 0).count() <= 1
    }
    pub fn new(ncat: u32) -> Self {
        Self(std::iter::repeat_n(0, ncat as usize).collect())
    }
}

impl VoteAggregator<DataFrame> for Votes {
    fn new(input: &DataFrame) -> Self {
        Votes::new(input.ncat)
    }
    fn ingest_vote(&mut self, v: u32) {
        self.0[v as usize] += 1;
    }
    fn merge(&mut self, other: &Self) {
        self.0
            .iter_mut()
            .zip(other.0.iter())
            .for_each(|(t, s)| *t += s);
    }
    fn collapse(&self) -> u32 {
        self.0
            .iter()
            .enumerate()
            .fold(None, |best, (cls, count)| {
                if best.map(|x: (usize, u32)| x.1).unwrap_or(0) < *count {
                    Some((cls, *count))
                } else {
                    best
                }
            })
            .map(|x| x.0 as u32)
            .unwrap_or(u32::MAX)
    }
}
