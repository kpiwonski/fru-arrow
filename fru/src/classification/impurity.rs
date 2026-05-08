use std::ops::Add;

use super::{DecisionSlice, Votes};
use crate::{attribute::DfPivot, classification::ClsDecisionBasicType, tools::MidpointThreshold};
use minarrow::BooleanArray;
use xrf::{Mask, RfRng, VoteAggregator};

pub fn scan_bin(x: &BooleanArray<()>, ys: &DecisionSlice, mask: &Mask) -> Option<(DfPivot, f64)> {
    let mut left = Votes::new(ys.ncat);
    let mut xt = 0_usize;
    let n = mask.len();
    mask.iter()
        .map(|&e| x[e])
        .zip(ys.values.iter())
        .for_each(|(x, &y)| {
            if x {
                left.ingest_vote(y);
                xt += 1;
            }
        });
    let score: f64 = ys
        .summary
        .0
        .iter()
        .zip(left.0.iter())
        .map(|(total, &left)| {
            let for_false = (n - xt) as f64;
            let for_true = xt as f64;
            let n = n as f64;
            let right = (total - left) as f64;
            let left = left as f64;
            (left / for_true) * (left / n) + (right / for_false) * (right / n)
        })
        .sum();
    Some((DfPivot::Logical, score))
}

pub fn scan_categorical<T: Copy + Ord + TryInto<usize> + Into<DfPivot> + MidpointThreshold>(
    x: &[T],
    xc: usize,
    ys: &DecisionSlice,
    mask: &Mask,
    _rng: &mut RfRng,
) -> Option<(DfPivot, f64)> {
    if xc > 10 {
        //When there is too many combinations, just treat it as ordered
        return scan_integer(x, ys, mask);
    }
    if xc < 2 {
        return None;
    }
    let n = mask.len();
    let mut va: Vec<Votes> = std::iter::repeat_with(|| Votes::new(ys.ncat))
        .take(xc)
        .collect();
    mask.iter()
        .map(|&e| x[e])
        .zip(ys.values.iter())
        .for_each(|(x, &y)| va[x.try_into().ok().unwrap()].ingest_vote(y));
    let sub_max: u64 = (1 << (xc - 1)) - 1;

    (0..sub_max)
        .map(|bitmask_id| bitmask_id + (1 << (xc - 1)))
        .fold(None, |acc: Option<(u64, f64)>, bitmask| {
            let left = va
                .iter()
                .enumerate()
                .filter(|(e, _)| bitmask & (1 << e) != 0)
                .fold(Votes::new(ys.ncat), |mut acc, (_, v)| {
                    acc.merge(v);
                    acc
                });
            let in_left: usize = left.0.iter().sum();

            let score: f64 = ys
                .summary
                .0
                .iter()
                .zip(left.0.iter())
                .map(|(&all, &left)| {
                    let ahead = (n - in_left) as f64;
                    let scanned = in_left as f64;
                    let n = n as f64;
                    let right = (all - left) as f64;
                    let left = left as f64;
                    (left / scanned) * (left / n) + (right / ahead) * (right / n)
                })
                .sum();
            if score > acc.map(|x| x.1).unwrap_or(f64::NEG_INFINITY) {
                return Some((bitmask, score));
            }
            acc
        })
        .map(|(bitmask, score)| (DfPivot::Subset(bitmask), score))
}

pub fn scan_float<T: Copy + PartialOrd + Add<T, Output = T> + Into<f64>>(
    x: &[T],
    ys: &DecisionSlice,
    mask: &Mask,
) -> Option<(DfPivot, f64)> {
    let mut bound: Vec<(T, ClsDecisionBasicType)> = mask
        .iter()
        .zip(ys.values.iter())
        .map(|(&xe, &y)| (x[xe], y))
        .collect();
    bound.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let n = bound.len();
    let mut left = Votes::new(ys.ncat);
    let mut scanned = 0_usize;
    
    bound
        .windows(2)
        .map(|x| (x[0].0, x[1].0, x[0].1))
        .fold(None, |acc: Option<(f64, f64)>, (x, next_x, y)| {
            scanned += 1;
            left.ingest_vote(y);
            if x.partial_cmp(&next_x).unwrap().is_ne() {
                let score: f64 = ys
                    .summary
                    .0
                    .iter()
                    .zip(left.0.iter())
                    .map(|(&all, &left)| {
                        let ahead = (n - scanned) as f64;
                        let scanned = scanned as f64;
                        let n = n as f64;
                        let right = (all - left) as f64;
                        let left = left as f64;
                        (left / scanned) * (left / n) + (right / ahead) * (right / n)
                    })
                    .sum();
                if score > acc.map(|x| x.1).unwrap_or(f64::NEG_INFINITY) {
                    return Some((Into::<f64>::into(x + next_x) * 0.5, score));
                }
            }
            acc
        })
        .map(|(thresh, score)| (DfPivot::Real(thresh), score))
}

pub fn scan_integer<T: Copy + Ord + Into<DfPivot> + MidpointThreshold>(
    x: &[T],
    ys: &DecisionSlice,
    mask: &Mask,
) -> Option<(DfPivot, f64)> {
    let mut bound: Vec<(T, ClsDecisionBasicType)> = mask
        .iter()
        .zip(ys.values.iter())
        .map(|(&xe, &y)| (x[xe], y))
        .collect();
    bound.sort_unstable_by_key(|a| a.0);

    let n = bound.len();
    let mut left = Votes::new(ys.ncat);
    let mut scanned = 0_usize;
    
    bound
        .windows(2)
        .map(|x| (x[0].0, x[1].0, x[0].1))
        .fold(None, |acc: Option<(T, f64)>, (x, next_x, y)| {
            scanned += 1;
            left.ingest_vote(y);
            if x.cmp(&next_x).is_ne() {
                let score: f64 = ys
                    .summary
                    .0
                    .iter()
                    .zip(left.0.iter())
                    .map(|(&all, &left)| {
                        let ahead = (n - scanned) as f64;
                        let scanned = scanned as f64;
                        let n = n as f64;
                        let right = (all - left) as f64;
                        let left = left as f64;
                        (left / scanned) * (left / n) + (right / ahead) * (right / n)
                    })
                    .sum();
                if score > acc.map(|x| x.1).unwrap_or(f64::NEG_INFINITY) {
                    return Some((x.midpoint_threshold(next_x), score));
                }
            }
            acc
        })
        .map(|(thresh, score)| (thresh.into(), score))
}
