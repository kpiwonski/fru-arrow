use std::ops::Add;

use super::DecisionSlice;
use super::VarAggregator;
use crate::attribute::DfPivot;
use crate::regression::RegDecisionBasicType;
use minarrow::BooleanArray;
use xrf::{Mask, RfRng};

pub fn scan_bin(x: &BooleanArray<()>, ys: &DecisionSlice, mask: &Mask) -> Option<(DfPivot, f64)> {
    //TODO: Maybe use ys's summary?
    let mut left = VarAggregator::new();
    let mut right = VarAggregator::new();
    mask.iter()
        .map(|&e| x[e] != false)
        .zip(ys.values.iter())
        .for_each(|(x, &y)| {
            if x {
                left.ingest(y);
            } else {
                right.ingest(y);
            }
        });
    let score: f64 = -(left.var_n() + right.var_n());
    //Big variance is bad, we measure var(Y)-var*(left)-var*(right), so - (var*left+var*right)
    Some((DfPivot::Logical, score))
}

pub fn scan_categorical<T: Copy + Ord + TryInto<usize> + Into<DfPivot>>(
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
    let mut va: Vec<VarAggregator> = std::iter::repeat_with(|| VarAggregator::new())
        .take(xc)
        .collect();
    mask.iter()
        .map(|&e| x[e])
        .zip(ys.values.iter())
        .for_each(|(x, &y)| va[x.try_into().ok().unwrap()].ingest(y));
    let sub_max: u64 = (1 << (xc - 1)) - 1;

    (0..sub_max)
        .map(|bitmask_id| bitmask_id + (1 << (xc - 1)))
        .fold(None, |acc: Option<(u64, f64)>, bitmask| {
            let left = va
                .iter()
                .enumerate()
                .filter(|(e, _)| bitmask & (1 << e) != 0)
                .fold(VarAggregator::new(), |mut acc, (_, v)| {
                    acc.merge(v);
                    acc
                });
            let right = va
                .iter()
                .enumerate()
                .filter(|(e, _)| bitmask & (1 << e) == 0)
                .fold(VarAggregator::new(), |mut acc, (_, v)| {
                    acc.merge(v);
                    acc
                });
            let score: f64 = -(left.var_n() + right.var_n());
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
    let mut bound: Vec<(T, RegDecisionBasicType)> = mask
        .iter()
        .zip(ys.values.iter())
        .map(|(&xe, &y)| (x[xe], y))
        .collect();
    bound.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut left = VarAggregator::new();
    let mut right = ys.summary.clone();
    let ans = bound
        .windows(2)
        .map(|x| (x[0].0, x[1].0, x[0].1))
        .fold(None, |acc: Option<(f64, f64)>, (x, next_x, y)| {
            left.ingest(y);
            right.degest(y);
            if x.partial_cmp(&next_x).unwrap().is_ne() {
                let score: f64 = -(left.var_n() + right.var_n());
                if score > acc.map(|x| x.1).unwrap_or(f64::NEG_INFINITY) {
                    return Some((Into::<f64>::into(x + next_x) * 0.5, score));
                }
            }
            acc
        })
        .map(|(thresh, score)| (DfPivot::Real(thresh), score));
    ans
}

pub fn scan_integer<T: Copy + Ord + Into<DfPivot>>(
    x: &[T],
    ys: &DecisionSlice,
    mask: &Mask,
) -> Option<(DfPivot, f64)> {
    let mut bound: Vec<(T, RegDecisionBasicType)> = mask
        .iter()
        .zip(ys.values.iter())
        .map(|(&xe, &y)| (x[xe], y))
        .collect();
    bound.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let mut left = VarAggregator::new();
    let mut right = ys.summary.clone();
    let ans = bound
        .windows(2)
        .map(|x| (x[0].0, x[1].0, x[0].1))
        .fold(None, |acc: Option<(T, f64)>, (x, next_x, y)| {
            left.ingest(y);
            right.degest(y);
            if x.cmp(&next_x).is_ne() {
                let score: f64 = -(left.var_n() + right.var_n());
                if score > acc.map(|x| x.1).unwrap_or(f64::NEG_INFINITY) {
                    return Some((x, score));
                }
            }
            acc
        })
        .map(|(thresh, score)| (thresh.into(), score));
    ans
}
