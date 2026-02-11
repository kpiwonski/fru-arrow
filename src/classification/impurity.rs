use super::{DecisionSlice, Votes};
use crate::{attribute::DfPivot, classification::DecisionBasicType};
use minarrow::BooleanArray;
use xrf::{Mask, RfRng, VoteAggregator};

pub fn scan_bin(x: &BooleanArray<()>, ys: &DecisionSlice, mask: &Mask) -> Option<(DfPivot, f64)> {
    let mut left = Votes::new(ys.ncat);
    let mut xt = 0_usize;
    let n = mask.len();
    mask.iter()
        .map(|&e| x[e] != false)
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

pub fn scan_factor(
    x: &[i64],
    xc: u32,
    ys: &DecisionSlice,
    mask: &Mask,
    _rng: &mut RfRng,
) -> Option<(DfPivot, f64)> {
    if xc > 10 {
        //When there is too many combinations, just treat it as ordered
        return scan_i64(x, ys, mask);
    }
    if xc < 2 {
        return None;
    }
    let n = mask.len();
    let mut va: Vec<Votes> = std::iter::repeat_with(|| Votes::new(ys.ncat))
        .take(xc as usize)
        .collect();
    mask.iter()
        .map(|&e| x[e])
        .zip(ys.values.iter())
        .for_each(|(x, &y)| va[x as usize].ingest_vote(y));
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
            let in_left: u64 = left.0.iter().sum();

            let score: f64 = ys
                .summary
                .0
                .iter()
                .zip(left.0.iter())
                .map(|(&all, &left)| {
                    let ahead = (n - in_left as usize) as f64;
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

pub fn scan_f64(x: &[f64], ys: &DecisionSlice, mask: &Mask) -> Option<(DfPivot, f64)> {
    let mut bound: Vec<(f64, DecisionBasicType)> = mask
        .iter()
        .zip(ys.values.iter())
        .map(|(&xe, &y)| (x[xe], y))
        .collect();
    bound.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

    let n = bound.len();
    let mut left = Votes::new(ys.ncat);
    let mut scanned = 0_usize;
    let ans = bound
        .windows(2)
        .map(|x| (x[0].0, x[1].0, x[0].1))
        .fold(None, |acc: Option<(f64, f64)>, (x, next_x, y)| {
            scanned += 1;
            left.ingest_vote(y);
            if x.total_cmp(&next_x).is_ne() {
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
                    return Some((0.5 * (x + next_x), score));
                }
            }
            acc
        })
        .map(|(thresh, score)| (DfPivot::Real(thresh), score));
    ans
}

pub fn scan_i64(x: &[i64], ys: &DecisionSlice, mask: &Mask) -> Option<(DfPivot, f64)> {
    let mut bound: Vec<(i64, DecisionBasicType)> = mask
        .iter()
        .zip(ys.values.iter())
        .map(|(&xe, &y)| (x[xe], y))
        .collect();
    bound.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let n = bound.len();
    let mut left = Votes::new(ys.ncat);
    let mut scanned = 0_usize;
    let ans = bound
        .windows(2)
        .map(|x| (x[0].0, x[1].0, x[0].1))
        .fold(None, |acc: Option<(i64, f64)>, (x, next_x, y)| {
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
                    return Some((x, score));
                }
            }
            acc
        })
        .map(|(thresh, score)| (DfPivot::Integer(thresh), score));
    ans
}
