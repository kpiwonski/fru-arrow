use std::marker::PhantomData;

use minarrow::{Array, BooleanArray, FloatArray, IntegerArray, NumericArray};
use xrf::{FeatureSampler, RfInput, RfRng};

#[derive(Debug)]
pub enum DfPivot {
    Logical,
    Real(f64),
    Integer(i64),
    Subset(u64),
}

pub struct SplittingIterator<'a, M> {
    pair: DfSplittingPair<'a>,
    mask_iter: M,
}

enum DfSplittingPair<'a> {
    Logical(&'a BooleanArray<()>),
    Numeric(&'a FloatArray<f64>, f64),
    Integer(&'a IntegerArray<i64>, i64),
    Factor((&'a IntegerArray<i64>, u64)),
}
impl<'a, M> SplittingIterator<'a, M> {
    pub fn new(x: &'a Array, pivot: &DfPivot, mask_iter: M) -> Self {
        let pair = match x {
            Array::NumericArray(num) => match (num, pivot) {
                (NumericArray::Float64(arr), &DfPivot::Real(xt)) => {
                    DfSplittingPair::Numeric(arr, xt)
                }
                (NumericArray::Int64(arr), &DfPivot::Integer(xt)) => {
                    DfSplittingPair::Integer(arr, xt)
                }
                (NumericArray::Int64(arr), &DfPivot::Subset(sub)) => {
                    DfSplittingPair::Factor((arr, sub))
                }
                _ => panic!("Unsupported array type!"),
            },
            Array::BooleanArray(arr) => match pivot {
                &DfPivot::Logical => DfSplittingPair::Logical(arr),
                _ => panic!("Unsupported array type!"),
            },
            _ => panic!("Unsupported array type!"),
        };

        Self { pair, mask_iter }
    }
}

impl<'a, 'b, M> Iterator for SplittingIterator<'a, M>
where
    M: Iterator<Item = &'b usize>,
{
    type Item = bool;
    fn next(&mut self) -> Option<bool> {
        if let Some(&e) = self.mask_iter.next() {
            let ans = match self.pair {
                DfSplittingPair::Logical(x) => x[e] != false,
                DfSplittingPair::Numeric(x, xt) => {
                    //TODO: Check polarity
                    x[e] > xt
                }
                DfSplittingPair::Integer(x, xt) => {
                    //TODO: Check polarity
                    x[e] > xt
                }
                DfSplittingPair::Factor((x, split)) => (split as u64) & (1 << x[e]) != 0,
            };
            Some(ans)
        } else {
            None
        }
    }
}

pub struct FYSampler<I> {
    mixed: Vec<u32>,
    left: usize,
    marker: PhantomData<I>,
}

impl<I: RfInput<FeatureId = u32>> FYSampler<I> {
    pub fn new(input: &I) -> Self {
        Self {
            mixed: (0..input.feature_count()).map(|x| x as u32).collect(),
            left: input.feature_count(),
            marker: PhantomData,
        }
    }
}

impl<I: RfInput<FeatureId = u32>> FeatureSampler<I> for FYSampler<I> {
    fn random_feature(&mut self, rng: &mut RfRng) -> I::FeatureId {
        let sel = rng.up_to(self.left);
        let ans = self.mixed[sel];
        self.left = self.left.checked_sub(1).unwrap();
        self.mixed.swap(sel, self.left);
        ans
    }
    fn reload(&mut self) {
        self.left = self.mixed.len();
    }
    fn reset(&mut self) {
        self.mixed = (0..self.mixed.len()).map(|x| x as u32).collect();
        self.left = self.mixed.len();
    }
}
