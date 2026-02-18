use std::marker::PhantomData;

use minarrow::{
    Array, BooleanArray, CategoricalArray, FloatArray, IntegerArray, NumericArray, TextArray,
};
use xrf::{FeatureSampler, RfInput, RfRng};

#[derive(Debug)]
pub enum DfPivot {
    Logical,
    Real(f64),
    Integer(i64),
    UInteger(u64),
    Subset(u64),
}

pub struct SplittingIterator<'a, M> {
    pair: DfSplittingPair<'a>,
    mask_iter: M,
}

enum DfSplittingPair<'a> {
    Boolean(&'a BooleanArray<()>),
    Float32(&'a FloatArray<f32>, f64),
    Float64(&'a FloatArray<f64>, f64),
    Integer8(&'a IntegerArray<i8>, i64),
    Integer16(&'a IntegerArray<i16>, i64),
    Integer32(&'a IntegerArray<i32>, i64),
    Integer64(&'a IntegerArray<i64>, i64),
    UInteger8(&'a IntegerArray<u8>, u64),
    UInteger16(&'a IntegerArray<u16>, u64),
    UInteger32(&'a IntegerArray<u32>, u64),
    UInteger64(&'a IntegerArray<u64>, u64),
    Categorical8(&'a CategoricalArray<u8>, u64),
    Categorical16(&'a CategoricalArray<u16>, u64),
    Categorical32(&'a CategoricalArray<u32>, u64),
    Categorical64(&'a CategoricalArray<u64>, u64),
}
impl<'a, M> SplittingIterator<'a, M> {
    pub fn new(x: &'a Array, pivot: &DfPivot, mask_iter: M) -> Self {
        let pair = match x {
            Array::NumericArray(num) => match (num, pivot) {
                (NumericArray::Float32(arr), &DfPivot::Real(xt)) => {
                    DfSplittingPair::Float32(arr, xt)
                }
                (NumericArray::Float64(arr), &DfPivot::Real(xt)) => {
                    DfSplittingPair::Float64(arr, xt)
                }
                (NumericArray::Int8(arr), &DfPivot::Integer(xt)) => {
                    DfSplittingPair::Integer8(arr, xt)
                }
                (NumericArray::Int16(arr), &DfPivot::Integer(xt)) => {
                    DfSplittingPair::Integer16(arr, xt)
                }
                (NumericArray::Int32(arr), &DfPivot::Integer(xt)) => {
                    DfSplittingPair::Integer32(arr, xt)
                }
                (NumericArray::Int64(arr), &DfPivot::Integer(xt)) => {
                    DfSplittingPair::Integer64(arr, xt)
                }
                (NumericArray::UInt8(arr), &DfPivot::UInteger(xt)) => {
                    DfSplittingPair::UInteger8(arr, xt)
                }
                (NumericArray::UInt16(arr), &DfPivot::UInteger(xt)) => {
                    DfSplittingPair::UInteger16(arr, xt)
                }
                (NumericArray::UInt32(arr), &DfPivot::UInteger(xt)) => {
                    DfSplittingPair::UInteger32(arr, xt)
                }
                (NumericArray::UInt64(arr), &DfPivot::UInteger(xt)) => {
                    DfSplittingPair::UInteger64(arr, xt)
                }
                _ => panic!("Unsupported array type!"),
            },
            Array::TextArray(cat) => match (cat, pivot) {
                (TextArray::Categorical8(arr), &DfPivot::Subset(sub)) => {
                    DfSplittingPair::Categorical8(arr, sub)
                }
                (TextArray::Categorical16(arr), &DfPivot::Subset(sub)) => {
                    DfSplittingPair::Categorical16(arr, sub)
                }
                (TextArray::Categorical32(arr), &DfPivot::Subset(sub)) => {
                    DfSplittingPair::Categorical32(arr, sub)
                }
                (TextArray::Categorical64(arr), &DfPivot::Subset(sub)) => {
                    DfSplittingPair::Categorical64(arr, sub)
                }
                _ => panic!("Unsupported array type!"),
            },
            Array::BooleanArray(arr) => match pivot {
                &DfPivot::Logical => DfSplittingPair::Boolean(arr),
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
                DfSplittingPair::Boolean(x) => x[e] != false,
                DfSplittingPair::Float32(x, xt) => x[e] as f64 > xt,
                DfSplittingPair::Float64(x, xt) => x[e] > xt,
                DfSplittingPair::Integer8(x, xt) => x[e] as i64 > xt,
                DfSplittingPair::Integer16(x, xt) => x[e] as i64 > xt,
                DfSplittingPair::Integer32(x, xt) => x[e] as i64 > xt,
                DfSplittingPair::Integer64(x, xt) => x[e] > xt,
                DfSplittingPair::UInteger8(x, xt) => x[e] as u64 > xt,
                DfSplittingPair::UInteger16(x, xt) => x[e] as u64 > xt,
                DfSplittingPair::UInteger32(x, xt) => x[e] as u64 > xt,
                DfSplittingPair::UInteger64(x, xt) => x[e] > xt,
                DfSplittingPair::Categorical8(x, split) => split & (1 << x[e] as u64) != 0,
                DfSplittingPair::Categorical16(x, split) => split & (1 << x[e] as u64) != 0,
                DfSplittingPair::Categorical32(x, split) => split & (1 << x[e] as u64) != 0,
                DfSplittingPair::Categorical64(x, split) => split & (1 << x[e]) != 0,
            };
            Some(ans)
        } else {
            None
        }
    }
}

pub struct FYSampler<I> {
    mixed: Vec<usize>,
    left: usize,
    marker: PhantomData<I>,
}

impl<I: RfInput<FeatureId = usize>> FYSampler<I> {
    pub fn new(input: &I) -> Self {
        Self {
            mixed: (0..input.feature_count()).collect(),
            left: input.feature_count(),
            marker: PhantomData,
        }
    }
}

impl<I: RfInput<FeatureId = usize>> FeatureSampler<I> for FYSampler<I> {
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
        self.mixed = (0..self.mixed.len()).collect();
        self.left = self.mixed.len();
    }
}

macro_rules! impl_from_uint_for_dfpivot {
    ($($t:ty),* $(,)?) => {
        $(
            impl From<$t> for DfPivot {
                fn from(value: $t) -> Self {
                    DfPivot::UInteger(value as u64)
                }
            }
        )*
    };
}

macro_rules! impl_from_int_for_dfpivot {
    ($($t:ty),* $(,)?) => {
        $(
            impl From<$t> for DfPivot {
                fn from(value: $t) -> Self {
                    DfPivot::Integer(value as i64)
                }
            }
        )*
    };
}

impl_from_int_for_dfpivot!(i8, i16, i32, i64);
impl_from_uint_for_dfpivot!(u8, u16, u32, u64);
