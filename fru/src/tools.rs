pub trait MidpointThreshold {
    fn midpoint_threshold(self, b: Self) -> Self;
}

macro_rules! midpoint_threshold_impl {
    ($SelfT:ty) => {
        impl MidpointThreshold for $SelfT {
            #[inline]
            fn midpoint_threshold(self, b: $SelfT) -> $SelfT {
                //Hacker's delight approach
                // creates (a+b)/2)_threshold, which is what is needed
                // for negative inputs due to how integer pivot is executed
                ((self ^ b) >> 1) + (self & b)
            }
        }
    };
}

midpoint_threshold_impl! { i8 }
midpoint_threshold_impl! { i16 }
midpoint_threshold_impl! { i32}
midpoint_threshold_impl! { i64 }

midpoint_threshold_impl! { u8 }
midpoint_threshold_impl! { u16 }
midpoint_threshold_impl! { u32}
midpoint_threshold_impl! { u64 }
