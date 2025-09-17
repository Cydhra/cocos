#[cfg(feature = "simd")]
mod simd;
#[cfg(feature = "simd")]
pub(crate) use simd::*;

// scalar fallback for stable build, since portable simd is still nightly
#[cfg(not(feature = "simd"))]
mod scalar;
#[cfg(not(feature = "simd"))]
pub(crate) use scalar::*;

#[cfg(test)]
mod tests {
    use crate::vectors::dot_prod;

    #[test]
    fn test_dot_product() {
        // test dot product of vector larger than the VECTOR_SIZE
        assert_eq!(
            dot_prod(
                &[1., 2., 3., 4., 5., 1., 2., 3., 4., 5.],
                &[2., 3., 4., 5., 6., 2., 3., 4., 5., 6.]
            ),
            1. * 2.
                + 2. * 3.
                + 3. * 4.
                + 4. * 5.
                + 5. * 6.
                + 1. * 2.
                + 2. * 3.
                + 3. * 4.
                + 4. * 5.
                + 5. * 6.
        );
    }
}
