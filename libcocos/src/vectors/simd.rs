use std::simd::Simd;

const VECTOR_SIZE: usize = 8;

#[inline]
pub fn dot_prod(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());

    a[..]
        .chunks_exact(VECTOR_SIZE)
        .map(Simd::<f64, VECTOR_SIZE>::load_or_default)
        .zip(
            b[..]
                .chunks_exact(VECTOR_SIZE)
                .map(Simd::<f64, VECTOR_SIZE>::load_or_default),
        )
        .map(|(a, b)| a * b)
        .sum::<Simd<f64, VECTOR_SIZE>>()
        .to_array()
        .iter()
        .sum()
}
