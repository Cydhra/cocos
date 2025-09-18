#[inline]
pub(crate) fn dot_prod(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());

    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

#[inline]
pub(crate) fn max(a: &[f64]) -> f64 {
    debug_assert!(!a.is_empty());
    *a.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
}
