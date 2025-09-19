//! This module implements the necessary numerical algorithms to estimate the signed distance `d`
//! and the curvature `c` for the log-likelihood regions of all trees, which are in turn used to
//! estimate the AU p-value for each tree.
//!
//! The most convenient method to estimate p-values for a [`todo`], is calling
//! [`get_au_value`], which will return a p-value for each tree in the table.
//! When using the `rayon` feature, the corresponding function is called [`par_get_au_value`]
//!
//! [`get_au_value`]: get_au_value
//! [`par_get_au_value`]: par_get_au_value

use crate::BootstrapReplicates;

mod math;
use crate::au::math::cdf;

mod newton;
pub use newton::fit_model_newton;

#[cfg(feature = "rayon")]
pub use newton::par_fit_model_newton;

mod wls;
pub use wls::fit_model_wls;

#[cfg(feature = "rayon")]
pub use wls::par_fit_model_wls;

/// Perform the AU test on all inputs in the [`BootstrapReplicates`]. This method fits parameters with
/// [`fit_model_wls`] to obtain starting values for a Newton optimizer that obtains the final values
/// for the parameters.
///
/// The parameters are then transformed into the final p-value for the AU test and returned in a
/// vector (same order as the input table).
///
/// # Parameters
/// - `bootstrap_replicates`: A set of matrices of bootstrap replicates of all input sequences, one
///   matrix per bootstrap scale.
///
/// # Parallelization
/// A parallel version of this function is available with the `rayon` feature as
/// [`par_get_au_value`].
/// Note that this step is so fast that parallelization is usually worth it only at several
/// thousand trees.
///
/// [`BpTable`]: todo
pub fn get_au_value(
    bootstrap_replicates: &BootstrapReplicates,
) -> Result<Vec<f64>, argmin_math::Error> {
    let params = fit_model_wls(bootstrap_replicates);
    let init = params.into_iter().map(|r| r.unwrap_or_else(|dummy| dummy));

    let results = fit_model_newton(bootstrap_replicates, init)?;
    results.iter().map(|(c, d)| Ok(1.0 - cdf(d - c))).collect()
}

/// Perform the AU test on all trees in the [``] in parallel. //TODO
/// For full documentation see [`get_au_value`].
///
/// # Parallelization
/// This method uses the global rayon thread pool.
/// Note that this step is so fast that parallelization is usually worth it only at several
/// thousand trees.
///
/// # Parameters
/// - `bootstrap_replicates`: A set of matrices of bootstrap replicates of all input sequences, one
///   matrix per bootstrap scale.
///
/// [`get_au_value`]: get_au_value
#[cfg(feature = "rayon")]
pub fn par_get_au_value(
    bootstrap_replicates: &BootstrapReplicates,
) -> Result<Vec<f64>, argmin_math::Error> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let params = par_fit_model_wls(bootstrap_replicates);
    let init = params
        .into_par_iter()
        .map(|r| r.unwrap_or_else(|dummy| dummy));

    let results = par_fit_model_newton(bootstrap_replicates, init)?;
    results.iter().map(|(c, d)| Ok(1.0 - cdf(d - c))).collect()
}
