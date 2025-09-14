use crate::BpTable;

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

/// Perform the AU test on all trees in the [`BpTable`]. This method fits parameters with
/// [`fit_model_wls`] to obtain starting values for a Newton optimizer that obtains the final values
/// for the parameters.
///
/// The parameters are then transformed into the final p-value for the AU test and returned in a
/// vector (same order as the input table).
///
/// # Parameters
/// - `bp_values`: A table of `R` bootstrap proportions for each tree in the selection problem.
///
/// # Parallelization
/// A parallel version of this function is available with the `rayon` feature as
/// [`par_get_au_value`].
/// Note that this step is so fast that parallelization is usually worth it only at several
/// thousand trees.
///
/// [`BpTable`]: BpTable
pub fn get_au_value(bp_values: &BpTable) -> Result<Vec<f64>, argmin_math::Error> {
    let params = fit_model_wls(bp_values);
    let init = params.into_iter().map(|r| {
        match r {
            Ok(res) => res,
            Err(dummy) => dummy,
        }
    });

    let results = fit_model_newton(bp_values, init)?;
    results.iter().map(|(c, d)| Ok(1.0 - cdf(d - c))).collect()
}

/// Perform the AU test on all trees in the [`BpTable`] in parallel.
/// For full documentation see [`get_au_value`].
///
/// # Parallelization
/// This method uses the global rayon thread pool.
/// Note that this step is so fast that parallelization is usually worth it only at several
/// thousand trees.
///
/// # Parameters
/// - `bp_values`: A table of `R` bootstrap proportions for each tree in the selection problem.
///
/// [`BpTable`]: crate::BpTable
/// [`get_au_value`]: get_au_value
#[cfg(feature = "rayon")]
pub fn par_get_au_value(bp_values: &BpTable) -> Result<Vec<f64>, argmin_math::Error> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let params = par_fit_model_wls(bp_values);
    let init = params
        .into_par_iter()
        .map(|r| {
            match r {
                Ok(res) => res,
                Err(dummy) => dummy,
            }
        });

    let results = par_fit_model_newton(bp_values, init)?;
    results.iter().map(|(c, d)| Ok(1.0 - cdf(d - c))).collect()
}
