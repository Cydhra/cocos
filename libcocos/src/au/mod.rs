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

pub mod newton;
pub use newton::fit_model_newton;

#[cfg(feature = "rayon")]
pub use newton::par_fit_model_newton;

pub mod error;
mod wls;

pub use wls::fit_model_wls;

use crate::au::error::MathError;
use crate::au::newton::NewtonProblem;
use crate::au::wls::fit_model_bp_wls;
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
pub fn get_au_value(bootstrap_replicates: &BootstrapReplicates) -> Result<Vec<f64>, MathError> {
    let closest_scale = bootstrap_replicates
        .scales()
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (1.0 - *a).abs().total_cmp(&(1.0 - *b).abs()))
        .unwrap()
        .0;
    let num_replicates = bootstrap_replicates.replication_counts()[closest_scale];

    (0..bootstrap_replicates.num_trees)
        .into_iter()
        // .take(1)
        .map(|tree| {
            let mut threshold = bootstrap_replicates
                .get_vectors(tree)
                .nth(closest_scale)
                .unwrap()[num_replicates / 2];

            let mut last_p_value = 0.0;
            let mut last_error = 0.0;
            let mut last_threshold = 0.0;
            let mut last_degrees_of_freedom = 0;
            let mut target_threshold = 0.0;
            let mut loop_count = 0;

            loop {
                let bp_values = bootstrap_replicates.compute_bp_values(tree, threshold);

                let params = fit_model_bp_wls(
                    &bp_values,
                    bootstrap_replicates.scales(),
                    bootstrap_replicates.replication_counts(),
                );
                let (c, d) = params.unwrap_or_else(|dummy| dummy);
                let mut problem = NewtonProblem::new(
                    &bp_values,
                    bootstrap_replicates.scales(),
                    bootstrap_replicates.replication_counts(),
                    d,
                    c,
                );
                problem.solve()?;

                let df = problem.degrees_of_freedom();
                let error = problem.standard_error();

                let p_value = -problem.p_value();

                // check whether our likelihood is unsolvable, or
                if df < 0
                    // whether the p value is decreasing despite the threshold also decreasing
                    || ((p_value - last_p_value) * (threshold - last_threshold) > 0.0
                    // while the change in p-value is significant (i.e., larger than standard error)
                    && (p_value - last_p_value).abs() > 0.1 * last_error
                    // and the function was fine before
                    && last_degrees_of_freedom >= 0)
                {
                    // turn back a bit toward the previous threshold and prevent the threshold
                    // from crossing the non-monotone region again
                    target_threshold = threshold;
                    threshold = 0.5 * threshold + 0.5 * last_threshold;
                    loop_count += 1;

                    // TODO this shouldnt be necessary because eventually we should exit from
                    //  this condition and don't get stuck in it
                    if loop_count > 50 {
                        // TODO warn about failed convergence
                        return Ok(problem.p_value());
                    }
                    continue;
                }

                last_threshold = threshold;

                if (p_value - last_p_value).abs() < 0.01 * last_error {
                    // we have reached stability, test whether the threshold is already close to the
                    // target. if so, just do it with the intended threshold, otherwise return
                    if (threshold < 1e-10) {
                        last_p_value = p_value;
                        last_error = error;
                        threshold = 0.0;
                        last_degrees_of_freedom = df;
                        continue;
                    } else {
                        return Ok(problem.p_value());
                    }
                } else {
                    threshold = 0.5 * threshold + 0.5 * target_threshold;
                }

                last_p_value = p_value;
                last_error = error;
                last_degrees_of_freedom = df;

                if threshold < 1e-10 {
                    return Ok(problem.p_value());
                }

                loop_count += 1;
                if loop_count > 50 {
                    // TODO warn about failed convergence
                    return Ok(problem.p_value());
                }
            }
        })
        .collect()
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
pub fn par_get_au_value(bootstrap_replicates: &BootstrapReplicates) -> Result<Vec<f64>, MathError> {
    use rayon::prelude::*;

    let params = par_fit_model_wls(bootstrap_replicates);
    let init = params
        .into_par_iter()
        .map(|r| r.unwrap_or_else(|dummy| dummy));

    let results = par_fit_model_newton(bootstrap_replicates, init)?;
    results.iter().map(|(c, d)| Ok(1.0 - cdf(d - c))).collect()
}
