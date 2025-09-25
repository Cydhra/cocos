//! This module implements the necessary numerical algorithms to estimate the signed distance `d`
//! and the curvature `c` for the log-likelihood regions of all trees, which are in turn used to
//! estimate the AU p-value for each tree.
//!
//! The most convenient method to estimate p-values for a [`todo`], is calling
//! [`get_au_value`], which will return a p-value for each tree in the table.
//! When using the `rayon` feature, the corresponding function is called [`par_get_au_value`]
//!
//! [`get_au_value`]: get_au_values
//! [`par_get_au_value`]: par_get_au_values

use crate::{BootstrapReplicates, EPSILON};

mod math;

pub mod newton;

pub mod error;
mod wls;

use crate::au::error::MathError;
use crate::au::newton::NewtonProblem;
use crate::au::wls::fit_model_bp_wls;

/// Select the scaling factor of a [`BootstrapReplicates`] instance that is closest to 1, and
/// return its index and its [replication count].
/// Using these parameters, we can select the initial threshold for the BP convergence during AU
/// estimation.
///
/// [`BootstrapReplicates`]: BootstrapReplicates
/// [replication count]: BootstrapReplicates::replication_counts
fn select_threshold_element(bootstrap_replicates: &BootstrapReplicates) -> (usize, usize) {
    let closest_scale = bootstrap_replicates
        .scales()
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (1.0 - *a).abs().total_cmp(&(1.0 - *b).abs()))
        .unwrap()
        .0;
    let num_replicates = bootstrap_replicates.replication_counts()[closest_scale];

    (closest_scale, num_replicates)
}

/// Perform the AU test on one input of the [`BootstrapReplicates`] instance.
///
/// The method begins fitting the signed distance and curvature parameters of the bootstrap input
/// vector using the [WLS method].
/// The parameters are then used as starting values for a [Newton optimization] which obtains the
/// final p-value.
/// Additionally, this is repeated at different quantiles of the empirical distribution of
/// Bootstrap Proportions.
/// The p-values at different quantiles are compared and the function returns the p-value if
/// it converges at a final value.
///
/// If the p-value does not converge, a [`MathError::ConvergenceFailed`] error is returned.
///
/// # Parameters
/// - `bootstrap_replicates`: A set of matrices of bootstrap replicates of all input sequences, one
///   matrix per bootstrap scale.
/// - `tree` the index of the input sequence in `bootstrap_replicates`
/// - `initial_threshold` the bootstrap count at the quantile at which the algorithm starts. A good
///   candidate for this parameter is the median bootstrap count at scaling factor 1.
///
/// # Result
/// Returns the p-value of the given input sequence or a [`MathError`] is the algorithm encountered
/// an unsolvable edge case.
/// The `MathError` may still contain a p-value.
/// It is left to the application designer to decide whether this p-value can be trusted.
pub fn calc_au_value(
    bootstrap_replicates: &BootstrapReplicates,
    tree: usize,
    initial_threshold: f64,
) -> Result<f64, MathError> {
    let mut threshold = initial_threshold;
    let mut last_p_value = 0.0;
    let mut last_error = 0.0;
    let mut last_threshold = 0.0;
    let mut last_degrees_of_freedom = 0;
    let mut target_threshold = 0.0;

    for _ in 0..100 {
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
        problem.solve();

        let p_value = problem.p_value();

        // check whether our likelihood is unsolvable, or
        if problem.degrees_of_freedom() < 0
            // whether the p value is decreasing despite the threshold also decreasing
            || ((last_p_value - p_value) * (threshold - last_threshold) > 0.0
            // while the change in p-value is significant (i.e., larger than standard error)
            && (p_value - last_p_value).abs() > 0.1 * last_error
            // and the function was fine before
            && last_degrees_of_freedom >= 0)
        {
            // turn back a bit toward the previous threshold and prevent the threshold
            // from crossing the non-monotone region again
            target_threshold = threshold;
            threshold = 0.5 * threshold + 0.5 * last_threshold;
            continue;
        }

        last_threshold = threshold;

        if (last_p_value - p_value).abs() < 0.01 * last_error {
            // we have reached convergence of the p-value
            return Ok(problem.p_value());
        } else {
            threshold = 0.5 * threshold + 0.5 * target_threshold;
        }

        last_p_value = p_value;
        last_error = problem.standard_error();
        last_degrees_of_freedom = problem.degrees_of_freedom();

        if (threshold - target_threshold).abs() < EPSILON {
            // we have reached the canonical BP value
            return Ok(problem.p_value());
        }
    }

    Err(MathError::ConvergenceFailed {
        p_value: last_p_value,
    })
}

/// Perform the AU test on all inputs in the [`BootstrapReplicates`] instance.
///
/// The method begins fitting the signed distance and curvature parameters of any bootstrap
/// vector using the [WLS method].
/// The parameters are then used as starting values for a [Newton optimization] which obtains the
/// final values.
/// Additionally, this is repeated at different quantiles of the empirical distribution of
/// Bootstrap Proportions.
/// The p-values at different quantiles are compared and the function returns the p-value if
/// it converges at a final value.
///
/// If the p-value does not converge, a [`MathError::ConvergenceFailed`] error is returned.
///
/// # Parameters
/// - `bootstrap_replicates`: A set of matrices of bootstrap replicates of all input sequences, one
///   matrix per bootstrap scale.
///
/// # Return
/// Returns a list of p-values or [`MathErrors`] that describe the problem with the test.
/// Many `MathErrors` still contain a p-value.
/// It is left to the application designer to decide whether those p-values can be trusted.
///
/// # Parallelization
/// A parallel version of this function is available with the `rayon` feature as
/// [`par_get_au_values`].
/// Note that this step is so fast that parallelization is usually worth it only at several
/// thousand trees.
///
/// [WLS method]: fit_model_bp_wls
/// [Newton optimization]: newton::fit_model_bp_newton
/// [`MathError::ConvergenceFailed`]: MathError::ConvergenceFailed
/// [`MathErrors`]: MathError
pub fn get_au_values(bootstrap_replicates: &BootstrapReplicates) -> Box<[Result<f64, MathError>]> {
    let (closest_scale, num_replicates) = select_threshold_element(bootstrap_replicates);

    (0..bootstrap_replicates.num_trees)
        .map(|tree| {
            let threshold = bootstrap_replicates
                .get_vectors(tree)
                .nth(closest_scale)
                .unwrap()[num_replicates / 2];

            calc_au_value(bootstrap_replicates, tree, threshold)
        })
        .collect()
}

/// Perform the AU test on all trees in the [`BootstrapReplicates`] instance in parallel.
/// For full documentation see [`get_au_values`].
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
/// [`get_au_values`]: get_au_values
#[cfg(feature = "rayon")]
pub fn par_get_au_values(
    bootstrap_replicates: &BootstrapReplicates,
) -> Box<[Result<f64, MathError>]> {
    use rayon::prelude::*;
    let (closest_scale, num_replicates) = select_threshold_element(bootstrap_replicates);

    (0..bootstrap_replicates.num_trees)
        .into_par_iter()
        .map(|tree| {
            let threshold = bootstrap_replicates
                .get_vectors(tree)
                .nth(closest_scale)
                .unwrap()[num_replicates / 2];

            calc_au_value(bootstrap_replicates, tree, threshold)
        })
        .collect()
}
