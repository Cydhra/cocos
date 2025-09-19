//! Implementation of datatypes and functions used by the Newton-Raphson solver which estimates
//! the `c` and `d` values for the AU test.

use crate::BootstrapReplicates;
use crate::au::math::{Matrix2by2, Vec2, cdf, pdf};
use argmin::core::{Error, Executor, Gradient, Hessian, State};
use argmin::solver::newton::Newton;

struct NewtonProblem<'input> {
    bp_values: &'input [f64],
    scales: &'input [f64],
    num_replicates: &'input [usize],
}

impl<'tree> Gradient for NewtonProblem<'tree> {
    type Param = Vec2;
    type Gradient = Vec2;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin_math::Error> {
        let Vec2(c, d) = *param;
        let (gradient_c, gradient_d) = self.gradient_sum_function(c, d);
        Ok(Vec2(gradient_c, gradient_d))
    }
}

impl<'input> Hessian for NewtonProblem<'input> {
    type Param = Vec2;
    type Hessian = Matrix2by2;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error> {
        let &Vec2(c, d) = param;

        let (hess_cc, hess_cd, hess_dd) =
            Self::hessian_sum_function(self.bp_values, self.scales, self.num_replicates, c, d);

        Ok(Matrix2by2(hess_cc, hess_cd, hess_cd, hess_dd))
    }
}

impl<'input> NewtonProblem<'input> {
    /// Create a new problem instance, which can be used in a [Newton Solver] to obtain values for the
    /// distance and curvature parameters of the AU test.
    ///
    /// For details refer to https://doi.org/10.1080/10635150290069913 Appendix 9.
    ///
    /// [Newton Solver]: argmin::solver::newton::Newton
    pub fn new(
        bp_values: &'input [f64],
        scales: &'input [f64],
        num_replicates: &'input [usize],
    ) -> Self {
        Self {
            bp_values,
            scales,
            num_replicates,
        }
    }

    /// Gradient component of the sum of the function, parameterized with the inner pi function.
    #[inline]
    fn gradient_sum_function(&self, c: f64, d: f64) -> (f64, f64) {
        let mut gradient_c = 0.0;
        let mut gradient_d = 0.0;

        for (&count, (&scale, &num_replicates)) in self
            .bp_values
            .iter()
            .zip(self.scales.iter().zip(self.num_replicates))
        {
            let pi = Self::pi_k(c, d, scale);

            if pi > 0.0 && pi < 1.0 {
                let scale_root = scale.sqrt();
                let gradient_core = -pdf(d * scale_root + c / scale_root)
                    * (count - num_replicates as f64 * pi)
                    / (pi * (1.0 - pi));

                gradient_c += gradient_core / scale_root;
                gradient_d += gradient_core * scale_root;
            } else {
                // prevent division by zero and other numerical issues
                continue;
            }
        }

        (gradient_c, gradient_d)
    }

    #[inline]
    fn hessian_sum_function(
        bp_values: &[f64],
        scales: &[f64],
        replications: &[usize],
        c: f64,
        d: f64,
    ) -> (f64, f64, f64) {
        let mut hessian_cc = 0.0;
        let mut hessian_dc = 0.0;
        let mut hessian_dd = 0.0;
        for (&count, (&scale, &num_replicates)) in
            bp_values.iter().zip(scales.iter().zip(replications))
        {
            let pi = Self::pi_k(c, d, scale);
            if pi > 0.0 && pi < 1.0 {
                let core = Self::hessian_core(c, d, scale, num_replicates, count);

                hessian_cc += core / scale;
                hessian_dc += core;
                hessian_dd += core * scale;
            } else {
                // prevent division by zero and other numerical issues
                continue;
            }
        }

        (hessian_cc, hessian_dc, hessian_dd)
    }

    /// Likelihood cumulative distribution function of the two parameters d, c.
    ///
    /// For details refer to https://doi.org/10.1080/10635150290069913 Appendix 9.
    #[inline(always)]
    fn pi_k(c: f64, d: f64, scale: f64) -> f64 {
        let scale_root = scale.sqrt();

        cdf(-(d * scale_root + c / scale_root))
    }

    /// The common part of all three hessian derivatives
    #[inline(always)]
    fn hessian_core(c: f64, d: f64, scale: f64, num_replicates: usize, count: f64) -> f64 {
        let pi_k = Self::pi_k(c, d, scale);

        let scale_root = scale.sqrt();
        let linear = d * scale_root + c / scale_root;
        let density = pdf(linear);

        // careful with numerical details here:
        // pi_k * pi_k can become zero, while pi_k is still greater than zero, so we must not
        // compute X * (pi_k * pi_k) at any point.
        // The same is true for (1.0 - pi_k), so we must not compute X * (1 - pi_k)², and if X
        // is pi_k², we need to be extra careful, doing multiple divisions instead of multiplications
        density
            * (density * (-count + 2.0 * count * pi_k - num_replicates as f64 * pi_k * pi_k)
                / pi_k
                / pi_k
                / (1.0 - pi_k)
                / (1.0 - pi_k)
                + linear * (count - num_replicates as f64 * pi_k) / (pi_k * (1.0 - pi_k)))
    }
}

/// Estimate the parameters `d` (signed distance) and `c` (a curvature constant) which are used
/// in the AU p-value estimation using the Newton-Raphson method for a single tree.
///
/// # Parameters
/// - `tree_bp` all BP values for the tree, one for each bootstrap scaling factor
/// - `scales` the bootstrap scaling factors
/// - `replication_counts` the number of bootstrap replicates for each scaling factor
/// - `c` the initial value for c
/// - `d` the initial value for d
///
/// # Return
/// Returns a tuple containing the `c` and `d` value estimates for the tree.
///
/// # References
/// For details refer to https://doi.org/10.1080/10635150290069913, Appendix 9
pub fn fit_model_to_tree(
    bootstrap_counts: &[f64],
    scales: &[f64],
    replication_counts: &[usize],
    c: f64,
    d: f64,
) -> Result<(f64, f64), Error> {
    let problem = NewtonProblem::new(bootstrap_counts, scales, replication_counts);

    let init = Vec2(c, d);
    let solver = Newton::<f64>::new();

    let result = Executor::new(problem, solver)
        .configure(|state| state.param(init).max_iters(30))
        .run()?;

    let Some(&Vec2(c, d)) = result.state().get_best_param() else {
        panic!("solver returned None")
    };

    Ok((c, d))
}

/// Estimate the parameters `d` (signed distance) and `c` (a curvature constant) which are used
/// in the AU p-value estimation using the Newton-Raphson method for many trees.
///
/// # Parameters
/// - `bootstrap_replicates` A set of matrices of bootstrap replicates of all input sequences, one
///   matrix per bootstrap scale.
/// - `start_params` any iterable that contains a pair of starting parameters for `(c, d)`,
///   which are used to start the newton optimization.
///
/// # Parallelization
/// A parallel version of this function is available with the `rayon` feature as
/// [`par_fit_model_newton`].
/// Note that this step is so fast that parallelization is usually worth it only at several
/// thousand trees.
///
/// # Return
/// Returns a vector of tuples containing the `c` and `d` value estimates for each tree.
///
/// # References
/// For details refer to <https://doi.org/10.1080/10635150290069913>, Appendix 9
///
/// See also [`fit_model_wls`].
///
/// [`par_fit_model_newton`]: par_fit_model_newton
/// [`fit_model_wls`]: super::fit_model_wls
pub fn fit_model_newton<I: IntoIterator<Item = (f64, f64)>>(
    bootstrap_replicates: &BootstrapReplicates,
    start_params: I,
) -> Result<Vec<(f64, f64)>, Error> {
    (0..bootstrap_replicates.num_trees())
        .zip(start_params)
        .map(|(tree_index, (c, d))| {
            fit_model_to_tree(
                &bootstrap_replicates.compute_bp_values(tree_index, 0.0),
                bootstrap_replicates.scales(),
                bootstrap_replicates.replication_counts(),
                c,
                d,
            )
        })
        .collect::<Result<Vec<(f64, f64)>, Error>>()
}

/// Estimate the parameters `d` (signed distance) and `c` (a curvature constant) which are used
/// in the AU p-value estimation using the Newton-Raphson method for many trees in parallel.
/// Note that this step is so fast that parallelization is usually worth it only at several
/// thousand trees.
///
/// # Parameters
/// - `bootstrap_replicates`: A set of matrices of bootstrap replicates of all input sequences, one
///   matrix per bootstrap scale.
/// - `start_params` any iterable that contains a pair of starting parameters for `(c, d)`,
///   which are used to start the newton optimization.
///
/// # Return
/// Returns a vector of tuples containing the `c` and `d` value estimates for each tree.
///
/// # Parallelization
/// This method uses the global rayon thread pool.
///
/// # References
/// For details refer to <https://doi.org/10.1080/10635150290069913>, Appendix 9
///
/// See also [`par_fit_model_wls`].
///
/// [`par_fit_model_wls`]: super::par_fit_model_wls
#[cfg(feature = "rayon")]
pub fn par_fit_model_newton<I>(
    bootstrap_replicates: &BootstrapReplicates,
    start_params: I,
) -> Result<Vec<(f64, f64)>, Error>
where
    I: rayon::iter::IntoParallelIterator<Item = (f64, f64)>,
    <I as rayon::iter::IntoParallelIterator>::Iter: rayon::iter::IndexedParallelIterator,
{
    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

    (0..bootstrap_replicates.num_trees())
        .into_par_iter()
        .zip(start_params)
        .map(|(tree_index, (c, d))| {
            fit_model_to_tree(
                &bootstrap_replicates.compute_bp_values(tree_index, 0.0),
                bootstrap_replicates.scales(),
                bootstrap_replicates.replication_counts(),
                c,
                d,
            )
        })
        .collect::<Result<Vec<(f64, f64)>, Error>>()
}
