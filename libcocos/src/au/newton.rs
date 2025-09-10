//! Implementation of datatypes and functions used by the Newton-Raphson solver which estimates
//! the `c` and `d` values for the AU test.

use crate::BpTable;
use crate::au::math::{Matrix2by2, Vec2, cdf, pdf};
use crate::bootstrap::DEFAULT_REPLICATES;
use argmin::core::{Error, Executor, Gradient, Hessian, State};
use argmin::solver::newton::Newton;

struct NewtonProblem<'tree> {
    scales: &'tree [f64],
    bp_values: &'tree [f64],
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

impl<'tree> Hessian for NewtonProblem<'tree> {
    type Param = Vec2;
    type Hessian = Matrix2by2;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error> {
        let &Vec2(c, d) = param;

        let (hess_cc, hess_cd, hess_dd) =
            Self::hessian_sum_function(self.bp_values, self.scales, c, d);

        Ok(Matrix2by2(hess_cc, hess_cd, hess_cd, hess_dd))
    }
}

impl<'tree> NewtonProblem<'tree> {
    /// Create a new problem instance, which can be used in a [Newton Solver] to obtain values for the
    /// distance and curvature parameters of the AU test.
    ///
    /// For details refer to https://doi.org/10.1080/10635150290069913 Appendix 9.
    ///
    /// [Newton Solver]: argmin::solver::newton::Newton
    pub fn new(bp_values: &'tree [f64], scales: &'tree [f64]) -> Self {
        Self { bp_values, scales }
    }

    /// Gradient component of the sum of the function, parameterized with the inner pi function.
    #[inline]
    fn gradient_sum_function(&self, c: f64, d: f64) -> (f64, f64) {
        let mut gradient_c = 0.0;
        let mut gradient_d = 0.0;

        for (&bp, &scale) in self.bp_values.iter().zip(self.scales) {
            let pi = Self::pi_k(c, d, scale);

            if pi > 0.0 && pi < 1.0 {
                let scale_root = scale.sqrt();
                let gradient_core = -pdf(d * scale_root + c / scale_root)
                    * (DEFAULT_REPLICATES as f64 * bp - DEFAULT_REPLICATES as f64 * pi)
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
    fn hessian_sum_function(bp_values: &[f64], scales: &[f64], c: f64, d: f64) -> (f64, f64, f64) {
        let mut hessian_cc = 0.0;
        let mut hessian_dc = 0.0;
        let mut hessian_dd = 0.0;
        for (&bp, &scale) in bp_values.iter().zip(scales) {
            let count = DEFAULT_REPLICATES as f64 * bp;

            let pi = Self::pi_k(c, d, scale);
            if pi > 0.0 && pi < 1.0 {
                let core = Self::hessian_core(c, d, scale, count);

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
    fn hessian_core(c: f64, d: f64, scale: f64, count: f64) -> f64 {
        let pi_k = Self::pi_k(c, d, scale);

        let scale_root = scale.sqrt();
        let linear = d * scale_root + c / scale_root;
        let density = pdf(linear);

        // careful with numerical details here:
        // pi_k * pi_k can become zero, while pi_k is still greater than zero, so we must not
        // compute X * (pi_k * pi_k) at any point.
        // The same is true for (1.0 - pi_k), so we must not compute X * (1 - pi_k)², and if X
        // is pi_k², we need to mix the factors
        density
            * (density * (-count + 2.0 * count * pi_k - DEFAULT_REPLICATES as f64 * pi_k * pi_k)
                / (pi_k * (1.0 - pi_k) * pi_k * (1.0 - pi_k))
                + linear * (count - DEFAULT_REPLICATES as f64 * pi_k) / (pi_k * (1.0 - pi_k)))
    }
}

/// Estimate the parameters `d` (signed distance) and `c` (a curvature constant) which are used
/// in the AU p-value estimation using the Newton-Raphson method.
///
/// # Parameters
/// - `bp_values` a table containing all bp values for all trees in the AU test
///
/// # Return
/// Returns a vector of tuples containing the `c` and `d` value estimates for each tree.
///
/// For details refer to https://doi.org/10.1080/10635150290069913.
pub fn estimate_curv_dist_newton<I: IntoIterator<Item = (f64, f64)>>(
    bp_values: &BpTable,
    start_params: I,
) -> Result<Vec<(f64, f64)>, Error> {
    let cd_vals = (0..bp_values.num_trees())
        .zip(start_params.into_iter())
        .map(|(tree_index, (c, d))| {
            let problem =
                NewtonProblem::new(bp_values.tree_bp_values(tree_index), bp_values.scales());

            let init = Vec2(c, d);
            let solver = Newton::<f64>::new();

            let result = Executor::new(problem, solver)
                .configure(|state| state.param(init).max_iters(30))
                .run()?;

            let Some(&Vec2(c, d)) = result.state().get_best_param() else {
                panic!("solver returned None")
            };

            Ok((c, d))
        })
        .collect::<Result<Vec<(f64, f64)>, Error>>();

    cd_vals
}
