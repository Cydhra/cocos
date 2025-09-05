//! Implementation of datatypes and functions used by the Newton-Raphson solver which estimates
//! the `c` and `d` values for the AU test.

use crate::BpTable;
use crate::au::math::{Matrix2by2, Vec2, cdf, pdf, pdf_diff};
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
        let gradient_c =
            Self::gradient_sum_function(self.bp_values, self.scales, c, d, Self::gradient_pi_c);
        let gradient_d =
            Self::gradient_sum_function(self.bp_values, self.scales, c, d, Self::gradient_pi_d);
        Ok(Vec2(gradient_c, gradient_d))
    }
}

impl<'tree> Hessian for NewtonProblem<'tree> {
    type Param = Vec2;
    type Hessian = Matrix2by2;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error> {
        let &Vec2(c, d) = param;

        let hess_cc = Self::hessian_sum_function(
            self.bp_values,
            self.scales,
            c,
            d,
            Self::gradient_pi_c,
            Self::gradient_pi_c,
            Self::gradient_pi_c,
            Self::gradient_pi_c,
            Self::hessian_pi_cc,
            Self::hessian_pi_cc,
        );
        let hess_cd = Self::hessian_sum_function(
            self.bp_values,
            self.scales,
            c,
            d,
            Self::gradient_pi_d,
            Self::gradient_pi_c,
            Self::gradient_pi_d,
            Self::gradient_pi_c,
            Self::hessian_pi_cd,
            Self::hessian_pi_cd,
        );
        let hess_dd = Self::hessian_sum_function(
            self.bp_values,
            self.scales,
            c,
            d,
            Self::gradient_pi_d,
            Self::gradient_pi_d,
            Self::gradient_pi_d,
            Self::gradient_pi_d,
            Self::hessian_pi_dd,
            Self::hessian_pi_dd,
        );

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
    fn gradient_sum_function(
        bp_values: &[f64],
        scales: &[f64],
        c: f64,
        d: f64,
        pi_gradient: fn(f64, f64, f64) -> f64,
    ) -> f64 {
        // negative of the actual gradient because we want to find a maximum instead of a minimum.
        -bp_values
            .iter()
            .zip(scales)
            .map(|(&bp, &scale)| {
                let pi = Self::pi_k(c, d, scale);

                DEFAULT_REPLICATES as f64 * pi_gradient(c, d, scale) * (bp - pi) / (pi * (1.0 - pi))
            })
            .sum::<f64>()
    }

    #[inline]
    fn hessian_sum_function(
        bp_values: &[f64],
        scales: &[f64],
        c: f64,
        d: f64,
        pi_grad_1_1: fn(f64, f64, f64) -> f64,
        pi_grad_1_2: fn(f64, f64, f64) -> f64,
        pi_grad_2_1: fn(f64, f64, f64) -> f64,
        pi_grad_2_2: fn(f64, f64, f64) -> f64,
        pi_hess_1: fn(f64, f64, f64) -> f64,
        pi_hess_2: fn(f64, f64, f64) -> f64,
    ) -> f64 {
        // negative of the actual hessian because we want to find a maximum instead of a minimum.
        -bp_values
            .iter()
            .zip(scales)
            .map(|(&bp, &scale)| {
                let pi_k = Self::pi_k(c, d, scale);
                let pi_k_sq = pi_k * pi_k;

                DEFAULT_REPLICATES as f64
                    * (-(1.0 - bp) * pi_grad_1_1(c, d, scale) * pi_grad_1_2(c, d, scale)
                        / (1.0 - pi_k_sq)
                        - bp * pi_grad_2_1(c, d, scale) * pi_grad_2_2(c, d, scale)
                            / (1.0 - pi_k_sq)
                        - (1.0 - bp) * pi_hess_1(c, d, scale) / (1.0 - pi_k)
                        + bp * pi_hess_2(c, d, scale) / (1.0 - pi_k))
            })
            .sum::<f64>()
    }

    /// Likelihood cumulative distribution function of the two parameters d, c.
    ///
    /// For details refer to https://doi.org/10.1080/10635150290069913 Appendix 9.
    #[inline(always)]
    fn pi_k(c: f64, d: f64, scale: f64) -> f64 {
        let scale_root = scale.sqrt();

        1.0 - cdf(d * scale_root + c / scale_root)
    }

    /// Component of the gradient of [`pi_k`] with respect to the parameter `c`.
    ///
    /// [`pi_k`]: Self::pi_k
    #[inline(always)]
    fn gradient_pi_c(c: f64, d: f64, scale: f64) -> f64 {
        let scale_root = scale.sqrt();

        -pdf(d * scale_root + c / scale_root) / scale_root
    }

    /// Component of the gradient of [`pi_k`] with respect to the parameter `d`.
    ///
    /// [`pi_k`]: Self::pi_k
    #[inline(always)]
    fn gradient_pi_d(c: f64, d: f64, scale: f64) -> f64 {
        let scale_root = scale.sqrt();

        -pdf(d * scale_root + c / scale_root) * scale_root
    }

    /// Component of the hessian matrix of [`pi_k`] derived twice with respect to `c`.
    ///
    /// [`pi_k`]: Self::pi_k
    #[inline(always)]
    fn hessian_pi_cc(c: f64, d: f64, scale: f64) -> f64 {
        let scale_root = scale.sqrt();

        -pdf_diff(d * scale_root + c / scale_root) / scale
    }

    /// Component of the hessian matrix of [`pi_k`] derived once with respect to `c` and once to `d`.
    ///
    /// [`pi_k`]: Self::pi_k
    #[inline(always)]
    fn hessian_pi_cd(c: f64, d: f64, scale: f64) -> f64 {
        let scale_root = scale.sqrt();

        -pdf_diff(d * scale_root + c / scale_root)
    }

    /// Component of the hessian matrix of [`pi_k`] derived twice with respect to `d`.
    ///
    /// [`pi_k`]: Self::pi_k
    #[inline(always)]
    fn hessian_pi_dd(c: f64, d: f64, scale: f64) -> f64 {
        let scale_root = scale.sqrt();

        -pdf_diff(d * scale_root + c / scale_root) * scale
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
pub fn estimate_curv_dist_newton(bp_values: &BpTable) -> Result<Vec<(f64, f64)>, Error> {
    let cd_vals = (0..10)
        .map(|i| {
            let problem = NewtonProblem::new(bp_values.tree_bp_values(i), bp_values.scales());
            let init = Vec2(1.5, 2.0);
            let solver = Newton::<f64>::new().with_gamma(0.1)?;

            let result = Executor::new(problem, solver)
                .configure(|state| state.param(init).max_iters(800))
                .run()?;

            let Some(&Vec2(c, d)) = result.state().get_best_param() else {
                panic!("solver returned None")
            };

            Ok((c, d))
        })
        .collect::<Result<Vec<(f64, f64)>, Error>>();

    cd_vals
}
