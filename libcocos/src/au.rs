use crate::BpTable;
use crate::bootstrap::DEFAULT_REPLICATES;
use argmin::core::State;
use argmin::core::{Executor, Gradient, Hessian};
use argmin::solver::newton::Newton;
use argmin_math::{ArgminDot, ArgminInv, ArgminMul, ArgminSub, Error};
use statrs::distribution::{ContinuousCDF, Normal};
use std::f64::consts::PI;

#[derive(Debug, Clone)]
struct Vec2(f64, f64);

#[derive(Debug, Clone)]
struct Matrix2by2(f64, f64, f64, f64);

impl ArgminMul<Vec2, Vec2> for f64 {
    fn mul(&self, other: &Vec2) -> Vec2 {
        Vec2(other.0 * self, other.1 * self)
    }
}

impl ArgminDot<Vec2, f64> for Vec2 {
    fn dot(&self, other: &Vec2) -> f64 {
        self.0 * other.0 + self.1 * other.1
    }
}

impl ArgminSub<Vec2, Vec2> for Vec2 {
    fn sub(&self, other: &Vec2) -> Vec2 {
        Vec2(self.0 - other.0, self.1 - other.1)
    }
}

impl ArgminInv<Matrix2by2> for Matrix2by2 {
    fn inv(&self) -> Result<Matrix2by2, argmin_math::Error> {
        let det = self.0 * self.3 - self.1 * self.2;
        Ok(Matrix2by2(
            self.3 / det,
            -self.1 / det,
            -self.2 / det,
            self.0 / det,
        ))
    }
}

impl ArgminDot<Vec2, Vec2> for Matrix2by2 {
    fn dot(&self, other: &Vec2) -> Vec2 {
        Vec2(
            self.0 * other.0 + self.1 * other.1,
            self.2 * other.0 + self.3 * other.1,
        )
    }
}

struct DCProblem<'tree> {
    scales: &'tree [f64],
    bp_values: &'tree [f64],
}

impl<'tree> Gradient for DCProblem<'tree> {
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

impl<'tree> Hessian for DCProblem<'tree> {
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

impl<'tree> DCProblem<'tree> {
    const fn new(bp_values: &'tree [f64], scales: &'tree [f64]) -> Self {
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
        bp_values
            .iter()
            .zip(scales)
            .map(|(&bp, &scale)| {
                DEFAULT_REPLICATES as f64 * pi_gradient(c, d, scale) / Self::pi_k(c, d, scale)
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
        bp_values
            .iter()
            .zip(scales)
            .map(|(&bp, &scale)| {
                let pi_k = Self::pi_k(c, d, scale);
                let pi_k_sq = pi_k * pi_k;

                DEFAULT_REPLICATES as f64
                    * (-(1.0 - bp) * pi_grad_1_1(c, d, scale) * pi_grad_1_2(c, d, scale) / pi_k_sq
                        - bp * pi_grad_2_1(c, d, scale) * pi_grad_2_2(c, d, scale) / pi_k_sq
                        + (1.0 - bp) * pi_hess_1(c, d, scale) / pi_k
                        + bp * pi_hess_2(c, d, scale) / pi_k)
            })
            .sum::<f64>()
    }

    #[inline(always)]
    fn exponential(c: f64, d: f64, scale_root: f64) -> f64 {
        let frac = d * scale_root + c / scale_root;

        (-(frac * frac) / 2.0).exp()
    }

    #[inline(always)]
    fn gradient_inner_factor(c: f64, d: f64, scale: f64) -> f64 {
        c + scale * d
    }

    #[inline(always)]
    fn hessian_inner_factor(c: f64, d: f64, scale_root: f64) -> f64 {
        let f = c / scale_root + scale_root * d;
        f * f
    }

    #[inline(always)]
    fn pi_k(c: f64, d: f64, scale: f64) -> f64 {
        let scale_root = scale.sqrt();
        let pi2root = (2.0 * PI).sqrt();

        1.0 - (Self::exponential(c, d, scale_root) / pi2root)
    }

    #[inline(always)]
    fn gradient_pi_c(c: f64, d: f64, scale: f64) -> f64 {
        let pi2root = (2.0 * PI).sqrt();
        let scale_root = scale.sqrt();

        -Self::exponential(c, d, scale_root) * Self::gradient_inner_factor(c, d, scale)
            / (scale * pi2root)
    }

    #[inline(always)]
    fn gradient_pi_d(c: f64, d: f64, scale: f64) -> f64 {
        let pi2root = (2.0 * PI).sqrt();
        let scale_root = scale.sqrt();

        -Self::exponential(c, d, scale_root) * Self::gradient_inner_factor(c, d, scale) / pi2root
    }

    #[inline(always)]
    fn hessian_pi_cc(c: f64, d: f64, scale: f64) -> f64 {
        let pi2root = (2.0 * PI).sqrt();
        let scale_root = scale.sqrt();
        let exponential = Self::exponential(c, d, scale_root);
        let scaled_pi_root = pi2root * scale;

        (exponential * Self::hessian_inner_factor(c, d, scale_root) / scaled_pi_root)
            - (exponential / scaled_pi_root)
    }

    #[inline(always)]
    fn hessian_pi_cd(c: f64, d: f64, scale: f64) -> f64 {
        let pi2root = (2.0 * PI).sqrt();
        let scale_root = scale.sqrt();
        let exponential = Self::exponential(c, d, scale_root);

        (exponential * Self::hessian_inner_factor(c, d, scale_root) / pi2root)
            - (exponential / pi2root)
    }

    #[inline(always)]
    fn hessian_pi_dd(c: f64, d: f64, scale: f64) -> f64 {
        let pi2root = (2.0 * PI).sqrt();
        let scale_root = scale.sqrt();
        let exponential = Self::exponential(c, d, scale_root);

        (scale * exponential * Self::hessian_inner_factor(c, d, scale_root) / pi2root)
            - (scale * exponential / pi2root)
    }
}

/// Estimate the parameters `d` (signed distance) and `c` (a curvature constant) which are used
/// in the AU p-value estimation.
///
/// For details refer to https://doi.org/10.1080/10635150290069913.
pub fn estimate_dc(bp_values: &BpTable) -> Result<Vec<(f64, f64)>, Error> {
    let cd_vals = (0..10)
        .map(|i| {
            let problem = DCProblem::new(bp_values.tree_bp_values(i), bp_values.scales());
            let init = Vec2(1.0, 1.0);
            let solver = Newton::<f64>::new();

            let result = Executor::new(problem, solver)
                .configure(|state| state.param(init).max_iters(10))
                .run()?;

            let Some(&Vec2(c, d)) = result.state().get_best_param() else {
                panic!("solver returned None")
            };

            Ok((c, d))
        })
        .collect::<Result<Vec<(f64, f64)>, Error>>();

    cd_vals
}

pub fn get_au_value(bp_values: &BpTable) -> Result<Vec<f64>, Error> {
    let normal = Normal::new(0.0, 1.0);
    let results = estimate_dc(bp_values)?;

    results
        .iter()
        .map(|(c, d)| Ok(1.0 - normal?.cdf(d - c)))
        .collect()
}
