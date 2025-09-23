//! Implementations of mathematical operations and datatypes required for the numerical algorithms
//! used by the AU test.

use statrs::consts;
use statrs::function::erf;

#[derive(Debug, Clone)]
pub(super) struct Vec2(pub f64, pub f64);

impl Vec2 {
    pub(crate) fn sub(&self, other: &Vec2) -> Vec2 {
        Vec2(self.0 - other.0, self.1 - other.1)
    }
}

#[derive(Debug, Clone)]
pub(super) struct Matrix2by2(pub f64, pub f64, pub f64, pub f64);

impl Matrix2by2 {
    pub(crate) fn inv(&self) -> Result<Matrix2by2, argmin_math::Error> {
        let det = self.0 * self.3 - self.1 * self.2;
        if det.abs() == 0.0 {
            println!("warning: hessian is singular");
        }

        Ok(Matrix2by2(
            self.3 / det,
            -self.1 / det,
            -self.2 / det,
            self.0 / det,
        ))
    }

    pub(crate) fn dot(&self, other: &Vec2) -> Vec2 {
        Vec2(
            self.0 * other.0 + self.1 * other.1,
            self.2 * other.0 + self.3 * other.1,
        )
    }
}

/// The cumulative distribution function of the standard normal distribution.
#[inline(always)]
pub(crate) fn cdf(x: f64) -> f64 {
    0.5 * erf::erfc((-x) / (std::f64::consts::SQRT_2))
}

/// The probability density function of the standard normal distribution.
#[inline(always)]
pub(crate) fn pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / consts::SQRT_2PI
}

// The probability covered by the quantile given as `x` of the standard normal distribution.
#[inline(always)]
pub(crate) fn quantile(x: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&x), "Invalid quantile");
    -(std::f64::consts::SQRT_2 * erf::erfc_inv(2.0 * x))
}
