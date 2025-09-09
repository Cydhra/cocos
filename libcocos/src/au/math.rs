//! Implementations of mathematical operations and datatypes required for the numerical algorithms
//! used by the AU test.

use argmin_math::{ArgminDot, ArgminInv, ArgminMul, ArgminSub};
use statrs::consts;
use statrs::function::erf;

#[derive(Debug, Clone)]
pub(super) struct Vec2(pub f64, pub f64);

#[derive(Debug, Clone)]
pub(super) struct Matrix2by2(pub f64, pub f64, pub f64, pub f64);

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
}

impl ArgminDot<Vec2, Vec2> for Matrix2by2 {
    fn dot(&self, other: &Vec2) -> Vec2 {
        Vec2(
            self.0 * other.0 + self.1 * other.1,
            self.2 * other.0 + self.3 * other.1,
        )
    }
}

impl ArgminMul<f64, Vec2> for Vec2 {
    fn mul(&self, other: &f64) -> Vec2 {
        Vec2(self.0 * *other, self.1 * *other)
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
    debug_assert!(0.0 <= x && x <= 1.0, "Invalid quantile");
    -(std::f64::consts::SQRT_2 * erf::erfc_inv(2.0 * x))
}
