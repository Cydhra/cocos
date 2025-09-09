//! Implementations of mathematical operations and datatypes required for the numerical algorithms
//! used by the AU test.

use crate::vectors::dot_prod;
use argmin::core::Error;
use argmin_math::{ArgminDot, ArgminInv, ArgminMul, ArgminSub, ArgminTranspose};
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

/// A 2 by M matrix with runtime-defined column number.
#[derive(Debug, Clone)]
pub(super) struct Matrix2byN {
    row1: Box<[f64]>,
    row2: Box<[f64]>,
    transposed: bool,
}

impl Matrix2byN {
    pub(crate) fn new(row1: Box<[f64]>, row2: Box<[f64]>) -> Self {
        Self {
            row1,
            row2,
            transposed: false,
        }
    }
}

impl FromIterator<(f64, f64)> for Matrix2byN {
    fn from_iter<T: IntoIterator<Item = (f64, f64)>>(iter: T) -> Self {
        let mut row1 = Vec::new();
        let mut row2 = Vec::new();
        iter.into_iter().for_each(|(c, d)| {
            row1.push(c);
            row2.push(d);
        });

        Matrix2byN::new(row1.into_boxed_slice(), row2.into_boxed_slice())
    }
}

/// Pseudo implementation of transpose. Since we only ever use transpose in the Gauss-Newton method,
/// and it only ever applies it to the Jacobi Matrix, and the Jacobi Matrix only ever is 2 by N,
/// and we only need that to multiply it by itself, or with the residuals, we can just implicitly assume
/// that it is transposed during the multiplication.
impl ArgminTranspose<Matrix2byN> for Matrix2byN {
    fn t(mut self) -> Matrix2byN {
        self.transposed = !self.transposed;
        self
    }
}

impl ArgminDot<Matrix2byN, Matrix2byN> for Matrix2byN {
    fn dot(&self, other: &Matrix2byN) -> Matrix2byN {
        if self.transposed || other.transposed {
            let row1 = [
                dot_prod(&self.row1, &other.row1),
                dot_prod(&self.row1, &other.row2),
            ];
            let row2 = [
                dot_prod(&self.row2, &other.row1),
                dot_prod(&self.row2, &other.row2),
            ];
            Matrix2byN::new(Box::new(row1), Box::new(row2))
        } else {
            assert_eq!(
                self.row1.len(),
                2,
                "Non-transposed matrix multiplication is implemented only for 2 by 2 matrices"
            );
            let r11 = self.row1[0] * other.row1[0] + self.row1[1] * other.row2[0];
            let r12 = self.row1[0] * other.row1[1] + self.row1[1] * other.row2[1];
            let r21 = self.row2[0] * other.row1[0] + self.row2[1] * other.row2[0];
            let r22 = self.row2[0] * other.row1[1] + self.row2[1] * other.row2[1];

            Matrix2byN::new(Box::new([r11, r12]), Box::new([r21, r22]))
        }
    }
}

impl ArgminDot<Vec<f64>, Vec2> for Matrix2byN {
    fn dot(&self, other: &Vec<f64>) -> Vec2 {
        assert_eq!(self.row1.len(), other.len());

        let c = dot_prod(&self.row1, &other);
        let d = dot_prod(&self.row2, &other);

        Vec2(c, d)
    }
}

impl ArgminDot<Vec2, Vec2> for Matrix2byN {
    fn dot(&self, other: &Vec2) -> Vec2 {
        assert_eq!(self.row1.len(), 2);

        let c = self.row1[0] * other.0 + self.row1[1] * other.1;
        let d = self.row2[0] * other.0 + self.row2[1] * other.1;

        Vec2(c, d)
    }
}

impl ArgminInv<Matrix2byN> for Matrix2byN {
    fn inv(&self) -> Result<Matrix2byN, Error> {
        assert_eq!(
            self.row1.len(),
            2,
            "inversion is only implemented for 2 by 2 matrices"
        );
        let determinant = self.row1[0] * self.row2[1] - self.row1[1] * self.row2[0];
        Ok(Matrix2byN::new(
            Box::new([self.row2[1] / determinant, -self.row1[1] / determinant]),
            Box::new([-self.row2[0] / determinant, self.row1[0] / determinant]),
        ))
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

/// The inverse CDF or quantile function of the standard normal distribution.
#[inline(always)]
pub(crate) fn inv_cdf(x: f64) -> f64 {
    assert!(0.0 <= x && x <= 1.0);
    -(std::f64::consts::SQRT_2 * erf::erfc_inv(2.0 * x))
}
