//! The library's error type.
//! The type comprises all possible mathematical errors and edge cases which could compromise the
//! test results.

use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum MathError {
    HessianSingular,
    ConvergenceFailed { p_value: f64 },
}

impl Display for MathError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MathError::HessianSingular => {
                write!(f, "Hessian matrix is singular")
            }
            MathError::ConvergenceFailed { .. } => {
                write!(f, "Convergence of likelihood function failed")
            }
        }
    }
}

impl Error for MathError {}
