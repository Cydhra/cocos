//! The library's error type.
//! The type comprises all possible mathematical errors and edge cases which could compromise the
//! test results.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// The error type of the AU test.
/// The different variants describe different errors that can come up during the calculation.
/// Some errors still contain a p-value, which may or may not be numerically stable.
#[derive(Debug)]
pub enum MathError {
    /// Encountered during the Newton optimization if the Hessian is singular, and can thus not be
    /// inverted. This should be handled by the AU test by just assuming that the Newton method
    /// converged at a local extremum.
    /// However, this can also happen if the starting parameters are ill-defined.
    HessianSingular,

    /// Encountered when the convergence test of the p-value does not converge.
    /// This can mean the likelihood surface is degenerated and the p-value is likely not numerically
    /// stable.
    ConvergenceFailed {
        /// The last p-value obtained during the convergence test. It may be unstable if the
        /// likelihood surface is severely degenerated.
        p_value: f64,
    },
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
