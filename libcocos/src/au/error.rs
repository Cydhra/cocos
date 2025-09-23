use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum MathError {
    HessianSingular,
}

impl Display for MathError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MathError::HessianSingular => {
                write!(f, "Hessian matrix is singular")
            }
        }
    }
}

impl Error for MathError {}
