//! Implementation of datatypes and functions used by the Gauss-Newton solver which estimates
//! the `c` and `d` values for the AU test.

use crate::BpTable;
use crate::au::math::{Matrix2byN, Vec2, inv_cdf, pdf};
use crate::bootstrap::DEFAULT_REPLICATES;
use argmin::core::{Error, Executor, Jacobian, Operator, State};
use argmin::solver::gaussnewton::GaussNewton;

pub(super) struct RssProblem<'tree> {
    /// The BP values for the given problem instance (i.e., of one tree for which we are calculating
    /// the AU p-value). One BP value per scaling constant.
    bp_values: &'tree [f64],

    /// The scaling constants of the multiscale bootstrap process.
    scales: &'tree [f64],

    /// The variance weights resulting from the BP values. Since these do not depend on `c` and `d`,
    /// we can pre-calculate them in the constructor
    weights: Vec<f64>,
}

impl<'tree> Jacobian for RssProblem<'tree> {
    type Param = Vec2;
    type Jacobian = Matrix2byN;

    fn jacobian(&self, param: &Self::Param) -> Result<Self::Jacobian, Error> {
        let &Vec2(c, d) = param;
        let mut row1 = vec![0.0; self.bp_values.len()].into_boxed_slice();
        let mut row2 = vec![0.0; self.bp_values.len()].into_boxed_slice();

        self.weights
            .iter()
            .zip(self.bp_values.iter())
            .zip(self.scales.iter())
            .enumerate()
            .for_each(|(i, ((&w, &bp), &scale))| {
                row1[i] = self.gradient_c(c, d, w, bp, scale);
                row2[i] = self.gradient_d(c, d, w, bp, scale);
            });

        Ok(Matrix2byN::new(row1, row2))
    }
}

impl<'tree> Operator for RssProblem<'tree> {
    type Param = Vec2;
    type Output = Vec<f64>;

    /// Compute the sum of squared residuals using the given parameters `c` and `d`.
    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let &Vec2(c, d) = param;

        let residuals = self
            .weights
            .iter()
            .zip(self.bp_values.iter())
            .zip(self.scales.iter())
            .map(|((&w, &bp), &scale)| {
                let scale_root = scale.sqrt();
                let x = d * scale_root + c / scale_root;
                let f = x - inv_cdf(1.0 - bp);
                w * f * f
            })
            .collect();

        Ok(residuals)
    }
}

impl<'tree> RssProblem<'tree> {
    pub(super) fn new(bp_values: &'tree [f64], scales: &'tree [f64]) -> Self {
        let weights = Self::calculate_weights(bp_values);
        Self {
            bp_values,
            scales,
            weights,
        }
    }

    /// Calculate the inverse variance of the BP values which are used as weights in the RSS
    /// optimization.
    ///
    /// For details refer to https://doi.org/10.1080/10635150290069913.
    fn calculate_weights(bp: &[f64]) -> Vec<f64> {
        bp.iter()
            .map(|&x| {
                let d = pdf(inv_cdf(x));
                (d * d * DEFAULT_REPLICATES as f64) / (x * (1.0 - x))
            })
            .collect()
    }

    /// Gradient component with respect to `c` of the residual sum function, for a specific residual
    /// given through its weight, bp value and scale.
    fn gradient_c(&self, c: f64, d: f64, weight: f64, bp: f64, scale: f64) -> f64 {
        let scale_root = scale.sqrt();
        2.0 * weight * (c + scale * d - scale_root * inv_cdf(1.0 - bp)) / scale
    }

    /// Gradient component with respect to `d` of the residual sum function, for a specific residual
    /// given through its weight, bp value and scale.
    fn gradient_d(&self, c: f64, d: f64, weight: f64, bp: f64, scale: f64) -> f64 {
        let scale_root = scale.sqrt();
        2.0 * weight * (c + scale * d - scale_root * inv_cdf(1.0 - bp))
    }
}

pub fn estimate_curv_dist_rss(bp_values: &BpTable) -> Result<Vec<(f64, f64)>, Error> {
    let estimates = (0..bp_values.num_trees)
        .map(|i| {
            let problem = RssProblem::new(bp_values.tree_bp_values(i), bp_values.scales());
            let init = Vec2(1.0, 1.0);
            let solver = GaussNewton::<f64>::new().with_gamma(0.1).unwrap();

            // RSS usually converges, so the high limit is usually not reached.
            let result = Executor::new(problem, solver)
                .configure(|state| state.param(init).max_iters(1000))
                .run()?;

            let Some(&Vec2(c, d)) = result.state().get_best_param() else {
                panic!("solver returned None")
            };

            Ok((c, d))
        })
        .collect::<Result<Vec<(f64, f64)>, Error>>();

    estimates
}
