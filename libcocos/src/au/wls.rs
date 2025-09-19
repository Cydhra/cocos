//! This module solves the WLS problem for the AU test. Since the problem only has two parameters,
//! we can solve the matrix analytically and need not use a numerical solution.

use crate::BootstrapReplicates;
use crate::au::math::{pdf, quantile};

/// Problem instance for the weighted least squares regression that fits curvature and signed
/// distance to the observed BP values.
pub(super) struct WlsProblem<'input> {
    bootstrap_replicates: &'input BootstrapReplicates,
    scale_roots: Box<[f64]>,
}

impl<'input> WlsProblem<'input> {
    pub(super) fn new(bootstrap_replicates: &'input BootstrapReplicates) -> Self {
        let scale_roots = bootstrap_replicates
            .scales
            .iter()
            .map(|&r| r.sqrt())
            .collect::<Box<[_]>>();

        Self {
            bootstrap_replicates,
            scale_roots,
        }
    }

    /// Compute the transformations from observed BP probability into a signed distance (including
    /// curvature correction).
    /// Specifically, Shimodaira's model assumes that the probability that a tree is the correct
    /// model for the observed sequence data is `cdf(corrected distance)`,
    /// where the `corrected distance` is the distance of the tree's log-likelihood from the boundary
    /// of the region (in the space of all log-likelihoods) that belongs to the tree.
    /// This is because if the distance towards the boundary of region `i` is very short,
    /// we are more likely to assign tree `i` as the correct tree due to sampling error
    /// (i.e., sampling error in the infinite sites model).
    ///
    /// This method takes a Bootstrap Proportion `bp` as an observation of this probability and
    /// inverts the probability function to obtain the `corrected distance`.
    /// Next we can fit two parameters `c` and `d` to the observed distances at all scales
    /// (hence why we bootstrap at multiple scales), to obtain the original `c` and `d` that make
    /// up `corrected distance = d * sqrt(scale) + c / sqrt(scale)`.
    #[inline(always)]
    fn compute_distance(bp: f64) -> f64 {
        -quantile(bp)
    }

    //noinspection DuplicatedCode
    /// Fit two parameters `c: curvature` and `d: signed distance` to the observed probability
    /// of tree `i`.
    /// That is, given the bootstrap proportion (BP) of tree `i` at each scale, assume the BP value is
    /// proportional to the probability `Pr` that tree `i` is the best tree for the sequence data,
    /// and now fit `c` and `d` to `Pr = cdf(d * sqrt(scale) + c / sqrt(scale))`.
    ///
    /// # Return
    /// Returns Ok((c, d)) if the regression was computed successfully, or Err((0.0, 0.0)) if there
    /// is no solution.
    fn fit_parameters_to_tree(&self, i: usize) -> Result<(f64, f64), (f64, f64)> {
        // compute weights and observed distances
        let mut observed_distances =
            vec![0.0; self.bootstrap_replicates.num_scales()].into_boxed_slice();
        let mut weight_vector =
            vec![0.0; self.bootstrap_replicates.num_scales()].into_boxed_slice();

        let mut non_zero_observations = 0;
        for (scale_index, &count) in self
            .bootstrap_replicates
            .compute_bp_values(i, 0.0)
            .iter()
            .enumerate()
        {
            let bp = count / self.bootstrap_replicates.replication_counts[scale_index] as f64;
            if bp < 1E-16 {
                // guard against division by zero, both weight and distance is observed
                // to be zero in those cases
                continue;
            }
            observed_distances[scale_index] = Self::compute_distance(bp);
            let x = pdf(observed_distances[scale_index]);
            weight_vector[scale_index] =
                (x * x * self.bootstrap_replicates.replication_counts()[scale_index] as f64)
                    / ((1.0 - bp) * bp);
            non_zero_observations += 1;
        }

        // we can't compute estimates if bp is zero.
        if non_zero_observations < 2 {
            return Err((0.0, 0.0));
        }

        // compute weighted model matrix

        // The model assumes that the probability of tree `i` being the correct tree is
        // cdf(signed distance), and the signed distance is `d * sqrt(scale) + c / sqrt(scale)`.
        // This is weighted according to the paper by Shimodaira, and then we minimize the squared
        // residuals when fitting this model to the observed distances.
        // Here we compute the model matrix as M = X * W * Xtop, where W is the diagonal weight matrix
        // and X is the model, which is a two-row matrix containing `sqrt(scale)` in the top row
        // (modelling d) and `1/sqrt(scale)` in the bottom row (modelling c). We compute the product
        // directly to save some unnecessary dot products

        // M is:    (model_d      off_diagonal)
        //          (off_diagonal model_c     )

        let model_d = weight_vector
            .iter()
            .zip(self.bootstrap_replicates.scales().iter())
            .map(|(&w, &scale)| w * scale)
            .sum::<f64>();
        let off_diagonal = weight_vector.iter().sum::<f64>();
        let model_c = weight_vector
            .iter()
            .zip(self.bootstrap_replicates.scales().iter())
            .map(|(&w, &scale)| w / scale)
            .sum::<f64>();

        // compute the observations

        // we have the observed distances and have to multiply them with sqrt(scale) and 1/sqrt(scale) and the weights
        let observed_d = observed_distances
            .iter()
            .zip(self.scale_roots.iter())
            .zip(weight_vector.iter())
            .map(|((&dist, &s), &weight)| weight * s * dist)
            .sum::<f64>();

        let observed_c = observed_distances
            .iter()
            .zip(self.scale_roots.iter())
            .zip(weight_vector.iter())
            .map(|((&dist, &s), &weight)| weight * (1.0 / s) * dist)
            .sum::<f64>();

        // compute estimates for c and d by inverting the model matrix and multiplying it with
        // the observations to solve M * beta = observations, where beta is the vector containing
        // c and d. The following is the multiplication `inv(M) * observations`.

        let determinant = model_d * model_c - off_diagonal * off_diagonal;
        if determinant.abs() == 0.0 {
            return Err((0.0, 0.0));
        }

        let estimate_d = (model_c * observed_d - off_diagonal * observed_c) / determinant;
        let estimate_c = (-off_diagonal * observed_d + model_d * observed_c) / determinant;

        Ok((estimate_c, estimate_d))
    }
}

/// Fit curvature (`c`) and signed distance (`d`) parameters to observed bootstrap probabilities
/// using the Weighted Least Squares method.
/// The bootstrap values are given in a [`BpTable`] and one pair of parameters is fit per
/// replicate set (i.e., per tree).
///
/// # Parameters
/// - `bootstrap_replicates` A set of matrices of bootstrap replicates of all input sequences, one
///   matrix per bootstrap scale.
/// - `threshold` The threshold to use for evaluating bootstrap proportions, where a threshold of
///   zero means canonical BP values.
///
/// # Return
/// A vector of `Result<(f64, f64), (f64, f64)>`.
/// If the regression was calculated successfully, the result is `Ok((c, d))`.
/// If the regression has no solution, for example because there are not enough BP ratios greater
/// than zero, an Error with a dummy value is returned.
/// In this case, the parameters should be estimated using Newton's method with the dummy value
/// as the initial guess.
///
/// # Parallel Fitting
/// A parallel version of this function is available with the `rayon` feature.
/// Note that this step is so fast that parallelization is usually worth it only at several
/// thousand trees.
///
/// # References
/// For details refer to <https://doi.org/10.1080/10635150290069913>.
///
/// See also [`fit_model_newton`].
///
/// [`BpTable`]: BpTable
/// [`fit_model_newton`]: super::fit_model_newton
pub fn fit_model_wls(
    bootstrap_replicates: &BootstrapReplicates,
) -> Vec<Result<(f64, f64), (f64, f64)>> {
    let problem = WlsProblem::new(bootstrap_replicates);

    let mut result = Vec::with_capacity(bootstrap_replicates.num_trees());
    for tree in 0..bootstrap_replicates.num_trees() {
        result.push(problem.fit_parameters_to_tree(tree));
    }

    result
}

/// Fit curvature (`c`) and signed distance (`d`) parameters to observed bootstrap probabilities
/// using the Weighted Least Squares method in parallel.
/// Note that this step is so fast that parallelization is usually worth it only at several
/// thousand trees.
///
/// For a full explanation, see [`fit_model_wls`].
///
/// [``fit_model_wls`]: fit_model_wls
#[cfg(feature = "rayon")]
pub fn par_fit_model_wls(
    bootstrap_replicates: &BootstrapReplicates,
) -> Vec<Result<(f64, f64), (f64, f64)>> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let problem = WlsProblem::new(bootstrap_replicates);

    (0..bootstrap_replicates.num_trees())
        .into_par_iter()
        .map(|tree| problem.fit_parameters_to_tree(tree))
        .collect()
}
