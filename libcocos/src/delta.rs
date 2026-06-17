//! This module contains types for normalized bootstrap tables.
//! After [bootstrapping], we have obtained a list of replicates.
//! Each replicate is a set of likelihoods, one for each input sequence.
//! We normalize these sets by computing the delta between each likelihood and the best
//! competing likelihood of the replicate.
//! For more information see [`BootstrapDeltaTable`].
//!
//! Furthermore, this module contains a function to estimate the [distribution of bootstrap proportions]
//! for a given table of replicates.
//! The distribution is used during the AU test.
//!
//! [bootstrapping]: crate::bootstrap
//! [`BootstrapDeltaTable`]: BootstrapDeltaTable
//! [distribution of bootstrap proportions]: BootstrapDeltaTable::smooth_biased_bp

use crate::bootstrap::SingleScaleBootstrap;

/// The bootstrap table grants access to vectors of delta log-likelihoods of the bootstrap replicates
/// generated for the inputs.
/// It contains a set of matrices, one matrix per replicate scale.
/// Each matrix contains `N` row vectors, one per input sequence.
/// Each row vector contains `B` entries, one per generated replicate.
/// The matrices do not necessarily share the column count, since each scale may choose its
/// replicate count independently.
/// One entry of a vector `i` represents the compound (i.e., sum of sequence) likelihood of one
/// bootstrap replicate of input sequence `i`, there are `B(s)` bootstrap replicates per scale `s`,
/// and there are `k` replicate scales.
///
/// # Deltas
/// The delta is the difference between the replicate likelihood and the best competing likelihood
/// of the same replicate.
/// For example, in a replicate with three compound likelihoods: `(-5.0, -4.0, -5.1)`,
/// the corresponding deltas are `(1.0, -1.0, 1.1)`.
/// The first and last entry are compared to the best entry (`-4.0`),
/// while the second entry is compared with the best remaining entry (in this case the first one).
///
/// # Bootstrap
/// Read the [module documentation] to understand how bootstrap replicates are generated.
///
/// # Multiscale Bootstrap
/// This type holds one matrix per scale, representing a full multiscale bootstrap.
/// The scales that were used to generate the replicate deltas can be queried using
/// [`scales`].
///
/// A scale of `1.0` means the replicate likelihoods were computed from replicate sequences
/// of the same length as the input sequences.
/// For larger (smaller) scales, the replicate sequences are longer (shorter) accordingly.
/// The likelihoods are scaled back by the replicate scale, so they are comparable.
///
/// # Sorting
/// The delta vectors have been sorted in ascending order for each tree individually.
/// Thus, the columns of the delta log-likelihood matrices bear no meaning.
///
/// [module documentation]: crate::bootstrap
pub trait BootstrapDeltaTable {
    /// Get the list of normalized delta log-likelihoods for each replicate scale for the specified
    /// input index.
    /// That is, for each replication scale, a sorted slice with `replication_count` entries is
    /// returned for the given `input_index` which holds the log-likelihood delta to the next
    /// competing input (negative, if this input is better).
    /// Here, `replication_count` is the number of replicates specified for the given replication
    /// scale.
    fn get_delta_vectors(&self, input_index: usize) -> impl Iterator<Item = &[f64]>;

    /// Compute the Bootstrap Proportions (BP Values) from a smoothed empirical distribution function
    /// derived from the bootstrap deltas.
    ///
    /// Note that this function returns bootstrap counts rather than proportions.
    /// To obtain the proportion,
    /// divide the count through the replication count of the respective scale.
    ///
    /// Warning: This method assumes all replicate vectors are sorted in ascending order.
    /// If the vectors are not sorted, the method returns nonsensical results.
    ///
    /// # Threshold
    /// At threshold `0`, this function computes the canonical bootstrap counts (smoothed)
    /// where each count is the number of bootstrap replicates where the resampled input yielded the
    /// maximum likelihood.
    ///
    /// At higher (or lower) thresholds, an artificial bias is introduced:
    /// It biases the BP value away from the true estimate
    /// by (dis)counting replicates where the input had
    /// only a very slight likelihood delta to the best scoring input.
    ///
    /// To allow accurately estimating the distance of the input's likelihood vector from the
    /// hypothesis' region boundary (see (<https://doi.org/10.1080/10635150290069913>),
    /// the BP values in the AU-Test start with a bias corresponding to the median delta.
    /// That is, if an input has a median log-likelihood delta of +50 points to the best competing
    /// input across the replicates, AU estimation starts with a bias of +50:
    /// All replicates with a delta lower than +50 points are counted toward the Bootstrap Proportion.
    /// (A negative threshold means replicates are discounted unless the input was better than the
    /// best competing input by at least the threshold).
    /// Then, the bias is lowered until the algorithm either reaches a bias of 0, or the smallest
    /// bias that does not come with high uncertainty of the p-value.
    ///
    /// # Smoothing
    /// To avoid numerical issues when estimating the parameters required by the AU test,
    /// the counts are smoothed.
    /// Smoothing linearly interpolates between two concrete counts depending on the threshold value.
    /// That is, if the threshold is 50.0, and 100 bootstrap replicates have a delta lower than 50.0,
    /// the result is interpolated between 100 and 101 depending on how close each are to the threshold.
    ///
    /// # Parameters
    /// - `input_index` the index of the input sequence to the AU test for which to compute the BP
    ///   values.
    /// - `bias` the maximum difference in likelihood from the optimal likelihood that a
    ///   replicate can have to still count towards the Bootstrap Proportion.
    fn smooth_biased_bp(&self, input_index: usize, bias: f64) -> Box<[f64]> {
        self.get_delta_vectors(input_index)
            .map(|normal_lnl| {
                let len = normal_lnl.len();
                let discrete_count = normal_lnl.iter().position(|&x| x > bias).unwrap_or(len);

                let smoothed = if discrete_count < len {
                    if discrete_count == 0 {
                        if normal_lnl[1] > normal_lnl[0] {
                            0.5 + (bias - normal_lnl[0]) / (normal_lnl[1] - normal_lnl[0])
                        } else {
                            0.0
                        }
                    } else if normal_lnl[discrete_count] > normal_lnl[discrete_count - 1] {
                        -0.5 + discrete_count as f64
                            + (bias - normal_lnl[discrete_count - 1])
                                / (normal_lnl[discrete_count] - normal_lnl[discrete_count - 1])
                    } else {
                        0.5 + discrete_count as f64
                    }
                } else if normal_lnl[len - 1] - normal_lnl[len - 2] > 0.0 {
                    len as f64 - 0.5
                        + (bias - normal_lnl[len - 1]) / (normal_lnl[len - 1] - normal_lnl[len - 2])
                } else {
                    len as f64
                };

                if smoothed > len as f64 {
                    len as f64
                } else if smoothed < 0.0 {
                    0.0
                } else {
                    smoothed
                }
            })
            .collect()
    }

    /// The number of scaling factors to the multiscale bootstrap process.
    fn num_scales(&self) -> usize;

    /// Get the [bootstrap scales] to the multiscale bootstrap process in the order of the replicate
    /// matrices.
    ///
    /// [bootstrap scales]: crate::bootstrap
    fn bootstrap_scales(&self) -> &[f64];

    /// Get the numbers of replicates for each [scaling factor].
    ///
    /// [scaling factor]: crate::bootstrap
    fn replicate_counts(&self) -> &[usize];

    /// Get the number of input sequences to the bootstrap process that generated this instance.
    fn num_trees(&self) -> usize;
}

/// A set of bootstrap replicate delta log-likelihood matrices.
/// More specifically, this struct contains one matrix of bootstrap replicate deltas per scaling factor.
/// Each matrix contains `B` likelihood deltas for each of the `N` input sequences,
/// where `B` is the replication count of that matrix,
/// and `N` is the number of input sequences to the bootstrapping scheme.
///
/// [module documentation]: crate::bootstrap
/// [`scales`]: Self::bootstrap_scales
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReplicateDeltas {
    /// A set of matrices, each matrix containing all bootstrap replicates for all trees of a single
    /// scaling factor, one matrix per scaling factor.
    replicates: Box<[Box<[f64]>]>,

    /// An array of scaling factors. For each factor, each tree generates `B` bootstrap replicates
    /// with a sequence length equal to the original sequence length multiplied by the factor.
    /// The number of replicates `B` is stored in [`scales`].
    scales: Box<[f64]>,

    /// An array with the same size as [`scales`], indicating how many bootstrap replicates each
    /// tree generates per scaling factor.
    replication_counts: Box<[usize]>,

    /// Number of rows in the [`bp_values`] matrix.
    num_trees: usize,
}

impl ReplicateDeltas {
    /// Initialize a new empty bootstrap list, initialized with the given array of scales and number
    /// of replicates.
    /// The bootstrap matrices can be initialized with ???
    ///
    /// # Parameters
    /// - `scales` The array of scaling factors that were used during bootstrapping. Each tree has
    ///   one BP value per scaling factor.
    /// - `num_replicates` The array of replication numbers, i.e., the `i`-th value indicates how
    ///   many bootstrap replicates were generated for the `i`-th BP value of each tree.
    /// - `num_tree` for how many trees the matrix is to be generated.
    ///
    /// [`scale_bp_values_mut`]: Self::scale_bp_values_mut
    pub fn new(scales: Box<[f64]>, replication_counts: Box<[usize]>, num_trees: usize) -> Self {
        // allocate the arrays for the bootstrap statistics
        let mut replicate_vector = Vec::with_capacity(scales.len());
        for &count in &replication_counts {
            replicate_vector.push(vec![0f64; count * num_trees].into_boxed_slice());
        }

        Self {
            replicates: replicate_vector.into_boxed_slice(),
            scales,
            replication_counts,
            num_trees,
        }
    }

    /// Get access to the vectors containing the bootstrap replicates for each tree at a
    /// given `scale_index`. That is, given the index `scale_index` of a scaling factor,
    /// get an iterator over all [normalized] bootstrap likelihood vectors associated with the
    /// inputs to the bootstrap algorithm (one vector per input sequence).
    ///
    /// [normalized]: bootstrap::normalize_replicates
    pub fn get_bootstrap_vectors(&self, scale_index: usize) -> impl Iterator<Item = &[f64]> {
        let num_replicates = self.replication_counts[scale_index];
        self.replicates[scale_index].chunks_exact(num_replicates)
    }

    /// Get mutable access to the vectors containing the bootstrap replicates for each tree at a
    /// given `scale_index`. That is, given the index `scale_index` of a scaling factor,
    /// get an iterator over all [normalized] bootstrap likelihood vectors associated with the
    /// inputs to the bootstrap algorithm (one vector per input sequence).
    ///
    /// [normalized]: bootstrap::normalize_replicates
    pub fn get_bootstrap_vectors_mut(
        &mut self,
        scale_index: usize,
    ) -> impl Iterator<Item = &mut [f64]> {
        let num_replicates = self.replication_counts[scale_index];
        self.replicates[scale_index].chunks_exact_mut(num_replicates)
    }
}

impl BootstrapDeltaTable for ReplicateDeltas {
    fn get_delta_vectors(&self, input_index: usize) -> impl Iterator<Item = &[f64]> {
        self.replicates
            .iter()
            .zip(self.replication_counts.iter())
            .map(move |(matrix, &count)| &matrix[input_index * count..(input_index + 1) * count])
    }

    fn num_scales(&self) -> usize {
        self.scales.len()
    }

    fn bootstrap_scales(&self) -> &[f64] {
        &self.scales
    }

    fn replicate_counts(&self) -> &[usize] {
        &self.replication_counts
    }

    fn num_trees(&self) -> usize {
        self.num_trees
    }
}

/// Given a matrix of replicates, subtract the maximum of each full replicate of the likelihood for
/// the given tree and write the result into `target`.
/// The tree is identified by `vector_index`, meaning every `vector_index`-th element of each
/// replicate in `replicate_likelihoods`.
/// The maximum of each `replicate` is pre-calculated in the `maxima` array.
///
/// This method is the kernel used by [`compute_delta_table`] and [`par_compute_delta_table`].
fn compute_likelihood_deltas(
    target: &mut [f64],
    replicate_likelihoods: &SingleScaleBootstrap,
    maxima: &[(f64, f64)],
    vector_index: usize,
) {
    target
        .iter_mut()
        .zip(replicate_likelihoods.all_replicates())
        .enumerate()
        .for_each(|(i, (target, replicate))| {
            let (best, follow_up) = maxima[i];
            *target = if replicate[vector_index] == best {
                follow_up
            } else {
                best
            } - replicate[vector_index];
        });
}

fn compute_likelihood_deltas_scaled(
    target: &mut [f64],
    replicate_likelihoods: &SingleScaleBootstrap,
    maxima: &[(f64, f64)],
    mean: f64,
    factor: f64,
    vector_index: usize,
) {
    target
        .iter_mut()
        .zip(replicate_likelihoods.all_replicates())
        .enumerate()
        .for_each(|(i, (target, replicate))| {
            let (best, follow_up) = maxima[i];
            let scaled_lh = mean + factor * (replicate[vector_index] - mean);
            *target = if scaled_lh == best { follow_up } else { best } - scaled_lh;
        });
}

/// Select the largest two entries of a slice.
fn column_max(column: &[f64]) -> (f64, f64) {
    let mut best = f64::NEG_INFINITY;
    let mut follow_up = f64::NEG_INFINITY;

    for &likelihood in column {
        if likelihood >= best {
            follow_up = best;
            best = likelihood;
        } else if likelihood > follow_up {
            follow_up = likelihood;
        }
    }

    (best, follow_up)
}

fn scaled_column_max(column: &[f64], means: &[f64], scale: f64) -> (f64, f64) {
    let mut best = f64::NEG_INFINITY;
    let mut follow_up = f64::NEG_INFINITY;

    for (row, &likelihood) in column.iter().enumerate() {
        let likelihood = means[row] + scale * (likelihood - means[row]);

        if likelihood >= best {
            follow_up = best;
            best = likelihood;
        } else if likelihood > follow_up {
            follow_up = likelihood;
        }
    }

    (best, follow_up)
}

/// Convert the replicate likelihoods into the delta table.
/// Refer to the [`ReplicateDeltas`] documentation for details.
///
/// In short, this method:
///  - transposes the matrix of likelihoods
///  - normalizes the replicate likelihoods to deltas around the best likelihood
///  - sorts the replicate (delta-)likelihoods within each row (so per input sequence) in
///    ascending order
///
/// The results are written into the provided `delta_table` instance into the
/// `scale_index`-th matrix.
/// The parameter is an out-parameter to facilitate writing the results of multiple
/// [single-scale] bootstrap runs into the same, pre-allocated table.
///
/// # Parameters
/// - `delta_table` the [`BootstrapReplicates`] matrix set where the results are written to.
/// - `replicate_likelihoods` the bootstrap replicates as generated by [`bootstrap`], meaning an
///   array with `B` replicate sets, each containing one likelihood per input sequence.
/// - `scale_index` the index of the scaling factor used for bootstrapping in the scaling factor
///   array.
///
/// [`BootstrapReplicates`]: ReplicateDeltas
/// [`single-scale`]: crate::bootstrap::bootstrap
/// [`bootstrap`]: crate::bootstrap::bootstrap
pub fn compute_delta_table(
    delta_table: &mut ReplicateDeltas,
    replicate_likelihoods: &SingleScaleBootstrap,
    scale_index: usize,
) {
    // Calculate the maximum likelihood for each bootstrap replicate. Technically the paper calls
    // for calculating the maximum without the element that is being compared with, but since it
    // is never important whether the statistic is zero or below zero, we can just use the maximum
    // every time, accepting that the best input for the replicate gets likelihood zero
    let boot_max: Box<[_]> = replicate_likelihoods
        .all_replicates()
        .map(|replicate| column_max(replicate))
        .collect();

    // subtract the maximum from each replicate likelihood for each tree, such that all bootstrap
    // replicates are distributed around 0
    delta_table
        .get_bootstrap_vectors_mut(scale_index)
        .enumerate()
        .for_each(|(vector_index, vector)| {
            compute_likelihood_deltas(vector, replicate_likelihoods, &boot_max, vector_index);
        });
    delta_table
        .get_bootstrap_vectors_mut(scale_index)
        .for_each(|vector| {
            vector.sort_unstable_by(|a, b| a.total_cmp(b));
        });
}

/// Convert a single-scale bootstrap replicate set into a multi-scale [`ReplicateDeltas`] table
/// using the approximation outlined by [Shimodaira](<https://doi.org/10.1080/10635150290069913>).
///
/// The replicates are re-scaled using the `target_scales` and each re-scaled set of replicates is
/// then converted into likelihood deltas.
/// Refer to the [`ReplicateDeltas`] documentation for details about the deltas.
///
/// The `target_scales` array contains indices into the scaling factors that have been given to the
/// `bootstrap_replicates`. One of the `target_scales` is the `reference_scale`. The deltas of the
/// reference scale are being calculated exactly like in the non-approximated algorithm.
/// All other scales are being approximated by rescaling the likelihoods to the target scale,
/// and then added to the delta table. The `target_scales` array does not have to contain all
/// scales; it is possible to call this function multiple times with different `target_scales` and
/// a different reference_scale to perform the approximation using multiple bootstrap runs.
///
/// # Parameters
/// - `delta_table` the [`BootstrapReplicates`] matrix set where the results are written to.
/// - `replicate_likelihoods` the bootstrap replicates as generated by [`bootstrap`], meaning an
///   array with `B` replicate sets, each containing one likelihood per input sequence. They have
///   been generated by a single-scale bootstrap at the scale defined by `reference_scale`
/// - `reference_scale` the index of the scaling factor used for bootstrapping the replicates.
/// - `target_scales` the indices into the `scales` array of `bootstrap_replicates`. Each scale
///   that is included in this array will be approximated using the rescaling approximation.
///
/// [`ReplicateDeltas`]: ReplicateDeltas
pub fn compute_approximate_delta_table(
    bootstrap_replicates: &mut ReplicateDeltas,
    replicate_likelihoods: &SingleScaleBootstrap,
    reference_scale: usize,
    target_scales: &[usize],
) {
    // calculate means
    let mut means = vec![0.0; bootstrap_replicates.num_trees()];
    replicate_likelihoods
        .all_replicates()
        .fold(&mut means, |acc, rep| {
            acc.iter_mut().zip(rep).for_each(|(v, r)| *v += r);
            acc
        });
    means
        .iter_mut()
        .for_each(|m| *m /= bootstrap_replicates.replication_counts[reference_scale] as f64);

    // calculate scalars
    let bootstrap_scale = bootstrap_replicates.scales[reference_scale];
    let rescale_factors: Vec<_> = bootstrap_replicates
        .scales
        .iter()
        .map(|scale| (bootstrap_scale / scale).sqrt())
        .collect();

    for &scale_index in target_scales {
        if scale_index == reference_scale {
            compute_delta_table(bootstrap_replicates, replicate_likelihoods, scale_index)
        } else {
            let boot_max: Box<[_]> = replicate_likelihoods
                .all_replicates()
                .map(|replicate| scaled_column_max(replicate, &means, rescale_factors[scale_index]))
                .collect();

            bootstrap_replicates
                .get_bootstrap_vectors_mut(scale_index)
                .enumerate()
                .for_each(|(vector_index, vector)| {
                    compute_likelihood_deltas_scaled(
                        vector,
                        replicate_likelihoods,
                        &boot_max,
                        means[vector_index],
                        rescale_factors[scale_index],
                        vector_index,
                    );
                });

            bootstrap_replicates
                .get_bootstrap_vectors_mut(scale_index)
                .for_each(|vector| {
                    vector.sort_unstable_by(|a, b| a.total_cmp(b));
                });
        }
    }
}

/// Convert the replicate likelihoods into the format expected by [`BootstrapReplicates`] in
/// parallel.
///
/// For a full explanation refer to [`compute_delta_table`].
///
/// [`BootstrapReplicates`]: ReplicateDeltas
#[cfg(feature = "rayon")]
pub fn par_compute_delta_table(
    replicate_matrix: &mut ReplicateDeltas,
    replicate_likelihoods: &SingleScaleBootstrap,
    scale_index: usize,
) {
    use rayon::prelude::*;

    // for comments on this method see sequential version

    // this method is run sequentially because it is very low effort and a par_bridge does not
    // preserve ordering, which would necessitate another sequential sorting step otherwise
    let boot_max: Box<[_]> = replicate_likelihoods
        .all_replicates()
        .map(|replicate| column_max(replicate))
        .collect();

    replicate_matrix
        .get_bootstrap_vectors_mut(scale_index)
        .enumerate()
        .par_bridge()
        .for_each(|(vector_index, vector)| {
            compute_likelihood_deltas(vector, replicate_likelihoods, &boot_max, vector_index);
        });
    replicate_matrix
        .get_bootstrap_vectors_mut(scale_index)
        .par_bridge()
        .for_each(|vector| {
            vector.sort_unstable_by(|a, b| a.total_cmp(b));
        });
}
