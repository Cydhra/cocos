#![cfg_attr(feature = "simd", feature(portable_simd))]
#![warn(missing_docs)]
#![allow(clippy::inline_always)]

//! This library implements the approximately unbiased test by H. Shimodaira (<https://doi.org/10.1080/10635150290069913>).
//! The main contribution is the full parallel implementation of the test, which is gated behind
//! the crate feature `rayon`.
//! Furthermore, the crate feature `simd` enables a `portable_simd` vector implementation of the
//! test, which provides a substantial speed boost (since over 90% of the runtime is spent
//! in dot products).
//! Because `portable_simd` is a nightly feature,
//! the implementation falls back to a scalar implementation on stable.
//!
//! Optionally, the library supports `serde` for its two structures [`SiteLikelihoodTable`] and
//! [`todo`].
//!
//! The library takes pre-parsed log-likelihood vectors as input ([`SiteLikelihoodTable`])
//! and can therefore be used to apply the AU test to every selection problem that uses the
//! RELL bootstrap method (<https://doi.org/10.1007/BF02109483>).
//!
//! A separate binary crate with a CLI is available which applies the test to phylogenetic trees.

use crate::au::error::MathError;
use crate::au::get_au_value;
use crate::bootstrap::bp_test;
use rand::Rng;
use std::ops::{Index, IndexMut};

pub mod au;

pub mod bootstrap;

pub(crate) mod vectors;

/// A table containing the per-site (log-)likelihoods of `N` phylogenetic trees, with `M` alignment
/// sites each. The table is used during bootstrap to generate bootstrap replicates of the alignment
/// quickly.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SiteLikelihoodTable {
    likelihoods: Box<[f64]>,
    num_trees: usize,
    num_sites: usize,
}

impl SiteLikelihoodTable {
    /// Initialize a new site likelihood table for `num_trees` trees with `num_sites` per-site
    /// likelihoods each.
    pub fn new(num_trees: usize, num_sites: usize) -> Self {
        Self {
            likelihoods: vec![0f64; num_trees * num_sites].into_boxed_slice(),
            num_trees,
            num_sites,
        }
    }

    /// Return an iterator over all tree vectors contained in the table.
    /// Each vector contains all site-likelihoods of the tree.
    pub fn trees(&self) -> Box<[&[f64]]> {
        self.likelihoods.chunks_exact(self.num_sites).collect()
    }

    /// Get the number of trees in the table
    pub fn num_trees(&self) -> usize {
        self.num_trees
    }

    /// Get the number of likelihood values per tree
    pub fn num_sites(&self) -> usize {
        self.num_sites
    }

    /// Access the given `site` of the alignment.
    /// The method returns an iterator over all trees' per-site log-likelihoods at `site` in their
    /// sequence.
    pub fn get_site(&self, site: usize) -> impl Iterator<Item = &f64> {
        self.likelihoods.iter().skip(site).step_by(self.num_sites)
    }

    /// Access the given `site` of the alignment.
    /// The method returns a mutable iterator over all trees' per-site log-likelihoods at `site`
    /// in their sequence.
    pub fn get_site_mut(&mut self, site: usize) -> impl Iterator<Item = &mut f64> {
        self.likelihoods
            .iter_mut()
            .skip(site)
            .step_by(self.num_sites)
    }
}

impl Index<usize> for SiteLikelihoodTable {
    type Output = SiteLikelihoods;

    fn index(&self, index: usize) -> &Self::Output {
        &self.likelihoods[index * self.num_sites..(index + 1) * self.num_sites]
    }
}

impl IndexMut<usize> for SiteLikelihoodTable {
    fn index_mut(&mut self, index: usize) -> &mut <Self as Index<usize>>::Output {
        &mut self.likelihoods[index * self.num_sites..(index + 1) * self.num_sites]
    }
}

/// A slice of per-site likelihoods of one tree
pub type SiteLikelihoods = [f64];

/// A slice with the same length as tree site-likelihood vectors, containing integer resampling
/// weights drawn uniformly at random (with replacement). The weights sum to the bootstrap sequence
/// length.
pub type ResamplingWeights = Box<[f64]>;

/// A set of normalized bootstrap replicate likelihood matrices.
/// More specifically, this struct contains one matrix of bootstrap replicates per scaling factor.
/// Each matrix contains `B` likelihoods for `N` input sequences,
/// where `B` is the replication count for the scaling factor of that matrix,
/// and `N` is the number of input sequences to the bootstrapping.
///
/// The likelihood values are normalized, that is, the likelihoods are moved towards zero,
/// where the best tree of each replicate has likelihood 0, and the other values are the difference
/// to the best likelihood.
/// This way, calculating the canonical BP values is equivalent to counting the zeros in each
/// tree's matrix row.
/// For more information about calculating BP values, refer to [`compute_bp_values`].
///
/// The likelihood vectors of each input sequence have to be sorted.
/// Writing unsorted sequences into this matrix prevents calculation of BP values.
///
/// [`compute_bp_values`]: BootstrapReplicates::compute_bp_values
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BootstrapReplicates {
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

impl BootstrapReplicates {
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

    /// Get access to the bootstrap statistics of the input sequence with index `input_index`
    /// at all bootstrap scales.
    pub fn get_vectors(&self, input_index: usize) -> impl Iterator<Item = &[f64]> {
        self.replicates
            .iter()
            .zip(self.replication_counts.iter())
            .map(move |(matrix, &count)| &matrix[input_index * count..(input_index + 1) * count])
    }

    /// Compute the Bootstrap Proportions from the empirical bootstrap distribution at the given
    /// threshold.
    /// At threshold `0`, the function computes the canonical bootstrap proportions
    /// where each count is the number of bootstrap replicates where the resampled input yielded the
    /// maximum likelihood.
    ///
    /// To allow accurately estimating the distance of the input's likelihood vector from the
    /// hypothesis' region boundary (see (<https://doi.org/10.1080/10635150290069913>),
    /// the BP values are first over-estimated by increasing the threshold.
    /// Increasing it allows trees which have slightly suboptimal likelihoods
    /// in a bootstrap replicate to count the replicate as well.
    /// The BP values then continuously approach the true canonical BP values by lowering the threshold.
    ///
    /// This method computes the bootstrap counts of the tree `input_index` while also counting
    /// bootstrap replicates where the tree has an up to `threshold` lower log-likelihood than the
    /// best tree in that replicate. Note that the method does not compute proportions, but counts.
    ///
    /// To avoid numerical issues during numerical optimization, the count value is interpolated
    /// to convert the empirical distribution of bootstrap counts into a continuous distribution.
    ///
    /// Warning: This method assumes all per-input normalized replicates are sorted in ascending order.
    /// If the vectors are not sorted, the method returns nonsensical results.
    ///
    /// # Parameters
    /// - `input_index` the index of the input sequence to the AU test for which to compute the BP
    ///   values.
    /// - `threshold` the maximum difference in likelihood from the optimal likelihood that a
    ///   replicate can have to still count towards the Bootstrap Proportion.
    pub fn compute_bp_values(&self, input_index: usize, threshold: f64) -> Box<[f64]> {
        self.get_vectors(input_index)
            .map(|normal_lnl| {
                let len = normal_lnl.len();
                let discrete_count = normal_lnl
                    .iter()
                    .position(|&x| x > threshold)
                    .unwrap_or(len);
                let smoothed = if discrete_count < len - 1 {
                    if normal_lnl[discrete_count + 1] > normal_lnl[discrete_count] {
                        0.5 + discrete_count as f64
                            + (threshold - normal_lnl[discrete_count])
                                / (normal_lnl[discrete_count + 1] - normal_lnl[discrete_count])
                    } else {
                        if discrete_count > 0 {
                            0.5 + discrete_count as f64
                        } else {
                            0.0
                        }
                    }
                } else {
                    if normal_lnl[len - 1] - normal_lnl[len - 2] > 0.0 {
                        0.5 + len as f64
                            + (threshold - normal_lnl[len - 2])
                                / (normal_lnl[len - 1] - normal_lnl[len])
                    } else {
                        len as f64
                    }
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
    pub fn num_scales(&self) -> usize {
        self.scales.len()
    }

    /// Get the scaling factors to the multiscale bootstrap process in the order of the replicate
    /// matrices.
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Get the numbers of replicates for each [scaling factor].
    ///
    /// [scaling factor]: Self::scales
    pub fn replication_counts(&self) -> &[usize] {
        &self.replication_counts
    }

    /// Get the number of input sequences to the bootstrap process that generated this instance.
    pub fn num_trees(&self) -> usize {
        self.num_trees
    }
}

/// Calculate the AU p-values for a given table of log-likelihoods using the RELL bootstrap method
/// and subsequent AU test.
/// This is a convenience method to call bootstrapping and p-value calculation in one call.
///
/// # Parameters
/// - `rng` the random number generator to use for bootstrapping
/// - `likelihoods` the [`SiteLikelihoodTable`] that contains the log-likelihoods to resample
/// - `bootstrap_scales` a slice containing the scaling factors for the multiscale bootstrap
/// - `replication_counts` a slice containing a number for each scale in `bootstrap_scales`
///   indicating how many replicates to generate for that scale.
///
/// # Return
/// Returns a vector of p-values with one p-value for each tree in the input table, or an error
/// if at least one of the calculations failed.
pub fn au_test<R>(
    rng: &mut R,
    likelihoods: &SiteLikelihoodTable,
    bootstrap_scales: &[f64],
    replication_counts: &[usize],
) -> Box<[Result<f64, MathError>]>
where
    R: Rng,
{
    let bootstrap_replicates = bp_test(rng, likelihoods, bootstrap_scales, replication_counts);
    get_au_value(&bootstrap_replicates)
}

/// Calculate the AU p-values for a given table of log-likelihoods using the RELL bootstrap method
/// and subsequent AU test in parallel.
/// This is a convenience method to call bootstrapping and p-value calculation in one call.
///
/// For full documentation refer to [`au_test`]
#[cfg(feature = "rayon")]
pub fn par_au_test<R>(
    rng: &mut R,
    likelihoods: &SiteLikelihoodTable,
    bootstrap_scales: &[f64],
    replication_counts: &[usize],
) -> Box<[Result<f64, MathError>]>
where
    R: Rng + Clone + Send,
{
    use crate::au::par_get_au_value;
    use crate::bootstrap::par_bp_test;

    let bootstrap_replicates = par_bp_test(rng, likelihoods, bootstrap_scales, replication_counts);
    par_get_au_value(&bootstrap_replicates)
}
