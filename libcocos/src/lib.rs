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
//! [`BpTable`].
//!
//! The library takes pre-parsed log-likelihood vectors as input ([`SiteLikelihoodTable`])
//! and can therefore be used to apply the AU test to every selection problem that uses the
//! RELL bootstrap method (<https://doi.org/10.1007/BF02109483>).
//!
//! A separate binary crate with a CLI is available which applies the test to phylogenetic trees.

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

/// A matrix containing one or more BP values per input tree, one for each scale factor in the
/// multiscale bootstrapping process.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BpTable {
    /// A matrix of bootstrap proportions, where each row contains `R` bootstrap proportions per
    /// tree and there are `N` rows, where `N` is the number of trees.
    bp_values: Box<[f64]>,

    /// An array of scaling factors. For each factor, each tree generates `B` bootstrap replicates
    /// with a sequence length equal to the original sequence length multiplied by the factor.
    /// The number of replicates `B` is stored in [`scales`].
    scales: Box<[f64]>,

    /// An array with the same size as [`scales`], indicating how many bootstrap replicates each
    /// tree generates per scaling factor.
    num_replicates: Box<[usize]>,

    /// Number of rows in the [`bp_values`] matrix.
    num_trees: usize,
}

impl BpTable {
    /// Initialize a new empty BP matrix, initialized with the given array of scales and number
    /// of replicates.
    /// The matrix can be initialized with the [`scale_bp_values_mut`] iterator access.
    ///
    /// # Parameters
    /// - `scales` The array of scaling factors that were used during bootstrapping. Each tree has
    ///   one BP value per scaling factor.
    /// - `num_replicates` The array of replication numbers, i.e., the `i`-th value indicates how
    ///   many bootstrap replicates were generated for the `i`-th BP value of each tree.
    /// - `num_tree` for how many trees the matrix is to be generated.
    ///
    /// [`scale_bp_values_mut`]: Self::scale_bp_values_mut
    pub fn new(scales: Box<[f64]>, num_replicates: Box<[usize]>, num_trees: usize) -> Self {
        assert_eq!(
            scales.len(),
            num_replicates.len(),
            "Each scale needs an associated number of replicates"
        );

        Self {
            bp_values: vec![0.0; num_trees * scales.len()].into_boxed_slice(),
            scales,
            num_replicates,
            num_trees,
        }
    }

    /// How many trees are in the table
    pub fn num_trees(&self) -> usize {
        self.num_trees
    }

    /// How many resampling scale factors are in the table
    pub fn num_scales(&self) -> usize {
        self.scales.len()
    }

    /// Get all resampling scale factors that were used in resampling.
    /// Each tree has one BP value per scale factor.
    /// There are [`num_scales`] factors in the returned slice.
    ///
    /// [`num_scales`]: Self::num_scales
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Get all bootstrap replication numbers, one per scaling factor.
    /// Each tree has one BP value for scaling factor `i` that was calculated from `n` bootstrap
    /// replicates where `n = self.num_replicates()[i]`.
    /// There are [`num_scales`] numbers in the returned slice.
    ///
    /// The replication numbers are `usize`, for convenient use as an array index, and since each
    /// bootstrap replicate needs to be stored at some point, they are constrained to the platform's
    /// address size anyway.
    ///
    /// [`num_scales`]: Self::num_scales
    pub fn num_replicates(&self) -> &[usize] {
        &self.num_replicates
    }

    /// Get all BP values for the tree at index `tree`.
    pub fn tree_bp_values(&self, tree: usize) -> &[f64] {
        &self.bp_values[self.num_scales() * tree..self.num_scales() * (tree + 1)]
    }

    /// Get mutable access to all BP values for the tree at index `tree`.
    pub fn tree_bp_values_mut(&mut self, tree: usize) -> &mut [f64] {
        let num_scales = self.num_scales();
        &mut self.bp_values[num_scales * tree..num_scales * (tree + 1)]
    }

    /// Get access to all BP values for a given scale factor.
    /// Each tree has one BP value in this iterator, in the order of the trees.
    pub fn scale_bp_values(&self, scale_index: usize) -> impl Iterator<Item = &f64> + use<'_> {
        let step = self.num_scales();
        self.bp_values.iter().skip(scale_index).step_by(step)
    }

    /// Get mutable access to all BP values for a given scale factor.
    /// Each tree has one BP value in this iterator, in the order of the trees.
    pub fn scale_bp_values_mut(&mut self, scale_index: usize) -> impl Iterator<Item = &mut f64> {
        let step = self.num_scales();
        self.bp_values.iter_mut().skip(scale_index).step_by(step)
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
/// - `bootstrap_replicates` a slice containing a number for each scale in `bootstrap_scales`
///   indicating how many replicates to generate for that scale.
///
/// # Return
/// Returns a vector of p-values with one p-value for each tree in the input table, or an error
/// if at least one of the calculations failed.
pub fn au_test<R>(
    rng: &mut R,
    likelihoods: &SiteLikelihoodTable,
    bootstrap_scales: &[f64],
    bootstrap_replicates: &[usize],
) -> Result<Vec<f64>, argmin_math::Error>
where
    R: Rng,
{
    let bp_table = bp_test(rng, likelihoods, bootstrap_scales, bootstrap_replicates);
    get_au_value(&bp_table)
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
    bootstrap_replicates: &[usize],
) -> Result<Vec<f64>, argmin_math::Error>
where
    R: Rng + Clone + Send,
{
    use crate::au::par_get_au_value;
    use crate::bootstrap::par_bp_test;

    let bp_table = par_bp_test(rng, likelihoods, bootstrap_scales, bootstrap_replicates);
    par_get_au_value(&bp_table)
}
