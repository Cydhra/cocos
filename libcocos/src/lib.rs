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
use crate::au::get_au_values;
use crate::bootstrap::bp_test;
use rand::Rng;
use std::ops::{Index, IndexMut};

pub mod au;

pub mod bootstrap;

pub(crate) mod vectors;

/// An epsilon for tests for zero. Values smaller than this value are considered to be zero. Note
/// that this is not the machine precision, and has been chosen arbitrarily.
pub(crate) const EPSILON: f64 = 1e-16;

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
    get_au_values(&bootstrap_replicates)
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
    use crate::au::par_get_au_values;
    use crate::bootstrap::par_bp_test;

    let bootstrap_replicates = par_bp_test(rng, likelihoods, bootstrap_scales, replication_counts);
    par_get_au_values(&bootstrap_replicates)
}
