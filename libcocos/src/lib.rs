#![cfg_attr(feature = "simd", feature(portable_simd))]

use std::ops::{Index, IndexMut};

pub mod au;

pub mod bootstrap;

pub mod vectors;

/// A table containing the per-site (log-)likelihoods of `N` phylogenetic trees, with `M` alignment
/// sites each. The table is used during bootstrap to generate bootstrap replicates of the alignment
/// quickly.
#[derive(Clone, Debug)]
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

pub type BootstrapReplicates = Box<[Box<[f64]>]>;

/// A matrix containing one or more BP values per input tree, one for each scale factor in the
/// multiscale bootstrapping process.
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
    pub fn num_replicates(&self) -> &[usize] {
        &self.num_replicates
    }

    /// Ge tall BP values for the tree at index `tree`.
    pub fn tree_bp_values(&self, tree: usize) -> &[f64] {
        &self.bp_values[self.num_scales() * tree..self.num_scales() * (tree + 1)]
    }

    /// Get mutable access to all BP values for a given scale factor.
    /// Each tree has one BP value in this iterator, in the order of the trees.
    pub fn scale_bp_values_mut(&mut self, scale_index: usize) -> impl Iterator<Item = &mut f64> {
        let step = self.num_scales();
        self.bp_values.iter_mut().skip(scale_index).step_by(step)
    }
}
