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
    bp_values: Box<[f64]>,
    scales: Box<[f64]>,
    num_trees: usize,
}

impl BpTable {
    pub fn new(scales: Box<[f64]>, num_trees: usize) -> Self {
        Self {
            bp_values: vec![0.0; num_trees * scales.len()].into_boxed_slice(),
            scales,
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
    pub fn scales(&self) -> &[f64] {
        &self.scales
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

impl Index<usize> for BpTable {
    type Output = [f64];

    fn index(&self, index: usize) -> &Self::Output {
        &self.bp_values[index * self.num_scales()..(index + 1) * self.num_scales()]
    }
}

impl IndexMut<usize> for BpTable {
    fn index_mut(&mut self, index: usize) -> &mut <Self as Index<usize>>::Output {
        let scales = self.num_scales();
        &mut self.bp_values[index * scales..(index + 1) * scales]
    }
}
