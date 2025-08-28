#![cfg_attr(feature = "simd", feature(portable_simd))]

use std::ops::{Index, IndexMut};

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

pub type SiteLikelihoods = [f64];

pub type BootstrapVec = Vec<f64>;

pub type BootstrapWeights = [f64];
