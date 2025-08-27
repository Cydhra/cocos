#![cfg_attr(feature = "simd", feature(portable_simd))]

pub mod bootstrap;

pub mod vectors;

pub type SiteLikelihoods = [f64];

pub type BootstrapVec = Vec<f64>;

pub type BootstrapWeights = [f64];