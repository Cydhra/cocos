//! A test case which runs the available data files against libcocos and tests whether
//! the sequential outputs and the parallel outputs generate the same p-values.
//! It does so by doing two one-sided welch's t-tests that attempt to reject the hypothesis that the
//! mean of the parallel outputs is outside the equivalence margin of the sequential outputs.
//! The tests do use the same seeds for sequential and parallel outputs.
//! However, we do not guarantee equal outputs at the moment,
//! so we use two t-tests to compare results.

use libcocos::bootstrap::{DEFAULT_FACTORS, DEFAULT_REPLICATES};
use libcocos::{au_test, par_au_test};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use rstest::*;
use std::cmp::max;
use std::num::NonZero;
use std::path::PathBuf;
use std::thread::available_parallelism;

/// Number of p-values taken per configuration.
const NUM_SAMPLES: usize = 50;

mod common;

#[rstest]
fn compare_with_sequential(#[files("data/*.siteLH")] site_likelihoods: PathBuf) {
    // setup thread pool to enforce parallelism even on single-core systems
    let available = available_parallelism().unwrap_or(NonZero::<usize>::new(8).unwrap());
    let threads = max(4, available.get());
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap_or_else(|err| {
            eprintln!("cannot set global thread pool: {:?}", err);
            assert!(
                rayon::current_num_threads() > 1,
                "cannot run parallelism check with only one thread"
            );
        });

    // read site-likelihoods
    let per_site_lnl = common::read_slh(&site_likelihoods);
    let num_trees = per_site_lnl.num_trees();

    // run cocos in parallel over trees with sequential invocations of the AU test

    // generate independent seeds for the threads to ensure different runs. seeds are determinisitic
    // though to avoid random test fluctuations.
    let mut seed_rng = StdRng::from_seed([0u8; 32]);
    let seeds: Vec<_> = (0..NUM_SAMPLES).map(|_| seed_rng.random()).collect();

    let sequential_runs: Vec<_> = seeds
        .par_iter()
        .map(|seed| {
            let mut rng = StdRng::from_seed(*seed);
            au_test(
                &mut rng,
                &per_site_lnl,
                &DEFAULT_FACTORS,
                &DEFAULT_REPLICATES,
            )
        })
        .collect();

    let sequential_statistics = common::calculate_statistics(&sequential_runs, num_trees);

    // parallel runs
    let parallel_runs: Vec<_> = seeds
        .into_iter()
        .map(|seed| {
            let mut rng = StdRng::from_seed(seed);
            par_au_test(
                &mut rng,
                &per_site_lnl,
                &DEFAULT_FACTORS,
                &DEFAULT_REPLICATES,
            )
        })
        .collect();

    let parallel_statistics = common::calculate_statistics(&parallel_runs, num_trees);

    // collected hypotheses that cannot be rejected here for debug output
    common::reject_hypotheses(
        common::EQUIVALENCE_MARGIN,
        common::CONFIDENCE,
        &sequential_statistics,
        &parallel_statistics,
        "sequential",
        "parallel",
    );
}
