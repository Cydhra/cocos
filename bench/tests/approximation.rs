//! A test case which runs the available data files against libcocos and tests whether
//! the full bootstrap outputs and the approximate bootstrap outputs generate the same p-values.
//! It does so by doing two one-sided welch's t-tests that attempt to reject the hypothesis that the
//! mean of the approximated outputs is outside the equivalence margin of the canonical outputs.
//! The tests do use the same seeds for canonical and approximate outputs.
//! However, we do not guarantee equal outputs at the moment,
//! so we use two t-tests to compare results.

use libcocos::au::get_au_values;
use libcocos::au_test;
use libcocos::bootstrap::{DEFAULT_FACTORS, DEFAULT_REPLICATES, bootstrap};
use libcocos::delta::{ReplicateDeltas, compute_approximate_delta_table};
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

const NUM_REPLICATES: usize = 10_000;

const SAMPLED_FACTOR: usize = 5;

mod common;

#[rstest]
#[ignore] // this test is expected to fail, since the approximation cannot be as good as true multiscale
fn compare_with_canonical(#[files("data/*.siteLH")] site_likelihoods: PathBuf) {
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

    // generate independent seeds for the threads
    let mut seed_rng = rand::rng();
    let seeds: Vec<_> = (0..NUM_SAMPLES).map(|_| seed_rng.random()).collect();

    let canonical_runs: Vec<_> = seeds
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

    // calculate sequential mean and variance
    let canonical_statistics = common::calculate_statistics(&canonical_runs, num_trees);

    // approximate runs
    let approx_runs: Vec<_> = seeds
        .into_iter()
        .map(|seed| {
            let mut rng = StdRng::from_seed(seed);

            let mut replicate_matrix = ReplicateDeltas::new(
                DEFAULT_FACTORS.to_vec().into_boxed_slice(),
                vec![NUM_REPLICATES; DEFAULT_FACTORS.len()].into_boxed_slice(),
                per_site_lnl.num_trees(),
            );

            let bootstrap_replicates = bootstrap(
                &mut rng,
                &per_site_lnl,
                NUM_REPLICATES,
                DEFAULT_FACTORS[SAMPLED_FACTOR],
            );
            compute_approximate_delta_table(
                &mut replicate_matrix,
                &bootstrap_replicates,
                SAMPLED_FACTOR,
            );

            get_au_values(&replicate_matrix)
        })
        .collect();

    // calculate parallel mean and variance
    let approx_statistics = common::calculate_statistics(&approx_runs, num_trees);

    // collected hypotheses that cannot be rejected here for debug output
    common::reject_hypotheses(
        common::EQUIVALENCE_MARGIN,
        common::CONFIDENCE,
        &canonical_statistics,
        &approx_statistics,
        "canonical",
        "approximate",
    );
}
