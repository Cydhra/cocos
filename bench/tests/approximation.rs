//! A test case which runs the available data files against libcocos and tests whether
//! the full bootstrap outputs and the approximate bootstrap outputs of the rescaling approximation
//! generate the same p-values.
//! A second test compares the approximate cocos outputs against the approximate consel outputs.
//! Both test cases are expected to fail and are thus ignored by default.
//! This test case exists to test how much worse different approximation schemes are.
//! The approximation scheme is implemented in [`compute_approximate_pval`].

use crate::common::{read_consel_results, read_slh};
use libcocos::au::error::MathError;
use libcocos::au::get_au_values;
use libcocos::bootstrap::{DEFAULT_FACTORS, DEFAULT_REPLICATES, bootstrap};
use libcocos::delta::{ReplicateDeltas, compute_approximate_delta_table};
use libcocos::{SiteLikelihoodTable, au_test};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, rng};
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

mod common;

/// Compute p-values using the approximation of consel or a variant thereof.
fn compute_approximate_pval(
    per_site_lnl: &SiteLikelihoodTable,
    seeds: &[[u8; 32]],
) -> Vec<Box<[Result<f64, MathError>]>> {
    seeds
        .into_iter()
        .map(|seed| {
            let mut rng = StdRng::from_seed(*seed);

            let mut replicate_matrix = ReplicateDeltas::new(
                DEFAULT_FACTORS.to_vec().into_boxed_slice(),
                vec![NUM_REPLICATES; DEFAULT_FACTORS.len()].into_boxed_slice(),
                per_site_lnl.num_trees(),
            );

            let bootstrap_replicates =
                bootstrap(&mut rng, &per_site_lnl, NUM_REPLICATES, DEFAULT_FACTORS[5]);
            compute_approximate_delta_table(
                &mut replicate_matrix,
                &bootstrap_replicates,
                5,
                &(0..5).collect::<Vec<_>>(),
            );

            let bootstrap_replicates =
                bootstrap(&mut rng, &per_site_lnl, NUM_REPLICATES, DEFAULT_FACTORS[9]);
            compute_approximate_delta_table(
                &mut replicate_matrix,
                &bootstrap_replicates,
                9,
                &(5..10).collect::<Vec<_>>(),
            );

            get_au_values(&replicate_matrix)
        })
        .collect()
}

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
    let approx_runs: Vec<_> = compute_approximate_pval(&per_site_lnl, &seeds);

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

// this test is expected to fail, since the p-values are not necessarily normally distributed
// and thus the t-test cannot compare the results. This is expected and true for both consel
// and cocos.
#[rstest]
#[ignore]
fn compare_with_consel_approx(#[files("data/*.siteLH")] fixture: PathBuf) {
    let per_site_lnl = read_slh(&fixture);
    let num_trees = per_site_lnl.num_trees();
    let consel_statistics = read_consel_results(&fixture, num_trees, true);
    let num_samples = consel_statistics.get_num_samples();

    // generate independent seeds for the threads to ensure different runs. seeds are determinisitic
    // though to avoid random test fluctuations.
    let mut seed_rng = rng();
    let seeds: Vec<_> = (0..num_samples).map(|_| seed_rng.random()).collect();

    // approximate runs
    let approx_runs: Vec<_> = compute_approximate_pval(&per_site_lnl, &seeds);

    // calculate parallel mean and variance
    let approx_statistics = common::calculate_statistics(&approx_runs, num_trees);

    // collected hypotheses that cannot be rejected here for debug output
    common::reject_hypotheses(
        common::EQUIVALENCE_MARGIN,
        common::CONFIDENCE,
        &consel_statistics,
        &approx_statistics,
        "consel (approx)",
        "cocos (approx)",
    );
}
