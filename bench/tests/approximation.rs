//! A test case which runs the available data files against libcocos and tests whether
//! the full bootstrap outputs and the approximate bootstrap outputs generate the same p-values.
//! It does so by doing two one-sided welch's t-tests that attempt to reject the hypothesis that the
//! mean of the approximated outputs is outside the equivalence margin of the canonical outputs.
//! The tests do use the same seeds for canonical and approximate outputs.
//! However, we do not guarantee equal outputs at the moment,
//! so we use two t-tests to compare results.

use bench::reject_hypotheses;
use libcocos::au::error::MathError;
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
use std::fs::File;
use std::io::BufReader;
use std::num::NonZero;
use std::path::PathBuf;
use std::thread::available_parallelism;

/// Number of p-values taken per configuration.
const NUM_SAMPLES: usize = 50;

/// Margin accepted for the difference in p-values between cocos and consel.
/// We accept 2% (additive) difference of p-values.
/// This is expected to not drastically alter the amount of rejected trees
/// (especially since we see lower variance for low p-values).
const EQUIVALENCE_MARGIN: f64 = 0.02;

/// Confidence value for the t tests.
/// Note that this means that the null-hypothesis is rejected incorrectly with probability 1%,
/// but this doesn't mean that failing to reject it has an equally high probability.
/// Failing to reject the hypothesis should always be treated as a problem with the algorithm.
const CONFIDENCE: f64 = 0.99;

const NUM_REPLICATES: usize = 10_000;

const SAMPLED_FACTOR: usize = 5;

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
    let per_site_lnl = cocos_parse::parse_puzzle(BufReader::new(
        File::open(&site_likelihoods).expect("cannot read fixture"),
    ))
    .expect("cannot parse siteLH file");
    let num_trees = per_site_lnl.num_trees();

    // run cocos in parallel over trees with sequential invocations of the AU test
    let mut canonical_mean = vec![0.0; num_trees];
    let mut canonical_variance = vec![0.0; num_trees];

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
    canonical_runs.iter().for_each(|p_values| {
        for (item, result) in p_values.iter().enumerate() {
            let au = match result.as_ref() {
                Ok(p_value) => *p_value,
                Err(error) => match error {
                    MathError::HessianSingular => panic!("AU test failed due to singular hessian"),
                    MathError::ConvergenceFailed { p_value } => *p_value,
                },
            };
            canonical_mean[item] += au;
            canonical_variance[item] += au * au;
        }
    });

    for i in 0..num_trees {
        // variance with Bessel's correction
        canonical_variance[i] -= canonical_mean[i] * canonical_mean[i] / NUM_SAMPLES as f64;
        canonical_variance[i] /= (NUM_SAMPLES - 1) as f64;

        // calculate mean and variance
        canonical_mean[i] /= NUM_SAMPLES as f64;
    }

    // parallel runs
    let mut approx_mean = vec![0.0; num_trees];
    let mut approx_variance = vec![0.0; num_trees];

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
    approx_runs.iter().for_each(|p_values| {
        for (item, result) in p_values.iter().enumerate() {
            let au = match result.as_ref() {
                Ok(p_value) => *p_value,
                Err(error) => match error {
                    MathError::HessianSingular => panic!("AU test failed due to singular hessian"),
                    MathError::ConvergenceFailed { p_value } => *p_value,
                },
            };
            approx_mean[item] += au;
            approx_variance[item] += au * au;
        }
    });

    for i in 0..num_trees {
        // variance with Bessel's correction
        approx_variance[i] -= approx_mean[i] * approx_mean[i] / NUM_SAMPLES as f64;
        approx_variance[i] /= (NUM_SAMPLES - 1) as f64;

        // calculate mean and variance
        approx_mean[i] /= NUM_SAMPLES as f64;
    }

    // collected hypotheses that cannot be rejected here for debug output
    reject_hypotheses(
        NUM_SAMPLES,
        EQUIVALENCE_MARGIN,
        CONFIDENCE,
        &canonical_mean,
        &canonical_variance,
        &approx_mean,
        &approx_variance,
        "canonical",
        "approximate",
    );
}
