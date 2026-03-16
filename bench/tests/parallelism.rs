//! A test case which runs the available data files against libcocos and tests whether
//! the sequential outputs and the parallel outputs generate the same p-values.
//! It does so by doing two one-sided welch's t-tests that attempt to reject the hypothesis that the
//! mean of the parallel outputs is outside the equivalence margin of the sequential outputs.
//! The tests do use the same seeds for sequential and parallel outputs.
//! However, we do not guarantee equal outputs at the moment,
//! so we use two t-tests to compare results.

use bench::reject_hypotheses;
use libcocos::au::error::MathError;
use libcocos::bootstrap::{DEFAULT_FACTORS, DEFAULT_REPLICATES};
use libcocos::{au_test, par_au_test};
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
    let per_site_lnl = cocos_parse::parse_puzzle(BufReader::new(
        File::open(&site_likelihoods).expect("cannot read fixture"),
    ))
    .expect("cannot parse siteLH file");
    let num_trees = per_site_lnl.num_trees();

    // run cocos in parallel over trees with sequential invocations of the AU test
    let mut sequential_mean = vec![0.0; num_trees];
    let mut sequential_variance = vec![0.0; num_trees];

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

    // calculate sequential mean and variance
    sequential_runs.iter().for_each(|p_values| {
        for (item, result) in p_values.iter().enumerate() {
            let au = match result.as_ref() {
                Ok(p_value) => *p_value,
                Err(error) => match error {
                    MathError::HessianSingular => panic!("AU test failed due to singular hessian"),
                    MathError::ConvergenceFailed { p_value } => *p_value,
                },
            };
            sequential_mean[item] += au;
            sequential_variance[item] += au * au;
        }
    });

    for i in 0..num_trees {
        // variance with Bessel's correction
        sequential_variance[i] -= sequential_mean[i] * sequential_mean[i] / NUM_SAMPLES as f64;
        sequential_variance[i] /= (NUM_SAMPLES - 1) as f64;

        // calculate mean and variance
        sequential_mean[i] /= NUM_SAMPLES as f64;
    }

    // parallel runs
    let mut parallel_mean = vec![0.0; num_trees];
    let mut parallel_variance = vec![0.0; num_trees];

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

    // calculate parallel mean and variance
    parallel_runs.iter().for_each(|p_values| {
        for (item, result) in p_values.iter().enumerate() {
            let au = match result.as_ref() {
                Ok(p_value) => *p_value,
                Err(error) => match error {
                    MathError::HessianSingular => panic!("AU test failed due to singular hessian"),
                    MathError::ConvergenceFailed { p_value } => *p_value,
                },
            };
            parallel_mean[item] += au;
            parallel_variance[item] += au * au;
        }
    });

    for i in 0..num_trees {
        // variance with Bessel's correction
        parallel_variance[i] -= parallel_mean[i] * parallel_mean[i] / NUM_SAMPLES as f64;
        parallel_variance[i] /= (NUM_SAMPLES - 1) as f64;

        // calculate mean and variance
        parallel_mean[i] /= NUM_SAMPLES as f64;
    }

    // collected hypotheses that cannot be rejected here for debug output
    reject_hypotheses(
        NUM_SAMPLES,
        EQUIVALENCE_MARGIN,
        CONFIDENCE,
        &sequential_mean,
        &sequential_variance,
        &parallel_mean,
        &parallel_variance,
        "sequential",
        "parallel",
    );
}
