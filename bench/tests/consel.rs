//! A test case which runs the available data files against consel and libcocos and tests whether
//! the libcocos outputs have the same distribution as the consel outputs.
//! It does so by doing two one-sided welch's t-tests that attempt to reject the hypothesis that the mean
//! of cocos' output is outside the confidence interval of consel's mean output.
//! Consel's output is provided in N CSV files of pre-computed consel runs with random seeds,
//! which will be matched with N cocos runs.

use crate::common::read_slh;
use crate::common::{TreeStatistics, read_consel_results};
use libcocos::au::error::MathError;
use libcocos::au_test;
use libcocos::bootstrap::{DEFAULT_FACTORS, DEFAULT_REPLICATES};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rstest::*;
use std::path::PathBuf;

mod common;

#[rstest]
fn compare_with_consel(#[files("data/*.siteLH")] site_likelihoods: PathBuf) {
    let per_site_lnl = read_slh(&site_likelihoods);
    let num_trees = per_site_lnl.num_trees();
    let consel_statistics = read_consel_results(&site_likelihoods, num_trees);
    let num_samples = consel_statistics.get_num_samples();

    // run cocos in parallel (we assume test execution is sequential so we can leverage threads.
    // We also run AU test sequentially, because we probably have more runs than CPUs).
    let mut cocos_statistics = TreeStatistics::new(num_trees);

    // generate independent seeds for the threads to ensure different runs. seeds are determinisitic
    // though to avoid random test fluctuations.
    let mut seed_rng = StdRng::from_seed([0u8; 32]);
    let seeds: Vec<_> = (0..num_samples).map(|_| seed_rng.random()).collect();

    let p_value_runs: Vec<_> = seeds
        .into_par_iter()
        .map(|seed| {
            let mut rng = StdRng::from_seed(seed);
            au_test(
                &mut rng,
                &per_site_lnl,
                &DEFAULT_FACTORS,
                &DEFAULT_REPLICATES,
            )
        })
        .collect();

    // calculate means and variances
    p_value_runs.iter().for_each(|p_values| {
        for (item, result) in p_values.iter().enumerate() {
            let au = match result.as_ref() {
                Ok(p_value) => *p_value,
                Err(error) => match error {
                    MathError::HessianSingular => panic!("AU test failed due to singular hessian"),
                    MathError::ConvergenceFailed { p_value } => *p_value,
                },
            };
            cocos_statistics.add_sample(item, au);
        }
    });

    cocos_statistics.finalize();

    // collected hypotheses that cannot be rejected here for debug output
    common::reject_hypotheses(
        common::EQUIVALENCE_MARGIN,
        common::CONFIDENCE,
        &consel_statistics,
        &cocos_statistics,
        "consel",
        "cocos",
    );
}
