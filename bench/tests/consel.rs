//! A test case which runs the available data files against consel and libcocos and tests whether
//! the libcocos outputs have the same distribution as the consel outputs.
//! It does so by doing two one-sided welch's t-tests that attempt to reject the hypothesis that the mean
//! of cocos' output is outside the confidence interval of consel's mean output.
//! Consel's output is provided in 50 CSV files of pre-computed consel runs with random seeds.

use bench::reject_hypotheses;
use csv::Trim;
use libcocos::au::error::MathError;
use libcocos::au_test;
use libcocos::bootstrap::{DEFAULT_FACTORS, DEFAULT_REPLICATES};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rstest::*;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

/// Number of p-values taken from cocos.
/// It is expected that the consel outputs provide equal amounts of samples.
/// The implementation of the Welch–Satterthwaite equation relies on this fact.
const NUM_SAMPLES: usize = 50;

/// Margin accepted for the difference in p-values between cocos and consel.
/// We accept 2% (additive) difference of p-values.
/// This is expected to not drastically alter the amount of rejected trees between the tools
/// (especially since we see lower variance for low p-values).
const EQUIVALENCE_MARGIN: f64 = 0.02;

/// Confidence value for the t tests.
/// Note that this means that the null-hypothesis is rejected incorrectly with probability 1%,
/// but this doesn't mean that failing to reject it has an equally high probability.
/// Failing to reject the hypothesis should always be treated as a problem with the algorithm.
const CONFIDENCE: f64 = 0.99;

/// Consel output is saved to a CSV file with these five values per record. The records in the file
/// are sorted by rank, not by item.
#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct ConselRecord {
    rank: usize,
    item: usize,
    obs: f64,
    au: f64,
    np: f64,
}

#[rstest]
fn compare_with_consel(#[files("data/*.siteLH")] site_likelihoods: PathBuf) {
    // get environment
    let mut file_name = site_likelihoods
        .file_name()
        .expect("test called with invalid fixture")
        .to_str()
        .expect("file name is not representable");

    // find directory with consel output
    if site_likelihoods.extension().is_some() {
        let suffix = format!(
            ".{}",
            site_likelihoods.extension().unwrap().to_str().unwrap()
        );
        file_name = file_name.strip_suffix(&suffix).unwrap();
    }

    // find consel output
    let consel_dir = site_likelihoods
        .parent()
        .expect("fixture cannot be located")
        .join(file_name);

    // read site-likelihoods
    let per_site_lnl = cocos_parse::parse_puzzle(BufReader::new(
        File::open(&site_likelihoods).expect("cannot read fixture"),
    ))
    .expect("cannot parse siteLH file");
    let num_trees = per_site_lnl.num_trees();

    // read in consel outputs
    let mut consel_mean = vec![0.0; num_trees];
    let mut consel_variance = vec![0.0; num_trees];

    for i in 0..NUM_SAMPLES {
        let result_file = consel_dir.join(format!("run{}.csv", i));

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .trim(Trim::Fields)
            .from_reader(File::open(&result_file).expect("cannot open consel output"));

        // read in the results and store them in the samples
        for record in reader.deserialize::<ConselRecord>() {
            let record = record.expect("malformed consel output");
            consel_mean[record.item - 1] += record.au;
            consel_variance[record.item - 1] += record.au * record.au;
        }
    }

    // calculate consel mean and variance
    for i in 0..num_trees {
        consel_variance[i] -= consel_mean[i] * consel_mean[i] / NUM_SAMPLES as f64;
        consel_variance[i] /= (NUM_SAMPLES - 1) as f64;

        consel_mean[i] /= NUM_SAMPLES as f64;
    }

    // run cocos in parallel (we assume test execution is sequential so we can leverage threads.
    // We also run AU test sequentially, because we probably have more runs than CPUs).
    let mut cocos_mean = vec![0.0; num_trees];
    let mut cocos_variance = vec![0.0; num_trees];

    // generate independent seeds for the threads
    let mut seed_rng = rand::rng();
    let seeds: Vec<_> = (0..NUM_SAMPLES).map(|_| seed_rng.random()).collect();

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
            cocos_mean[item] += au;
            cocos_variance[item] += au * au;
        }
    });

    // calculate cocos mean and variance
    for i in 0..num_trees {
        // variance with Bessel's correction
        cocos_variance[i] -= cocos_mean[i] * cocos_mean[i] / NUM_SAMPLES as f64;
        cocos_variance[i] /= (NUM_SAMPLES - 1) as f64;

        // calculate mean and variance
        cocos_mean[i] /= NUM_SAMPLES as f64;
    }

    // collected hypotheses that cannot be rejected here for debug output
    reject_hypotheses(
        NUM_SAMPLES,
        EQUIVALENCE_MARGIN,
        CONFIDENCE,
        &consel_mean,
        &consel_variance,
        &cocos_mean,
        &cocos_variance,
        "consel",
        "cocos",
    );
}
