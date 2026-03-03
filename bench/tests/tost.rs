//! A test case which runs the available data files against consel and libcocos and tests whether
//! the libcocos outputs have the same distribution as the consel outputs.
//! It does so by doing two one-sided welch's t-tests that attempt to reject the hypothesis that the mean
//! of cocos' output is outside the confidence interval of consel's mean output.
//! Consel's output is provided in 10 CSV files of pre-computed consel runs with random seeds.

use csv::Trim;
use libcocos::au_test;
use libcocos::bootstrap::{DEFAULT_FACTORS, DEFAULT_REPLICATES};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rstest::*;
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

/// Number of p-values taken from cocos.
/// It is expected that the consel outputs provide equal amounts of samples.
/// The implementation of the Welch–Satterthwaite equation relies on this fact.
const NUM_SAMPLES: usize = 25;

/// Margin accepted for the difference in p-values between cocos and consel.
/// We accept 2.5% (additive) difference of p-values.
/// This is expected to not drastically alter the amount of rejected trees between the tools
/// (especially since we see lower variance for low p-values).
const EQUIVALENCE_MARGIN: f64 = 0.025;

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
fn test_distribution(#[files("data/*.siteLH")] site_likelihoods: PathBuf) {
    // get environment
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"));
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
    let data_dir = repository_root.join("data");
    let consel_dir = data_dir.join(file_name);

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
            let au = result.as_ref().expect("calculating AU value failed");
            cocos_mean[item] += au;
            cocos_variance[item] += au * au;
        }
    });

    // collected hypotheses that cannot be rejected here for debug output
    let mut unrejected_hypotheses = Vec::new();

    for i in 0..num_trees {
        // variance with Bessel's correction
        cocos_variance[i] -= cocos_mean[i] * cocos_mean[i] / NUM_SAMPLES as f64;
        cocos_variance[i] /= (NUM_SAMPLES - 1) as f64;

        // calculate mean and variance
        cocos_mean[i] /= NUM_SAMPLES as f64;

        // pooled corrected standard deviation of the distributions
        let standard_error_consel_squared = consel_variance[i] / NUM_SAMPLES as f64;
        let standard_error_cocos_squared = cocos_variance[i] / NUM_SAMPLES as f64;
        let standard_error_delta_squared =
            standard_error_consel_squared + standard_error_cocos_squared;
        let standard_error_delta =
            (standard_error_consel_squared + standard_error_cocos_squared).sqrt();

        // calculate degrees of freedom assuming unequal variances using Welch–Satterthwaite equation
        let individual_degrees_of_freedom = (NUM_SAMPLES - 1) as f64; // degrees of freedom of the independent distributions

        // the degrees of freedom of a linear combination, simplified because the individual degrees of freedom
        // are the same for all summands and thus can be factored out of the denominator.
        // reference: https://en.wikipedia.org/wiki/Welch%E2%80%93Satterthwaite_equation
        // simplified: https://en.wikipedia.org/wiki/Welch%27s_t-test#Calculations
        let pooled_degrees_of_freedom = individual_degrees_of_freedom
            * (standard_error_delta_squared * standard_error_delta_squared)
            / (standard_error_consel_squared * standard_error_consel_squared
                + standard_error_cocos_squared * standard_error_cocos_squared);

        // calculate the test statistics as a confidence interval with radius of the accepted margin
        // reference: https://en.wikipedia.org/wiki/Equivalence_test#TOST_procedure
        let lower_statistic =
            (consel_mean[i] - (cocos_mean[i] - EQUIVALENCE_MARGIN)) / standard_error_delta;
        let upper_statistic =
            (consel_mean[i] - (cocos_mean[i] + EQUIVALENCE_MARGIN)) / standard_error_delta;

        // reject the hypothesis that the thresholds are exceeded significantly
        let t_distribution = StudentsT::new(0.0, 1.0, pooled_degrees_of_freedom)
            .expect("cannot instance the Student's t distribution");
        let critical_threshold = t_distribution.inverse_cdf(CONFIDENCE);

        // test whether the hypotheses that the bounds are exceeded can be rejected
        let lower_bound_rejected = lower_statistic > critical_threshold;
        let upper_bound_rejected = upper_statistic < -critical_threshold;

        if !lower_bound_rejected || !upper_bound_rejected {
            unrejected_hypotheses.push((
                lower_bound_rejected,
                upper_bound_rejected,
                i,
                consel_mean[i],
                consel_variance[i],
                cocos_mean[i],
                cocos_variance[i],
            ));
        }
    }

    assert_eq!(
        unrejected_hypotheses.len(),
        0,
        "failed to reject inequality hypotheses for {} trees. Unrejected:\n{}",
        unrejected_hypotheses.len(),
        unrejected_hypotheses.iter().map(
            |(
                lower_bound_rejected,
                upper_bound_rejected,
                item,
                consel_mean,
                consel_variance,
                cocos_mean,
                cocos_variance,
            )| {
                format!(
                    "Failed to reject {} of tree {:02}.\tConsel: {:.6} (var: {:.6}),\tCocos: {:.6} (var: {:.6})",
                    if *lower_bound_rejected && *upper_bound_rejected {
                        "both bounds"
                    } else if *lower_bound_rejected {
                        "lower bound"
                    } else {
                        "upper bound"
                    },
                    item,
                    consel_mean,
                    consel_variance,
                    cocos_mean,
                    cocos_variance,
                )
            }
        ).collect::<Vec<_>>().join("\n")
    )
}
