//! A test case which runs the available data files against consel and libcocos and tests whether
//! the libcocos outputs have the same distribution as the consel outputs.
//! It does so by doing two one-sided welch's t-tests that attempt to reject the hypothesis that the mean
//! of cocos' output is outside the confidence interval of consel's mean output.

use csv::Trim;
use libcocos::au_test;
use libcocos::bootstrap::{DEFAULT_FACTORS, DEFAULT_REPLICATES};
use rstest::*;
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

/**
 * Number of p-values taken from consel and cocos to estimate the distribution of p-values for fixed
 * Inputs
 */
const NUM_SAMPLES: usize = 10;

/**
 * Margin accepted for the difference in p-values between cocos and consel.
 */
const EQUIVALENCE_MARGIN: f64 = 0.025;

/**
 * Confidence value for the t tests
 */
const CONFIDENCE: f64 = 0.95;

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

    // create directory for the consel output of this site-likelihood file
    if site_likelihoods.extension().is_some() {
        let suffix = format!(
            ".{}",
            site_likelihoods.extension().unwrap().to_str().unwrap()
        );
        file_name = file_name.strip_suffix(&suffix).unwrap();
    }

    // read site-likelihoods
    let per_site_lnl = cocos_parse::parse_puzzle(BufReader::new(
        File::open(&site_likelihoods).expect("cannot read fixture"),
    ))
    .expect("cannot parse siteLH file");
    let num_trees = per_site_lnl.num_trees();

    let mut consel_mean = vec![0.0; num_trees];
    let mut consel_variance = vec![0.0; num_trees];

    // read consel results
    let data_dir = repository_root.join("data");
    let consel_dir = data_dir.join(file_name);

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

    // calculate mean and variance
    for i in 0..num_trees {
        consel_mean[i] /= NUM_SAMPLES as f64;
        consel_variance[i] /= NUM_SAMPLES as f64;
        consel_variance[i] -= consel_mean[i] * consel_mean[i];
    }

    // run cocos
    let mut cocos_mean = vec![0.0; num_trees];
    let mut cocos_variance = vec![0.0; num_trees];

    let mut rng = rand::rng();

    for _ in 0..NUM_SAMPLES {
        let p_values = au_test(
            &mut rng,
            &per_site_lnl,
            &DEFAULT_FACTORS,
            &DEFAULT_REPLICATES,
        );
        for (item, result) in p_values.iter().enumerate() {
            let au = result.as_ref().expect("calculating AU value failed");
            cocos_mean[item] += au;
            cocos_variance[item] += au * au;
        }
    }

    for i in 0..num_trees {
        // calculate mean and variance
        cocos_mean[i] /= NUM_SAMPLES as f64;
        // variance with Bessel's correction
        cocos_variance[i] /= (NUM_SAMPLES - 1) as f64;
        cocos_variance[i] -= cocos_mean[i] * cocos_mean[i];

        // pooled corrected standard deviation of the distributions
        let standard_error_consel_squared = consel_variance[i] / NUM_SAMPLES as f64;
        let standard_error_cocos_squared = cocos_variance[i] / NUM_SAMPLES as f64;
        let standard_error_delta_squared =
            standard_error_consel_squared + standard_error_cocos_squared;
        let standard_error_delta =
            (standard_error_consel_squared + standard_error_cocos_squared).sqrt();

        // calculate degrees of freedom assuming unequal variances using Welch–Satterthwaite equation
        let individual_degrees_of_freedom = (NUM_SAMPLES - 1) as f64; // degrees of freedom of the independent distributions
        let pooled_degrees_of_freedom = individual_degrees_of_freedom
            * (standard_error_delta_squared * standard_error_delta_squared)
            / (standard_error_consel_squared * standard_error_consel_squared
                + standard_error_cocos_squared * standard_error_cocos_squared);

        // calculate the test statistics as a confidence interval with radius of the accepted margin
        let lower_statistic =
            (consel_mean[i] - (cocos_mean[i] - EQUIVALENCE_MARGIN)) / standard_error_delta;
        let upper_statistic =
            (consel_mean[i] - (cocos_mean[i] + EQUIVALENCE_MARGIN)) / standard_error_delta;

        // reject the hypothesis that the thresholds are exceeded significantly
        let t_distribution = StudentsT::new(0.0, 1.0, pooled_degrees_of_freedom)
            .expect("cannot instance the Student's t distribution");
        let critical_threshold = t_distribution.inverse_cdf(CONFIDENCE);

        print!(
            "mean is lower: {:.3} > {:.3} = {}\t",
            lower_statistic,
            critical_threshold,
            if lower_statistic > critical_threshold {
                "rejected"
            } else {
                "problem"
            }
        );
        print!(
            "mean is higher: {:.3} < {:.3} = {}\t",
            upper_statistic,
            -critical_threshold,
            if upper_statistic < -critical_threshold {
                "rejected"
            } else {
                "problem"
            }
        );

        println!(
            "item {}: consel mean: {:.5}, var: {:.6}\t-\tcocos mean: {:.5}, var: {:.6}",
            i, consel_mean[i], consel_variance[i], cocos_mean[i], cocos_variance[i]
        );
    }
}
