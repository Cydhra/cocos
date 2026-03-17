#![allow(dead_code)]
use csv::Trim;
use libcocos::SiteLikelihoodTable;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

/// Margin accepted for the difference in p-values between cocos and consel.
/// We accept 2% (additive) difference of p-values.
/// This is expected to not drastically alter the amount of rejected trees between the tools
/// (especially since we see lower variance for low p-values).
pub(crate) const EQUIVALENCE_MARGIN: f64 = 0.02;

/// Confidence value for the t tests.
/// Note that this means that the null-hypothesis is rejected incorrectly with probability 1%,
/// but this doesn't mean that failing to reject it has an equally high probability.
/// Failing to reject the hypothesis should always be treated as a problem with the algorithm.
pub(crate) const CONFIDENCE: f64 = 0.99;

/// Read in the siteLH file from a path
pub(crate) fn read_slh(fixture: &PathBuf) -> SiteLikelihoodTable {
    // read site-likelihoods
    cocos_parse::parse_puzzle(BufReader::new(
        File::open(&fixture).expect("cannot read fixture"),
    ))
    .expect("cannot parse siteLH file")
}

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

/// Read in the consel results provided in the data/ directory of this repository, in a subfolder
/// named after the fixture.
pub fn read_consel_results(fixture: &PathBuf, num_trees: usize, approx: bool) -> TreeStatistics {
    // find consel output
    let mut file_name = fixture
        .file_name()
        .expect("test called with invalid fixture")
        .to_str()
        .expect("file name is not representable");
    if fixture.extension().is_some() {
        let suffix = format!(".{}", fixture.extension().unwrap().to_str().unwrap());
        file_name = file_name.strip_suffix(&suffix).unwrap();
    }

    let consel_dir = fixture
        .parent()
        .expect("fixture cannot be located")
        .join(if approx {
            file_name.to_owned() + "_approx"
        } else {
            file_name.to_owned()
        });

    // read in consel outputs
    let mut consel_statistics = TreeStatistics::new(num_trees);

    // find consel samples to determine the number of samples to take from cocos
    let consel_samples: Vec<_> = fs::read_dir(consel_dir)
        .expect("cannot list consel output directory")
        .filter(|f| {
            f.as_ref()
                .expect("cannot list consel output directory content")
                .file_name()
                .to_str()
                .expect("cannot parse OS string")
                .ends_with("csv")
        })
        .collect();

    for result_file in consel_samples {
        let result_file = result_file.expect("cannot list consel output").path();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .trim(Trim::Fields)
            .from_reader(File::open(&result_file).expect("cannot open consel output"));

        // read in the results and store them in the samples
        for record in reader.deserialize::<ConselRecord>() {
            let record = record.expect("malformed consel output");
            consel_statistics.add_sample(record.item - 1, record.au);
        }
    }

    consel_statistics.finalize();
    consel_statistics
}

pub fn calculate_statistics(
    cocos_results: &[Box<[Result<f64, MathError>]>],
    num_trees: usize,
) -> TreeStatistics {
    let mut statistics = TreeStatistics::new(num_trees);

    cocos_results.iter().for_each(|p_values| {
        for (item, result) in p_values.iter().enumerate() {
            let au = match result.as_ref() {
                Ok(p_value) => *p_value,
                Err(error) => match error {
                    MathError::HessianSingular => panic!("AU test failed due to singular hessian"),
                    MathError::ConvergenceFailed { p_value } => *p_value,
                },
            };
            statistics.add_sample(item, au);
        }
    });

    statistics.finalize();
    statistics
}

pub struct TreeStatistics {
    mean: Vec<f64>,
    variance: Vec<f64>,
    samples: Vec<usize>,
    finalized: bool,
}

impl TreeStatistics {
    /// Create new empty statistics struct with space for `num_inputs` means and variances.
    pub fn new(num_inputs: usize) -> Self {
        if num_inputs == 0 {
            panic!("cannot calculate statistics for no inputs");
        }

        Self {
            mean: vec![0.0; num_inputs],
            variance: vec![0.0; num_inputs],
            samples: vec![0; num_inputs],
            finalized: false,
        }
    }

    /// Add the measurement `value` for the input at index `index`.
    /// Adds the measurement to mean and variance for this input.
    pub fn add_sample(&mut self, index: usize, value: f64) {
        self.mean[index] += value;
        self.variance[index] += value * value;
        self.samples[index] += 1;
    }

    /// Finalize the statistical moments for all inputs
    pub fn finalize(&mut self) {
        let num_samples = self.samples[0];
        self.samples.iter().for_each(|s| {
            assert_eq!(
                *s, num_samples,
                "not all inputs have an equal amount of samples"
            )
        });

        assert!(!self.finalized, "cannot finalize twice");

        for i in 0..self.mean.len() {
            self.variance[i] -= self.mean[i] * self.mean[i] / num_samples as f64;
            self.variance[i] /= (num_samples - 1) as f64;

            self.mean[i] /= num_samples as f64;
        }

        self.finalized = true;
    }

    pub fn get_num_samples(&self) -> usize {
        assert!(
            self.finalized,
            "number of samples is indeterminate before finalizing"
        );
        self.samples[0]
    }

    pub fn means(&self) -> &[f64] {
        &self.mean
    }

    pub fn variances(&self) -> &[f64] {
        &self.variance
    }
}

use libcocos::au::error::MathError;
use statrs::distribution::{ContinuousCDF, StudentsT};

pub fn reject_hypotheses(
    equivalence_margin: f64,
    confidence: f64,
    reference_statistics: &TreeStatistics,
    testing_statistics: &TreeStatistics,
    reference_name: &str,
    testing_name: &str,
) {
    // num samples
    let num_samples = reference_statistics.get_num_samples();
    assert_eq!(
        num_samples,
        testing_statistics.get_num_samples(),
        "t-test implementation expects both statistics to have equal number of samples"
    );

    assert!(
        reference_statistics.finalized && testing_statistics.finalized,
        "statistics not finalized"
    );

    // result list
    let mut unrejected_hypotheses = Vec::new();
    let num_trees = reference_statistics.means().len();

    for i in 0..num_trees {
        // pooled corrected standard deviation of the distributions
        let standard_error_reference_squared =
            reference_statistics.variances()[i] / num_samples as f64;
        let standard_error_testing_squared = testing_statistics.variances()[i] / num_samples as f64;
        let standard_error_delta_squared =
            standard_error_reference_squared + standard_error_testing_squared;
        let standard_error_delta =
            (standard_error_reference_squared + standard_error_testing_squared).sqrt();

        let lower_bound_rejected;
        let upper_bound_rejected;

        if standard_error_testing_squared == 0.0 && standard_error_reference_squared == 0.0 {
            // if both variances are zero, the t-test collapses, however we don't have any uncertainty
            // and can just compare the means directly.
            lower_bound_rejected = testing_statistics.means()[i]
                >= reference_statistics.means()[i] - equivalence_margin;
            upper_bound_rejected = testing_statistics.means()[i]
                <= reference_statistics.means()[i] + equivalence_margin;
        } else {
            // calculate degrees of freedom assuming unequal variances using Welch–Satterthwaite equation
            let individual_degrees_of_freedom = (num_samples - 1) as f64; // degrees of freedom of the independent distributions

            // the degrees of freedom of a linear combination, simplified because the individual degrees of freedom
            // are the same for all summands and thus can be factored out of the denominator.
            // reference: https://en.wikipedia.org/wiki/Welch%E2%80%93Satterthwaite_equation
            // simplified: https://en.wikipedia.org/wiki/Welch%27s_t-test#Calculations
            let pooled_degrees_of_freedom = individual_degrees_of_freedom
                * (standard_error_delta_squared * standard_error_delta_squared)
                / (standard_error_reference_squared * standard_error_reference_squared
                    + standard_error_testing_squared * standard_error_testing_squared);

            // calculate the test statistics as a confidence interval with radius of the accepted margin
            // reference: https://en.wikipedia.org/wiki/Equivalence_test#TOST_procedure
            let lower_statistic = (reference_statistics.means()[i]
                - (testing_statistics.means()[i] - equivalence_margin))
                / standard_error_delta;
            let upper_statistic = (reference_statistics.means()[i]
                - (testing_statistics.means()[i] + equivalence_margin))
                / standard_error_delta;

            // reject the hypothesis that the thresholds are exceeded significantly
            let t_distribution = StudentsT::new(0.0, 1.0, pooled_degrees_of_freedom)
                .expect("cannot instance the Student's t distribution");
            let critical_threshold = t_distribution.inverse_cdf(confidence);

            // test whether the hypotheses that the bounds are exceeded can be rejected
            lower_bound_rejected = lower_statistic > critical_threshold;
            upper_bound_rejected = upper_statistic < -critical_threshold;
        }

        if !lower_bound_rejected || !upper_bound_rejected {
            unrejected_hypotheses.push((
                lower_bound_rejected,
                upper_bound_rejected,
                i,
                reference_statistics.means()[i],
                reference_statistics.variances()[i],
                testing_statistics.means()[i],
                testing_statistics.variances()[i],
            ));
        } else {
            println!(
                "Rejected inequality: tree {i} is not significantly better or worse \
                 ({reference_name} mean: {:.6}, variance: {:.9}; \
                  {testing_name} mean: {:.6}, variance: {:.9})",
                reference_statistics.means()[i],
                reference_statistics.variances()[i],
                testing_statistics.means()[i],
                testing_statistics.variances()[i]
            )
        }
    }

    assert_eq!(
        unrejected_hypotheses.len(),
        0,
        "failed to reject inequality hypotheses for {} trees. Unrejected:\n{}",
        unrejected_hypotheses.len(),
        unrejected_hypotheses.iter().map(|(
                                              lower_bound_rejected,
                                              upper_bound_rejected,
                                              item,
                                              consel_mean,
                                              consel_variance,
                                              cocos_mean,
                                              cocos_variance,
                                          )| {
            format!(
                "Failed to reject {} of tree {:02}.\t{reference_name}: {:.6} (var: {:.6}),\t{testing_name}: {:.6} (var: {:.6})",
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
