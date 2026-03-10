use statrs::distribution::{ContinuousCDF, StudentsT};

pub fn reject_hypotheses(
    num_samples: usize,
    equivalence_margin: f64,
    confidence: f64,
    reference_mean: &[f64],
    reference_variance: &[f64],
    testing_mean: &[f64],
    testing_variance: &[f64],
    reference_name: &str,
    testing_name: &str,
) {
    // result list
    let mut unrejected_hypotheses = Vec::new();
    let num_trees = reference_mean.len();

    for i in 0..num_trees {
        // pooled corrected standard deviation of the distributions
        let standard_error_consel_squared = reference_variance[i] / num_samples as f64;
        let standard_error_cocos_squared = testing_variance[i] / num_samples as f64;
        let standard_error_delta_squared =
            standard_error_consel_squared + standard_error_cocos_squared;
        let standard_error_delta =
            (standard_error_consel_squared + standard_error_cocos_squared).sqrt();

        // calculate degrees of freedom assuming unequal variances using Welch–Satterthwaite equation
        let individual_degrees_of_freedom = (num_samples - 1) as f64; // degrees of freedom of the independent distributions

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
            (reference_mean[i] - (testing_mean[i] - equivalence_margin)) / standard_error_delta;
        let upper_statistic =
            (reference_mean[i] - (testing_mean[i] + equivalence_margin)) / standard_error_delta;

        // reject the hypothesis that the thresholds are exceeded significantly
        let t_distribution = StudentsT::new(0.0, 1.0, pooled_degrees_of_freedom)
            .expect("cannot instance the Student's t distribution");
        let critical_threshold = t_distribution.inverse_cdf(confidence);

        // test whether the hypotheses that the bounds are exceeded can be rejected
        let lower_bound_rejected = lower_statistic > critical_threshold;
        let upper_bound_rejected = upper_statistic < -critical_threshold;

        if !lower_bound_rejected || !upper_bound_rejected {
            unrejected_hypotheses.push((
                lower_bound_rejected,
                upper_bound_rejected,
                i,
                reference_mean[i],
                reference_variance[i],
                testing_mean[i],
                testing_variance[i],
            ));
        } else {
            println!(
                "Rejected inequality: tree {i} is not significantly better or worse \
                 ({reference_name} mean: {:.6}, variance: {:.6}; \
                  {testing_name} mean: {:.6}, variance: {:.6}",
                reference_mean[i], reference_variance[i], testing_mean[i], testing_variance[i]
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
