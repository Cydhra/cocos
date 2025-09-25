//! Verify that the input order of trees has no effect on the end result.
//! We read in several input files, and then calculate the p-values under different permutations
//! and verify that the p-values do not differ significantly (in the first 9 decimal places).

use bench::{load_from_path, resample_lnl_vectors};
use libcocos::au::error::MathError;
use libcocos::bootstrap::{DEFAULT_FACTORS, DEFAULT_REPLICATES};
use libcocos::{SiteLikelihoodTable, par_au_test};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

const PATHS: &[&str] = &["data/flat-phylo.siteLH", "data/medium.siteLH"];

const NUM_PERMUTATIONS: usize = 5;

fn main() -> anyhow::Result<()> {
    for path in PATHS {
        println!("Loading {:?}...", path);
        let site_lh = load_from_path(path)?;

        println!("Performing au test...");
        let p_values = au_test(&site_lh)
            .into_iter()
            .map(|x| match x {
                Ok(p) => p,
                Err(MathError::ConvergenceFailed { p_value }) => p_value,
                _ => {
                    assert!(false, "AU Test failed with error: {}", x.unwrap_err());
                    unreachable!()
                }
            })
            .collect::<Vec<_>>();

        for i in 0..NUM_PERMUTATIONS {
            println!("Shuffling input trees {}/{NUM_PERMUTATIONS}...", i + 1);
            // for reproducibility of the results, we fix the seed
            let mut rng = ChaCha8Rng::from_seed([i as u8; 32]);
            let mut permutation = (0..site_lh.num_trees()).collect::<Vec<_>>();
            permutation.shuffle(&mut rng);

            println!("Performing au test...");
            let new_input = resample_lnl_vectors(&site_lh, &permutation);
            let new_p_values = au_test(&new_input)
                .into_iter()
                .map(|x| match x {
                    Ok(p) => p,
                    Err(MathError::ConvergenceFailed { p_value }) => p_value,
                    _ => {
                        assert!(false, "AU Test failed with error: {}", x.unwrap_err());
                        unreachable!()
                    }
                })
                .collect::<Vec<_>>();

            println!("Comparing results...");
            for (index, &permuted_index) in permutation.iter().enumerate() {
                assert!(
                    (p_values[permuted_index] - new_p_values[index]).abs() < 1E-9,
                    "Shuffling inputs yielded different p-value: rep={i}, index={index}, au value={}, new au value={}",
                    p_values[permuted_index],
                    new_p_values[index],
                );
            }
        }
    }

    println!("Success!");
    Ok(())
}

fn au_test(likelihoods: &SiteLikelihoodTable) -> Box<[Result<f64, MathError>]> {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    par_au_test(&mut rng, likelihoods, &DEFAULT_FACTORS, &DEFAULT_REPLICATES)
}
