use bench::{load_from_path, resample_lnl_vectors};
use libcocos::SiteLikelihoodTable;
use libcocos::bootstrap::{DEFAULT_FACTORS, DEFAULT_REPLICATES};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

const PATHS: &[&str] = &["data/flat-phylo.siteLH"];

const NUM_PERMUTATIONS: usize = 10;

fn main() -> anyhow::Result<()> {
    for path in PATHS {
        println!("Loading {:?}...", path);
        let site_lh = load_from_path(path)?;

        println!("Performing au test...");
        let p_values = au_test(&site_lh)?;

        for i in 0..NUM_PERMUTATIONS {
            println!("Shuffling site-likelihoods {}/{NUM_PERMUTATIONS}...", i + 1);
            // for reproducibility of the results, we fix the seed
            let mut rng = ChaCha8Rng::from_seed([i as u8; 32]);
            let mut permutation = (0..site_lh.num_sites()).collect::<Vec<_>>();
            permutation.shuffle(&mut rng);

            println!("Performing au test...");
            let new_input = resample_lnl_vectors(&site_lh, &permutation);
            let new_p_values = au_test(&new_input)?;

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

fn au_test(likelihoods: &SiteLikelihoodTable) -> anyhow::Result<Vec<f64>> {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    libcocos::au_test(&mut rng, likelihoods, &DEFAULT_FACTORS, &DEFAULT_REPLICATES)
}
