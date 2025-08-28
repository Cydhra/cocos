use crate::vectors::dot_prod;
use crate::{BootstrapVec, BootstrapWeights, SiteLikelihoodTable, SiteLikelihoods};
use rand::Rng;
use rand::distr::Uniform;

pub const DEFAULT_FACTORS: [f64; 10] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4];

/// Generate a vector of per-site weights, indicating how often each site of an alignment got
/// selected in bootstrap replication.
///
/// # Parameters
/// - `num_sites` how many sites the original alignment has, minimum of 1
/// - `replication_factor` the ratio between the number of original sites and the number of sites in
///   the bootstrap replicate. Cannot be negative or zero.
///
/// # Panic
/// Panics if `num_sites` is zero, or `replication_factor` isn't a strictly positive number.
pub fn generate_selection_vector<R: Rng>(
    rng: &mut R,
    num_sites: usize,
    replication_factor: f64,
) -> BootstrapVec {
    assert!(num_sites > 0, "cannot bootstrap an alignment of size 0");
    assert!(
        replication_factor > 0.0,
        "replication_factor cannot be negative or zero"
    );

    let mut selection_vector = vec![0.0; num_sites];
    let distribution = Uniform::new(0, num_sites).unwrap();

    rng.sample_iter(distribution)
        .take((num_sites as f64 * replication_factor) as usize)
        .for_each(|site| selection_vector[site] += 1.0);

    selection_vector
}

/// Compute the log likelihood of a bootstrap replicate.
///
/// # Parameters
/// - `site_lh` a vector containing site log-likelihood values for a tree
/// - `selection` a vector containing weights for each site of the site-likelihood vector indicating
///   how often the site was chosen during bootstrap selection. This vector can be generated with
///   [`generate_selection_vector`]
///
/// # Panic
/// Panics if the `site_lh` vector and the `selection` vector have different lengths.
/// [`generate_selection_vector`]: generate_selection_vector
pub fn compute_replicate_likelihood(
    site_lh: &SiteLikelihoods,
    selection: &BootstrapWeights,
) -> f64 {
    debug_assert!(
        site_lh.len() == selection.len(),
        "selection vector must match site likelihood vector in length"
    );
    dot_prod(site_lh, selection)
}

/// Generate `num_replicates` bootstrap replicates for each log-likelihood sequence and calculate
/// their log-likelihood value.
///
/// # Parameters
/// - `rng` random number generator state
/// - `likelihoods` a slice of [SiteLikelihoods] vectors
/// - `num_replicates` how many replicates to generate
/// - `num_sites` how many entries the likelihood vectors have
/// - `replication_factor` the ratio between the original alignment length and the length of bootstrap sequences
///
/// # Return
/// Returns a vector containing vectors of likelihoods for each bootstrap replicate (i.e., for each
/// tree).
#[inline]
fn bootstrap_slice<R: Rng>(
    rng: &mut R,
    likelihoods: &[&SiteLikelihoods],
    num_replicates: usize,
    num_sites: usize,
    replication_factor: f64,
) -> Vec<BootstrapVec> {
    let mut results = Vec::with_capacity(num_replicates);

    for _ in 0..num_replicates {
        let weights = generate_selection_vector(rng, num_sites, replication_factor);

        // compute the sum of site log-likelihoods weighted by the given selection vector
        // and scale it by the replication_factor to make it comparable to the original log-likelihood
        let bootstrap_replicate = likelihoods
            .iter()
            .map(|site_lh| compute_replicate_likelihood(site_lh, &weights) / replication_factor)
            .collect();

        results.push(bootstrap_replicate);
    }

    results
}

/// Generate `num_replicates` bootstrap replicates for each log-likelihood sequence and calculate
/// their log-likelihood value.
///
/// # Parameters
/// - `rng` random number generator state
/// - `likelihoods` a matrix of site log-likelihoods for each phylogenetic tree.
/// - `num_replicates` how many replicates to generate
/// - `replication_factor` the ratio between the original alignment length and the length of bootstrap sequences
///
/// # Return
/// Returns a vector containing vectors of likelihoods for each bootstrap replicate (i.e., for each
/// tree).
///
/// # Panic
/// Panics if `num_replicates` is 0, or the `replication_factor` is negative or zero, or the
/// trees have zero site likelihoods.
pub fn bootstrap<R: Rng>(
    rng: &mut R,
    likelihoods: &SiteLikelihoodTable,
    num_replicates: usize,
    replication_factor: f64,
) -> Vec<BootstrapVec> {
    assert!(num_replicates > 0, "cannot bootstrap with 0 replicates");
    assert!(
        replication_factor > 0.0,
        "replication_factor cannot be negative or zero"
    );

    let num_sites = likelihoods.num_sites();
    assert!(num_sites > 0, "cannot bootstrap with 0 site likelihoods");

    bootstrap_slice(
        rng,
        &likelihoods.trees(),
        num_replicates,
        num_sites,
        replication_factor,
    )
}

#[cfg(feature = "rayon")]
pub fn par_bootstrap<R: Rng + Clone + Send>(
    rng: &mut R,
    likelihoods: &SiteLikelihoodTable,
    num_replicates: usize,
    replication_factor: f64,
) -> Vec<BootstrapVec> {
    use rayon::current_num_threads;
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
    use rayon::slice::ParallelSlice;

    assert!(num_replicates > 0, "cannot bootstrap with 0 replicates");
    assert!(
        replication_factor > 0.0,
        "replication_factor cannot be negative or zero"
    );

    let num_sites = likelihoods.num_sites();
    assert!(num_sites > 0, "cannot bootstrap with 0 site likelihoods");

    let regular_chunk_len = likelihoods.num_trees().div_ceil(current_num_threads());
    let trees = likelihoods.trees();
    let chunked_iter = trees.par_chunks(regular_chunk_len).enumerate();

    // divide the trees into chunks and let threads compute replicates for a subset of trees in parallel.
    // this has the advantage that the threads can generate equal resamplings from cloned RNGs,
    // instead of sharing resampling vectors between threads.
    // It has the disadvantage that we split work across each replicate and thus concatenation of
    // the final chunks is less efficient.
    let partial_replicates = chunked_iter
        .map_with(rng.clone(), |rng, (chunk_index, chunk)| {
            let partial_replicates =
                bootstrap_slice(rng, chunk, num_replicates, num_sites, replication_factor);
            (chunk_index, partial_replicates)
        })
        .collect::<Vec<_>>();

    let mut results = vec![vec![0f64; likelihoods.num_trees()]; num_replicates];

    // concatenate the trees from each chunk to make all replicates complete. This time we can
    // divide work between threads by splitting across replicates
    results
        .par_iter_mut()
        .enumerate()
        .for_each(|(replicate, target_tree_likelihoods)| {
            partial_replicates
                .iter()
                .for_each(|(chunk_index, bootstrap_vec)| {
                    let offset = chunk_index * regular_chunk_len;
                    let trees = &bootstrap_vec[replicate];
                    target_tree_likelihoods[offset..offset + trees.len()].copy_from_slice(trees);
                });
        });

    results
}

/// For each tree, calculate the number of replicates where that tree is the maximum likelihood
/// tree.
///
/// # Parameters
/// - `bootstrap_replicates` all bootstrap replicates for each tree.
///
/// # Return
/// A vector indicating for each tree how often it is the maximum likelihood within the bootstrap
/// replicates
pub fn count_max_replicates(bootstrap_replicates: &[BootstrapVec]) -> Vec<u32> {
    let num_replicates = bootstrap_replicates[0].len();
    assert!(num_replicates > 0, "cannot calculate BP with 0 replicates");

    let mut bp_vector = vec![0u32; num_replicates];

    add_max_replicates(bootstrap_replicates, &mut bp_vector);

    bp_vector
}

/// For each tree, calculate the number of replicates where that tree is the maximum likelihood
/// tree and add those values to an existing BP vector.
///
/// # Parameters
/// - `bootstrap_replicates` all bootstrap replicates for each tree.
/// - `bp_vector` a vector containing an entry for each tree in the replicates
pub fn add_max_replicates(bootstrap_replicates: &[BootstrapVec], bp_vector: &mut [u32]) {
    assert!(
        !bootstrap_replicates.is_empty(),
        "cannot bootstrap without site likelihoods"
    );
    assert_eq!(
        bp_vector.len(),
        bootstrap_replicates[0].len(),
        "the bp_vector has a different number of entries ({}) than the bootstrap replicates ({}).",
        bp_vector.len(),
        bootstrap_replicates[0].len()
    );

    bootstrap_replicates.iter().for_each(|rep| {
        let best_tree = rep
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;
        bp_vector[best_tree] += 1;
    });
}

/// Calculate the bootstrap proportion of each tree for one or multiple bootstrap tables.
pub fn calc_bootstrap_proportion<'a, I: IntoIterator<Item = &'a [BootstrapVec]>>(
    bootstraps: I,
) -> Vec<f64> {
    let mut iter = bootstraps.into_iter();
    let mut replicates = 0;
    let mut bp_vector = if let Some(first) = iter.next() {
        replicates += first.len();
        count_max_replicates(first)
    } else {
        panic!("cannot calculate BP values from zero bootstrap tables.");
    };

    iter.for_each(|bootstrap| {
        replicates += bootstrap.len();
        add_max_replicates(bootstrap, &mut bp_vector);
    });

    bp_vector
        .into_iter()
        .map(|bp| bp as f64 / replicates as f64)
        .collect()
}
