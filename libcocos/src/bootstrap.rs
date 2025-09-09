use crate::vectors::dot_prod;
use crate::{
    BootstrapReplicates, BpTable, ResamplingWeights, SiteLikelihoodTable, SiteLikelihoods,
};
use rand::Rng;
use rand::distr::Uniform;

pub const DEFAULT_FACTORS: [f64; 10] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4];

pub const DEFAULT_REPLICATES: usize = 10_000;

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
) -> ResamplingWeights {
    assert!(num_sites > 0, "cannot bootstrap an alignment of size 0");
    assert!(
        replication_factor > 0.0,
        "replication_factor cannot be negative or zero"
    );

    let mut selection_vector = vec![0.0; num_sites].into_boxed_slice();
    let distribution = Uniform::new(0, num_sites).unwrap();

    rng.sample_iter(distribution)
        .take((num_sites as f64 * replication_factor) as usize)
        .for_each(|site| selection_vector[site] += 1.0);

    selection_vector
}

/// Compute the log likelihood of a bootstrap replicate. A bootstrap replicate is encoded as a
/// weight vector containing an integer weight per site. Computing the likelihood of the replicate
/// is thus reduced to the dot product of the weight vector (which defines how often bootstrapping
/// chose each site) and the original site likelihood vector.
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
    selection: &ResamplingWeights,
) -> f64 {
    debug_assert!(
        site_lh.len() == selection.len(),
        "selection vector must match site likelihood vector in length"
    );
    dot_prod(site_lh, selection)
}

/// Generate `num_replicates` bootstrap replicates for each log-likelihood sequence and calculate
/// their log-likelihood value. This implements the actual work of bootstrapping, but operates on
/// a slice of [`SiteLikelihoods`] vectors, making this function the kernel to the sequential
/// and parallel bootstrapping algorithms.
///
/// # Parameters
/// - `rng` random number generator state
/// - `likelihoods` a slice of [`SiteLikelihoods`] vectors
/// - `num_replicates` how many replicates to generate
/// - `num_sites` how many entries the likelihood vectors have
/// - `replication_factor` the ratio between the original alignment length and the length of bootstrap sequences
///
/// # Return
/// Returns a vector containing vectors of likelihoods for each bootstrap replicate (i.e., for each
/// tree).
///
/// [`SiteLikelihoods`]: SiteLikelihoods
#[inline]
fn bootstrap_slice<R: Rng>(
    rng: &mut R,
    likelihoods: &[&SiteLikelihoods],
    num_replicates: usize,
    num_sites: usize,
    replication_factor: f64,
) -> BootstrapReplicates {
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

    results.into_boxed_slice()
}

/// Generate `num_replicates` bootstrap replicates for each log-likelihood sequence and calculate
/// their log-likelihood value.
///
/// # Parameters
/// - `rng` random number generator state
/// - `likelihoods` a matrix of site log-likelihoods, one vector of site log-likelihoods per input
///   tree.
/// - `num_replicates` how many replicates to generate per tree
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
) -> BootstrapReplicates {
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
) -> BootstrapReplicates {
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

    let mut results = vec![vec![0f64; likelihoods.num_trees()].into_boxed_slice(); num_replicates];

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

    results.into_boxed_slice()
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
pub fn count_max_replicates(bootstrap_replicates: &[Box<[f64]>], num_trees: usize) -> Vec<u32> {
    let mut bp_vector = vec![0u32; num_trees];

    bootstrap_replicates.iter().for_each(|rep| {
        let best_tree = rep
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;
        bp_vector[best_tree] += 1;
    });

    bp_vector
}

/// Convert the bootstrap counts to a bootstrap proportion. This includes the continuity correction
/// implemented in the original `consel` software.
///
/// # Parameters
/// - `bp_table` the bp value table that is being filled with the proportions.
/// - `bp_vector` an iterable containing the counts of how often each tree was the best in a
///   bootstrap replicate
/// - `scale_index` the index of the scaling factor we are currently converting. Each set of
///   bootstrap replicates is converted independently, and this index is used to write the
///   bootstrap proportion in the correct entry in each tree's table row.
/// - `num_replicates` how many replicates were generated for the scaling factor at the given index.
///
/// # References
/// For the correction, see the original [consel source code](https://github.com/shimo-lab/consel/blob/1a532a4fe9e7d4e9101f2bbe64613f3b0cfc6274/src/consel.c#L976):
fn count_to_proportion<I: IntoIterator<Item = u32>>(
    bp_table: &mut BpTable,
    bp_vector: I,
    scale_index: usize,
    num_replicates: usize,
) {
    bp_vector
        .into_iter()
        .zip(bp_table.scale_bp_values_mut(scale_index))
        .for_each(|(count, bp_entry)| {
            *bp_entry = count as f64 / num_replicates as f64;
        });
}

/// Calculate the bootstrap proportion of each tree for one or multiple bootstrap tables.
pub fn calc_bootstrap_proportion(
    bp_table: &mut BpTable,
    bootstrap_replicates: &BootstrapReplicates,
    scale_index: usize,
) {
    let num_replicates = bootstrap_replicates.len();
    let bp_vector = count_max_replicates(bootstrap_replicates, bp_table.num_trees());

    count_to_proportion(bp_table, bp_vector, scale_index, num_replicates);
}

#[cfg(feature = "rayon")]
pub fn par_calc_bootstrap_proportion(
    bp_table: &mut BpTable,
    bootstrap_replicates: &BootstrapReplicates,
    num_bootstrap: usize,
) {
    use rayon::current_num_threads;
    use rayon::iter::ParallelIterator;
    use rayon::slice::ParallelSlice;

    let num_replicates = bootstrap_replicates.len();
    let num_trees = bp_table.num_trees();
    let chunk_size = bootstrap_replicates.len() / current_num_threads();
    let bp_vector = bootstrap_replicates
        .par_chunks(chunk_size)
        .map(|chunk| count_max_replicates(chunk, bp_table.num_trees()))
        .reduce(
            || vec![0u32; num_trees],
            |mut acc, item| {
                acc.iter_mut().zip(item).for_each(|(a, b)| *a += b);
                acc
            },
        );

    count_to_proportion(bp_table, bp_vector, num_bootstrap, num_replicates);
}
