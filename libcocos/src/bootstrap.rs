//! This module handles the RELL bootstrap method.
//! Methods in this module take a matrix of log-likelihoods and approximate a bootstrap replicate
//! by drawing from the log-likelihoods and summing them to obtain the likelihood of a replicate.
//!
//! This was designed for phylogenetic trees, where drawing per-site log-likelihoods approximates
//! the bootstrap resampling of the Multiple Sequence Alignment, even if the model parameters
//! are not optimized for the resampled dataset.
//! However, it can be applied to other problems that allow sampling log-likelihoods of independent
//! events to approximate a bootstrap resampling of the original dataset.
//! The module makes no assumptions about the source of the log-likelihood and resamples at random
//! with the provided random number generator.

use crate::vectors::{dot_prod, max};
use crate::{BootstrapReplicates, ResamplingWeights, SiteLikelihoodTable, SiteLikelihoods};
use rand::Rng;
use rand::distr::Uniform;

/// The default bootstrap scales recommended by H. Shimodaira in the CONSEL software.
pub const DEFAULT_FACTORS: [f64; 10] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4];

/// The default bootstrap replicate counts for each scale factor. The values were recommended
/// by H. Shimodaira in <https://doi.org/10.1080/10635150290069913>, and are also used in CONSEL.
pub const DEFAULT_REPLICATES: [usize; 10] = [
    10_000, 10_000, 10_000, 10_000, 10_000, 10_000, 10_000, 10_000, 10_000, 10_000,
];

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
) -> Box<[Box<[f64]>]> {
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

/// Given a matrix of N log-likelihood sequences,
/// generate `num_replicates` bootstrap replicates for each sequence and calculate
/// their log-likelihood value.
///
/// # Parameters
/// - `rng` random number generator state
/// - `likelihoods` a matrix of site log-likelihoods, one vector of site log-likelihoods per input
///   tree.
/// - `num_replicates` how many replicates to generate per input sequence
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
) -> Box<[Box<[f64]>]> {
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

/// Given a matrix of N log-likelihood sequences,
/// generate `num_replicates` bootstrap replicates for each sequence and calculate
/// their log-likelihood value.
///
/// # Parameters
/// - `rng` random number generator state
/// - `likelihoods` a matrix of site log-likelihoods, one vector of site log-likelihoods per input
///   tree.
/// - `num_replicates` how many replicates to generate per tree
/// - `replication_factor` the ratio between the original alignment length and the length of bootstrap sequences
///
/// # Panic
/// Panics if `num_replicates` is 0, or the `replication_factor` is negative or zero, or the
/// trees have zero site likelihoods, or if the result array is too small to store all
#[cfg(feature = "rayon")]
pub fn par_bootstrap<R: Rng + Clone + Send>(
    rng: &R,
    likelihoods: &SiteLikelihoodTable,
    num_replicates: usize,
    replication_factor: f64,
) -> Box<[Box<[f64]>]> {
    use rayon::current_num_threads;
    use rayon::prelude::*;

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
        .collect::<Box<_>>();

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

/// Given a matrix of replicates, subtract the maximum of each full replicate of the likelihood for
/// the given tree and write the result into `target`.
/// The tree is identified by `vector_index`, meaning every `vector_index`-th element of each
/// replicate in `replicate_likelihoods`.
/// The maximum of each `replicate` is pre-calculated in the `maxima` array.
///
/// This method is the kernel used by [`normalize_replicates`] and [`par_normalize_replicates`].
fn normalize_replicate_vector(
    target: &mut [f64],
    replicate_likelihoods: &[Box<[f64]>],
    maxima: &[f64],
    vector_index: usize,
) {
    target
        .iter_mut()
        .zip(replicate_likelihoods.iter())
        .enumerate()
        .for_each(|(i, (target, replicate))| *target = maxima[i] - replicate[vector_index]);
}

/// Convert the replicate likelihoods into the format expected by [`BootstrapReplicates`].
/// This includes:
///  - Transposing the matrix of likelihoods
///  - Moving the likelihoods toward zero, such that the Maximum Likelihood bootstrap replicate of
///    each set of replicates has likelihood zero, and all others have the (positive) log-likelihood
///    difference to the best one.
///  - Sorting the replicate (delta-)likelihoods within each row (so per input sequence) in
///    ascending order
///
/// The results are written into the provided `bootstrap_replicates` instance into the
/// `scale_index`-th matrix.
///
/// # Parameters
/// - `bootstrap_replicates` the [`BootstrapReplicates`] matrix set where the results are written to.
/// - `replicate_likelihoods` the bootstrap replicates as generated by [`bootstrap`], meaning an
///   array with `B` replicate sets, each containing one likelihood per input sequence.
/// - `scale_index` the index of the scaling factor used for bootstrapping in the scaling factor
///   array.
///
/// [`BootstrapReplicates`]: BootstrapReplicates
/// [`bootstrap`]: bootstrap
pub fn normalize_replicates(
    bootstrap_replicates: &mut BootstrapReplicates,
    replicate_likelihoods: &[Box<[f64]>],
    scale_index: usize,
) {
    // Calculate the maximum likelihood for each bootstrap replicate. Technically the paper calls
    // for calculating the maximum without the element that is being compared with, but since it
    // is never important whether the statistic is zero or below zero, we can just use the maximum
    // every time, accepting that the best input for the replicate gets likelihood zero
    let boot_max: Box<[_]> = replicate_likelihoods
        .iter()
        .map(|replicate| max(replicate))
        .collect();

    // subtract the maximum from each replicate likelihood for each tree, such that all bootstrap
    // replicates are distributed around 0
    bootstrap_replicates
        .get_bootstrap_vectors_mut(scale_index)
        .enumerate()
        .for_each(|(vector_index, vector)| {
            normalize_replicate_vector(vector, replicate_likelihoods, &boot_max, vector_index);
        });
    bootstrap_replicates
        .get_bootstrap_vectors_mut(scale_index)
        .for_each(|vector| {
            vector.sort_unstable_by(|a, b| a.total_cmp(b));
        });
}

/// Convert the replicate likelihoods into the format expected by [`BootstrapReplicates`] in
/// parallel.
///
/// For a full explanation refer to [`normalize_replicates`].
///
/// [`BootstrapReplicates`]: BootstrapReplicates
#[cfg(feature = "rayon")]
pub fn par_normalize_replicates(
    replicate_likelihoods: &[Box<[f64]>],
    replicate_matrix: &mut BootstrapReplicates,
    scale_index: usize,
) {
    use rayon::prelude::*;

    // for comments on this method see sequential version
    let boot_max: Box<[_]> = replicate_likelihoods
        .par_iter()
        .map(|replicate| max(replicate))
        .collect();
    replicate_matrix
        .get_bootstrap_vectors_mut(scale_index)
        .enumerate()
        .par_bridge()
        .for_each(|(vector_index, vector)| {
            normalize_replicate_vector(vector, replicate_likelihoods, &boot_max, vector_index);
        });
    replicate_matrix
        .get_bootstrap_vectors_mut(scale_index)
        .par_bridge()
        .for_each(|vector| {
            vector.sort_unstable_by(|a, b| a.total_cmp(b));
        });
}

/// Convenience method to perform the multiscale BP-test.
/// This method calls [`bootstrap`] and [`calc_bootstrap_proportion`] once for each scale in
/// `bootstrap_scales`, generating a number of replicates as indicated by the corresponding value in
/// `bootstrap_replicates`. All results are stored in an instance of [`todo`], which is returned.
///
/// # Parameters
/// - `rng` the random number generator to use during the BP test
/// - `likelihoods` a matrix of `N` input sequences of log-likelihoods that are being resampled
///   by the bootstrap resampling
/// - `bootstrap_scales` the scaling factors of the multiscale bootstrap procedure.
/// - `replication_counts` how many replicates to generate for each corresponding scaling factor
///
/// # Return
/// The [`todo`] containing the bootstrap proportions of all input sequences for each scale
/// individually.
///
/// [`bootstrap`]: bootstrap
/// [`calc_bootstrap_proportion`]: calc_bootstrap_proportion
pub fn bp_test<R>(
    rng: &mut R,
    likelihoods: &SiteLikelihoodTable,
    bootstrap_scales: &[f64],
    replication_counts: &[usize],
) -> BootstrapReplicates
where
    R: Rng,
{
    let mut replicate_matrix = BootstrapReplicates::new(
        bootstrap_scales.to_vec().into_boxed_slice(),
        replication_counts.to_vec().into_boxed_slice(),
        likelihoods.num_trees(),
    );

    for (scale_index, (&bootstrap_scale, &num_replicates)) in bootstrap_scales
        .iter()
        .zip(replication_counts.iter())
        .enumerate()
    {
        let replicates = bootstrap(rng, likelihoods, num_replicates, bootstrap_scale);
        normalize_replicates(&mut replicate_matrix, &replicates, scale_index);
    }

    replicate_matrix
}

/// Convenience method to perform the multiscale BP-test in parallel.
/// This method calls [`par_bootstrap`] and [`par_calc_bootstrap_proportion`] once for each scale in
/// `bootstrap_scales`, generating a number of replicates as indicated by the corresponding value in
/// `bootstrap_replicates`. All results are stored in an instance of [`BpTable`], which is returned.
///
/// # Parameters
/// - `rng` the random number generator to use during the BP test
/// - `likelihoods` a matrix of `N` input sequences of log-likelihoods that are being resampled
///   by the bootstrap resampling
/// - `bootstrap_scales` the scaling factors of the multiscale bootstrap procedure.
/// - `replication_counts` how many replicates to generate for each corresponding scaling factor
///
/// # Return
/// The [`BpTable`] containing the bootstrap proportions of all input sequences for each scale
/// individually.
///
/// [`par_bootstrap`]: par_bootstrap
/// [`par_calc_bootstrap_proportion`]: par_calc_bootstrap_proportion
/// [`BpTable`]: todo
#[cfg(feature = "rayon")]
pub fn par_bp_test<R>(
    rng: &R,
    likelihoods: &SiteLikelihoodTable,
    bootstrap_scales: &[f64],
    replication_counts: &[usize],
) -> BootstrapReplicates
where
    R: Rng + Clone + Send,
{
    use crate::bootstrap::par_bootstrap;

    let mut replicate_matrix = BootstrapReplicates::new(
        bootstrap_scales.to_vec().into_boxed_slice(),
        replication_counts.to_vec().into_boxed_slice(),
        likelihoods.num_trees(),
    );

    for (scale_index, (&bootstrap_scale, &num_replicates)) in bootstrap_scales
        .iter()
        .zip(replication_counts.iter())
        .enumerate()
    {
        // TODO we aren't using the rng correctly here, we would have to consume it and return
        //  the used rng to guarantee the different scales arent generating the same prefix
        //  of their individual distribution
        let replicates = par_bootstrap(rng, likelihoods, num_replicates, bootstrap_scale);
        par_normalize_replicates(&replicates, &mut replicate_matrix, scale_index);
    }

    replicate_matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rng;

    macro_rules! assert_eq_eps {
        ($slice:expr, $($rest:tt)*) => {
            let s = $slice;
            let rest = $($rest)*;
            assert!(s.iter().zip(rest).all(|(a, b)| (a - b).abs() < 1E-9), "lists differ more than epsilon: {:?} != {:?}", s, rest);
        };
    }

    #[test]
    fn test_selection_weight() {
        // test whether the selection vector have the same length as the original vector and sum
        // to the rescaled vector length (the sum is the number of sites selected for the rescaled
        // vector).

        let mut rng = rng();

        let v = generate_selection_vector(&mut rng, 100, 1.0);
        assert_eq!(v.len(), 100);
        assert_eq!(v.iter().sum::<f64>(), 100.0);

        let v = generate_selection_vector(&mut rng, 100, 2.0);
        assert_eq!(v.len(), 100);
        assert_eq!(v.iter().sum::<f64>(), 200.0);

        let v = generate_selection_vector(&mut rng, 200, 0.5);
        assert_eq!(v.len(), 200);
        assert_eq!(v.iter().sum::<f64>(), 100.0);
    }

    #[test]
    fn test_normalize_replicates() {
        // normalize replicates is supposed to move all replicate likelihoods towards 0, such that
        // the maximum likelihood of each replicate is negative (second_highest - max), and all
        // other likelihoods are slightly positive (max - other).
        // Our code contains an optimization which moves the maximum to 0 instead of negative, since
        // it is never important whether the value is actually negative or just equal to zero.

        let replicates = [
            vec![-2.0, -1.9, -2.0].into_boxed_slice(),
            vec![-2.0, -2.0, -1.0].into_boxed_slice(),
            vec![-2.0, -1.0, -1.0].into_boxed_slice(),
            vec![-2.0, -1.0, -0.5].into_boxed_slice(),
        ];

        let mut replicate_matrix = BootstrapReplicates::new(Box::new([1.0]), Box::new([4]), 3);
        normalize_replicates(&mut replicate_matrix, replicates.as_slice(), 0);

        let mut iter = replicate_matrix.get_bootstrap_vectors(0);

        // likelihoods should be normalized, so zero for the highest and positive difference for
        // the lower ones, and sorted in ascending order
        assert_eq_eps!(iter.next().unwrap(), &[0.1, 1.0, 1.0, 1.5]);
        assert_eq_eps!(iter.next().unwrap(), &[0.0, 0.0, 0.5, 1.0]);
        assert_eq_eps!(iter.next().unwrap(), &[0.0, 0.0, 0.0, 0.1]);
    }
}
