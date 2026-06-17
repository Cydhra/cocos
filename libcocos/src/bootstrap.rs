//! This module handles the RELL bootstrap method.
//! Methods in this module take a table of log-likelihoods and approximate a bootstrap replicate
//! by resampling columns of the table and summing the rows of the resulting resampled table
//! to obtain the likelihoods of each input within the replicate.
//!
//! Given a set of [bootstrap scales] and a matching set of [replicate counts], bootstrapping
//! will generate a set `B` of bootstrap replicates per `scale`, where `B` is the replicate count
//! of the respective bootstrap scale.
//! Each replicate is resampled (with replacement) from columns of the input log-likelihood table,
//! and the  bootstrap scale determines the factor to the column count used for resampling.
//! For example, a scale of 1.5 means the resampling uses 1.5 times the number of columns of the input
//! sequences.
//!
//! # Deltas
//! The replicates are normalized after all of them have been sampled.
//! Normalization means that for each replicate (one set of log-likelihoods, one for each input)
//! the best-scoring input (highest likelihood) is determined, and then deltas are computed as
//! the difference of log-likelihood for each input to the best input.
//! The delta for the best input is the negative difference to the second-best tree (which might be
//! zero if the second-best input has the same exact likelihood).
//! The deltas are then sorted per-input, meaning for each input, one sorted vector of likelihood
//! deltas per scale is computed.
//!
//! # RELL Bootstrap
//! This was designed for phylogenetic trees, where drawing per-site log-likelihoods approximates
//! the bootstrap resampling of the Multiple Sequence Alignment, even if the model parameters
//! are not optimized for the resampled dataset.
//! However, it can be applied to other problems that allow sampling log-likelihoods of independent
//! events to approximate a bootstrap resampling of the original dataset.
//! The module makes no assumptions about the source of the log-likelihood and resamples at random
//! with the provided random number generator.
//!
//! [bootstrap scales]: DEFAULT_FACTORS
//! [replicate counts]: DEFAULT_REPLICATES

use crate::delta::{ReplicateDeltas, compute_approximate_delta_table, compute_delta_table};
use crate::vectors::dot_prod;
use crate::{ResamplingWeights, SiteLikelihoodTable, SiteLikelihoods};
use rand::Rng;
use rand::distr::Uniform;

/// The default bootstrap scales recommended by H. Shimodaira in the CONSEL software.
pub const DEFAULT_FACTORS: [f64; 10] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4];

/// The default bootstrap replicate counts for each scale factor. The values were recommended
/// by H. Shimodaira in <https://doi.org/10.1080/10635150290069913>, and are also used in CONSEL.
pub const DEFAULT_REPLICATES: [usize; 10] = [
    10_000, 10_000, 10_000, 10_000, 10_000, 10_000, 10_000, 10_000, 10_000, 10_000,
];

/// A set of bootstrap replicates of a given scale
pub struct SingleScaleBootstrap {
    replicates: Box<[f64]>,
    num_inputs: usize,
}

impl SingleScaleBootstrap {
    /// New empty single-scale bootstrap table for `num_inputs` input sequences at `num_replicates`
    /// replicates per input.
    pub fn empty(num_inputs: usize, num_replicates: usize) -> Self {
        let replicates = vec![0f64; num_inputs * num_replicates];

        Self {
            replicates: replicates.into_boxed_slice(),
            num_inputs,
        }
    }

    /// Access the bootstrap replicate at the given index.
    /// A replicate contains one likelihood for each input sequence
    pub fn replicate(&self, index: usize) -> &[f64] {
        &self.replicates[index * self.num_inputs..(index + 1) * self.num_inputs]
    }

    /// Mutably access the bootstrap replicate at the given index.
    /// A replicate contains one likelihood for each input sequence
    pub fn replicate_mut(&mut self, index: usize) -> &mut [f64] {
        &mut self.replicates[index * self.num_inputs..(index + 1) * self.num_inputs]
    }

    /// Iterator over all replicates in the table
    pub fn all_replicates(&self) -> impl Iterator<Item = &[f64]> {
        self.replicates.chunks_exact(self.num_inputs)
    }

    /// Iterator with mutable slices of all replicates in the table
    pub fn all_replicates_mut(&mut self) -> impl Iterator<Item = &mut [f64]> {
        self.replicates.chunks_exact_mut(self.num_inputs)
    }
}

/// Generate a random vector of per-site weights, indicating how often each site of an alignment got
/// selected in bootstrap replication.
/// This vector can be multiplied with each input sequence to generate a set of `N` replicate
/// likelihoods that are each drawn from the same columns across all input sequences.
/// When repeated `B` times, a `N x B` matrix of bootstrap replicates is generated.
///
/// # Replicate Scales
/// The bootstrapping of the AU test is a multiscale bootstrap scheme.
/// The `bootstrap_scale` of the bootstrap process indicates how many columns are selected (with replacement)
/// from the input sequences.
/// Therefore, the sum of entries in the weight vector corresponds to `bootstrap_scale * M`,
/// where `M` is the input sequence length.
///
/// # Parameters
/// - `num_sites` how many sites the original alignment has, minimum of 1
/// - `bootstrap_scale` the ratio between the number of original sites and the number of sites in
///   the bootstrap replicate. Cannot be negative or zero.
///
/// # Panic
/// Panics if `num_sites` is zero, or `bootstrap_scale` isn't a strictly positive number.
pub fn generate_weight_vector<R: Rng>(
    rng: &mut R,
    num_sites: usize,
    bootstrap_scale: f64,
) -> ResamplingWeights {
    assert!(num_sites > 0, "cannot bootstrap an alignment of size 0");
    assert!(
        bootstrap_scale > 0.0,
        "replication_factor cannot be negative or zero"
    );

    let mut selection_vector = vec![0.0; num_sites].into_boxed_slice();
    let distribution = Uniform::new(0, num_sites).unwrap();

    rng.sample_iter(distribution)
        .take((num_sites as f64 * bootstrap_scale) as usize)
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
///   [`generate_weight_vector`]
///
/// # Panic
/// Panics if the `site_lh` vector and the `selection` vector have different lengths.
///
/// [`generate_weight_vector`]: generate_weight_vector
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
/// their compound log-likelihood value.
/// This implements the actual work of bootstrapping
/// but operates on a slice of [`SiteLikelihoods`] vectors,
/// making this function the kernel to the sequential and parallel bootstrapping algorithms.
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
) -> SingleScaleBootstrap {
    let mut results = SingleScaleBootstrap::empty(likelihoods.len(), num_replicates);

    for rep in 0..num_replicates {
        let weights = generate_weight_vector(rng, num_sites, replication_factor);

        // compute the sum of site log-likelihoods weighted by the given selection vector
        // and scale it by the replication_factor to make it comparable to the original log-likelihood
        likelihoods
            .iter()
            .zip(results.replicate_mut(rep))
            .for_each(|(site_lh, target)| {
                *target = compute_replicate_likelihood(site_lh, &weights) / replication_factor
            })
    }

    results
}

/// Given a matrix of N log-likelihood sequences,
/// generate `num_replicates` bootstrap replicates for each sequence and calculate
/// their compound log-likelihood value.
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
) -> SingleScaleBootstrap {
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
) -> SingleScaleBootstrap {
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

    let mut results = SingleScaleBootstrap::empty(likelihoods.num_trees(), num_replicates);

    // concatenate the trees from each chunk to make all replicates complete. This time we can
    // divide work between threads by splitting across replicates
    results
        .all_replicates_mut()
        .enumerate()
        .par_bridge()
        .for_each(|(replicate, concatenated_likelihoods)| {
            partial_replicates
                .iter()
                .for_each(|(chunk_index, bootstrap_vec)| {
                    let offset = chunk_index * regular_chunk_len;
                    let trees = &bootstrap_vec.replicate(replicate);
                    concatenated_likelihoods[offset..offset + trees.len()].copy_from_slice(trees);
                });
        });

    results
}

/// Convenience method to perform the multiscale bootstrap including calculation of the likelihood deltas.
/// This method calls [`bootstrap`] and [`compute_delta_table`] once for each scale in
/// `bootstrap_scales`, generating a number of replicates as indicated by the corresponding value in
/// `bootstrap_replicates`.
///
/// # Parameters
/// - `rng` the random number generator to use during the BP test
/// - `likelihoods` a matrix of `N` input sequences of log-likelihoods that are being resampled
///   by the bootstrap resampling
/// - `bootstrap_scales` the replicate scaling factors of the multiscale bootstrap procedure.
/// - `replication_counts` how many replicates to generate for each corresponding scaling factor
///
/// # Return
/// The [likelihood delta table] which contains the normalized bootstrap likelihoods. The type contains
/// one table per replicate scale.
///
/// [`bootstrap`]: bootstrap
/// [`compute_delta_table`]: compute_delta_table
/// [likelihood delta table]: ReplicateDeltas
pub fn multiscale_bootstrap<R>(
    rng: &mut R,
    likelihoods: &SiteLikelihoodTable,
    bootstrap_scales: &[f64],
    replication_counts: &[usize],
) -> ReplicateDeltas
where
    R: Rng,
{
    let mut replicate_matrix = ReplicateDeltas::new(
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
        compute_delta_table(&mut replicate_matrix, &replicates, scale_index);
    }

    replicate_matrix
}

/// Convenience method to perform the multiscale bootstrap including calculation of the likelihood deltas
/// in parallel.
/// More details in the [single-threaded function].
///
/// # Parameters
/// - `rng` the random number generator to use during the BP test
/// - `likelihoods` a matrix of `N` input sequences of log-likelihoods that are being resampled
///   by the bootstrap resampling
/// - `bootstrap_scales` the replicate scaling factors of the multiscale bootstrap procedure.
/// - `replication_counts` how many replicates to generate for each corresponding scaling factor
///
/// # Return
/// The [likelihood delta table] which contains the normalized bootstrap likelihoods. The type contains
/// one table per replicate scale.
///
/// [single-threaded function]: multiscale_bootstrap
/// [likelihood delta table]: ReplicateDeltas
#[cfg(feature = "rayon")]
pub fn par_multiscale_bootstrap<R>(
    rng: &R,
    likelihoods: &SiteLikelihoodTable,
    bootstrap_scales: &[f64],
    replication_counts: &[usize],
) -> ReplicateDeltas
where
    R: Rng + Clone + Send,
{
    use crate::bootstrap::par_bootstrap;
    use crate::delta::par_compute_delta_table;

    let mut replicate_matrix = ReplicateDeltas::new(
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
        par_compute_delta_table(&mut replicate_matrix, &replicates, scale_index);
    }

    replicate_matrix
}

/// Convenience method to perform the multiscale bootstrap including calculation of the likelihood deltas
/// with the [rescaling approximation].
/// This method performs [`bootstrap`] only once, and computes the multiscale [delta table]
/// using rescaling as outlined in the [rescaling approximation].
///
/// # Parameters
/// - `rng` the random number generator to use during the BP test
/// - `likelihoods` a matrix of `N` input sequences of log-likelihoods that are being resampled
///   by the bootstrap resampling
/// - `bootstrap_scales` the replicate scaling factors of the multiscale bootstrap procedure. The
///   scale indexed by `reference_scale` will be used for the bootstrap process, the likelihoods
///   of the other scales will be approximated.
/// - `reference_scale` which scale to use for bootstrapping
/// - `replication_counts` how many replicates to generate for each corresponding scaling factor
///
/// # Return
/// The [likelihood delta table] which contains the normalized bootstrap likelihoods.
/// The type contains one table per replicate scale.
///
/// [delta table]: ReplicateDeltas
/// [likelihood delta table]: ReplicateDeltas
/// [rescaling approximation]: compute_approximate_delta_table
/// [`bootstrap`]: bootstrap
pub fn approx_multiscale_bootstrap<R>(
    rng: &mut R,
    likelihoods: &SiteLikelihoodTable,
    bootstrap_scales: &[f64],
    reference_scale: usize,
    replication_count: usize,
) -> ReplicateDeltas
where
    R: Rng + Clone + Send,
{
    let mut replicate_matrix = ReplicateDeltas::new(
        bootstrap_scales.to_vec().into_boxed_slice(),
        vec![replication_count; bootstrap_scales.len()].into_boxed_slice(),
        likelihoods.num_trees(),
    );

    let replicates = bootstrap(
        rng,
        likelihoods,
        replication_count,
        bootstrap_scales[reference_scale],
    );
    compute_approximate_delta_table(
        &mut replicate_matrix,
        &replicates,
        reference_scale,
        &(0..bootstrap_scales.len()).collect::<Vec<_>>(),
    );
    replicate_matrix
}

/// Convenience method to perform the multiscale bootstrap including calculation of the likelihood deltas
/// with the [rescaling approximation] in parallel.
///
/// More details in the [single-threaded function].
///
/// # Parameters
/// - `rng` the random number generator to use during the BP test
/// - `likelihoods` a matrix of `N` input sequences of log-likelihoods that are being resampled
///   by the bootstrap resampling
/// - `bootstrap_scales` the replicate scaling factors of the multiscale bootstrap procedure. The
///   scale indexed by `reference_scale` will be used for the bootstrap process, the likelihoods
///   of the other scales will be approximated.
/// - `reference_scale` which scale to use for bootstrapping
/// - `replication_counts` how many replicates to generate for each corresponding scaling factor
///
/// # Return
/// The [likelihood delta table] which contains the normalized bootstrap likelihoods.
/// The type contains one table per replicate scale.
///
/// [single-threaded function]: approx_multiscale_bootstrap
/// [rescaling approximation]: compute_approximate_delta_table
/// [likelihood delta table]: ReplicateDeltas
#[cfg(feature = "rayon")]
pub fn par_approx_multiscale_bootstrap<R>(
    rng: &mut R,
    likelihoods: &SiteLikelihoodTable,
    bootstrap_scales: &[f64],
    reference_scale: usize,
    replication_count: usize,
) -> ReplicateDeltas
where
    R: Rng + Clone + Send,
{
    let mut replicate_matrix = ReplicateDeltas::new(
        bootstrap_scales.to_vec().into_boxed_slice(),
        vec![replication_count; bootstrap_scales.len()].into_boxed_slice(),
        likelihoods.num_trees(),
    );

    let replicates = par_bootstrap(
        rng,
        likelihoods,
        replication_count,
        bootstrap_scales[reference_scale],
    );

    // TODO make parallel version of this
    compute_approximate_delta_table(
        &mut replicate_matrix,
        &replicates,
        reference_scale,
        &(0..bootstrap_scales.len()).collect::<Vec<_>>(),
    );
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

        let v = generate_weight_vector(&mut rng, 100, 1.0);
        assert_eq!(v.len(), 100);
        assert_eq!(v.iter().sum::<f64>(), 100.0);

        let v = generate_weight_vector(&mut rng, 100, 2.0);
        assert_eq!(v.len(), 100);
        assert_eq!(v.iter().sum::<f64>(), 200.0);

        let v = generate_weight_vector(&mut rng, 200, 0.5);
        assert_eq!(v.len(), 200);
        assert_eq!(v.iter().sum::<f64>(), 100.0);
    }

    #[test]
    fn test_normalize_replicates() {
        // normalize replicates is supposed to calculate observed likelihood differences when
        // compared with the global maximum likelihood (replicate) tree, or with the second best
        // tree in case of the best tree.

        #[rustfmt::skip]
        let replicates = Box::new([
            -2.0, -1.9, -2.0,
            -2.0, -2.0, -1.0,
            -2.0, -1.0, -1.0,
            -2.0, -1.0, -0.5,
        ]);
        let replicate_table = SingleScaleBootstrap {
            replicates,
            num_inputs: 3,
        };

        let mut replicate_matrix = ReplicateDeltas::new(Box::new([1.0]), Box::new([4]), 3);
        compute_delta_table(&mut replicate_matrix, &replicate_table, 0);

        let mut iter = replicate_matrix.get_bootstrap_vectors(0);

        // likelihoods should be normalized, so zero for the highest and positive difference for
        // the lower ones, and sorted in ascending order
        assert_eq_eps!(iter.next().unwrap(), &[0.1, 1.0, 1.0, 1.5]);
        assert_eq_eps!(iter.next().unwrap(), &[-0.1, 0.0, 0.5, 1.0]);
        assert_eq_eps!(iter.next().unwrap(), &[-1.0, -0.5, 0.0, 0.1]);
    }
}
