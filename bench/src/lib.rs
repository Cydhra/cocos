use libcocos::SiteLikelihoodTable;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Load and parse site-likelihoods from given path.
pub fn load_from_path(path: &str) -> anyhow::Result<SiteLikelihoodTable> {
    let file = File::open(Path::new(path))?;
    let reader = BufReader::new(file);
    cocos_parse::parse_puzzle(reader)
}

/// Resample a [`SiteLikelihoodTable`] using an index list.
/// The `i`-th vector of the resulting likelihood table is drawn from the original table as the
/// vector at position `selection[i]`.
/// The resulting table thus has `selection.len()` likelihood vectors.
pub fn resample_lnl_vectors(
    original: &SiteLikelihoodTable,
    selection: &[usize],
) -> SiteLikelihoodTable {
    let mut result = SiteLikelihoodTable::new(selection.len(), original.num_sites());

    for (i, &vector) in selection.iter().enumerate() {
        result[i].copy_from_slice(&original[vector]);
    }

    result
}
