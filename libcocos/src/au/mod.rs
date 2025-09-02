use crate::BpTable;
use statrs::distribution::{ContinuousCDF, Normal};

mod math;

mod newton;
pub use newton::estimate_curv_dist;

pub fn get_au_value(bp_values: &BpTable) -> Result<Vec<f64>, argmin_math::Error> {
    let normal = Normal::new(0.0, 1.0);
    let results = estimate_curv_dist(bp_values)?;

    results
        .iter()
        .map(|(c, d)| Ok(1.0 - normal?.cdf(d - c)))
        .collect()
}
