use crate::BpTable;
use statrs::distribution::{ContinuousCDF, Normal};

mod math;

mod newton;
pub use newton::estimate_curv_dist_newton;

mod rss;
pub use rss::estimate_curv_dist_rss;

pub fn get_au_value(bp_values: &BpTable) -> Result<Vec<f64>, argmin_math::Error> {
    let params = estimate_curv_dist_rss(bp_values)?;
    let results = estimate_curv_dist_newton(bp_values, params)?;

    let normal = Normal::new(0.0, 1.0).unwrap();
    results
        .iter()
        .map(|(c, d)| Ok(1.0 - normal.cdf(d - c)))
        .collect()
}
