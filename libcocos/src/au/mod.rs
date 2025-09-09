use crate::BpTable;

mod math;
use crate::au::math::cdf;

mod newton;
pub use newton::estimate_curv_dist_newton;

mod wls;
pub use wls::estimate_curv_dist_wls;

pub fn get_au_value(bp_values: &BpTable) -> Result<Vec<f64>, argmin_math::Error> {
    let params = estimate_curv_dist_wls(bp_values);
    let results = estimate_curv_dist_newton(bp_values, params)?;

    results.iter().map(|(c, d)| Ok(1.0 - cdf(d - c))).collect()
}
