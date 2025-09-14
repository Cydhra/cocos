use crate::BpTable;

mod math;
use crate::au::math::cdf;

mod newton;
pub use newton::fit_model_newton;

mod wls;
pub use wls::fit_model_wls;

pub fn get_au_value(bp_values: &BpTable) -> Result<Vec<f64>, argmin_math::Error> {
    let params = fit_model_wls(bp_values);
    let init = params.into_iter().map(|r| {
        if r.is_ok() {
            r.unwrap()
        } else {
            r.unwrap_err()
        }
    });

    let results = fit_model_newton(bp_values, init)?;

    results.iter().map(|(c, d)| Ok(1.0 - cdf(d - c))).collect()
}
