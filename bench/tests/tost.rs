//! A test case which runs the available data files against consel and libcocos and tests whether
//! the libcocos outputs have the same distribution as the consel outputs.
//! It does so by doing two one-sided t-tests that attempt to reject the hypothesis that the mean
//! of cocos' output is outside the confidence interval of consel's mean output.

use csv::Trim;
use libcocos::au_test;
use libcocos::bootstrap::{DEFAULT_FACTORS, DEFAULT_REPLICATES};
use rstest::*;
use std::fs::{File, create_dir};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

const NUM_SAMPLES: usize = 10;

/// Consel output is saved to a CSV file with these five values per record. The records in the file
/// are sorted by rank, not by item.
#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct ConselRecord {
    rank: usize,
    item: usize,
    obs: f64,
    au: f64,
    np: f64,
}

#[rstest]
fn test_distribution(#[files("data/*.siteLH")] site_likelihoods: PathBuf) {
    // get environment
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let script_dir = repository_root.join("scripts");
    let consel_script = script_dir.join("runconsel.sh");
    let mut file_name = site_likelihoods
        .file_name()
        .expect("test called with invalid fixture")
        .to_str()
        .expect("file name is not representable");

    // create directory for the consel output of this site-likelihood file
    if site_likelihoods.extension().is_some() {
        let suffix = format!(
            ".{}",
            site_likelihoods.extension().unwrap().to_str().unwrap()
        );
        file_name = file_name.strip_suffix(&suffix).unwrap();
    }
    let scratch_dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(file_name);
    let _ = create_dir(&scratch_dir);

    // read site-likelihoods
    let per_site_lnl = cocos_parse::parse_puzzle(BufReader::new(
        File::open(&site_likelihoods).expect("cannot read fixture"),
    ))
    .expect("cannot parse siteLH file");
    let num_trees = per_site_lnl.num_trees();

    let mut consel_mean = vec![0.0; num_trees];
    let mut consel_variance = vec![0.0; num_trees];

    // run consel NUM_SAMPLES times and collect results
    for i in 0..NUM_SAMPLES {
        let run = scratch_dir.join(format!("run{}", i));
        let output = Command::new(&consel_script)
            .arg(&site_likelihoods)
            .arg(&run)
            .output()
            .expect("failed to run consel");
        assert_eq!(
            output.status,
            ExitStatus::default(),
            "consel script did not finish successfully"
        );

        let mut result_file = run.clone();
        result_file.set_extension("csv");

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .trim(Trim::Fields)
            .from_reader(File::open(&result_file).expect("cannot open consel output "));

        // read in the results and store them in the samples
        for record in reader.deserialize::<ConselRecord>() {
            let record = record.expect("malformed consel output");
            consel_mean[record.item - 1] += record.au;
            consel_variance[record.item - 1] += record.au * record.au;
        }
    }

    // calculate mean and variance
    for i in 0..num_trees {
        consel_mean[i] /= NUM_SAMPLES as f64;
        consel_variance[i] /= NUM_SAMPLES as f64;
        consel_variance[i] -= consel_mean[i] * consel_mean[i];
    }

    // run cocos
    let mut cocos_mean = vec![0.0; num_trees];
    let mut cocos_variance = vec![0.0; num_trees];

    let mut rng = rand::rng();
    for _ in 0..NUM_SAMPLES {
        let p_values = au_test(
            &mut rng,
            &per_site_lnl,
            &DEFAULT_FACTORS,
            &DEFAULT_REPLICATES,
        );
        for (item, result) in p_values.iter().enumerate() {
            let au = result.as_ref().expect("calculating AU value failed");
            cocos_mean[item] += au;
            cocos_variance[item] += au * au;
        }
    }

    // calculate mean and variance
    for i in 0..num_trees {
        cocos_mean[i] /= NUM_SAMPLES as f64;
        cocos_variance[i] /= NUM_SAMPLES as f64;
        cocos_variance[i] -= cocos_mean[i] * cocos_mean[i];

        println!(
            "item {}: consel mean: {}, var: {}\t-\tcocos mean: {}, var: {}",
            i, consel_mean[i], consel_variance[i], cocos_mean[i], cocos_variance[i]
        );
    }
}
