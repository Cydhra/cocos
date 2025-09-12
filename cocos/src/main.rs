use clap::*;
use libcocos::BpTable;
use libcocos::au::get_au_value;
use libcocos::bootstrap::{
    DEFAULT_FACTORS, DEFAULT_REPLICATES, bootstrap, calc_bootstrap_proportion, par_bootstrap,
    par_calc_bootstrap_proportion,
};
use rand::{RngCore, SeedableRng, rng};
use rand_chacha::ChaCha8Rng;
use rayon::{ThreadPoolBuilder, current_num_threads};
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{BufReader, Read, stdin};
use std::path::PathBuf;
use std::process::exit;
use std::time::Instant;

mod parse;

#[derive(Parser, Debug)]
#[command(version, arg_required_else_help = true)]
struct CliArgs {
    /// Input file to parse.
    /// If "-" is provided, stdin will be used.
    #[clap(long, short = 'i', value_parser = parse_input)]
    input: Input,

    /// Output file to write the selection ranking to.
    /// If "-" is provided, stdout will be used.
    #[clap(long, short = 'o', value_parser = parse_output)]
    output: Output,

    /// Number of threads to use for parallel processing. If set to 0 (default),
    /// the number of threads will be determined automatically.
    #[clap(long, short, default_value_t = 0)]
    threads: usize,

    /// Format for the site-loglikelihood input. Defaults to "puzzle", the format used by tools like
    /// Tree-PUZZLE and RAxML-ng.
    #[clap(long, short, default_value_t = Format::Puzzle)]
    format: Format,

    /// Seed for the random number generator used for bootstrapping. If not provided, a random seed
    /// is chosen.
    #[clap(long, short)]
    seed: Option<u64>,
}

#[derive(ValueEnum, Clone, Debug, PartialEq)]
enum Format {
    Puzzle,
}

impl Display for Format {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.to_possible_value()
            .expect("no values are skipped")
            .get_name()
            .fmt(f)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Input {
    FromFile(PathBuf),
    Stdin,
}

impl Display for Input {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Input::FromFile(p) => write!(f, "{:?}", p),
            Input::Stdin => {
                write!(f, "-")
            }
        }
    }
}

impl Input {
    fn as_reader(&self) -> Result<BufReader<Box<dyn Read>>, std::io::Error> {
        let read: Box<dyn Read> = match self {
            Input::FromFile(path) => Box::new(File::open(path)?),
            Input::Stdin => Box::new(stdin()),
        };

        Ok(BufReader::new(read))
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Output {
    IntoFile(PathBuf),
    Stdout,
}

impl Display for Output {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Output::IntoFile(p) => write!(f, "{:?}", p),
            Output::Stdout => {
                write!(f, "-")
            }
        }
    }
}

/// Parses the input parameter into a file or stdin.
/// If the input is "-", stdin will be used.
/// Otherwise, it checks if the file exists and is a file.
fn parse_input(input: &str) -> Result<Input, String> {
    if input == "-" {
        Ok(Input::Stdin)
    } else {
        let path = PathBuf::from(input);

        if !path.exists() {
            return Err(format!("\"{}\" does not exist", path.to_str().unwrap()));
        }

        if !path.is_file() {
            return Err(format!("\"{}\" is not a file", path.to_str().unwrap()));
        }

        Ok(Input::FromFile(path))
    }
}

/// Parses the output parameter into a file or stdout.
/// If the output is "-", stdout will be used.
/// Otherwise, it interprets the parameter as a file path.
fn parse_output(output: &str) -> Result<Output, String> {
    if output == "-" {
        Ok(Output::Stdout)
    } else {
        let path = PathBuf::from(output);
        Ok(Output::IntoFile(path))
    }
}

fn main() {
    let args = CliArgs::parse();

    let input = args.input.as_reader().unwrap_or_else(|e| {
        eprintln!("Failed to read input: {}", e);
        exit(1);
    });

    let likelihoods = match args.format {
        Format::Puzzle => parse::parse_puzzle(input),
    }
    .unwrap_or_else(|e| {
        eprintln!("Failed to parse input: {}", e);
        exit(1);
    });

    // seed for bootstrapping or random (fixed) seed. We choose a fixed seed with a thread_rng
    // instead of using thread_rng during bootstrapping to ensure all threads use the same seed.
    let seed = args.seed.unwrap_or_else(|| rng().next_u64());
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    if args.threads != 1 {
        ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap_or_else(|e| {
                eprintln!("Failed to build thread pool: {}", e);
                exit(1);
            });

        println!("Running bootstrap with {} threads", current_num_threads());
    } else {
        println!("Running bootstrap single-threaded");
    }

    let mut bp_table = BpTable::new(
        Box::new(DEFAULT_FACTORS),
        Box::new(DEFAULT_REPLICATES),
        likelihoods.num_trees(),
    );
    let start = Instant::now();

    for (scale_index, (&bootstrap_scale, &num_replicates)) in DEFAULT_FACTORS
        .iter()
        .zip(DEFAULT_REPLICATES.iter())
        .enumerate()
    {
        let replicates = if args.threads == 1 {
            bootstrap(&mut rng, &likelihoods, num_replicates, bootstrap_scale)
        } else {
            par_bootstrap(&mut rng, &likelihoods, num_replicates, bootstrap_scale)
        };

        if args.threads == 1 {
            calc_bootstrap_proportion(&mut bp_table, &replicates, scale_index);
        } else {
            par_calc_bootstrap_proportion(&mut bp_table, &replicates, scale_index);
        }
    }

    println!("Finished Bootstrapping in {:?}", start.elapsed());
    println!(
        "trees: {}, scales: {}",
        bp_table.num_trees(),
        bp_table.scales().len()
    );

    println!(
        "First 10 BP values: {:?}",
        bp_table.scale_bp_values_mut(0).take(10).collect::<Vec<_>>()
    );

    let p_value = get_au_value(&bp_table).unwrap();
    let indistinguishable = p_value.iter().filter(|&&v| v >= 0.05).count();

    println!("Tree Set Size: {indistinguishable}");

    println!(
        "first 10: {:?}",
        &p_value.iter().take(10).collect::<Vec<_>>()
    );
    println!("Total time {:?}", start.elapsed());
}
