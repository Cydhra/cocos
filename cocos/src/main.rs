use clap::*;
use libcocos::au::error::MathError;
use libcocos::au::{get_au_values, par_get_au_values};
use libcocos::bootstrap::{DEFAULT_FACTORS, DEFAULT_REPLICATES, bp_test, par_bp_test};
use rand::{RngCore, SeedableRng, rng};
use rand_chacha::ChaCha8Rng;
use rayon::{ThreadPoolBuilder, current_num_threads};
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write, stdin, stdout};
use std::path::PathBuf;
use std::process::exit;
use std::time::Instant;

mod output;

// only use eprintln to log, so stdout can be used for data output
macro_rules! log {
    ($($rest:tt)*) => {
        std::eprintln!($($rest)*)
    }
}

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

impl Output {
    fn as_writer(&self) -> Result<BufWriter<Box<dyn Write>>, std::io::Error> {
        let write: Box<dyn Write> = match self {
            Output::IntoFile(path) => Box::new(File::create(path)?),
            Output::Stdout => Box::new(stdout()),
        };

        Ok(BufWriter::new(write))
    }
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
        log!("Failed to read input: {}", e);
        exit(1);
    });

    let likelihoods = match args.format {
        Format::Puzzle => cocos_parse::parse_puzzle(input),
    }
    .unwrap_or_else(|e| {
        log!("Failed to parse input: {}", e);
        exit(1);
    });

    // seed for bootstrapping or random (fixed) seed. We choose a fixed seed with a thread_rng
    // instead of using thread_rng during bootstrapping to ensure all threads use the same seed.
    let seed = args.seed.unwrap_or_else(|| rng().next_u64());
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    log!("Random seed: {}", seed);

    if args.threads == 0 {
        // use physical CPU count because the code is almost exclusively using the AVX unit, which
        // exists at most once per physical core.
        let threads = num_cpus::get_physical();
        log!(
            "Limiting parallel execution to {} threads to avoid vector processor oversubscription.",
            threads
        );

        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap_or_else(|e| {
                log!("Failed to build thread pool: {}", e);
                exit(1);
            });
    } else if args.threads != 1 {
        ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap_or_else(|e| {
                log!("Failed to build thread pool: {}", e);
                exit(1);
            });
    }

    log!(
        "Bootstrapping {} trees at {} scales with {} threads.",
        likelihoods.num_trees(),
        DEFAULT_FACTORS.len(),
        current_num_threads(),
    );

    let start = Instant::now();
    let bootstrap_replicates = if args.threads == 1 {
        bp_test(
            &mut rng,
            &likelihoods,
            &DEFAULT_FACTORS,
            &DEFAULT_REPLICATES,
        )
    } else {
        par_bp_test(&rng, &likelihoods, &DEFAULT_FACTORS, &DEFAULT_REPLICATES)
    };
    log!("Finished Bootstrapping in {:?}.", start.elapsed());

    let au_values = if bootstrap_replicates.num_trees() >= 1000 {
        log!("Estimating necessary parameters in parallel...");
        par_get_au_values(&bootstrap_replicates)
    } else {
        log!("Not enough trees. Estimating necessary parameters sequentially...");
        get_au_values(&bootstrap_replicates)
    };

    let au_values = au_values
        .iter()
        .enumerate()
        .map(|(i, result)| match result {
            Ok(p_value) => *p_value,
            Err(error) => match error {
                MathError::HessianSingular => {
                    log!("Error: Failed to calculate p-value for tree {i}: {}", error);
                    0.0
                }
                MathError::ConvergenceFailed { p_value } => {
                    log!("Warning: likelihood function for tree {i} did not converge.");
                    *p_value
                }
            },
        })
        .collect::<Box<[_]>>();

    log!("Total time {:?}", start.elapsed());
    log!(
        "Credible Tree Set Size: {}",
        au_values.iter().filter(|&&v| v >= 0.05).count()
    );

    let writer = args.output.as_writer().unwrap_or_else(|e| {
        log!("Failed to open output file: {}", e);
        exit(1);
    });

    let canonical_bp_values = (0..bootstrap_replicates.num_trees())
        .map(|tree| {
            bootstrap_replicates.compute_bp_values(tree, 0.0)[4]
                / bootstrap_replicates.replication_counts()[4] as f64
        })
        .collect::<Vec<_>>();
    output::print_tsv(writer, canonical_bp_values, au_values).unwrap_or_else(|e| {
        log!("Failed to write to output file: {}", e);
        exit(1);
    });

    log!("Written output to {}.", args.output)
}
