use std::io::{BufWriter, Write};

/// Print the BP and AU results to a TSV file.
pub fn print_tsv<I1, I2, W>(mut output: BufWriter<W>, bp: I1, au: I2) -> std::io::Result<()>
where
    I1: IntoIterator<Item = f64>,
    I2: IntoIterator<Item = f64>,
    W: Write,
{
    writeln!(output, "BP\tAU")?;

    for (bp, au) in bp.into_iter().zip(au.into_iter()) {
        writeln!(output, "{}\t{}", bp, au)?;
    }

    Ok(())
}
