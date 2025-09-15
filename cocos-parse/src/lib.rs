use anyhow::anyhow;
use libcocos::SiteLikelihoodTable;
use std::io::{BufRead, BufReader, Read};
use std::ops::IndexMut;

/// Parse a puzzle-style site-loglikelihood
pub fn parse_puzzle<R: Read>(mut reader: BufReader<R>) -> anyhow::Result<SiteLikelihoodTable> {
    let mut header = Vec::with_capacity(16);
    reader.read_until(b'\n', &mut header)?;

    let header = String::from_utf8_lossy(&header);

    let mut head_iter = header.splitn(2, ' ');
    let num_trees: usize = head_iter
        .next()
        .ok_or(anyhow!("Invalid puzzle header."))?
        .parse()?;
    let num_sites: usize = head_iter
        .next()
        .ok_or(anyhow!("Invalid puzzle header."))?
        .trim()
        .parse()?;

    let mut line = String::with_capacity(num_sites * 10);
    let mut result = SiteLikelihoodTable::new(num_trees, num_sites);

    for tree_id in 0..num_trees {
        line.clear();
        let size = reader.read_line(&mut line)?;
        if size > 0 {
            line.pop();

            if line.ends_with('\r') {
                line.pop();
            }

            // get the name from the first column
            let mut record = line.splitn(2, ' ');
            let _record_name = record
                .next()
                .ok_or(anyhow!("Invalid puzzle record: no data"))?;
            let data = record
                .next()
                .ok_or(anyhow!("Invalid puzzle record: no data"))?
                .trim();

            // split the rest of the columns
            let sites_iter = data.splitn(num_sites, ' ');
            sites_iter
                .map(|site| site.parse())
                .collect::<Result<Box<[f64]>, _>>()
                .map_err(|e| {
                    anyhow!(
                        "Failed to parse site Likelihoods of tree {}: {}",
                        tree_id,
                        e
                    )
                })?
                .iter()
                .zip(result.index_mut(tree_id))
                .for_each(|(site_lh, result)| *result = *site_lh);
        } else {
            anyhow::bail!("Empty record where tree record was expected.");
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};

    #[test]
    fn test_simple_puzzle() {
        let site_lnl = b"2 4\n\
            tree1    -1.4 -1.3 -1.4 -1000.5\n\
            tree2    -1.6 -2.3 -10.4 -100.5\n\
            ";

        let reader = BufReader::new(Cursor::new(site_lnl));
        let result = parse_puzzle(reader);

        assert!(
            result.is_ok(),
            "Parsing puzzle failed: {}",
            result.unwrap_err()
        );

        let puzzle = result.unwrap();
        assert_eq!(puzzle.num_trees(), 2);

        let first_line = &puzzle[0];
        assert_eq!(first_line[0], -1.4);
        assert_eq!(first_line[1], -1.3);
        assert_eq!(first_line[2], -1.4);
        assert_eq!(first_line[3], -1000.5);

        let second_line = &puzzle[1];
        assert_eq!(second_line[0], -1.6);
        assert_eq!(second_line[1], -2.3);
        assert_eq!(second_line[2], -10.4);
        assert_eq!(second_line[3], -100.5);
    }
}
