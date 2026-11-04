//! Parser for the UAI MARKOV format used by `asia.uai` and `Promedus_14.uai`.
//!
//! Layout: header `MARKOV`, variable count, cardinalities, factor count,
//! then **all** factor scopes (arity + ordered variable ids), then **all**
//! factor tables (entry count + entries). Tables are row-major with the
//! *last* scope variable varying fastest; a factor's axis order is its scope
//! order, with the child variable listed first (e.g. scope `1 0` is `p(t|a)`
//! with axes `(t, a)`).

/// One factor: scope is the ordered list of variable ids (axis order of `table`).
#[derive(Debug, Clone, PartialEq)]
pub struct Factor {
    pub scope: Vec<usize>,
    /// Row-major entries; the last scope variable varies fastest.
    pub table: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UaiModel {
    pub nvars: usize,
    pub cards: Vec<usize>,
    pub factors: Vec<Factor>,
}

/// Parse a UAI MARKOV model file.
pub fn parse_uai(text: &str) -> Result<UaiModel, String> {
    let mut toks = text.split_whitespace();
    let header = toks.next().ok_or("empty file")?;
    if header != "MARKOV" {
        return Err(format!("expected MARKOV header, got {header}"));
    }
    let nvars: usize = toks
        .next()
        .ok_or("missing variable count")?
        .parse()
        .map_err(|e| format!("bad variable count: {e}"))?;
    let mut cards = Vec::with_capacity(nvars);
    for _ in 0..nvars {
        cards.push(
            toks
                .next()
                .ok_or("missing cardinality")?
                .parse()
                .map_err(|e| format!("bad cardinality: {e}"))?,
        );
    }
    let nfactors: usize = toks
        .next()
        .ok_or("missing factor count")?
        .parse()
        .map_err(|e| format!("bad factor count: {e}"))?;
    // All factor scopes come first.
    let mut factors = Vec::with_capacity(nfactors);
    for fi in 0..nfactors {
        let arity: usize = toks
            .next()
            .ok_or_else(|| format!("factor {fi}: missing arity"))?
            .parse()
            .map_err(|e| format!("factor {fi}: bad arity: {e}"))?;
        let mut scope = Vec::with_capacity(arity);
        for _ in 0..arity {
            let v: usize = toks
                .next()
                .ok_or_else(|| format!("factor {fi}: missing scope id"))?
                .parse()
                .map_err(|e| format!("factor {fi}: bad scope id: {e}"))?;
            if v >= nvars {
                return Err(format!("factor {fi}: scope id {v} out of range"));
            }
            scope.push(v);
        }
        factors.push(Factor {
            scope,
            table: Vec::new(),
        });
    }
    // Then all factor tables.
    for factors_fi in factors.iter_mut().enumerate() {
        let (fi, f) = factors_fi;
        let count: usize = toks
            .next()
            .ok_or_else(|| format!("factor {fi}: missing entry count"))?
            .parse()
            .map_err(|e| format!("factor {fi}: bad entry count: {e}"))?;
        let expected: usize = f.scope.iter().map(|&v| cards[v]).product();
        if count != expected {
            return Err(format!(
                "factor {fi}: entry count {count} != product of cardinalities {expected}"
            ));
        }
        for _ in 0..count {
            let x: f64 = toks
                .next()
                .ok_or_else(|| format!("factor {fi}: missing table entry"))?
                .parse()
                .map_err(|e| format!("factor {fi}: bad table entry: {e}"))?;
            f.table.push(x);
        }
    }
    if toks.next().is_some() {
        return Err("trailing tokens after last factor".into());
    }
    Ok(UaiModel {
        nvars,
        cards,
        factors,
    })
}

/// Parse a UAI evidence file: count, then (variable, state) pairs.
pub fn parse_evid(text: &str) -> Result<Vec<(usize, usize)>, String> {
    let mut toks = text.split_whitespace();
    let n: usize = toks
        .next()
        .ok_or("empty evidence file")?
        .parse()
        .map_err(|e| format!("bad evidence count: {e}"))?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let v: usize = toks
            .next()
            .ok_or("missing evidence variable")?
            .parse()
            .map_err(|e| format!("bad evidence variable: {e}"))?;
        let s: usize = toks
            .next()
            .ok_or("missing evidence state")?
            .parse()
            .map_err(|e| format!("bad evidence state: {e}"))?;
        out.push((v, s));
    }
    if toks.next().is_some() {
        return Err("trailing tokens in evidence file".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASIA: &str = include_str!("../asia.uai");

    #[test]
    fn parses_header_and_scopes() {
        let m = parse_uai(ASIA).unwrap();
        assert_eq!(m.nvars, 8);
        assert_eq!(m.cards, vec![2; 8]);
        assert_eq!(m.factors.len(), 8);
        let scopes: Vec<Vec<usize>> = m.factors.iter().map(|f| f.scope.clone()).collect();
        assert_eq!(
            scopes,
            vec![
                vec![0],      // p(a)
                vec![1, 0],   // p(t|a)
                vec![2],      // p(s)
                vec![3, 2],   // p(l|s)
                vec![4, 2],   // p(b|s)
                vec![5, 3, 1],// p(e|l,t)
                vec![6, 5],   // p(x|e)
                vec![7, 5, 4],// p(d|e,b)
            ]
        );
    }

    #[test]
    fn parses_tables_exactly() {
        let m = parse_uai(ASIA).unwrap();
        assert_eq!(m.factors[0].table, vec![0.01, 0.99]); // p(a)
        assert_eq!(m.factors[1].table, vec![0.05, 0.01, 0.95, 0.99]); // p(t|a)
        assert_eq!(m.factors[2].table, vec![0.5, 0.5]); // p(s)
        assert_eq!(m.factors[3].table, vec![0.1, 0.01, 0.9, 0.99]); // p(l|s)
        assert_eq!(m.factors[4].table, vec![0.6, 0.3, 0.4, 0.7]); // p(b|s)
        assert_eq!(
            m.factors[5].table,
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ); // p(e|l,t)
        assert_eq!(m.factors[6].table, vec![0.98, 0.05, 0.02, 0.95]); // p(x|e)
        assert_eq!(
            m.factors[7].table,
            vec![0.9, 0.7, 0.8, 0.1, 0.1, 0.3, 0.2, 0.9]
        ); // p(d|e,b)
    }

    #[test]
    fn parses_evidence_file() {
        assert_eq!(parse_evid("2\n41 1\n47 0\n").unwrap(), vec![(41, 1), (47, 0)]);
        assert!(parse_evid("2\n41 1\n").is_err());
    }

    #[test]
    fn rejects_bad_input() {
        assert!(parse_uai("BAYES\n8\n").is_err());
        assert!(parse_uai("").is_err());
        // entry count inconsistent with cardinalities
        let bad = "MARKOV\n1\n2\n1\n1 0\n3\n0.1 0.2 0.3\n";
        assert!(parse_uai(bad).is_err());
    }
}
