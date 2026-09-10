//! Triangular-lattice initial conditions (learning sheet, "The lattice").

/// Lattice positions plus the periodic box they tile.
pub struct Lattice {
    pub pos: Vec<[f64; 2]>,
    pub box_: [f64; 2],
}

/// N atoms (perfect square, even row count) at density `rho` on a
/// staggered triangular lattice in a `sqrt(N)*a x sqrt(N)*h` box.
pub fn triangular(n: usize, rho: f64) -> anyhow::Result<Lattice> {
    let side = (n as f64).sqrt();
    if side.fract() != 0.0 {
        anyhow::bail!("--n {n} must be a perfect square (sqrt(N) rows and columns)");
    }
    let rows = side as usize;
    if rows % 2 != 0 {
        anyhow::bail!(
            "--n {n} gives {rows} rows; use an even row count so \
             staggering continues across the periodic edge"
        );
    }
    let a = (2.0 / (3.0_f64.sqrt() * rho)).sqrt();
    let h = (3.0_f64.sqrt() / 2.0) * a;
    let mut pos = Vec::with_capacity(n);
    for j in 0..rows {
        for i in 0..rows {
            pos.push([(i as f64 + 0.5 * (j % 2) as f64) * a, j as f64 * h]);
        }
    }
    Ok(Lattice { pos, box_: [rows as f64 * a, rows as f64 * h] })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64, msg: &str) {
        assert!((a - b).abs() < tol, "{msg}: {a} vs {b}");
    }

    #[test]
    fn contract_geometry_matches_sheet() {
        let lat = triangular(100, 0.8).unwrap();
        let a = (2.0 / (3.0_f64.sqrt() * 0.8)).sqrt();
        close(lat.box_[0], 10.0 * a, 1e-9, "Lx exact");
        close(lat.box_[1], 10.0 * (3.0_f64.sqrt() / 2.0) * a, 1e-9, "Ly exact");
        // sheet eq. (4) prints values rounded to four decimals
        close(lat.box_[0], 12.0141, 1e-4, "Lx sheet");
        close(lat.box_[1], 10.4045, 1e-4, "Ly sheet");
    }

    #[test]
    fn rows_stagger_and_stay_in_box() {
        let lat = triangular(100, 0.8).unwrap();
        let a = (2.0 / (3.0_f64.sqrt() * 0.8)).sqrt();
        let h = (3.0_f64.sqrt() / 2.0) * a;
        assert_eq!(lat.pos[0], [0.0, 0.0]);      // x_00, y_00
        assert_eq!(lat.pos[1], [a, 0.0]);        // x_10
        assert_eq!(lat.pos[10], [a / 2.0, h]);   // j odd shifts by a/2
        assert_eq!(lat.pos[11], [a * 1.5, h]);   // x_11
        for [x, y] in &lat.pos {
            assert!((0.0..lat.box_[0]).contains(x), "x out of box: {x}");
            assert!((0.0..lat.box_[1]).contains(y), "y out of box: {y}");
        }
    }

    #[test]
    fn rejects_bad_atom_counts() {
        assert!(triangular(90, 0.8).is_err()); // not a perfect square
        assert!(triangular(9, 0.8).is_err());  // odd row count (3)
        assert!(triangular(16, 0.8).is_ok());  // 4x4 is fine
    }
}
