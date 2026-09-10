//! Cell list: bucket atoms by position so the force loop only visits
//! candidates within one cell of each atom (learning sheet Part 5).

/// Cells of width/height at least `rc`; atoms threaded through buckets
/// as a head/next linked list (no per-cell allocation, rebuilt per step).
pub(crate) struct CellList {
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) cell_w: f64,
    pub(crate) cell_h: f64,
    pub(crate) head: Vec<usize>, // usize::MAX = empty
    pub(crate) next: Vec<usize>,
}

impl CellList {
    /// nx = floor(L/rc) cells per axis (>= 2 because rc < min(L)/2 is
    /// validated by the run driver); cell sides L/n are then >= rc.
    pub(crate) fn new(box_: [f64; 2], rc: f64, pos: &[[f64; 2]]) -> Self {
        let nx = ((box_[0] / rc).floor() as usize).max(1);
        let ny = ((box_[1] / rc).floor() as usize).max(1);
        let cw = box_[0] / nx as f64;
        let ch = box_[1] / ny as f64;
        let mut head = vec![usize::MAX; nx * ny];
        let mut next = vec![usize::MAX; pos.len()];
        for (i, p) in pos.iter().enumerate() {
            let cx = ((p[0] / cw).floor() as usize).min(nx - 1);
            let cy = ((p[1] / ch).floor() as usize).min(ny - 1);
            let c = cy * nx + cx;
            next[i] = head[c];
            head[c] = i;
        }
        CellList { nx, ny, cell_w: cw, cell_h: ch, head, next }
    }

    /// The (up to) nine neighbour cells of (cx, cy), indices wrapped at
    /// the box edges and deduplicated when wrapping merges offsets
    /// (e.g. a two-cell-wide box maps -1 and +1 onto the same cell).
    pub(crate) fn neighbour_cells(&self, cx: usize, cy: usize) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::with_capacity(9);
        for dy in [-1i64, 0, 1] {
            for dx in [-1i64, 0, 1] {
                let wx = ((cx as i64 + dx).rem_euclid(self.nx as i64)) as usize;
                let wy = ((cy as i64 + dy).rem_euclid(self.ny as i64)) as usize;
                let c = (wy * self.nx + wx) as usize;
                if !out.contains(&c) {
                    out.push(c);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RC;

    /// splitmix64 for deterministic test positions.
    fn mix(s: &mut u64) -> f64 {
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn scatter(n: usize, box_: [f64; 2], seed: u64) -> Vec<[f64; 2]> {
        let mut s = seed;
        (0..n).map(|_| [mix(&mut s) * box_[0], mix(&mut s) * box_[1]]).collect()
    }

    #[test]
    fn contract_box_is_four_by_four() {
        let pos = scatter(100, [12.0141, 10.4045], 1);
        let cl = CellList::new([12.0141, 10.4045], RC, &pos);
        assert_eq!((cl.nx, cl.ny), (4, 4));
        assert!(cl.cell_w >= RC && cl.cell_h >= RC, "cells must be >= rc");
    }

    #[test]
    fn every_atom_lands_in_exactly_one_cell() {
        let box_ = [12.0141, 10.4045];
        let pos = scatter(100, box_, 2);
        let cl = CellList::new(box_, RC, &pos);
        let mut counts = vec![0usize; cl.nx * cl.ny];
        let mut seen = 0;
        for c in 0..cl.nx * cl.ny {
            let mut j = cl.head[c];
            while j != usize::MAX {
                counts[c] += 1;
                seen += 1;
                j = cl.next[j];
            }
        }
        assert_eq!(seen, 100);
        assert_eq!(cl.head.len(), cl.nx * cl.ny);
        assert!(cl.head.iter().any(|&h| h != usize::MAX));
    }

    #[test]
    fn two_cell_box_dedups_wrapped_offsets() {
        // box [6, 6]: nx = ny = 2; +-1 offsets wrap onto the same cells, so
        // the neighbour set must still contain each cell exactly once.
        let box_ = [6.0, 6.0];
        let pos = scatter(16, box_, 3);
        let cl = CellList::new(box_, RC, &pos);
        assert_eq!((cl.nx, cl.ny), (2, 2));
        for cy in 0..cl.ny {
            for cx in 0..cl.nx {
                let cells = cl.neighbour_cells(cx, cy);
                let mut sorted = cells.clone();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(sorted.len(), 4, "expected all 4 cells, got {sorted:?}");
            }
        }
    }

    #[test]
    fn wrapped_neighbours_stay_within_one_cell_step() {
        let box_ = [12.0141, 10.4045];
        let pos = scatter(10, box_, 4);
        let cl = CellList::new(box_, RC, &pos);
        for (cx, cy) in [(0, 0), (3, 3), (0, 3), (3, 0), (2, 1)] {
            for c in cl.neighbour_cells(cx, cy) {
                let (ncx, ncy) = (c % cl.nx, c / cl.nx);
                let ddx = (ncx as i64 - cx as i64).abs().min(cl.nx as i64 - (ncx as i64 - cx as i64).abs());
                let ddy = (ncy as i64 - cy as i64).abs().min(cl.ny as i64 - (ncy as i64 - cy as i64).abs());
                assert!(ddx <= 1 && ddy <= 1, "cell too far: ({cx}, {cy}) -> ({ncx}, {ncy})");
            }
        }
    }
}
