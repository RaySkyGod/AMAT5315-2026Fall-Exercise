//! Evaluates the md crate's Lennard-Jones functions on a 2-D grid around one
//! atom fixed at the origin, and writes CSV rows `x,y,u,fx,fy` to stdout,
//! where (fx, fy) is the force acting on a probe atom placed at (x, y).
//!
//! The repulsive core (r < 0.75 sigma) is omitted because u(r) -> +infinity
//! as r -> 0; the plot script masks it as a white disk.
//!
//! Regenerate week2/field.png (run from inside the md/ directory):
//!     cargo run --example field | python3 ../plot_field.py ../field.png

use md::{lj_energy, lj_force};

fn main() {
    let n = 221_u32; // 221 x 221 grid over [-2.2, 2.2]^2
    let (lo, hi) = (-2.2_f64, 2.2_f64);
    let core = 0.75_f64;

    println!("x,y,u,fx,fy");
    for iy in 0..n {
        let y = lo + (hi - lo) * iy as f64 / (n - 1) as f64;
        for ix in 0..n {
            let x = lo + (hi - lo) * ix as f64 / (n - 1) as f64;
            let r = (x * x + y * y).sqrt();
            if r < core {
                continue;
            }
            let f = lj_force(r); // scalar force, positive = repulsive
            let (fx, fy) = (f * x / r, f * y / r);
            println!("{x:.6},{y:.6},{:.9},{fx:.9},{fy:.9}", lj_energy(r));
        }
    }
}
