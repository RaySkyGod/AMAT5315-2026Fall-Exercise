//! Runs the two-atom dimer experiment and writes CSV to stdout for plotting.
//!
//! Panel 1: both integrators, 500 steps at dt = 0.01 (sampled every step).
//! Panel 2: velocity-Verlet alone, 5000 steps at dt = 0.01 (sampled every
//!          10th step); the plot scales its error by 1000.
//!
//! Initial state: dimer at rest with r0 = 1.5 sigma, reduced units.
//!
//! Regenerate week2/dimer.png (from inside the md/ directory):
//!     cargo run --example dimer | python3 ../plot_dimer.py ../dimer.png

use md::{run, ForwardEuler, System, VelocityVerlet};

const DT: f64 = 0.01;
const R0: f64 = 1.5;

fn main() {
    println!("# panel 1: t,euler,verlet  (500 steps, dt = {DT}, every step)");

    let euler = run(
        &mut ForwardEuler::new(),
        System::dimer_at_rest(R0),
        DT,
        500,
        1,
    );
    let verlet = run(
        &mut VelocityVerlet::new(&System::dimer_at_rest(R0)),
        System::dimer_at_rest(R0),
        DT,
        500,
        1,
    );
    for ((te, ee), (_, ev)) in euler.iter().zip(verlet.iter()) {
        println!("{te:.6e},{ee:.9e},{ev:.9e}");
    }

    println!("# panel 2: t,verlet  (5000 steps, dt = {DT}, every 10th step)");
    let long = run(
        &mut VelocityVerlet::new(&System::dimer_at_rest(R0)),
        System::dimer_at_rest(R0),
        DT,
        5000,
        10,
    );
    for (t, e) in &long {
        println!("{t:.6e},{e:.9e}");
    }

    let report = |label: &str, s: &[(f64, f64)]| {
        let max = s.iter().map(|p| p.1.abs()).fold(0.0_f64, f64::max);
        eprintln!("{label}: final |dE/E0| = {:.3e}, max = {:.3e}", s.last().unwrap().1.abs(), max);
    };
    report("forward-Euler  (t = 5) ", &euler);
    report("velocity-Verlet(t = 5) ", &verlet);
    report("velocity-Verlet(t = 50)", &long);
}
