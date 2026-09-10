//! Stage-timing harness: where the run spends its time, N = 400.
//!
//! samply's perf_event sampling is unavailable on this machine (kernel
//! perf_event_paranoid = 2, no sudo), so this harness times the stages
//! directly -- the same attribution the Firefox Profiler call tree would
//! give. It re-creates the profiled command's workload
//! (`md run --n 400 --eq-steps 200 --steps 1000`) with the production
//! functions from the library: `Fluid::forces` for the force stage, the
//! Verlet kick/drift/wrap arithmetic for integration, `TrajWriter` for I/O.
//!
//! Run:  cargo run --manifest-path md/Cargo.toml --release --example profile_stages

use std::time::Instant;

use md::{triangular, Fluid, Frame, RunConfig, TrajWriter, gaussian_velocities, remove_com, rescale, write_run};

/// One velocity-Verlet step split into integration vs force stages.
/// Mirrors md::fluid::Verlet::step; keep the two in sync.
fn timed_step(
    f: &mut Fluid,
    accel: &mut Vec<[f64; 2]>,
    dt: f64,
    t_force: &mut f64,
    t_integ: &mut f64,
) {
    let half = 0.5 * dt;
    let t = Instant::now();
    for k in 0..f.pos.len() {
        for d in 0..2 {
            f.vel[k][d] += accel[k][d] * half;
            f.pos[k][d] += f.vel[k][d] * dt;
        }
    }
    f.wrap();
    *t_integ += t.elapsed().as_secs_f64();

    let t = Instant::now();
    *accel = f.forces();
    *t_force += t.elapsed().as_secs_f64();

    let t = Instant::now();
    for k in 0..f.pos.len() {
        for d in 0..2 {
            f.vel[k][d] += accel[k][d] * half;
        }
    }
    *t_integ += t.elapsed().as_secs_f64();
}

fn main() -> anyhow::Result<()> {
    let (n, rho, temp, dt) = (400_usize, 0.8_f64, 0.5_f64, 0.01_f64);
    let (eq_steps, steps, sample_every) = (200_usize, 1000_usize, 50_usize);
    let (mut t_force, mut t_integ, mut t_io, mut t_measure, mut t_else) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

    let total = Instant::now();

    let t = Instant::now(); // setup: lattice, RNG, thermostat, files
    let lat = triangular(n, rho)?;
    let mut fluid = Fluid {
        pos: lat.pos,
        vel: gaussian_velocities(n, temp, 2026)?,
        box_: lat.box_,
    };
    remove_com(&mut fluid.vel);
    rescale(&mut fluid.vel, temp);
    let cfg = RunConfig {
        n, rho, box_: fluid.box_, dt, temperature: temp,
        eq_steps, steps, sample_every, seed: 2026,
        integrator: "velocity-verlet".into(),
    };
    let out = std::path::PathBuf::from("/tmp/md-prof");
    write_run(&out, &cfg)?;
    let mut writer = TrajWriter::new(&out)?;
    t_else += t.elapsed().as_secs_f64();

    let mut accel = {
        let t = Instant::now();
        let a = fluid.forces();
        t_force += t.elapsed().as_secs_f64();
        a
    };

    for k in 0..eq_steps {
        if k % 50 == 0 {
            let t = Instant::now();
            rescale(&mut fluid.vel, temp);
            t_else += t.elapsed().as_secs_f64();
        }
        timed_step(&mut fluid, &mut accel, dt, &mut t_force, &mut t_integ);
    }
    for step in 1..=steps {
        timed_step(&mut fluid, &mut accel, dt, &mut t_force, &mut t_integ);
        if step % sample_every == 0 {
            let t = Instant::now();
            let (e_pot, e_kin) = (fluid.e_pot(), fluid.e_kin());
            t_measure += t.elapsed().as_secs_f64();
            let t = Instant::now();
            writer.write_frame(&Frame {
                step: step as u64,
                t: step as f64 * dt,
                pos: fluid.pos.clone(),
                vel: fluid.vel.clone(),
                e_pot,
                e_kin,
            })?;
            t_io += t.elapsed().as_secs_f64();
        }
    }
    let elapsed = total.elapsed().as_secs_f64();
    let sum = t_force + t_integ + t_io + t_measure + t_else;

    println!("stage,seconds,share");
    for (name, secs) in [("forces", t_force), ("integrate", t_integ),
                         ("write_frame", t_io), ("measure (e_pot)", t_measure),
                         ("everything else", t_else)] {
        println!("{name},{secs:.4},{:.1}%", 100.0 * secs / sum);
    }
    println!("TOTAL,{elapsed:.4},100.0%");
    eprintln!("elapsed (wall) {elapsed:.3} s | force share {:.1}%", 100.0 * t_force / sum);
    Ok(())
}
