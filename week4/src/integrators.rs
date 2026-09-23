//! The `Integrator` trait of Part 1: forward Euler, the explicit midpoint
//! rule, classical fourth-order Runge-Kutta (Equation 10), and an RK4 whose
//! four stage weights are equal (used by the Part 1 accuracy study).
//!
//! States are complex vectors: the line of Part 1 is real (zero imaginary
//! part), and the fluid solver of Part 2 carries Fourier coefficients.

use num_complex::Complex64;

/// Rate function `F`: state -> d(state)/dt.
pub type RateFn<'a> = &'a dyn Fn(&[Complex64]) -> Vec<Complex64>;

/// One explicit time integrator: advance `u` by one step of size `h` using
/// the rate function `f`.
pub trait Integrator {
    fn step(&self, u: &[Complex64], f: RateFn, h: f64) -> Vec<Complex64>;
    fn name(&self) -> &'static str;
}

fn axpy(a: f64, x: &[Complex64], y: &[Complex64]) -> Vec<Complex64> {
    x.iter().zip(y).map(|(&xi, &yi)| a * xi + yi).collect()
}

/// Forward Euler: `u_{n+1} = u_n + h F(u_n)`.
pub struct Euler;

impl Integrator for Euler {
    fn step(&self, u: &[Complex64], f: RateFn, h: f64) -> Vec<Complex64> {
        axpy(h, &f(u), u)
    }
    fn name(&self) -> &'static str {
        "euler"
    }
}

/// Explicit midpoint rule (rk2): trial half step, use the rate found there.
pub struct Midpoint;

impl Integrator for Midpoint {
    fn step(&self, u: &[Complex64], f: RateFn, h: f64) -> Vec<Complex64> {
        let k1 = f(u);
        let u_half = axpy(h / 2.0, &k1, u);
        let k2 = f(&u_half);
        axpy(h, &k2, u)
    }
    fn name(&self) -> &'static str {
        "rk2"
    }
}

/// Classical fourth-order Runge-Kutta, stage weights 1, 2, 2, 1.
pub struct Rk4;

impl Integrator for Rk4 {
    fn step(&self, u: &[Complex64], f: RateFn, h: f64) -> Vec<Complex64> {
        let k1 = f(u);
        let k2 = f(&axpy(h / 2.0, &k1, u));
        let k3 = f(&axpy(h / 2.0, &k2, u));
        let k4 = f(&axpy(h, &k3, u));
        let mut out = u.to_vec();
        for i in 0..out.len() {
            out[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        out
    }
    fn name(&self) -> &'static str {
        "rk4"
    }
}

/// RK4 with the four stage weights equal (1, 1, 1, 1)/4: same four stages,
/// only second order -- the signature of a wrong-weight mistake.
pub struct EqualWeightsRk4;

impl Integrator for EqualWeightsRk4 {
    fn step(&self, u: &[Complex64], f: RateFn, h: f64) -> Vec<Complex64> {
        let k1 = f(u);
        let k2 = f(&axpy(h / 2.0, &k1, u));
        let k3 = f(&axpy(h / 2.0, &k2, u));
        let k4 = f(&axpy(h, &k3, u));
        let mut out = u.to_vec();
        for i in 0..out.len() {
            out[i] += (h / 4.0) * (k1[i] + k2[i] + k3[i] + k4[i]);
        }
        out
    }
    fn name(&self) -> &'static str {
        "equal-weight rk4"
    }
}

/// Select an integrator by its `--method` flag value.
pub fn by_name(method: &str) -> anyhow::Result<Box<dyn Integrator>> {
    match method {
        "euler" => Ok(Box::new(Euler)),
        "rk2" => Ok(Box::new(Midpoint)),
        "rk4" => Ok(Box::new(Rk4)),
        other => Err(anyhow::anyhow!("unknown method {other:?}: euler, rk2, or rk4")),
    }
}
