//! Safe wrappers around the Enzyme kernel compiled by build.rs.
//!
//! The kernel (src/kernel.rs) is an input to build.rs only: never `mod` it
//! from this crate. Wrappers own or borrow every buffer and check every
//! slice length before the FFI call; no pointer escapes a call.

unsafe extern "C" {
    fn enzyme_cube(x: f64, out: *mut f64);
    fn enzyme_step_primal(um: *const f64, u: *const f64, foot: *const f64, c: *const f64,
        qamp: f64, cfl: f64, dt2: f64, ad: *const f64, bd: *const f64,
        nx: usize, nz: usize, un: *mut f64);
    fn enzyme_step_jvp(um: *const f64, d_um: *const f64, u: *const f64, d_u: *const f64,
        foot: *const f64, c: *const f64, d_c: *const f64, qamp: f64, cfl: f64, dt2: f64,
        ad: *const f64, bd: *const f64, nx: usize, nz: usize, un: *mut f64, d_un: *mut f64);
    fn enzyme_step_vjp(um: *const f64, u: *const f64, foot: *const f64, c: *const f64,
        qamp: f64, cfl: f64, dt2: f64, ad: *const f64, bd: *const f64,
        nx: usize, nz: usize, seed: *mut f64, un_out: *mut f64,
        d_um: *mut f64, d_u: *mut f64, d_c: *mut f64);
}

/// Smoke test of the Enzyme pipeline: returns (x^3, d/dx x^3, adjoint) at x.
pub fn cube_smoke(x: f64) -> [f64; 3] {
    let mut out = [0.0; 3];
    unsafe { enzyme_cube(x, out.as_mut_ptr()) }
    out
}

fn check_len(what: &str, len: usize, n: usize) {
    assert_eq!(len, n, "{} has {} entries, expected nx*nz = {}", what, len, n);
}

/// One primal timestep: bitwise identical to `wave::step` (proven by test).
#[allow(clippy::too_many_arguments)]
pub fn step_primal(um: &[f64], u: &[f64], foot: &[f64], c: &[f64], qamp: f64,
                   cfl: f64, dt2: f64, ad: &[f64], bd: &[f64], nx: usize, nz: usize,
                   un: &mut [f64]) {
    let n = nx * nz;
    check_len("um", um.len(), n);
    check_len("u", u.len(), n);
    check_len("foot", foot.len(), n);
    check_len("c", c.len(), n);
    check_len("ad", ad.len(), n);
    check_len("bd", bd.len(), n);
    check_len("un", un.len(), n);
    unsafe {
        enzyme_step_primal(um.as_ptr(), u.as_ptr(), foot.as_ptr(), c.as_ptr(), qamp,
                           cfl, dt2, ad.as_ptr(), bd.as_ptr(), nx, nz, un.as_mut_ptr());
    }
}

/// One tangent (JVP) step: advances the primal `(um, u) -> un` and the
/// tangent `(d_um, d_u) -> d_un` with tangent velocity `d_c` in one call.
#[allow(clippy::too_many_arguments)]
pub fn step_jvp(um: &[f64], d_um: &[f64], u: &[f64], d_u: &[f64], foot: &[f64],
                c: &[f64], d_c: &[f64], qamp: f64, cfl: f64, dt2: f64,
                ad: &[f64], bd: &[f64], nx: usize, nz: usize,
                un: &mut [f64], d_un: &mut [f64]) {
    let n = nx * nz;
    for (name, len) in [("um", um.len()), ("d_um", d_um.len()), ("u", u.len()),
                        ("d_u", d_u.len()), ("foot", foot.len()), ("c", c.len()),
                        ("d_c", d_c.len()), ("ad", ad.len()), ("bd", bd.len()),
                        ("un", un.len()), ("d_un", d_un.len())] {
        check_len(name, len, n);
    }
    unsafe {
        enzyme_step_jvp(um.as_ptr(), d_um.as_ptr(), u.as_ptr(), d_u.as_ptr(),
                        foot.as_ptr(), c.as_ptr(), d_c.as_ptr(), qamp, cfl, dt2,
                        ad.as_ptr(), bd.as_ptr(), nx, nz, un.as_mut_ptr(),
                        d_un.as_mut_ptr());
    }
}

/// One adjoint (VJP) step through the timestep. Given the primal state
/// `(um, u)` and the output adjoint `un_bar`, writes the input adjoints
/// `(d_um, d_u, d_c)` (zeroed first: this is the VJP of a single step).
#[allow(clippy::too_many_arguments)]
pub fn step_vjp(um: &[f64], u: &[f64], foot: &[f64], c: &[f64], qamp: f64,
                cfl: f64, dt2: f64, ad: &[f64], bd: &[f64], nx: usize, nz: usize,
                un_bar: &[f64], d_um: &mut [f64], d_u: &mut [f64], d_c: &mut [f64]) {
    let n = nx * nz;
    for (name, len) in [("um", um.len()), ("u", u.len()), ("foot", foot.len()),
                        ("c", c.len()), ("ad", ad.len()), ("bd", bd.len()),
                        ("un_bar", un_bar.len()), ("d_um", d_um.len()),
                        ("d_u", d_u.len()), ("d_c", d_c.len())] {
        check_len(name, len, n);
    }
    // The kernel's reverse entry takes the output adjoint twice (seed slot
    // and the kept primal-output slot); fill both with a copy of un_bar.
    let mut seed = un_bar.to_vec();
    let mut un_out = un_bar.to_vec();
    for v in d_um.iter_mut().chain(d_u.iter_mut()).chain(d_c.iter_mut()) {
        *v = 0.0;
    }
    unsafe {
        enzyme_step_vjp(um.as_ptr(), u.as_ptr(), foot.as_ptr(), c.as_ptr(), qamp,
                        cfl, dt2, ad.as_ptr(), bd.as_ptr(), nx, nz,
                        seed.as_mut_ptr(), un_out.as_mut_ptr(),
                        d_um.as_mut_ptr(), d_u.as_mut_ptr(), d_c.as_mut_ptr());
    }
}
