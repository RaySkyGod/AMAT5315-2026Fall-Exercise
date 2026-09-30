//! Safe wrappers around the Enzyme kernel compiled by build.rs.
//!
//! The kernel (src/kernel.rs) is an input to build.rs only: never `mod` it
//! from this crate. Wrappers own or borrow every buffer and check every
//! slice length before the FFI call; no pointer escapes a call.

unsafe extern "C" {
    fn enzyme_cube(x: f64, out: *mut f64);
}

/// Smoke test of the Enzyme pipeline: returns (x^3, d/dx x^3, adjoint) at x.
pub fn cube_smoke(x: f64) -> [f64; 3] {
    let mut out = [0.0; 3];
    unsafe { enzyme_cube(x, out.as_mut_ptr()) }
    out
}
