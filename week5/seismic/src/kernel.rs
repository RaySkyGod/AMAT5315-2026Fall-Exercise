#![no_std]
#![feature(autodiff)]
use core::autodiff::{autodiff_forward, autodiff_reverse};

unsafe extern "C" {
    fn abort() -> !;
}
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { unsafe { abort() } }

#[autodiff_forward(cube_forward, Dual, Dual)]
#[autodiff_reverse(cube_reverse, Active, Active)]
fn cube(x: f64) -> f64 { x * x * x }

/// The damped acoustic timestep (bitwise identical to `wave::step`).
/// Active inputs for the derivatives: um, u, c (and the output un).
#[autodiff_forward(step_jvp, Dual, Dual, Const, Dual, Const, Const, Const, Const, Const, Const, Const, Dual)]
#[autodiff_reverse(step_vjp, Duplicated, Duplicated, Const, Duplicated, Const, Const, Const, Const, Const, Const, Const, DuplicatedOnly)]
#[allow(clippy::too_many_arguments)]
fn step(um: &[f64], u: &[f64], foot: &[f64], c: &[f64], qamp: f64,
        cfl: f64, dt2: f64, ad: &[f64], bd: &[f64], nx: usize, nz: usize,
        un: &mut [f64]) {
    for k in 0..nx * nz {
        un[k] = 0.0;
    }
    for j in 1..nz - 1 {
        for i in 1..nx - 1 {
            let k = j * nx + i;
            let lap = u[k + 1] + u[k - 1] + u[k + nx] + u[k - nx] - 4.0 * u[k];
            un[k] = (2.0 * u[k] - ad[k] * um[k]
                     + cfl * c[k] * c[k] * lap
                     + dt2 * qamp * foot[k]) / bd[k];
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_cube(x: f64, out: *mut f64) {
    let (y, dy) = cube_forward(x, 1.0);
    let (_, adj) = cube_reverse(x, 1.0);
    unsafe { *out = y; *out.add(1) = dy; *out.add(2) = adj; }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_step_primal(um: *const f64, u: *const f64,
    foot: *const f64, c: *const f64, qamp: f64, cfl: f64, dt2: f64,
    ad: *const f64, bd: *const f64, nx: usize, nz: usize, un: *mut f64) {
    let n = nx * nz;
    step(unsafe { core::slice::from_raw_parts(um, n) },
         unsafe { core::slice::from_raw_parts(u, n) },
         unsafe { core::slice::from_raw_parts(foot, n) },
         unsafe { core::slice::from_raw_parts(c, n) },
         qamp, cfl, dt2,
         unsafe { core::slice::from_raw_parts(ad, n) },
         unsafe { core::slice::from_raw_parts(bd, n) },
         nx, nz,
         unsafe { core::slice::from_raw_parts_mut(un, n) });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_step_jvp(um: *const f64, d_um: *const f64,
    u: *const f64, d_u: *const f64, foot: *const f64, c: *const f64,
    d_c: *const f64, qamp: f64, cfl: f64, dt2: f64, ad: *const f64,
    bd: *const f64, nx: usize, nz: usize, un: *mut f64, d_un: *mut f64) {
    let n = nx * nz;
    step_jvp(unsafe { core::slice::from_raw_parts(um, n) },
             unsafe { core::slice::from_raw_parts(d_um, n) },
             unsafe { core::slice::from_raw_parts(u, n) },
             unsafe { core::slice::from_raw_parts(d_u, n) },
             unsafe { core::slice::from_raw_parts(foot, n) },
             unsafe { core::slice::from_raw_parts(c, n) },
             unsafe { core::slice::from_raw_parts(d_c, n) },
             qamp, cfl, dt2,
             unsafe { core::slice::from_raw_parts(ad, n) },
             unsafe { core::slice::from_raw_parts(bd, n) },
             nx, nz,
             unsafe { core::slice::from_raw_parts_mut(un, n) },
             unsafe { core::slice::from_raw_parts_mut(d_un, n) });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_step_vjp(um: *const f64, u: *const f64,
    foot: *const f64, c: *const f64, qamp: f64, cfl: f64, dt2: f64,
    ad: *const f64, bd: *const f64, nx: usize, nz: usize,
    seed: *mut f64, un_out: *mut f64, d_um: *mut f64, d_u: *mut f64, d_c: *mut f64) {
    // The reverse signature is
    //   step_vjp(um, d_um, u, d_u, foot, c, d_c, qamp, cfl, dt2, ad, bd,
    //            nx, nz, un: &mut [f64], d_un: &mut [f64]).
    // The caller (std side) fills BOTH trailing slots with a copy of the
    // output adjoint `un_bar`: the seed slot is read by Enzyme, the kept
    // primal-output slot is never read (DuplicatedOnly).
    let n = nx * nz;
    unsafe {
        step_vjp(core::slice::from_raw_parts(um, n),
             core::slice::from_raw_parts_mut(d_um, n),
             core::slice::from_raw_parts(u, n),
             core::slice::from_raw_parts_mut(d_u, n),
             core::slice::from_raw_parts(foot, n),
             core::slice::from_raw_parts(c, n),
             core::slice::from_raw_parts_mut(d_c, n),
             qamp, cfl, dt2,
             core::slice::from_raw_parts(ad, n),
             core::slice::from_raw_parts(bd, n),
             nx, nz,
             core::slice::from_raw_parts_mut(seed, n),
             core::slice::from_raw_parts_mut(un_out, n));
    }
}
