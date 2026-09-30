use seismic::{enzyme, wave};

/// Deterministic pseudo-random inputs for the identity tests: an LCG mapped
/// to um,u ∈ [-1,1], foot ∈ [0,3], c ∈ [1,2], σ ∈ [0,0.4] with dt = 0.2 so
/// ad = 1-σdt, bd = 1+σdt.
fn inputs(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x = 12345u64;
    let mut rnd = move || {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((x >> 33) as f64) / (u32::MAX as f64)
    };
    let mut f = |lo: f64, hi: f64| lo + (hi - lo) * rnd();
    let um: Vec<_> = (0..n).map(|_| f(-1.0, 1.0)).collect();
    let u: Vec<_> = (0..n).map(|_| f(-1.0, 1.0)).collect();
    let foot: Vec<_> = (0..n).map(|_| f(0.0, 3.0)).collect();
    let c: Vec<_> = (0..n).map(|_| f(1.0, 2.0)).collect();
    let ad: Vec<_> = (0..n).map(|_| 1.0 - f(0.0, 0.4) * 0.2).collect();
    let bd: Vec<_> = ad.iter().map(|&a| 2.0 - a).collect();
    (um, u, foot, c, ad, bd)
}

#[test]
fn enzyme_cube_smoke() {
    let [y, dy, adj] = seismic::enzyme::cube_smoke(2.0);
    assert_eq!([y, dy, adj], [8.0, 12.0, 12.0]);
}

#[test]
fn kernel_primal_equals_plain_step_bitwise() {
    let (nx, nz) = (9, 7);
    let (um, u, foot, c, ad, bd) = inputs(nx * nz);
    let mut a = vec![0.0; nx * nz];
    enzyme::step_primal(&um, &u, &foot, &c, 0.7, 0.04, 0.01, &ad, &bd, nx, nz, &mut a);
    let mut b = vec![0.0; nx * nz];
    wave::step(&um, &u, &foot, &c, 0.7, 0.04, 0.01, &ad, &bd, nx, nz, &mut b);
    assert_eq!(a, b); // bitwise identical
}

#[test]
fn jvp_matches_hand_tangent() {
    // Independent hand tangent (test-only, per learning sheet):
    // d_un = (2 d_u - ad·d_um + cfl·(c²·lap(d_u) + 2c·d_c·lap(u)) + dt2·dqamp·foot)/bd,
    // with dqamp = 0 (qamp is Const) and boundary values zero.
    let (nx, nz) = (9, 7);
    let (um, u, foot, c, ad, bd) = inputs(nx * nz);
    let (d_um, d_u, _, d_c, _, _) = inputs(nx * nz); // second draw = tangents
    let mut un = vec![0.0; nx * nz];
    let mut d_un = vec![0.0; nx * nz];
    enzyme::step_jvp(&um, &d_um, &u, &d_u, &foot, &c, &d_c, 0.7, 0.04, 0.01,
                     &ad, &bd, nx, nz, &mut un, &mut d_un);
    let lap = |v: &[f64], k: usize| v[k + 1] + v[k - 1] + v[k + nx] + v[k - nx] - 4.0 * v[k];
    let mut hand = vec![0.0; nx * nz];
    for j in 1..nz - 1 {
        for i in 1..nx - 1 {
            let k = j * nx + i;
            hand[k] = (2.0 * d_u[k] - ad[k] * d_um[k]
                       + 0.04 * (2.0 * c[k] * d_c[k] * lap(&u, k)
                                 + c[k] * c[k] * lap(&d_u, k))) / bd[k];
        }
    }
    let scale = hand.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let max = hand.iter().zip(&d_un).map(|(h, e)| (h - e).abs()).fold(0.0_f64, f64::max);
    assert!(max / scale < 1e-12, "max rel diff {}", max / scale);
    // the jvp call also reproduces the primal bitwise
    let mut p = vec![0.0; nx * nz];
    wave::step(&um, &u, &foot, &c, 0.7, 0.04, 0.01, &ad, &bd, nx, nz, &mut p);
    assert_eq!(un, p);
}

#[test]
fn vjp_satisfies_dot_identity() {
    // <JVP(d_um,d_u,d_c), un_bar> == <(d_um,d_u,d_c), VJP(un_bar)>
    let (nx, nz) = (9, 7);
    let (um, u, foot, c, ad, bd) = inputs(nx * nz);
    let (d_um, d_u, _, d_c, _, _) = inputs(nx * nz);
    let (_, un_bar, _, _, _, _) = inputs(nx * nz);
    let mut un = vec![0.0; nx * nz];
    let mut d_un = vec![0.0; nx * nz];
    enzyme::step_jvp(&um, &d_um, &u, &d_u, &foot, &c, &d_c, 0.7, 0.04, 0.01,
                     &ad, &bd, nx, nz, &mut un, &mut d_un);
    let mut dum = vec![0.0; nx * nz];
    let mut du = vec![0.0; nx * nz];
    let mut dc = vec![0.0; nx * nz];
    enzyme::step_vjp(&um, &u, &foot, &c, 0.7, 0.04, 0.01, &ad, &bd, nx, nz,
                     &un_bar, &mut dum, &mut du, &mut dc);
    let lhs: f64 = d_un.iter().zip(&un_bar).map(|(a, b)| a * b).sum::<f64>();
    let rhs: f64 = dum.iter().zip(&d_um).map(|(a, b)| a * b).sum::<f64>()
        + du.iter().zip(&d_u).map(|(a, b)| a * b).sum::<f64>()
        + dc.iter().zip(&d_c).map(|(a, b)| a * b).sum::<f64>();
    assert!((lhs - rhs).abs() / lhs.abs() < 1e-12, "lhs {lhs} rhs {rhs}");
}
