use seismic::wave::{l2, sample, step, State};

fn setup() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (nx, nz) = (7usize, 7usize);
    let z = vec![0.0; nx * nz];
    let mut foot = vec![0.0; nx * nz];
    foot[3 * nx + 3] = 1.0;
    let c = vec![1.5; nx * nz];
    let one = vec![1.0; nx * nz];
    (z, foot, c, one.clone(), one.clone(), vec![0.0; nx * nz])
}

#[test]
fn two_steps_no_damping() {
    let (z, foot, c, ad, bd, _) = setup();
    let (nx, nz) = (7, 7);
    let mut u0 = vec![0.0; nx * nz];
    step(&z, &z, &foot, &c, 2.0, 0.04, 0.01, &ad, &bd, nx, nz, &mut u0);
    assert!((u0[3 * nx + 3] - 0.02).abs() < 1e-12);               // dt2*qamp*foot
    assert!(u0.iter().filter(|&&v| v != 0.0).count() == 1);
    let mut u1 = vec![0.0; nx * nz];
    step(&z, &u0, &foot, &c, 2.0, 0.04, 0.01, &ad, &bd, nx, nz, &mut u1);
    // center: 2*0.02 - 0 + 0.04*2.25*(-4*0.02) + 0.02 = 0.0528
    assert!((u1[3 * nx + 3] - 0.0528).abs() < 1e-12);
    // neighbor: cfl*c^2*(0.02) = 0.0018
    assert!((u1[3 * nx + 4] - 0.0018).abs() < 1e-12);
    assert!(u1[0] == 0.0);                                        // boundary stays zero
}

#[test]
fn one_step_with_damping() {
    let (z, foot, c, _, _, _) = setup();
    let (nx, nz) = (7, 7);
    let ad = vec![0.95; nx * nz];   // sigma = 0.5, dt = 0.1
    let bd = vec![1.05; nx * nz];
    let mut un = vec![1.0; nx * nz];  // garbage: must be fully overwritten
    step(&z, &z, &foot, &c, 2.0, 0.04, 0.01, &ad, &bd, nx, nz, &mut un);
    assert!((un[3 * nx + 3] - 0.02 / 1.05).abs() < 1e-12);
    assert!(un[0] == 0.0 && un[6 * nx + 6] == 0.0);
}

#[test]
fn state_rotate_and_sample_and_l2() {
    let mut s = State::zero(4, 3);
    assert!(s.um.iter().all(|&v| v == 0.0));
    let mut un = vec![0.0; 12];
    un[5] = 2.0;
    s.rotate(un);
    assert_eq!(s.u[5], 2.0);
    assert_eq!(s.um[5], 0.0);       // old u moved to um
    let y = sample(&s.u, 4, &[[1, 1], [3, 2], [1, 1]]);
    assert_eq!(y, vec![s.u[1 * 4 + 1], s.u[2 * 4 + 3], s.u[1 * 4 + 1]]);
    assert!((l2(&[3.0, 4.0]) - 5.0).abs() < 1e-12);
    assert!((l2(&[0.0; 5])).abs() < 1e-12);
}
