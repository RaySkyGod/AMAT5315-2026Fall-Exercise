#[test]
fn enzyme_cube_smoke() {
    let [y, dy, adj] = seismic::enzyme::cube_smoke(2.0);
    assert_eq!([y, dy, adj], [8.0, 12.0, 12.0]);
}
