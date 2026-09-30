use seismic::npy::write_f64;

#[test]
fn npy_header_and_payload() {
    let dir = std::env::temp_dir().join("seismic-npy-test");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("a.npy");
    write_f64(&p, &[2, 3], &(0..6).map(|i| i as f64).collect::<Vec<_>>()).unwrap();
    let b = std::fs::read(&p).unwrap();
    assert_eq!(&b[..6], b"\x93NUMPY");
    assert_eq!(&b[6..8], &[1, 0]);
    let hlen = u16::from_le_bytes([b[8], b[9]]) as usize;
    let header = String::from_utf8_lossy(&b[10..10 + hlen]);
    assert!(header.contains("'<f8'"));
    assert!(header.contains("'fortran_order': False"));
    assert!(header.contains("'shape': (2, 3)"));
    assert!(header.trim_end().ends_with('}'));
    assert_eq!(b.len(), 10 + hlen + 6 * 8);
    assert_eq!((10 + hlen) % 64, 0, "total prefix must be a multiple of 64");
}

#[test]
fn npy_f32_writes_four_bytes() {
    let dir = std::env::temp_dir().join("seismic-npy-test");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("b.npy");
    seismic::npy::write_f32(&p, &[4], &[0.5f32; 4]).unwrap();
    let b = std::fs::read(&p).unwrap();
    let hlen = u16::from_le_bytes([b[8], b[9]]) as usize;
    let header = String::from_utf8_lossy(&b[10..10 + hlen]);
    assert!(header.contains("'<f4'"));
    assert_eq!(b.len() - 10 - hlen, 4 * 4);
}
