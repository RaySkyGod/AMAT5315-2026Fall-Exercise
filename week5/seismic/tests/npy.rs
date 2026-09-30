use seismic::npy::write_f64;

#[test]
fn npy_roundtrip_read_f64() {
    let dir = std::env::temp_dir().join("seismic-npy-test");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("rt.npy");
    let data: Vec<f64> = (0..6).map(|i| (i as f64) * -0.25).collect();
    write_f64(&p, &[2, 3], &data).unwrap();
    let (shape, read) = seismic::npy::read_f64(&p).unwrap();
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(read, data);

    // a single-element file with a 1-d shape reads back too
    let p1 = dir.join("rt1.npy");
    write_f64(&p1, &[1], &[3.5]).unwrap();
    let (shape1, read1) = seismic::npy::read_f64(&p1).unwrap();
    assert_eq!(shape1, vec![1]);
    assert_eq!(read1, vec![3.5]);

    // a mismatched descr must be rejected
    let bad = dir.join("bad.npy");
    seismic::npy::write_f32(&bad, &[2, 2], &[0.0f32; 4]).unwrap();
    assert!(seismic::npy::read_f64(&bad).is_err());

    // a truncated payload must be rejected
    let trunc = dir.join("trunc.npy");
    write_f64(&trunc, &[3], &[1.0, 2.0, 3.0]).unwrap();
    let mut b = std::fs::read(&trunc).unwrap();
    b.truncate(b.len() - 8);
    std::fs::write(&trunc, &b).unwrap();
    assert!(seismic::npy::read_f64(&trunc).is_err());
}

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
