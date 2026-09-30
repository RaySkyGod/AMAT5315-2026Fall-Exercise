//! Minimal NumPy .npy reader and writer (version 1.0, little-endian, C order).

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};

fn shape_str(shape: &[usize]) -> String {
    match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    }
}

fn write_npy(path: &Path, descr: &str, shape: &[usize], bytes: &[u8]) -> Result<()> {
    let n_expected: usize = shape.iter().product();
    if bytes.len() % n_expected.max(1) != 0 {
        bail!("payload not a multiple of the element count");
    }
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr,
        shape_str(shape)
    );
    // total (magic + version + hlen + header) must be a multiple of 64, ending in \n
    let pad = 64 - (10 + header.len() + 1) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    if header.len() + 10 > u16::MAX as usize {
        bail!("npy header too long");
    }
    let mut f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    f.write_all(b"\x93NUMPY")?;
    f.write_all(&[1, 0])?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    f.write_all(bytes)?;
    Ok(())
}

pub fn write_f64(path: &Path, shape: &[usize], data: &[f64]) -> Result<()> {
    let n: usize = shape.iter().product();
    if data.len() != n {
        bail!("write_f64: {} elements does not match shape {:?}", data.len(), shape);
    }
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    write_npy(path, "<f8", shape, &bytes)
}

pub fn write_f32(path: &Path, shape: &[usize], data: &[f32]) -> Result<()> {
    let n: usize = shape.iter().product();
    if data.len() != n {
        bail!("write_f32: {} elements does not match shape {:?}", data.len(), shape);
    }
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    write_npy(path, "<f4", shape, &bytes)
}

/// Read a float64 .npy file: returns (shape, data). Validates the magic,
/// version, `'<'` endianness, `f8` descr and C order.
pub fn read_f64(path: &Path) -> Result<(Vec<usize>, Vec<f64>)> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).with_context(|| format!("read {}", path.display()))?;
    if buf.len() < 10 || &buf[..6] != b"\x93NUMPY" {
        bail!("{}: not an npy file", path.display());
    }
    if buf[6] != 1 || buf[7] != 0 {
        bail!("{}: only npy version 1.0 is supported", path.display());
    }
    let hlen = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let header = String::from_utf8_lossy(&buf[10..10 + hlen]).to_string();
    if !header.contains("'descr': '<f8'") {
        bail!("{}: expected descr '<f8', header is {}", path.display(), header.trim_end());
    }
    if !header.contains("'fortran_order': False") {
        bail!("{}: only C-order arrays are supported", path.display());
    }
    let open = header.find("'shape': (").context("shape key missing")? + "'shape': (".len();
    let close = header[open..].find(')').context("shape tuple not closed")? + open;
    let dims: Vec<usize> = header[open..close]
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect::<Result<_, _>>()
        .with_context(|| format!("{}: bad shape tuple", path.display()))?;
    let n: usize = dims.iter().product();
    let payload = &buf[10 + hlen..];
    if payload.len() != n * 8 {
        bail!(
            "{}: payload is {} bytes, shape {:?} needs {}",
            path.display(), payload.len(), dims, n * 8
        );
    }
    let mut data = Vec::with_capacity(n);
    for chunk in payload.chunks_exact(8) {
        data.push(f64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok((dims, data))
}
