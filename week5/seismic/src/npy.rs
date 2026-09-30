//! Minimal NumPy .npy writer (version 1.0, little-endian, C order).

use std::fs::File;
use std::io::Write;
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
