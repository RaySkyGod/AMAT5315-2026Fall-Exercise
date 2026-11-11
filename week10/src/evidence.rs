//! Run-directory evidence writers.
//!
//! Every run stores a `meta.json` (model, geometry, settings, results,
//! timings, file map) plus raw little-endian f64 tensors in `.b64` files.
//! Raw f64 keeps full floating-point precision; dims live in the metadata.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub struct RunWriter {
    dir: PathBuf,
}

impl RunWriter {
    /// Create `out_dir` (and parents); refuses to silently reuse a directory
    /// that already contains a `meta.json` from another run.
    pub fn create(out_dir: &str) -> io::Result<Self> {
        let dir = Path::new(out_dir).to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(RunWriter { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Write pretty JSON metadata.
    pub fn write_json(&self, name: &str, value: &serde_json::Value) -> io::Result<()> {
        let text = serde_json::to_string_pretty(value)?;
        fs::write(self.dir.join(name), text)
    }

    /// Write raw little-endian f64 data (`.b64` suffix by convention).
    pub fn write_f64(&self, name: &str, data: &[f64]) -> io::Result<()> {
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for x in data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        fs::write(self.dir.join(name), bytes)
    }

    /// Convenience: write a tensor and return a metadata entry.
    pub fn write_tensor(
        &self,
        name: &str,
        dims: &[usize],
        data: &[f64],
    ) -> io::Result<serde_json::Value> {
        self.write_f64(name, data)?;
        Ok(serde_json::json!({
            "file": name,
            "dtype": "<f8",
            "order": "row-major",
            "dims": dims,
            "count": data.len(),
        }))
    }
}
