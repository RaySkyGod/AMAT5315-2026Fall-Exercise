//! MP4 rendering: atoms panel + g(r) panel, PNG frames piped to ffmpeg.

use crate::rdf::g_r;
use crate::traj::load;
use anyhow::Context;
use image::{Rgba, RgbaImage};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

const W: u32 = 960;
const H: u32 = 480;
const FPS: &str = "10";

/// Speed -> colour (blue slow, red fast), scaled by the frame's max speed.
fn speed_colour(s: f64, s_max: f64) -> Rgba<u8> {
    let t = (s / s_max.max(1e-12)).clamp(0.0, 1.0);
    Rgba([
        (70.0 + 185.0 * t) as u8,
        (120.0 * (1.0 - t)) as u8,
        (230.0 * (1.0 - t)) as u8,
        255,
    ])
}

/// Filled disk (no anti-aliasing; radius in pixels).
fn draw_disk(img: &mut RgbaImage, cx: i32, cy: i32, r: i32, c: Rgba<u8>) {
    for y in (cy - r).max(0)..(cy + r + 1).min(img.height() as i32) {
        for x in (cx - r).max(0)..(cx + r + 1).min(img.width() as i32) {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= r * r {
                img.put_pixel(x as u32, y as u32, c);
            }
        }
    }
}

/// Thick line by drawing small disks along it.
fn draw_line(img: &mut RgbaImage, x0: f64, y0: f64, x1: f64, y1: f64, c: Rgba<u8>) {
    let steps = (x1 - x0).hypot(y1 - y0).ceil() as i32;
    for s in 0..=steps.max(1) {
        let t = s as f64 / steps.max(1) as f64;
        draw_disk(img, (x0 + t * (x1 - x0)) as i32, (y0 + t * (y1 - y0)) as i32, 1, c);
    }
}

/// 8x8 bitmap text from the font8x8 data crate (LSB = leftmost column).
pub(crate) fn draw_text(img: &mut RgbaImage, x0: i32, y0: i32, c: Rgba<u8>, s: &str) {
    use font8x8::unicode::UnicodeFonts;
    let fonts = font8x8::BASIC_FONTS;
    for (k, ch) in s.chars().enumerate() {
        if let Some(glyph) = fonts.get(ch) {
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..8 {
                    if bits & (1 << col) != 0 {
                        let x = x0 + k as i32 * 8 + col as i32;
                        let y = y0 + row as i32;
                        if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
                            img.put_pixel(x as u32, y as u32, c);
                        }
                    }
                }
            }
        }
    }
}

/// Render frame `k`: atoms coloured by speed (wrap-aware) beside g(r).
pub(crate) fn render_frame(
    frames: &[crate::traj::Frame],
    box_: &[f64; 2],
    rho: f64,
    k: usize,
) -> RgbaImage {
    let mut img = RgbaImage::from_pixel(W, H, Rgba([255, 255, 255, 255]));
    let f = &frames[k];

    // left panel: the box
    let (x0, y0, x1, y1) = (40.0_f64, 40.0_f64, 40.0 + W as f64 * 0.48, H as f64 - 40.0);
    let black = Rgba([40, 40, 40, 255]);
    draw_line(&mut img, x0, y0, x1, y0, black);
    draw_line(&mut img, x0, y1, x1, y1, black);
    draw_line(&mut img, x0, y0, x0, y1, black);
    draw_line(&mut img, x1, y0, x1, y1, black);
    let sx = (x1 - x0) / box_[0];
    let sy = (y1 - y0) / box_[1];
    let s_max = f
        .vel
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
        .fold(0.0_f64, f64::max);
    let r_px = 5.0_f64.max(sx.min(sy) * 0.35) as i32; // ~0.35 sigma in pixels
    for (p, v) in f.pos.iter().zip(f.vel.iter()) {
        let s = (v[0] * v[0] + v[1] * v[1]).sqrt();
        let c = speed_colour(s, s_max);
        let cx = x0 + p[0] * sx;
        let cy = y0 + p[1] * sy;
        // wrap-aware: also draw periodic copies near an edge
        for ox in [-box_[0], 0.0, box_[0]] {
            for oy in [-box_[1], 0.0, box_[1]] {
                let px = cx + ox * sx;
                let py = cy + oy * sy;
                if px > x0 - r_px as f64
                    && px < x1 + r_px as f64
                    && py > y0 - r_px as f64
                    && py < y1 + r_px as f64
                {
                    draw_disk(&mut img, px as i32, py as i32, r_px, c);
                }
            }
        }
    }
    draw_text(&mut img, 44, 12, black, "LJ fluid  rho=0.8  T=0.5");

    // right panel: g(r) from frames up to k (cumulative, like the physics)
    let g = g_r(&frames[..=k], box_, rho, 0.1);
    let (gx0, gy0, gx1, gy1) = (W as f64 * 0.55, 40.0, W as f64 - 40.0, H as f64 - 40.0);
    draw_line(&mut img, gx0, gy0, gx1, gy0, black);
    draw_line(&mut img, gx0, gy1, gx1, gy1, black);
    draw_line(&mut img, gx0, gy0, gx0, gy1, black);
    draw_line(&mut img, gx1, gy0, gx1, gy1, black);
    let r_max = box_[0].min(box_[1]) / 2.0;
    let g_max = g.iter().map(|(_, v)| *v).fold(1.0_f64, f64::max).max(1.5);
    let map = |r: f64, gv: f64| -> (f64, f64) {
        (gx0 + r / r_max * (gx1 - gx0), gy1 - (gv / g_max) * (gy1 - gy0))
    };
    // g = 1 reference
    let (_, ry) = map(0.0, 1.0);
    draw_line(&mut img, gx0, ry, gx1, ry, Rgba([180, 180, 180, 255]));
    for w in g.windows(2) {
        let (xa, ya) = map(w[0].0, w[0].1);
        let (xb, yb) = map(w[1].0, w[1].1);
        draw_line(&mut img, xa, ya, xb, yb, Rgba([20, 80, 180, 255]));
    }
    draw_text(&mut img, gx0 as i32, 12, black, "g(r)");
    img
}

/// Encode a frame as PNG bytes (testable without ffmpeg).
pub(crate) fn encode_png(img: &RgbaImage) -> anyhow::Result<Vec<u8>> {
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)?;
    Ok(buf)
}

/// Render every saved frame and pipe PNGs to a static ffmpeg.
pub fn render_video(dir: &Path, out: &Path) -> anyhow::Result<()> {
    let (cfg, frames) = load(dir)?;
    let rho = cfg.rho;
    let box_ = cfg.box_;
    let mut child = Command::new("ffmpeg")
        .arg("-y")
        .arg("-loglevel").arg("error")
        .arg("-f").arg("image2pipe")
        .arg("-framerate").arg(FPS)
        .arg("-i").arg("-")
        .arg("-pix_fmt").arg("yuv420p")
        .arg("-crf").arg("26")
        .arg("-r").arg(FPS)
        .arg(out)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .context("starting ffmpeg (install it and ensure it is on PATH)")?;
    {
        let stdin = child.stdin.as_mut().context("ffmpeg stdin")?;
        for k in 0..frames.len() {
            let png = encode_png(&render_frame(&frames, &box_, rho, k))?;
            stdin.write_all(&png)?;
        }
    }
    let status = child.wait().context("waiting for ffmpeg")?;
    if !status.success() {
        anyhow::bail!("ffmpeg exited with {status}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::triangular;
    use crate::traj::Frame;

    fn lattice_frame() -> (Frame, [f64; 2], f64) {
        let lat = triangular(100, 0.8).unwrap();
        let f = Frame {
            step: 50, t: 0.5, pos: lat.pos.clone(),
            vel: vec![[0.3, 0.0]; 100], e_pot: -1.0, e_kin: 4.5,
        };
        (f, lat.box_, 0.8)
    }

    #[test]
    fn png_encoding_magic_bytes() {
        let (f, box_, rho) = lattice_frame();
        let img = render_frame(&[f], &box_, rho, 0);
        let bytes = encode_png(&img).unwrap();
        assert_eq!(&bytes[..4], b"\x89PNG");
    }

    #[test]
    fn atoms_drawn_somewhere_non_background() {
        let (f, box_, rho) = lattice_frame();
        let img = render_frame(&[f], &box_, rho, 0);
        let bg = *img.get_pixel(2, 2);
        let mut non_bg = 0;
        for (_, _, p) in img.enumerate_pixels() {
            if *p != bg {
                non_bg += 1;
            }
        }
        assert!(non_bg > 1000, "only {non_bg} non-background pixels");
    }

    #[test]
    fn text_blitter_marks_pixels() {
        let mut img =
            image::RgbaImage::from_pixel(64, 16, image::Rgba([255, 255, 255, 255]));
        draw_text(&mut img, 2, 2, image::Rgba([0, 0, 0, 255]), "g(r)");
        let marked = img
            .enumerate_pixels()
            .filter(|(_, _, p)| p.0 != [255, 255, 255, 255])
            .count();
        assert!(marked > 4, "text drew {marked} pixels");
    }
}
