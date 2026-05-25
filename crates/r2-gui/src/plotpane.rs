//! Plot pane — rasterizes the engine's current plot.svg/hist.svg/etc.
//! and displays it as an egui image. Polls for changes after each
//! evaluation; manually refresh via Plot ▸ Refresh plot pane.

use eframe::egui;
use std::path::PathBuf;

pub struct PlotPane {
    last_path: Option<PathBuf>,
    last_mtime: Option<std::time::SystemTime>,
    texture: Option<egui::TextureHandle>,
}

impl PlotPane {
    pub fn new() -> Self {
        Self { last_path: None, last_mtime: None, texture: None }
    }

    /// Re-rasterize if any of plot.svg / hist.svg / boxplot.svg / barplot.svg
    /// in the current cwd has been modified since we last loaded.
    pub fn reload_if_changed(&mut self) {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let candidates = ["plot.svg", "hist.svg", "boxplot.svg", "barplot.svg"];
        // Pick the most recently modified of the candidates that exist.
        let mut latest: Option<(PathBuf, std::time::SystemTime)> = None;
        for name in &candidates {
            let p = cwd.join(name);
            if let Ok(meta) = std::fs::metadata(&p) {
                if let Ok(t) = meta.modified() {
                    if latest.as_ref().map(|(_, prev)| t > *prev).unwrap_or(true) {
                        latest = Some((p, t));
                    }
                }
            }
        }
        if let Some((path, mtime)) = latest {
            if self.last_mtime != Some(mtime) || self.last_path.as_ref() != Some(&path) {
                self.last_path = Some(path);
                self.last_mtime = Some(mtime);
                self.texture = None; // forces re-render on next ui()
            }
        }
    }

    pub fn force_reload(&mut self) {
        self.last_mtime = None;
        self.texture = None;
        self.reload_if_changed();
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Plot");
        ui.separator();
        // Build/refresh texture if needed.
        if self.texture.is_none() {
            if let Some(path) = &self.last_path {
                if let Some((rgba, w, h)) = rasterize_svg(path, 800, 600) {
                    let img = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
                    self.texture = Some(ui.ctx().load_texture("r2_plot", img, egui::TextureOptions::LINEAR));
                }
            }
        }

        match &self.texture {
            Some(tex) => {
                let size = tex.size_vec2();
                let avail = ui.available_size();
                let scale = (avail.x / size.x).min(avail.y / size.y).min(1.0);
                let display = size * scale;
                ui.add(egui::Image::new((tex.id(), display)));
            }
            None => {
                ui.add_space(40.0);
                ui.vertical_centered(|ui| {
                    ui.label("(no plot yet — type plot(rnorm(50)) below)");
                });
            }
        }
    }
}

/// Read an SVG file and rasterize to RGBA bytes at the requested size.
/// Returns (rgba, width, height) or None on any failure.
fn rasterize_svg(path: &std::path::Path, w: u32, h: u32) -> Option<(Vec<u8>, u32, u32)> {
    let svg = std::fs::read_to_string(path).ok()?;
    let mut opt = usvg::Options::default();
    opt.fontdb_mut().load_system_fonts();
    let tree = usvg::Tree::from_str(&svg, &opt).ok()?;
    let mut pixmap = tiny_skia::Pixmap::new(w, h)?;
    let svg_size = tree.size();
    let sx = w as f32 / svg_size.width();
    let sy = h as f32 / svg_size.height();
    let scale = sx.min(sy);
    let transform = tiny_skia::Transform::from_scale(scale, scale);
    resvg::render(&tree, transform, &mut pixmap.as_mut());
    Some((pixmap.take(), w, h))
}
