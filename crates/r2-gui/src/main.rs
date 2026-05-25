// On Windows, suppress the black console window that would otherwise flash
// when R2Gui.exe is launched from Explorer or the Start Menu. Without this
// attribute, Windows attaches a default console to the process — fine for a
// CLI app (r2.exe) but inappropriate for a GUI app. Debug builds keep the
// console so println!/eprintln! still surface during development.
#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

//! Ardon-R2 desktop GUI — egui/eframe-based standalone application.
//!
//! Single-window layout (RGui-inspired):
//!   ┌───────────────────────────────────────────────────────────┐
//!   │ File │ Edit │ Plot │ Packages │ Settings │ Help           │   menu bar
//!   ├───────────────────────────────────────────────────────────┤
//!   │ ┌─────────────────────────────────┐ ┌────────────────────┐│
//!   │ │ Console (REPL output + input)   │ │ Plot pane          ││
//!   │ │                                 │ │                    ││
//!   │ │ R2> rnorm(10)                   │ │  [SVG / PNG render]││
//!   │ │  [1] 0.42 -0.18 ...             │ │                    ││
//!   │ │                                 │ │                    ││
//!   │ │ R2> _                           │ │                    ││
//!   │ └─────────────────────────────────┘ └────────────────────┘│
//!   ├───────────────────────────────────────────────────────────┤
//!   │ cwd: C:\Users\you\Documents │ packages: 3 loaded │ R2 0.2 │
//!   └───────────────────────────────────────────────────────────┘

mod app;
mod console;
mod editor;
mod menubar;
mod plotpane;
mod settings;

use eframe::egui;

fn main() -> Result<(), eframe::Error> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 500.0])
            .with_title("Ardon-R2")
            .with_icon(load_icon()),
        ..Default::default()
    };
    eframe::run_native(
        "Ardon-R2",
        native_options,
        Box::new(|cc| Box::new(app::R2App::new(cc))),
    )
}

/// Try to load the R2 icon for the OS window title bar. Returns an
/// empty IconData if the bundled icon file isn't present so the build
/// never fails on a missing asset.
fn load_icon() -> egui::IconData {
    let bytes = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/logo.png"));
    match image::load_from_memory(bytes) {
        Ok(img) => {
            let rgba = img.into_rgba8();
            let (w, h) = (rgba.width(), rgba.height());
            egui::IconData { rgba: rgba.into_raw(), width: w, height: h }
        }
        Err(_) => egui::IconData::default(),
    }
}
