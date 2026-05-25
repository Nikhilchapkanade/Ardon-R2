//! Main App state and update loop.
//!
//! Layout (RGui / RStudio inspired):
//!
//!   ┌────────────────────────────────────────────────────────┐
//!   │ menu bar                                               │
//!   ├──────────────────────────────────┬─────────────────────┤
//!   │ Script editor                    │ Plot pane           │
//!   │ (top, large, scrollable)         │ (right side)        │
//!   ├──────────────────────────────────┤                     │
//!   │ Console (output only)            │                     │
//!   │ (bottom, smaller, scrollable)    │                     │
//!   ├──────────────────────────────────┴─────────────────────┤
//!   │ status bar                                             │
//!   └────────────────────────────────────────────────────────┘

use eframe::egui;
use r2_engine::Engine;
use r2_parser::Parser;
use r2_types::RVal;

use crate::console::Console;
use crate::editor::Editor;
use crate::menubar::draw_menubar;
use crate::plotpane::PlotPane;
use crate::settings::Settings;

pub struct R2App {
    pub engine: Engine,
    pub editor: Editor,
    pub console: Console,
    pub plot: PlotPane,
    pub settings: Settings,
    pub show_about: bool,
    pub show_settings: bool,
}

impl R2App {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self {
            engine: Engine::new(),
            editor: Editor::new(),
            console: Console::new(),
            plot: PlotPane::new(),
            settings: Settings::load_or_default(),
            show_about: false,
            show_settings: false,
        };
        // cwd init.
        if let Some(home) = &app.settings.working_directory {
            let _ = std::env::set_current_dir(home);
        } else if let Some(home) = pick_user_home() {
            let _ = std::env::set_current_dir(&home);
            app.settings.working_directory = Some(home.to_string_lossy().to_string());
        }
        // Theme.
        // (Actual call happens in ui() so we can use ctx.)
        app.console.push_output("Ardon-R2 0.2.0 — write code in the editor above, results appear here.");
        app.console.push_output(&format!("Working directory: {}",
            app.settings.working_directory.as_deref().unwrap_or("(unset)")));
        app.console.push_output("");
        app
    }

    pub fn run_source(&mut self, source: &str) {
        if source.trim().is_empty() { return; }
        self.console.push_input(source);
        let exprs = match Parser::parse(source) {
            Ok(e) => e,
            Err(e) => {
                self.console.push_error(&format!("Parse error: {}", e));
                return;
            }
        };
        for stmt in &exprs {
            match self.engine.eval(stmt) {
                Ok(val) => {
                    if !matches!(&val, RVal::Null) {
                        self.console.push_output(&format!("{}", val));
                    }
                }
                Err(err) => {
                    self.console.push_error(&format!("Error: {}", err));
                    break;
                }
            }
        }
        for w in self.engine.drain_warnings() {
            self.console.push_error(&format!("Warning: {}", w));
        }
        self.plot.reload_if_changed();
    }

    /// Save the current editor buffer; prompts via dialog if no path is set.
    pub fn save_current_script(&mut self, save_as: bool) {
        let path = if save_as || self.editor.path.is_none() {
            rfd::FileDialog::new()
                .add_filter("R2 scripts", &["r2", "R", "r"])
                .set_file_name("untitled.r2")
                .save_file()
        } else {
            self.editor.path.clone()
        };
        if let Some(p) = path {
            match self.editor.save_to(p.clone()) {
                Ok(_) => self.console.push_output(&format!("Saved: {}", p.display())),
                Err(e) => self.console.push_error(&format!("Save failed: {}", e)),
            }
        }
    }

    pub fn open_script_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("R2 scripts", &["r2", "R", "r"])
            .add_filter("All files", &["*"])
            .pick_file()
        {
            match self.editor.load_from(path.clone()) {
                Ok(_) => self.console.push_output(&format!("Opened: {}", path.display())),
                Err(e) => self.console.push_error(&format!("Open failed: {}", e)),
            }
        }
    }
}

impl eframe::App for R2App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Apply theme on every frame so the toggle in Settings is instant.
        if self.settings.dark_mode {
            ctx.set_visuals(egui::Visuals::dark());
        } else {
            ctx.set_visuals(egui::Visuals::light());
        }

        // Global shortcuts (Ctrl+S, Ctrl+O) that should work anywhere
        // in the app, not just inside the editor widget.
        let mods = ctx.input(|i| i.modifiers);
        if mods.command && ctx.input(|i| i.key_pressed(egui::Key::S)) {
            self.save_current_script(mods.shift);
        }
        if mods.command && ctx.input(|i| i.key_pressed(egui::Key::O)) {
            self.open_script_dialog();
        }

        // ── Menu bar ────────────────────────────────────────────────
        egui::TopBottomPanel::top("menubar").show(ctx, |ui| {
            draw_menubar(ui, self);
        });

        // ── Status bar ──────────────────────────────────────────────
        egui::TopBottomPanel::bottom("statusbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let cwd = self.settings.working_directory.as_deref().unwrap_or("(unset)");
                ui.label(format!("cwd: {}", cwd));
                ui.separator();
                ui.label(format!("{} lines", self.editor.line_count()));
                ui.separator();
                ui.label(if self.editor.dirty { "unsaved" } else { "saved" });
                ui.separator();
                ui.label("R2 0.2.0");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label("AGPL-3.0  |  github.com/devendratandle/Ardon-R2");
                });
            });
        });

        // ── Plot pane (right side) ──────────────────────────────────
        egui::SidePanel::right("plot_pane")
            .resizable(true)
            .default_width(520.0)
            .min_width(280.0)
            .show(ctx, |ui| {
                self.plot.ui(ui);
            });

        // ── Console at the bottom of the left column ────────────────
        egui::TopBottomPanel::bottom("console_pane")
            .resizable(true)
            .default_height(220.0)
            .min_height(80.0)
            .show(ctx, |ui| {
                self.console.ui(ui);
            });

        // ── Script Editor fills the remaining space ─────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            self.editor.ui(ui);
        });

        // Pick up anything the editor wants to run.
        if let Some(src) = self.editor.take_pending_run() {
            self.run_source(&src);
        }

        // Modal dialogs.
        if self.show_about {
            egui::Window::new("About Ardon-R2")
                .open(&mut self.show_about)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.heading("Ardon-R2");
                        ui.label("Version 0.2.0  (2026)");
                        ui.add_space(8.0);
                        ui.label("Statistical Computing, Reimagined");
                        ui.label("A pure-Rust reimplementation of R.");
                        ui.add_space(8.0);
                        ui.hyperlink_to("Project home", "https://github.com/devendratandle/Ardon-R2");
                        ui.hyperlink_to("Addon libraries", "https://github.com/devendratandle/Ardon-R2-libraries");
                        ui.add_space(8.0);
                        ui.label("© 2026 Devendra Tandale · AGPL-3.0");
                    });
                });
        }
        if self.show_settings {
            let mut open = self.show_settings;
            egui::Window::new("Settings")
                .open(&mut open)
                .collapsible(false)
                .resizable(true)
                .show(ctx, |ui| {
                    self.settings.ui(ui);
                });
            self.show_settings = open;
        }
    }
}

fn pick_user_home() -> Option<std::path::PathBuf> {
    if let Ok(custom) = std::env::var("R2_HOME") {
        let p = std::path::PathBuf::from(custom);
        if p.is_dir() { return Some(p); }
    }
    if let Ok(od) = std::env::var("OneDrive") {
        let p = std::path::PathBuf::from(&od).join("Documents");
        if p.is_dir() { return Some(p); }
    }
    if let Ok(user) = std::env::var("USERPROFILE") {
        let od = std::path::PathBuf::from(&user).join("OneDrive").join("Documents");
        if od.is_dir() { return Some(od); }
        let docs = std::path::PathBuf::from(user).join("Documents");
        if docs.is_dir() { return Some(docs); }
    }
    if let Ok(home) = std::env::var("HOME") {
        let docs = std::path::PathBuf::from(&home).join("Documents");
        if docs.is_dir() { return Some(docs); }
        let h = std::path::PathBuf::from(home);
        if h.is_dir() { return Some(h); }
    }
    None
}
