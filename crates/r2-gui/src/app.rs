//! Main App state and update loop.

use eframe::egui;
use r2_engine::Engine;
use r2_parser::Parser;
use r2_types::RVal;

use crate::console::Console;
use crate::menubar::draw_menubar;
use crate::plotpane::PlotPane;
use crate::settings::Settings;

pub struct R2App {
    pub engine: Engine,
    pub console: Console,
    pub plot: PlotPane,
    pub settings: Settings,
    /// "About" dialog open?
    pub show_about: bool,
    /// Settings window open?
    pub show_settings: bool,
}

impl R2App {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self {
            engine: Engine::new(),
            console: Console::new(),
            plot: PlotPane::new(),
            settings: Settings::load_or_default(),
            show_about: false,
            show_settings: false,
        };
        // Set initial cwd from settings.
        if let Some(home) = &app.settings.working_directory {
            let _ = std::env::set_current_dir(home);
        } else if let Some(home) = pick_user_home() {
            let _ = std::env::set_current_dir(&home);
            app.settings.working_directory = Some(home.to_string_lossy().to_string());
        }
        // Welcome banner.
        app.console.push_output("Ardon-R2 — Statistical Computing, Reimagined\n");
        app.console.push_output("Version 0.2.0 (2026) | Inspired by R. Built on Rust.\n");
        app.console.push_output("File ▸ Open script • Type code below • Enter to evaluate.\n");
        app.console.push_output(&format!("Working directory: {}\n\n",
            app.settings.working_directory.as_deref().unwrap_or("(unset)")));
        app
    }

    /// Run a chunk of R2 source. Captures stdout via the existing engine
    /// and appends both the input and the output to the console.
    pub fn run_source(&mut self, source: &str) {
        self.console.push_input(source);
        let exprs = match Parser::parse(source) {
            Ok(e) => e,
            Err(e) => {
                self.console.push_error(&format!("Parse error: {}\n", e));
                return;
            }
        };
        for stmt in &exprs {
            match self.engine.eval(stmt) {
                Ok(val) => {
                    if !matches!(&val, RVal::Null) {
                        self.console.push_output(&format!("{}\n", val));
                    }
                }
                Err(err) => {
                    self.console.push_error(&format!("Error: {}\n", err));
                    break;
                }
            }
        }
        for w in self.engine.drain_warnings() {
            self.console.push_error(&format!("Warning: {}\n", w));
        }

        // After every evaluation, refresh the plot pane in case a plot
        // function wrote to disk.
        self.plot.reload_if_changed();
    }
}

impl eframe::App for R2App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Menu bar at the top.
        egui::TopBottomPanel::top("menubar").show(ctx, |ui| {
            draw_menubar(ui, self);
        });

        // Status bar at the bottom.
        egui::TopBottomPanel::bottom("statusbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let cwd = self.settings.working_directory.as_deref().unwrap_or("(unset)");
                ui.label(format!("cwd: {}", cwd));
                ui.separator();
                ui.label("R2 0.2.0");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label("AGPL-3.0  |  github.com/devendratandle/Ardon-R2");
                });
            });
        });

        // Plot pane on the right, console on the left.
        egui::SidePanel::right("plot_pane")
            .resizable(true)
            .default_width(560.0)
            .show(ctx, |ui| {
                self.plot.ui(ui);
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            self.console.ui(ui, |_source| { /* deferred — see below */ });
            // Pull the pending source out of the console after the immediate-mode
            // UI run so we can mutate self.engine without a borrow conflict.
            if let Some(src) = self.console.take_pending_submit() {
                self.run_source(&src);
            }
        });

        // Optional modal dialogs.
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

/// Same resolution as r2-repl: OneDrive Documents first, then plain
/// Documents. Duplicated here to keep r2-gui's dep list narrow.
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
