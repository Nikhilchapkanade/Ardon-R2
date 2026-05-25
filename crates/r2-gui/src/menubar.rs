//! Menu bar — File / Edit / Plot / Packages / Settings / Help.
//!
//! Mirrors RGui's layout where it makes sense; trimmed for R2.

use eframe::egui;
use crate::app::R2App;

pub fn draw_menubar(ui: &mut egui::Ui, app: &mut R2App) {
    egui::menu::bar(ui, |ui| {
        // ── File ─────────────────────────────────────────────────────
        ui.menu_button("File", |ui| {
            if ui.button("New script…").clicked() {
                app.console.push_output("(New script: not yet wired — open a .r2 file via File ▸ Open script for now.)\n");
                ui.close_menu();
            }
            if ui.button("Open script…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("R2 scripts", &["r2", "R", "r"])
                    .add_filter("All files", &["*"])
                    .pick_file()
                {
                    match std::fs::read_to_string(&path) {
                        Ok(src) => app.run_source(&src),
                        Err(e)  => app.console.push_error(&format!("could not read {}: {}", path.display(), e)),
                    }
                }
                ui.close_menu();
            }
            ui.separator();
            if ui.button("Save console transcript…").clicked() {
                app.console.push_output("(Save transcript: planned for v0.2.1.)\n");
                ui.close_menu();
            }
            ui.separator();
            if ui.button("Quit").clicked() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });

        // ── Edit ─────────────────────────────────────────────────────
        ui.menu_button("Edit", |ui| {
            // egui handles Ctrl+C / Ctrl+V / Ctrl+X natively when a
            // TextEdit has focus. These menu items are mostly for
            // discoverability.
            ui.label(egui::RichText::new("Ctrl+C  Copy").weak());
            ui.label(egui::RichText::new("Ctrl+V  Paste").weak());
            ui.label(egui::RichText::new("Ctrl+X  Cut").weak());
            ui.label(egui::RichText::new("Ctrl+A  Select All").weak());
            ui.separator();
            if ui.button("Clear console").clicked() {
                app.console.clear();
                ui.close_menu();
            }
        });

        // ── Plot ─────────────────────────────────────────────────────
        ui.menu_button("Plot", |ui| {
            if ui.button("Refresh plot pane").clicked() {
                app.plot.force_reload();
                ui.close_menu();
            }
            if ui.button("Save current plot as PNG…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("PNG image", &["png"])
                    .set_file_name("plot.png")
                    .save_file()
                {
                    let p = path.to_string_lossy().replace('\\', "/");
                    app.run_source(&format!("save.plot(\"{}\", width = 1024, height = 768)", p));
                }
                ui.close_menu();
            }
            if ui.button("Save current plot as SVG…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("SVG vector", &["svg"])
                    .set_file_name("plot.svg")
                    .save_file()
                {
                    let p = path.to_string_lossy().replace('\\', "/");
                    app.run_source(&format!("save.plot(\"{}\")", p));
                }
                ui.close_menu();
            }
        });

        // ── Packages ─────────────────────────────────────────────────
        ui.menu_button("Packages", |ui| {
            if ui.button("List installed").clicked() {
                app.run_source("installed.packages()");
                ui.close_menu();
            }
            if ui.button("Install from local directory…").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    let p = path.to_string_lossy().replace('\\', "/");
                    app.run_source(&format!(
                        "install.packages(basename(\"{0}\"), path = \"{0}\")", p));
                }
                ui.close_menu();
            }
            ui.separator();
            ui.label(egui::RichText::new("From GitHub:").weak());
            ui.label(egui::RichText::new("install.packages(\"r2.survival\",").weak());
            ui.label(egui::RichText::new("  path=\"devendratandle/Ardon-R2-libraries\",").weak());
            ui.label(egui::RichText::new("  subdir=\"r2pkg-survival\")").weak());
        });

        // ── Settings ─────────────────────────────────────────────────
        ui.menu_button("Settings", |ui| {
            if ui.button("Preferences…").clicked() {
                app.show_settings = true;
                ui.close_menu();
            }
            if ui.button("Change working directory…").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    let _ = std::env::set_current_dir(&path);
                    app.settings.working_directory = Some(path.to_string_lossy().to_string());
                    app.settings.save();
                    app.console.push_output(&format!("Working directory now: {}\n", path.display()));
                }
                ui.close_menu();
            }
        });

        // ── Help ─────────────────────────────────────────────────────
        ui.menu_button("Help", |ui| {
            if ui.button("About Ardon-R2").clicked() {
                app.show_about = true;
                ui.close_menu();
            }
            if ui.button("Documentation (GitHub)").clicked() {
                let _ = open_url("https://github.com/devendratandle/Ardon-R2#readme");
                ui.close_menu();
            }
            if ui.button("Function reference (FUNCTIONS.md)").clicked() {
                let _ = open_url("https://github.com/devendratandle/Ardon-R2/blob/main/FUNCTIONS.md");
                ui.close_menu();
            }
            ui.separator();
            if ui.button("Report a bug").clicked() {
                let _ = open_url("https://github.com/devendratandle/Ardon-R2/issues/new");
                ui.close_menu();
            }
        });
    });
}

fn open_url(url: &str) -> std::io::Result<()> {
    #[cfg(target_os = "windows")]
    { std::process::Command::new("cmd").args(["/c", "start", "", url]).spawn()?; }
    #[cfg(target_os = "macos")]
    { std::process::Command::new("open").arg(url).spawn()?; }
    #[cfg(all(unix, not(target_os = "macos")))]
    { std::process::Command::new("xdg-open").arg(url).spawn()?; }
    Ok(())
}
