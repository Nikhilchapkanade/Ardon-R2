//! Menu bar — File / Edit / Run / Plot / Packages / Settings / Help.

use eframe::egui;
use crate::app::R2App;

pub fn draw_menubar(ui: &mut egui::Ui, app: &mut R2App) {
    egui::menu::bar(ui, |ui| {
        // ── File ─────────────────────────────────────────────────────
        ui.menu_button("File", |ui| {
            if ui.button("New script").clicked() {
                if app.editor.dirty {
                    app.console.push_output("(Editor has unsaved changes — Save first or use File ▸ Open to discard.)");
                } else {
                    app.editor.text = String::from("# New R2 script\n\n");
                    app.editor.path = None;
                    app.editor.dirty = false;
                }
                ui.close_menu();
            }
            if ui.button("Open script…    Ctrl+O").clicked() {
                app.open_script_dialog();
                ui.close_menu();
            }
            ui.separator();
            if ui.button("Save           Ctrl+S").clicked() {
                app.save_current_script(false);
                ui.close_menu();
            }
            if ui.button("Save as…   Ctrl+Shift+S").clicked() {
                app.save_current_script(true);
                ui.close_menu();
            }
            ui.separator();
            if ui.button("Quit").clicked() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });

        // ── Edit ─────────────────────────────────────────────────────
        ui.menu_button("Edit", |ui| {
            ui.label(egui::RichText::new("Ctrl+C  Copy").weak());
            ui.label(egui::RichText::new("Ctrl+V  Paste").weak());
            ui.label(egui::RichText::new("Ctrl+X  Cut").weak());
            ui.label(egui::RichText::new("Ctrl+A  Select All").weak());
            ui.label(egui::RichText::new("Ctrl+Z  Undo").weak());
            ui.label(egui::RichText::new("Ctrl+Y  Redo").weak());
            ui.separator();
            if ui.button("Clear console").clicked() {
                app.console.clear();
                ui.close_menu();
            }
        });

        // ── Run ─────────────────────────────────────────────────────
        ui.menu_button("Run", |ui| {
            if ui.button("Run current line  Ctrl+Enter").clicked() {
                // Trigger via a synthetic call.
                let line = last_non_comment_line(&app.editor.text);
                if !line.is_empty() {
                    app.run_source(&line);
                }
                ui.close_menu();
            }
            if ui.button("Run whole script  F5").clicked() {
                let src = app.editor.text.clone();
                app.run_source(&src);
                ui.close_menu();
            }
            ui.separator();
            if ui.button("Source open file…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("R2 scripts", &["r2", "R", "r"])
                    .pick_file()
                {
                    match std::fs::read_to_string(&path) {
                        Ok(src) => app.run_source(&src),
                        Err(e)  => app.console.push_error(&format!("could not read {}: {}", path.display(), e)),
                    }
                }
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
                    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                    app.run_source(&format!("install.packages(\"{}\", path = \"{}\")", name, p));
                }
                ui.close_menu();
            }
            ui.separator();
            ui.label(egui::RichText::new("From GitHub example:").weak());
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
                    app.console.push_output(&format!("Working directory now: {}", path.display()));
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

fn last_non_comment_line(text: &str) -> String {
    for line in text.lines().rev() {
        let t = line.trim();
        if !t.is_empty() && !t.starts_with('#') {
            return line.to_string();
        }
    }
    String::new()
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
