//! User preferences — persisted to `%APPDATA%\Ardon-R2\settings.json`
//! on Windows (or `$XDG_CONFIG_HOME/ardon-r2/settings.json` elsewhere).

use eframe::egui;
use std::path::PathBuf;

#[derive(Default, Clone)]
pub struct Settings {
    pub working_directory: Option<String>,
    pub font_scale:        f32,    // 1.0 = default
    pub dark_mode:         bool,
}

impl Settings {
    pub fn load_or_default() -> Self {
        let mut s = Settings { font_scale: 1.0, dark_mode: true, ..Default::default() };
        if let Some(path) = settings_path() {
            if let Ok(text) = std::fs::read_to_string(&path) {
                // Tiny hand-rolled key=value parser to avoid a serde dep.
                for line in text.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') { continue; }
                    if let Some((k, v)) = line.split_once('=') {
                        let k = k.trim();
                        let v = v.trim();
                        match k {
                            "working_directory" => s.working_directory = Some(v.to_string()),
                            "font_scale" => s.font_scale = v.parse().unwrap_or(1.0),
                            "dark_mode" => s.dark_mode = v == "true",
                            _ => {}
                        }
                    }
                }
            }
        }
        s
    }

    pub fn save(&self) {
        if let Some(path) = settings_path() {
            let _ = std::fs::create_dir_all(path.parent().unwrap());
            let mut text = String::new();
            text.push_str("# Ardon-R2 GUI preferences\n");
            if let Some(wd) = &self.working_directory {
                text.push_str(&format!("working_directory = {}\n", wd));
            }
            text.push_str(&format!("font_scale = {}\n", self.font_scale));
            text.push_str(&format!("dark_mode = {}\n", self.dark_mode));
            let _ = std::fs::write(path, text);
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Preferences");
        ui.separator();
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label("Working directory:");
            let mut wd = self.working_directory.clone().unwrap_or_default();
            if ui.text_edit_singleline(&mut wd).changed() {
                self.working_directory = Some(wd);
            }
            if ui.button("Browse…").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    let _ = std::env::set_current_dir(&path);
                    self.working_directory = Some(path.to_string_lossy().to_string());
                }
            }
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("Font scale:");
            ui.add(egui::Slider::new(&mut self.font_scale, 0.7..=2.0).step_by(0.05));
        });
        ui.ctx().set_pixels_per_point(self.font_scale);

        ui.add_space(8.0);
        ui.checkbox(&mut self.dark_mode, "Dark mode");
        if self.dark_mode { ui.ctx().set_visuals(egui::Visuals::dark()); }
        else               { ui.ctx().set_visuals(egui::Visuals::light()); }

        ui.add_space(16.0);
        if ui.button("Save").clicked() {
            self.save();
        }
    }
}

fn settings_path() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        let appdata = std::env::var("APPDATA").ok()?;
        Some(PathBuf::from(appdata).join("Ardon-R2").join("settings.toml"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        let base = std::env::var("XDG_CONFIG_HOME")
            .ok()
            .map(PathBuf::from)
            .or_else(|| std::env::var("HOME").ok().map(|h| PathBuf::from(h).join(".config")))?;
        Some(base.join("ardon-r2").join("settings.toml"))
    }
}
