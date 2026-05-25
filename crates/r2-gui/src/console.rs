//! Console pane — output transcript only. Input now happens in the
//! Script Editor pane (run via Ctrl+Enter / F5). This separation
//! matches RGui / RStudio convention: write code in the editor,
//! see results in the console.

use eframe::egui;

#[derive(Clone)]
enum Line {
    /// User's code echoed back with a "R2> " prefix.
    Input(String),
    /// Engine output.
    Output(String),
    /// Errors and warnings.
    Error(String),
}

pub struct Console {
    transcript: Vec<Line>,
    auto_scroll: bool,
}

impl Console {
    pub fn new() -> Self {
        Self {
            transcript: Vec::new(),
            auto_scroll: true,
        }
    }

    pub fn push_input(&mut self, s: &str) {
        // Echo each non-empty line with the prompt so multi-line scripts
        // read naturally in the transcript.
        for line in s.lines() {
            let line = line.trim_end();
            if line.is_empty() { continue; }
            self.transcript.push(Line::Input(line.to_string()));
        }
        self.auto_scroll = true;
    }
    pub fn push_output(&mut self, s: &str) { self.transcript.push(Line::Output(s.trim_end().to_string())); self.auto_scroll = true; }
    pub fn push_error(&mut self, s: &str)  { self.transcript.push(Line::Error(s.trim_end().to_string()));  self.auto_scroll = true; }

    pub fn clear(&mut self) {
        self.transcript.clear();
        self.auto_scroll = true;
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Console").strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("🗑 Clear").clicked() {
                    self.clear();
                }
            });
        });
        ui.separator();
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(self.auto_scroll)
            .show(ui, |ui| {
                ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
                    for line in &self.transcript {
                        match line {
                            Line::Input(s)  => ui.colored_label(egui::Color32::from_rgb(80, 140, 200), format!("R2> {}", s)),
                            Line::Output(s) => ui.monospace(s),
                            Line::Error(s)  => ui.colored_label(egui::Color32::from_rgb(220, 80, 80), s),
                        };
                    }
                });
            });
    }
}
