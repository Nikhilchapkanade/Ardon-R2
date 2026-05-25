//! REPL console widget — scrollable transcript above, one-line input
//! at the bottom. Submits on Enter (Shift+Enter inserts a newline for
//! multi-line expressions).

use eframe::egui;

/// One transcript line, tagged so we can render in distinct colors.
#[derive(Clone)]
enum Line {
    /// User input echoed with a "R2> " prefix.
    Input(String),
    /// Stdout / value print.
    Output(String),
    /// Errors and warnings.
    Error(String),
}

pub struct Console {
    transcript: Vec<Line>,
    /// In-progress input. Multi-line via Shift+Enter; submitted on plain Enter.
    input: String,
    /// Bumped each frame the transcript grows so the ScrollArea can pin to the bottom.
    auto_scroll: bool,
    /// History of submitted commands (most recent last).
    history: Vec<String>,
    /// Index into history when stepping via Up/Down. None = not browsing.
    history_cursor: Option<usize>,
    /// Set in `ui()` when the user pressed Enter — consumed once by App.
    pending_submit: Option<String>,
}

impl Console {
    pub fn new() -> Self {
        Self {
            transcript: Vec::new(),
            input: String::new(),
            auto_scroll: true,
            history: Vec::new(),
            history_cursor: None,
            pending_submit: None,
        }
    }

    pub fn push_input(&mut self, s: &str) { self.transcript.push(Line::Input(s.trim_end().to_string())); self.auto_scroll = true; }
    pub fn push_output(&mut self, s: &str) { self.transcript.push(Line::Output(s.trim_end().to_string())); self.auto_scroll = true; }
    pub fn push_error(&mut self, s: &str) { self.transcript.push(Line::Error(s.trim_end().to_string())); self.auto_scroll = true; }

    pub fn clear(&mut self) {
        self.transcript.clear();
        self.auto_scroll = true;
    }

    /// Called by the App after `ui()` to retrieve the line that should
    /// be evaluated. Returns Some(source) once per Enter press.
    pub fn take_pending_submit(&mut self) -> Option<String> {
        self.pending_submit.take()
    }

    pub fn ui<F: FnOnce(&str)>(&mut self, ui: &mut egui::Ui, _on_submit: F) {
        // Transcript area.
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .max_height(ui.available_height() - 80.0)
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

        ui.add_space(4.0);
        ui.separator();

        // Input area.
        ui.horizontal(|ui| {
            ui.label("R2>");
            let input_id = egui::Id::new("repl_input");
            let response = ui.add(
                egui::TextEdit::multiline(&mut self.input)
                    .desired_rows(2)
                    .desired_width(f32::INFINITY)
                    .id(input_id)
                    .font(egui::TextStyle::Monospace),
            );

            // Enter submits, Shift+Enter inserts newline.
            // egui's multiline TextEdit treats Enter as newline by default,
            // so we intercept the key BEFORE the widget consumes it.
            let mods = ui.input(|i| i.modifiers);
            let enter_pressed = ui.input(|i| i.key_pressed(egui::Key::Enter));
            if response.has_focus() && enter_pressed && !mods.shift {
                // Strip the trailing newline egui inserted before we caught it.
                if self.input.ends_with('\n') {
                    self.input.pop();
                }
                let src = std::mem::take(&mut self.input);
                if !src.trim().is_empty() {
                    self.history.push(src.clone());
                    self.history_cursor = None;
                    self.pending_submit = Some(src);
                }
                self.auto_scroll = true;
                // Re-focus the input for the next command.
                ui.memory_mut(|m| m.request_focus(input_id));
            }

            // Up/Down arrow history navigation when the cursor is at start/end.
            if response.has_focus() {
                if ui.input(|i| i.key_pressed(egui::Key::ArrowUp)) && !self.history.is_empty() {
                    let next = match self.history_cursor {
                        None => self.history.len() - 1,
                        Some(0) => 0,
                        Some(i) => i - 1,
                    };
                    self.history_cursor = Some(next);
                    self.input = self.history[next].clone();
                }
                if ui.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                    if let Some(i) = self.history_cursor {
                        if i + 1 < self.history.len() {
                            self.history_cursor = Some(i + 1);
                            self.input = self.history[i + 1].clone();
                        } else {
                            self.history_cursor = None;
                            self.input.clear();
                        }
                    }
                }
            }
        });
    }
}
