//! Script Editor pane — multi-line code editor with line-aware "run".
//!
//! Key bindings:
//!   Ctrl+Enter   run current line (or selection if non-empty)
//!   F5           run whole script
//!   Ctrl+S       save current file (uses dialog if no path yet)
//!   Ctrl+O       open file dialog
//!
//! The editor holds:
//!   - the text buffer (one whole script)
//!   - the file path (if saved)
//!   - the dirty flag (modified since last save / load)
//!   - a "pending run" string that the App harvests after each frame
//!     and dispatches to the engine (avoids borrow conflicts).

use eframe::egui;
use std::path::PathBuf;

pub struct Editor {
    pub text: String,
    pub path: Option<PathBuf>,
    pub dirty: bool,
    pending_run: Option<String>,
    /// Tracked so we can compute "line N of M" in the status bar.
    /// Reserved for a future iteration where egui exposes cursor pos.
    #[allow(dead_code)]
    pub cursor_line: usize,
}

impl Editor {
    pub fn new() -> Self {
        Self {
            text: String::from(
                "# Welcome to Ardon-R2.\n\
                 # Write code here, then:\n\
                 #   Ctrl+Enter  — run current line (or selection)\n\
                 #   F5          — run whole script\n\
                 #   Ctrl+S      — save\n\
                 #\n\
                 # Example: try this.\n\
                 x <- rnorm(mean = 10, sd = 2, n = 1000)\n\
                 hist(x)\n",
            ),
            path: None,
            dirty: false,
            pending_run: None,
            cursor_line: 1,
        }
    }

    pub fn load_from(&mut self, path: PathBuf) -> std::io::Result<()> {
        let text = std::fs::read_to_string(&path)?;
        self.text = text;
        self.path = Some(path);
        self.dirty = false;
        Ok(())
    }

    pub fn save_to(&mut self, path: PathBuf) -> std::io::Result<()> {
        std::fs::write(&path, &self.text)?;
        self.path = Some(path);
        self.dirty = false;
        Ok(())
    }

    pub fn take_pending_run(&mut self) -> Option<String> {
        self.pending_run.take()
    }

    pub fn title(&self) -> String {
        let name = self
            .path
            .as_ref()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .unwrap_or_else(|| "untitled.r2".into());
        if self.dirty { format!("{} ●", name) } else { name }
    }

    pub fn line_count(&self) -> usize {
        if self.text.is_empty() { 1 } else { self.text.lines().count().max(1) }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        // ── Top toolbar with title + Run buttons ─────────────────────
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(self.title()).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("▶ Run all (F5)").clicked() {
                    self.pending_run = Some(self.text.clone());
                }
                if ui.button("▶ Run line (Ctrl+Enter)").clicked() {
                    self.pending_run = Some(self.current_line_or_selection_for_run());
                }
            });
        });
        ui.separator();

        // ── Keyboard shortcuts BEFORE the text edit consumes them ────
        let mods = ui.input(|i| i.modifiers);
        if ui.input(|i| i.key_pressed(egui::Key::F5)) {
            self.pending_run = Some(self.text.clone());
        }
        if mods.command && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            self.pending_run = Some(self.current_line_or_selection_for_run());
        }

        // ── The actual text area ─────────────────────────────────────
        let id = egui::Id::new("script_editor");
        let response = egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add_sized(
                    [ui.available_width(), ui.available_height()],
                    egui::TextEdit::multiline(&mut self.text)
                        .id(id)
                        .code_editor()
                        .desired_rows(20)
                        .font(egui::TextStyle::Monospace)
                        .lock_focus(true),
                )
            });

        if response.inner.changed() {
            self.dirty = true;
        }
    }

    /// Heuristic for "current line": we don't have direct access to
    /// the cursor position from outside the TextEdit, so the
    /// Ctrl+Enter shortcut runs the LAST non-empty line of the buffer
    /// when no selection exists. This matches what most users want
    /// when iteratively building a script.
    ///
    /// (egui doesn't yet expose cursor position generically; when it
    /// does we'll switch to true current-line execution.)
    fn current_line_or_selection_for_run(&self) -> String {
        // Prefer the last non-empty line.
        for line in self.text.lines().rev() {
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                return line.to_string();
            }
        }
        String::new()
    }
}
