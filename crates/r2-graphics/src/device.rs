//! In-memory plot device for the graphics crate.
//!
//! Replaces the previous file-based plot-state model (which detected
//! "is a plot open" by reading `plot.svg` from cwd). The file model
//! caused test-isolation races on Windows when `cargo test` runs
//! sibling tests in parallel — one test's `plot.svg` would leak into
//! another test's "no plot open" precondition.
//!
//! The new model:
//!   1. A `PlotDevice` holds the SVG body, the canvas size, the
//!      `PlotParams` (everything `par()` can set), and a panel cursor
//!      for multi-panel `mfrow`/`mfcol` layouts.
//!   2. The device lives in `thread_local!` storage so concurrent
//!      tests do not collide, and the production REPL still has a
//!      single per-thread device.
//!   3. Plot functions call `begin_plot()` to obtain the rectangle
//!      they should draw into (respecting multi-panel layout) and
//!      `append_svg()` to write fragments. Overlays
//!      (`lines`/`points`/`abline`/`legend`) use `append_svg()`
//!      directly; the function errors if no plot is open.
//!   4. The full SVG is materialized on demand via `full_svg()` and
//!      flushed to disk by `save_to_file()` — either auto-saved by
//!      the plot function to preserve existing UX, or explicitly via
//!      `dev.off()` / `save_plot()`.

use std::cell::RefCell;

use r2_types::{ErrKind, R2Err};

/// Everything `par()` can set. Defaults mirror R's `par()` baseline so
/// scripts that do not call `par()` get R-compatible output.
#[derive(Debug, Clone)]
pub struct PlotParams {
    /// Multi-panel grid filled row-by-row. Mutually exclusive with `mfcol`.
    pub mfrow: Option<(u32, u32)>,
    /// Multi-panel grid filled column-by-column. Mutually exclusive with `mfrow`.
    pub mfcol: Option<(u32, u32)>,

    /// Inner margins in "lines" (bottom, left, top, right). R default `5.1, 4.1, 4.1, 2.1`.
    pub mar: [f64; 4],
    /// Outer margins. R default all zero.
    pub oma: [f64; 4],

    /// Text scale. R default 1.0.
    pub cex: f64,
    pub cex_axis: f64,
    pub cex_lab: f64,
    pub cex_main: f64,

    pub col: String,
    pub bg:  String,
    pub fg:  String,

    pub lty: String,
    pub lwd: f64,
    pub pch: i32,
    pub las: i32,

    /// If true, the next `plot()` overlays on the current panel instead of advancing.
    pub new: bool,
}

impl Default for PlotParams {
    fn default() -> Self {
        Self {
            mfrow: None,
            mfcol: None,
            mar: [5.1, 4.1, 4.1, 2.1],
            oma: [0.0; 4],
            cex: 1.0,
            cex_axis: 1.0,
            cex_lab: 1.0,
            cex_main: 1.2,
            col: "black".into(),
            bg:  "white".into(),
            fg:  "black".into(),
            lty: "solid".into(),
            lwd: 1.0,
            pch: 1,
            las: 0,
            new: false,
        }
    }
}

/// In-memory canvas. Holds the accumulated SVG body and the panel cursor.
#[derive(Debug, Clone)]
pub struct PlotDevice {
    /// Concatenated SVG fragments — placed between `<svg ...>` and `</svg>` at render time.
    pub svg_body: String,
    pub has_plot: bool,
    pub width: f64,
    pub height: f64,
    pub params: PlotParams,
    /// Index of the next panel to fill (0-indexed, wraps on `mfrow`/`mfcol` overflow).
    pub panel_cursor: u32,
}

impl Default for PlotDevice {
    fn default() -> Self {
        Self {
            svg_body: String::new(),
            has_plot: false,
            width: 600.0,
            height: 400.0,
            params: PlotParams::default(),
            panel_cursor: 0,
        }
    }
}

impl PlotDevice {
    /// Compute the rectangle the next plot should draw into.
    /// Returns `(x, y, panel_width, panel_height)` in canvas coordinates,
    /// and advances `panel_cursor` for the subsequent plot call.
    pub fn next_panel_rect(&mut self) -> (f64, f64, f64, f64) {
        let (rows, cols, col_major) = match (self.params.mfrow, self.params.mfcol) {
            (Some((r, c)), _) => (r as usize, c as usize, false),
            (_, Some((r, c))) => (r as usize, c as usize, true),
            (None, None) => return (0.0, 0.0, self.width, self.height),
        };
        let total = (rows * cols).max(1);
        let idx = (self.panel_cursor as usize) % total;
        let (row, col) = if col_major {
            (idx % rows, idx / rows)
        } else {
            (idx / cols, idx % cols)
        };
        let pw = self.width / cols as f64;
        let ph = self.height / rows as f64;
        let x = col as f64 * pw;
        let y = row as f64 * ph;
        self.panel_cursor = self.panel_cursor.wrapping_add(1);
        (x, y, pw, ph)
    }

    /// Materialize the full SVG document.
    pub fn full_svg(&self) -> String {
        let mut s = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
            self.width, self.height, self.width, self.height
        );
        s.push_str(&format!(r#"<rect width="100%" height="100%" fill="{}"/>"#, self.params.bg));
        s.push_str(&self.svg_body);
        s.push_str("</svg>");
        s
    }
}

thread_local! {
    pub(crate) static DEVICE: RefCell<PlotDevice> = RefCell::new(PlotDevice::default());
}

// ─── Public access surface (used by plots.rs, overlays.rs, params.rs) ──

pub fn with_device<R, F: FnOnce(&mut PlotDevice) -> R>(f: F) -> R {
    DEVICE.with(|d| f(&mut d.borrow_mut()))
}

/// Has any plot been opened in this device? Source of truth for overlay
/// preconditions — replaces the previous file-existence check.
pub fn current_has_plot() -> bool {
    DEVICE.with(|d| d.borrow().has_plot)
}

/// Begin a new plot. Returns the canvas-coordinate rectangle the plot
/// should draw into. Honors `par(mfrow=...)` / `par(mfcol=...)` multi-panel
/// layout: when the panel cursor is at 0 (or no multi-panel is set), the
/// SVG body is cleared. When in the middle of a panel cycle, the previous
/// panels' content is preserved and the new plot is placed in the next slot.
pub fn begin_plot() -> (f64, f64, f64, f64) {
    DEVICE.with(|d| {
        let mut dev = d.borrow_mut();
        let multipanel = dev.params.mfrow.is_some() || dev.params.mfcol.is_some();

        if !multipanel {
            dev.svg_body.clear();
        } else if dev.panel_cursor == 0 {
            dev.svg_body.clear();
        }
        dev.has_plot = true;
        dev.next_panel_rect()
    })
}

/// Append a raw SVG fragment to the device. Errors if no plot is open.
/// Used by overlay builtins (`lines`, `points`, `abline`, `legend`).
pub fn append_svg(fragment: &str) -> Result<(), R2Err> {
    DEVICE.with(|d| {
        let mut dev = d.borrow_mut();
        if !dev.has_plot {
            return Err(R2Err {
                msg: "no plot open — call plot() first".into(),
                kind: ErrKind::Runtime,
            });
        }
        dev.svg_body.push_str(fragment);
        Ok(())
    })
}

/// Flush the current device contents to a file. Does not modify device state.
pub fn save_to_file(path: &str) -> Result<(), std::io::Error> {
    let svg = DEVICE.with(|d| d.borrow().full_svg());
    std::fs::write(path, svg)
}

/// `dev.off()` equivalent — close the device and reset to default.
pub fn dev_off() {
    DEVICE.with(|d| *d.borrow_mut() = PlotDevice::default());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_device_has_no_plot() {
        // Tests share thread-local state — explicitly reset first.
        dev_off();
        assert!(!current_has_plot());
    }

    #[test]
    fn begin_plot_sets_has_plot_true_and_returns_full_canvas_by_default() {
        dev_off();
        let (x, y, w, h) = begin_plot();
        assert!(current_has_plot());
        assert_eq!((x, y), (0.0, 0.0));
        assert!(w > 0.0 && h > 0.0);
    }

    #[test]
    fn append_errors_when_no_plot_open() {
        dev_off();
        let r = append_svg("<circle cx=\"1\" cy=\"2\" r=\"3\"/>");
        assert!(r.is_err());
    }

    #[test]
    fn mfrow_2x2_advances_through_four_panels_then_wraps() {
        dev_off();
        with_device(|d| d.params.mfrow = Some((2, 2)));
        let r0 = begin_plot();
        let r1 = begin_plot();
        let r2 = begin_plot();
        let r3 = begin_plot();
        let r4 = begin_plot();
        // Row-major fill: (0,0), (0,c), (r,0), (r,c), then back to (0,0).
        assert_eq!((r0.0, r0.1), (0.0, 0.0));
        assert_eq!((r1.0, r1.1), (300.0, 0.0));
        assert_eq!((r2.0, r2.1), (0.0, 200.0));
        assert_eq!((r3.0, r3.1), (300.0, 200.0));
        assert_eq!((r4.0, r4.1), (0.0, 0.0));
        dev_off();
    }
}
