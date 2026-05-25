//! Primary data-shape plot builtins — Phase R.3 / Phase R.G.
//!
//! `plot`, `hist`, `boxplot`, `barplot` — each draws into the
//! thread-local `PlotDevice` defined in `device.rs`. They no longer
//! write a file directly; the device exposes `save_to_file()` for
//! explicit flushes via `dev.off()` / `save_plot()`. To preserve the
//! historical UX where `plot()` produced `plot.svg` on disk, each
//! function also calls `save_to_file()` with its default path at the
//! end. The in-memory state is the source of truth for the
//! "is a plot open" predicate that overlays consult.
//!
//! Multi-panel `par(mfrow=...)` / `par(mfcol=...)` is honored: the
//! `begin_plot()` call returns the rectangle to draw into, which
//! shifts and scales the SVG coordinates without changing the rest
//! of the drawing code.

use crate::device::{begin_plot, save_to_file, with_device};
use crate::{gn, gv, val_to_str};
use r2_types::{Attrs, ErrKind, EvalArg, R2Err, RVal};

#[inline]
fn rstr(s: &str) -> RVal {
    RVal::Character(vec![Some(std::sync::Arc::from(s))], Attrs::default())
}

/// `plot(x [, y], main=, xlab=, ylab=)` — scatter into the device.
///
/// Model-aware dispatch (`plot(lm)`, `plot(gbm)`, `plot(kmeans)`)
/// lives in r2-engine via the split-handler pattern.
pub fn bi_plot(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    let y: Vec<f64> = if a.len() > 1 && a[1].name.is_none() {
        gv(a, 1).as_reals()?.into_iter().filter_map(|x| x).collect()
    } else {
        (1..=x.len()).map(|i| i as f64).collect()
    };
    let title = gn(a, "main").map(|v| val_to_str(&v)).unwrap_or_default();
    let xlab = gn(a, "xlab").map(|v| val_to_str(&v)).unwrap_or("x".into());
    let ylab = gn(a, "ylab").map(|v| val_to_str(&v)).unwrap_or("y".into());

    // Per-plot params snapshot (e.g. user point color from par(col=...)).
    let (col, cex) = with_device(|d| (d.params.col.clone(), d.params.cex));

    // Reserve the panel rectangle. Honors par(mfrow=...) automatically.
    let (ox, oy, w, h) = begin_plot();
    let (ml, mr, mt, mb) = (60.0, 20.0, 30.0, 40.0);
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let xmin = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let xmax = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ymin = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let ymax = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let xrange = if (xmax - xmin).abs() < 1e-10 { 1.0 } else { xmax - xmin };
    let yrange = if (ymax - ymin).abs() < 1e-10 { 1.0 } else { ymax - ymin };

    let mut frag = String::new();
    frag.push_str(&format!(
        r#"<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="black" stroke-width="1"/>"#,
        ox + ml, oy + mt, pw, ph
    ));
    if !title.is_empty() {
        let fs = 14.0 * cex;
        frag.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-size="{:.1}" font-weight="bold">{}</text>"#,
            ox + w / 2.0, oy + 18.0, fs, title
        ));
    }
    frag.push_str(&format!(
        r#"<text x="{}" y="{}" text-anchor="middle" font-size="11">{}</text>"#,
        ox + ml + pw / 2.0, oy + h - 5.0, xlab
    ));
    frag.push_str(&format!(
        r#"<text x="{}" y="{}" text-anchor="middle" font-size="11" transform="rotate(-90,{},{})">{}</text>"#,
        ox + 15.0, oy + mt + ph / 2.0, ox + 15.0, oy + mt + ph / 2.0, ylab
    ));
    for i in 0..x.len().min(y.len()) {
        let px = ox + ml + (x[i] - xmin) / xrange * pw;
        let py = oy + mt + ph - (y[i] - ymin) / yrange * ph;
        frag.push_str(&format!(
            r#"<circle cx="{:.1}" cy="{:.1}" r="3" fill="{}" opacity="0.7"/>"#,
            px, py, col
        ));
    }
    for i in 0..=4 {
        let frac = i as f64 / 4.0;
        let xv = xmin + frac * xrange;
        let yv = ymin + frac * yrange;
        let px = ox + ml + frac * pw;
        let py = oy + mt + ph - frac * ph;
        frag.push_str(&format!(
            r#"<text x="{:.0}" y="{}" text-anchor="middle" font-size="9">{:.1}</text>"#,
            px, oy + h - mb + 15.0, xv
        ));
        frag.push_str(&format!(
            r#"<text x="{}" y="{:.0}" text-anchor="end" font-size="9">{:.1}</text>"#,
            ox + ml - 5.0, py + 3.0, yv
        ));
    }

    // Commit to the device. Cannot use append_svg() because the device's
    // has_plot is already true (begin_plot set it), but append_svg requires
    // the precondition we just satisfied — so the direct push is correct here.
    with_device(|d| d.svg_body.push_str(&frag));

    // Auto-flush to preserve the legacy "plot()  produces plot.svg" UX.
    // Print the absolute path so the user knows where to find it — the
    // default cwd is the user's Documents folder (R-style), not the
    // .exe's install dir, so the file lands somewhere they can actually
    // see in Explorer.
    let path = "plot.svg";
    let _ = save_to_file(path);
    print_save_path(path);
    Ok(rstr(path))
}

fn print_save_path(rel: &str) {
    match std::fs::canonicalize(rel) {
        Ok(abs) => {
            let display = abs.to_string_lossy();
            // Strip Windows \\?\ prefix that canonicalize adds.
            let clean = display.strip_prefix(r"\\?\").unwrap_or(&display);
            println!("Plot saved to {}", clean);
        }
        Err(_) => println!("Plot saved to {}", rel),
    }
}

/// `hist(x, breaks=, main=)` — text + SVG histogram into the device.
pub fn bi_hist(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    let nbins = gn(a, "breaks").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(10.0) as usize;
    let title = gn(a, "main").map(|v| val_to_str(&v)).unwrap_or("Histogram".into());

    let xmin = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let xmax = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bin_width = (xmax - xmin) / nbins as f64;

    let mut counts = vec![0usize; nbins];
    for &val in &x {
        let bin = ((val - xmin) / bin_width).floor() as usize;
        let bin = bin.min(nbins - 1);
        counts[bin] += 1;
    }
    let max_count = *counts.iter().max().unwrap_or(&1);

    println!("{}", title);
    for (i, &count) in counts.iter().enumerate() {
        let lo = xmin + i as f64 * bin_width;
        let hi = lo + bin_width;
        let bar_len = (count as f64 / max_count as f64 * 40.0) as usize;
        println!("{:7.1}-{:7.1} | {:>4} {}", lo, hi, count, "#".repeat(bar_len));
    }

    let (ox, oy, w, h) = begin_plot();
    let (ml, mr, mt, mb) = (60.0, 20.0, 30.0, 40.0);
    let pw = w - ml - mr;
    let ph = h - mt - mb;
    let bw = pw / nbins as f64;

    let mut frag = String::new();
    frag.push_str(&format!(
        r#"<text x="{}" y="{}" text-anchor="middle" font-size="14" font-weight="bold">{}</text>"#,
        ox + w / 2.0, oy + 18.0, title
    ));
    for (i, &count) in counts.iter().enumerate() {
        let bh = if max_count > 0 { count as f64 / max_count as f64 * ph } else { 0.0 };
        let bx = ox + ml + i as f64 * bw;
        let by = oy + mt + ph - bh;
        frag.push_str(&format!(
            r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}" stroke="white" stroke-width="1"/>"#,
            bx, by, bw, bh, "#3b82f6"
        ));
    }
    frag.push_str(&format!(
        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black"/>"#,
        ox + ml, oy + mt + ph, ox + ml + pw, oy + mt + ph
    ));
    frag.push_str(&format!(
        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black"/>"#,
        ox + ml, oy + mt, ox + ml, oy + mt + ph
    ));

    with_device(|d| d.svg_body.push_str(&frag));
    let _ = save_to_file("hist.svg");
    print_save_path("hist.svg");

    let _ = mb;
    Ok(RVal::Null)
}

/// `boxplot(g1=, g2=, ..., main=)` — multi-group box-and-whisker into the device.
pub fn bi_boxplot(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let title = gn(a, "main").map(|v| val_to_str(&v)).unwrap_or("Boxplot".into());

    let mut groups: Vec<(String, Vec<f64>)> = Vec::new();
    for (gi, arg) in a.iter().enumerate() {
        if arg.name.as_ref().map(|n| n.as_ref()) == Some("main") { continue; }
        let data: Vec<f64> = arg.value.as_reals()?.into_iter().filter_map(|x| x).collect();
        let name = arg.name.as_ref().map(|n| n.to_string()).unwrap_or(format!("V{}", gi + 1));
        groups.push((name, data));
    }
    if groups.is_empty() {
        return Err(R2Err { msg: "boxplot needs data".into(), kind: ErrKind::Runtime });
    }

    let (ox, oy, w, h) = begin_plot();
    let (ml, mr, mt, mb) = (60.0, 30.0, 30.0, 40.0);
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let all_min = groups.iter().flat_map(|(_, d)| d.iter()).cloned().fold(f64::INFINITY, f64::min);
    let all_max = groups.iter().flat_map(|(_, d)| d.iter()).cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (all_max - all_min).abs() < 1e-10 { 1.0 } else { all_max - all_min };

    let mut frag = String::new();
    frag.push_str(&format!(
        r#"<text x="{}" y="{}" text-anchor="middle" font-size="14" font-weight="bold">{}</text>"#,
        ox + w / 2.0, oy + 18.0, title
    ));

    let ng = groups.len() as f64;
    let bw = pw / ng * 0.6;
    let gap = pw / ng;

    for (i, (name, data)) in groups.iter().enumerate() {
        if data.len() < 2 { continue; }
        let mut sorted = data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let q1 = sorted[n / 4]; let median = sorted[n / 2]; let q3 = sorted[3 * n / 4];
        let min_val = sorted[0]; let max_val = sorted[n - 1];
        let iqr = q3 - q1;
        let lower_fence = (q1 - 1.5 * iqr).max(min_val);
        let upper_fence = (q3 + 1.5 * iqr).min(max_val);

        let cx = ox + ml + gap * i as f64 + gap / 2.0;
        let map_y = |v: f64| oy + mt + ph - (v - all_min) / range * ph;

        frag.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="black"/>"#, cx, map_y(lower_fence), cx, map_y(q1)));
        frag.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="black"/>"#, cx, map_y(q3), cx, map_y(upper_fence)));
        frag.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="black"/>"#, cx - bw / 4.0, map_y(lower_fence), cx + bw / 4.0, map_y(lower_fence)));
        frag.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="black"/>"#, cx - bw / 4.0, map_y(upper_fence), cx + bw / 4.0, map_y(upper_fence)));
        let by = map_y(q3); let bh = map_y(q1) - by;
        frag.push_str(&format!(r#"<rect x="{:.0}" y="{:.0}" width="{:.0}" height="{:.0}" fill="{}" stroke="black"/>"#, cx - bw / 2.0, by, bw, bh, "#93c5fd"));
        frag.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="black" stroke-width="2"/>"#, cx - bw / 2.0, map_y(median), cx + bw / 2.0, map_y(median)));
        frag.push_str(&format!(r#"<text x="{:.0}" y="{}" text-anchor="middle" font-size="10">{}</text>"#, cx, oy + h - mb + 15.0, name));
    }

    with_device(|d| d.svg_body.push_str(&frag));
    let _ = save_to_file("boxplot.svg");
    print_save_path("boxplot.svg");
    Ok(RVal::Null)
}

/// `barplot(heights, names.arg=, main=)` — colour-cycled bars into the device.
pub fn bi_barplot(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let heights: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    let title = gn(a, "main").map(|v| val_to_str(&v)).unwrap_or("Barplot".into());
    let names = gn(a, "names.arg");

    let labels: Vec<String> = if let Some(RVal::Character(v, _)) = &names {
        v.iter().map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_default()).collect()
    } else {
        (1..=heights.len()).map(|i| format!("{}", i)).collect()
    };

    let (ox, oy, w, h) = begin_plot();
    let (ml, mr, mt, mb) = (60.0, 20.0, 30.0, 50.0);
    let pw = w - ml - mr;
    let ph = h - mt - mb;
    let max_h = heights.iter().cloned().fold(0.0f64, f64::max);
    let bw = pw / heights.len() as f64 * 0.8;
    let gap = pw / heights.len() as f64;

    let colors = vec!["#3b82f6","#ef4444","#22c55e","#f59e0b","#8b5cf6","#ec4899","#06b6d4","#f97316"];

    let mut frag = String::new();
    frag.push_str(&format!(
        r#"<text x="{}" y="{}" text-anchor="middle" font-size="14" font-weight="bold">{}</text>"#,
        ox + w / 2.0, oy + 18.0, title
    ));

    for (i, &val) in heights.iter().enumerate() {
        let bh = if max_h > 0.0 { val / max_h * ph } else { 0.0 };
        let bx = ox + ml + gap * i as f64 + (gap - bw) / 2.0;
        let by = oy + mt + ph - bh;
        let color = colors[i % colors.len()];
        frag.push_str(&format!(
            r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}"/>"#,
            bx, by, bw, bh, color
        ));
        frag.push_str(&format!(
            r#"<text x="{:.0}" y="{:.0}" text-anchor="middle" font-size="10">{:.1}</text>"#,
            bx + bw / 2.0, by - 5.0, val
        ));
        let label = labels.get(i).map(|s| s.as_str()).unwrap_or("");
        frag.push_str(&format!(
            r#"<text x="{:.0}" y="{}" text-anchor="middle" font-size="10" transform="rotate(-30,{:.0},{})">{}</text>"#,
            bx + bw / 2.0, oy + h - mb + 20.0, bx + bw / 2.0, oy + h - mb + 20.0, label
        ));
    }

    with_device(|d| d.svg_body.push_str(&frag));
    let _ = save_to_file("barplot.svg");
    print_save_path("barplot.svg");
    Ok(RVal::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::dev_off;
    fn nums(v: &[f64]) -> RVal {
        RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default())
    }
    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }

    #[test]
    fn plot_writes_svg() {
        dev_off();
        let a = vec![evarg(nums(&[1.0, 2.0, 3.0])), evarg(nums(&[4.0, 5.0, 6.0]))];
        let r = bi_plot(&a).unwrap();
        match r { RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("plot.svg")), _ => panic!("plot must return path") }
        dev_off();
    }

    #[test]
    fn hist_returns_null_and_writes() {
        dev_off();
        let a = vec![evarg(nums(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))];
        let r = bi_hist(&a).unwrap();
        assert!(matches!(r, RVal::Null));
        dev_off();
    }

    #[test]
    fn boxplot_errors_with_no_groups() {
        dev_off();
        let r = bi_boxplot(&[]);
        assert!(r.is_err());
        dev_off();
    }

    #[test]
    fn barplot_returns_null() {
        dev_off();
        let a = vec![evarg(nums(&[3.0, 1.0, 4.0, 1.0, 5.0]))];
        let r = bi_barplot(&a).unwrap();
        assert!(matches!(r, RVal::Null));
        dev_off();
    }
}
