use r2_parser::Parser;
use r2_engine::Engine;
use r2_types::Expr;
use std::io::{self, Write};

fn main() {
    // Stack size set to 64MB via .cargo/config.toml linker flags
    // This avoids issues with _getch() FFI on spawned threads.
    //
    // Batch mode: `r2 <script.r2>` runs the script non-interactively, prints
    // results of non-silent top-level expressions to stdout, exits 1 on the
    // first eval error. Used for benchmarking and CI. Without arguments,
    // launches the interactive REPL.
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && !args[1].starts_with('-') {
        std::process::exit(run_script(&args[1]));
    }
    repl_main();
}

fn run_script(path: &str) -> i32 {
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("r2: cannot read {}: {}", path, e); return 1; }
    };
    let exprs = match Parser::parse(&source) {
        Ok(e) => e,
        Err(e) => { eprintln!("{}", e.display_with_source(&source)); return 1; }
    };
    let mut engine = Engine::new();
    for stmt in &exprs {
        match engine.eval(stmt) {
            Ok(val) => {
                if !is_silent(stmt) && !matches!(&val, r2_types::RVal::Null) {
                    println!("{}", val);
                }
            }
            Err(err) => { eprintln!("{}", err); return 1; }
        }
    }
    for w in engine.drain_warnings() { eprintln!("{}", w); }
    0
}

fn repl_main() {
    println!("\nArdon-R2 — Statistical Computing, Reimagined");
    println!("Version 0.1.1 (2026) | Inspired by R. Built on Rust.");
    println!("Created by Devendra Tandale | An AI-Assisted Project");
    println!("Assignment: both <- and = work. Mode: strict.");
    println!("Type q() to quit. Arrow keys for history.\n");

    let mut engine = Engine::new();
    let mut history: Vec<String> = Vec::new();
    let mut buffer = String::new();
    let mut continuation = false;

    loop {
        let prompt = if continuation { "R2+ " } else { "R2> " };
        let line = match read_line_with_history(prompt, &history) {
            Some(l) => l,
            None => break,
        };

        let trimmed = line.trim();
        if !continuation && (trimmed == "q()" || trimmed == "quit()") {
            println!("Goodbye.");
            break;
        }

        // R-style help: ?topic or ??topic → help("topic")
        let line = if !continuation && trimmed.starts_with("??") {
            let topic = trimmed[2..].trim();
            format!("help(\"{}\")", topic)
        } else if !continuation && trimmed.starts_with('?') && trimmed.len() > 1 {
            let topic = trimmed[1..].trim();
            format!("help(\"{}\")", topic)
        } else {
            line
        };
        let trimmed = line.trim();

        if !trimmed.is_empty() {
            if history.last().map(|s| s.as_str()) != Some(trimmed) {
                history.push(trimmed.to_string());
            }
        }

        buffer.push_str(&line);
        buffer.push('\n');

        match Parser::parse(&buffer) {
            Ok(stmts) => {
                continuation = false;
                for stmt in &stmts {
                    // Catch any panic to prevent REPL crash
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        engine.eval(stmt)
                    }));
                    match result {
                        Ok(Ok(val)) => {
                            if !is_silent(stmt) && !matches!(&val, r2_types::RVal::Null) {
                                println!("{}", val);
                            }
                        }
                        Ok(Err(e)) => eprintln!("{}", e),
                        Err(_) => eprintln!("Error: internal error (please report this bug)"),
                    }
                }
                for w in engine.drain_warnings() { eprintln!("{}", w); }
                buffer.clear();
            }
            Err(_) => {
                if incomplete(&buffer) {
                    continuation = true;
                } else {
                    if let Err(e) = Parser::parse(&buffer) {
                        // Rich format with source-line + caret underline.
                        eprintln!("{}", e.display_with_source(&buffer));
                    }
                    buffer.clear();
                    continuation = false;
                }
            }
        }
    }
}

fn is_silent(e: &Expr) -> bool {
    if matches!(e, Expr::Assign{..}|Expr::TypeDef{..}|Expr::MethodDef(_)|Expr::For{..}|Expr::While{..}) {
        return true;
    }
    // Calls to functions whose side effect IS the print (and whose return
    // value is uninteresting) shouldn't trigger auto-print of the return.
    // R does this via invisible(); we mark a small set by name.
    if let Expr::Call { func, .. } = e {
        if let Expr::Symbol(s) = func.as_ref() {
            return matches!(s.as_ref(), "print" | "cat" | "message" | "warning" | "writeLines" | "invisible");
        }
    }
    false
}

fn incomplete(s: &str) -> bool {
    let (mut p, mut b, mut k) = (0i32, 0i32, 0i32);
    let mut in_str = false; let mut q = ' ';
    for ch in s.chars() {
        if in_str { if ch == q { in_str = false; } continue; }
        match ch {
            '"'|'\'' => { in_str = true; q = ch; }
            '(' => p+=1, ')' => p-=1, '{' => b+=1, '}' => b-=1, '[' => k+=1, ']' => k-=1,
            '#' => break, _ => {}
        }
    }
    p > 0 || b > 0 || k > 0
}

// ═══════════════════════════════════════════════════════════════════════
// Windows line editor with arrow key history
// ═══════════════════════════════════════════════════════════════════════

#[cfg(windows)]
fn read_line_with_history(prompt: &str, history: &[String]) -> Option<String> {
    print!("{}", prompt);
    io::stdout().flush().unwrap();

    let mut line = String::new();
    let mut cursor = 0usize;
    let mut hist_idx: usize = history.len();
    let mut saved_line = String::new();

    loop {
        let ch = win_getch();
        match ch {
            13 => { println!(); return Some(line); }           // Enter
            3 => { println!("^C"); return Some(String::new()); } // Ctrl+C
            4 if line.is_empty() => { println!(); return None; } // Ctrl+D
            8 | 127 => {                                        // Backspace
                if cursor > 0 {
                    // Move back to previous char boundary
                    cursor -= 1;
                    while cursor > 0 && !line.is_char_boundary(cursor) { cursor -= 1; }
                    if line.is_char_boundary(cursor) { line.remove(cursor); }
                    redraw_line(prompt, &line, cursor);
                }
            }
            0 | 224 => {                                        // Special key prefix
                let key = win_getch();
                match key {
                    72 => {                                     // Up arrow
                        if !history.is_empty() && hist_idx > 0 {
                            if hist_idx == history.len() { saved_line = line.clone(); }
                            hist_idx -= 1;
                            line = history[hist_idx].clone();
                            cursor = line.len();
                            redraw_line(prompt, &line, cursor);
                        }
                    }
                    80 => {                                     // Down arrow
                        if hist_idx < history.len() {
                            hist_idx += 1;
                            line = if hist_idx == history.len() { saved_line.clone() } else { history[hist_idx].clone() };
                            cursor = line.len();
                            redraw_line(prompt, &line, cursor);
                        }
                    }
                    75 => {                                     // Left arrow
                        if cursor > 0 {
                            // Move back one character (could be multi-byte)
                            cursor -= 1;
                            while cursor > 0 && !line.is_char_boundary(cursor) { cursor -= 1; }
                            redraw_line(prompt, &line, cursor);
                        }
                    }
                    77 => {                                     // Right arrow
                        if cursor < line.len() {
                            cursor += 1;
                            while cursor < line.len() && !line.is_char_boundary(cursor) { cursor += 1; }
                            redraw_line(prompt, &line, cursor);
                        }
                    }
                    71 => { cursor = 0; redraw_line(prompt, &line, cursor); }       // Home
                    79 => { cursor = line.len(); redraw_line(prompt, &line, cursor); } // End
                    83 => {                                     // Delete
                        if cursor < line.len() && line.is_char_boundary(cursor) { line.remove(cursor); redraw_line(prompt, &line, cursor); }
                    }
                    _ => {}
                }
            }
            ch if ch >= 32 => {                                 // Printable
                let c = ch as u8 as char;
                if cursor <= line.len() && line.is_char_boundary(cursor) {
                    line.insert(cursor, c);
                    cursor += c.len_utf8();
                } else {
                    line.push(c);
                    cursor = line.len();
                }
                if cursor == line.len() { print!("{}", c); io::stdout().flush().unwrap(); }
                else { redraw_line(prompt, &line, cursor); }
            }
            _ => {}
        }
    }
}

#[cfg(windows)]
extern "C" { fn _getch() -> i32; }

#[cfg(windows)]
fn win_getch() -> i32 { unsafe { _getch() } }

#[cfg(windows)]
fn redraw_line(prompt: &str, line: &str, cursor: usize) {
    print!("\r{}{}\x1b[K", prompt, line);
    let back = line.len() - cursor;
    if back > 0 { print!("\x1b[{}D", back); }
    io::stdout().flush().unwrap();
}

// ═══════════════════════════════════════════════════════════════════════
// Unix fallback
// ═══════════════════════════════════════════════════════════════════════

#[cfg(not(windows))]
fn read_line_with_history(prompt: &str, _history: &[String]) -> Option<String> {
    use std::io::BufRead;
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut line = String::new();
    match io::stdin().lock().read_line(&mut line) {
        Ok(0) => None,
        Ok(_) => Some(line.trim_end().to_string()),
        Err(_) => None,
    }
}
