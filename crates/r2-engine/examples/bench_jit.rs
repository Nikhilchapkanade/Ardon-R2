// Phase C.2 micro-benchmark.
// Compiles the same loop both ways (JIT on and JIT off) and reports timings.
// Run: cargo run --release -p r2-engine --example bench_jit
//      or: R2_JIT=0 ... to override default.

use r2_engine::Engine;
use r2_parser::Parser;
use std::time::Instant;

fn run(script: &str, label: &str, jit: bool) -> (f64, String) {
    let mut e = Engine::new();
    e.set_jit_enabled(jit);
    let exprs = Parser::parse(script).expect("parse");
    let t0 = Instant::now();
    let mut last = String::new();
    for ex in exprs {
        match e.eval(&ex) {
            Ok(v) => last = format!("{:?}", v),
            Err(err) => {
                eprintln!("[{}] error: {}", label, err.msg);
                break;
            }
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    (elapsed, last)
}

fn main() {
    // Hot scalar loop: f(i, 2) summed across N iterations.
    let n = 50_000;
    let script = format!(
        r#"
f <- function(x, y) x*x + 2*x*y + y*y + 1
s <- 0
i <- 1
while (i <= {n}) {{
  s <- s + f(i, 2)
  i <- i + 1
}}
s
"#
    );

    println!("Hot scalar loop, N = {}", n);
    println!("Function: f(x,y) = x^2 + 2xy + y^2 + 1, called inside while-loop");
    println!();

    // Warm up.
    let _ = run(&script, "warmup", true);

    let (t_off, _) = run(&script, "JIT OFF", false);
    let (t_on,  _) = run(&script, "JIT ON ", true);

    println!("  JIT OFF: {:>7.3} s", t_off);
    println!("  JIT ON : {:>7.3} s", t_on);
    println!("  Speedup: {:>7.2}x", t_off / t_on);
}
