//! Integration tests for NSE forms `subset(df, cond)` and
//! `transform(df, name = expr)`.
//!
//! Both rely on engine-level pre-processing (see r2-engine/src/lib.rs
//! near line 459): when the call shape is `subset(df, ...)` or
//! `transform(df, ...)`, the engine first evaluates the data-frame
//! argument, then evaluates the remaining argument expressions in a
//! child env that shadows globals with df columns. Without this wiring,
//! `subset(df, x > 2)` would resolve `x` against the global env.
//!
//! Earlier versions of R2 documented these NSE paths as "deferred"; the
//! engine ships them now and these tests pin the behaviour.

use r2_engine::Engine;
use r2_parser::Parser;
use r2_types::RVal;

fn eval_last(script: &str) -> RVal {
    let mut e = Engine::new();
    let exprs = Parser::parse(script).expect("parse ok");
    let mut last = RVal::Null;
    for ex in exprs {
        last = e.eval(&ex).unwrap_or_else(|err| panic!("eval error: {}", err.msg));
    }
    last
}

#[test]
fn subset_nse_resolves_columns() {
    // df has columns x = c(1, 2, 3, 4, 5), y = c(10, 20, 30, 40, 50).
    // subset(df, x > 2) should keep rows where x > 2 → rows 3,4,5.
    // Without NSE, `x` would not resolve and the call would fail.
    let script = r#"
df <- data.frame(x = c(1, 2, 3, 4, 5), y = c(10, 20, 30, 40, 50))
result <- subset(df, x > 2)
result$x
"#;
    match eval_last(script) {
        RVal::Numeric(v, _) => {
            let got: Vec<Option<f64>> = v.iter().copied().collect();
            assert_eq!(got, vec![Some(3.0), Some(4.0), Some(5.0)], "subset filtered rows");
        }
        other => panic!("expected $x numeric, got {:?}", other),
    }
}

#[test]
fn subset_nse_compound_condition() {
    // subset(df, x > 1 & y < 50) → rows where 1 < x and y < 50.
    let script = r#"
df <- data.frame(x = c(1, 2, 3, 4, 5), y = c(10, 20, 30, 40, 50))
result <- subset(df, x > 1 & y < 50)
result$x
"#;
    match eval_last(script) {
        RVal::Numeric(v, _) => {
            let got: Vec<Option<f64>> = v.iter().copied().collect();
            // x > 1: rows 2..5 (x in {2,3,4,5}); y < 50: rows 1..4 (y in {10,20,30,40})
            // AND: rows 2..4 → x in {2,3,4}
            assert_eq!(got, vec![Some(2.0), Some(3.0), Some(4.0)]);
        }
        other => panic!("expected $x numeric, got {:?}", other),
    }
}

#[test]
fn transform_nse_evaluates_expr_against_columns() {
    // transform(df, z = x + y) should append z = x + y referring to df cols.
    let script = r#"
df <- data.frame(x = c(1, 2, 3), y = c(10, 20, 30))
result <- transform(df, z = x + y)
result$z
"#;
    match eval_last(script) {
        RVal::Numeric(v, _) => {
            let got: Vec<Option<f64>> = v.iter().copied().collect();
            assert_eq!(got, vec![Some(11.0), Some(22.0), Some(33.0)]);
        }
        other => panic!("expected $z numeric, got {:?}", other),
    }
}

#[test]
fn transform_nse_overwrites_existing_column() {
    // transform(df, x = x * 2) overwrites x rather than appending.
    let script = r#"
df <- data.frame(x = c(1, 2, 3), y = c(10, 20, 30))
result <- transform(df, x = x * 2)
result$x
"#;
    match eval_last(script) {
        RVal::Numeric(v, _) => {
            let got: Vec<Option<f64>> = v.iter().copied().collect();
            assert_eq!(got, vec![Some(2.0), Some(4.0), Some(6.0)]);
        }
        other => panic!("expected $x numeric, got {:?}", other),
    }
}
