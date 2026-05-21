//! dplyr-style row/column operations — Phase R.2 step 3.
//!
//! - `filter(df, mask)`        — keep rows where mask is TRUE
//! - `select(df, "c1", ...)`   — pick columns by name
//! - `arrange(df, col, ...)`   — sort by column values, optional decreasing
//!
//! All pure: no engine reference; uses `RVal::as_reals()` /
//! `RVal::as_logicals()` (Phase R.1 step 2).

use r2_types::*;
use std::sync::Arc;

#[inline]
fn first_arg(args: &[EvalArg]) -> RVal {
    args.first().map(|a| a.value.clone()).unwrap_or(RVal::Null)
}

#[inline]
fn nth_arg(args: &[EvalArg], i: usize) -> RVal {
    args.get(i).map(|a| a.value.clone()).unwrap_or(RVal::Null)
}

#[inline]
fn arg_named(args: &[EvalArg], name: &str) -> Option<RVal> {
    args.iter().find(|a| a.name.as_ref().map(|n| n.as_ref()) == Some(name))
        .map(|a| a.value.clone())
}

/// `filter(df, mask)` — keep rows where mask is TRUE.
pub fn bi_filter(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let df = match first_arg(a) {
        RVal::DataFrame(df) => df,
        _ => return Err(R2Err { msg: "filter() needs data.frame".into(), kind: ErrKind::Runtime }),
    };
    let mask = nth_arg(a, 1).as_logicals()?;
    let keep: Vec<usize> = mask.iter().enumerate()
        .filter(|(_, m)| **m == Some(true))
        .map(|(i, _)| i)
        .collect();

    let columns: Vec<(Arc<str>, RVal)> = df.columns.iter().map(|(name, col)| {
        let new_col = match col {
            RVal::Numeric(v, _)   => RVal::Numeric(keep.iter().map(|&r| if r < v.len() { v[r] } else { None }).collect(), Attrs::default()),
            RVal::Integer(v, _)   => RVal::Integer(keep.iter().map(|&r| if r < v.len() { v[r] } else { None }).collect(), Attrs::default()),
            RVal::Character(v, _) => RVal::Character(keep.iter().map(|&r| if r < v.len() { v[r].clone() } else { None }).collect(), Attrs::default()),
            RVal::Logical(v, _)   => RVal::Logical(keep.iter().map(|&r| if r < v.len() { v[r] } else { None }).collect(), Attrs::default()),
            _ => col.clone(),
        };
        (name.clone(), new_col)
    }).collect();

    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
}

/// `select(df, "col1", "col2", ...)` — pick columns by name.
pub fn bi_select(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let df = match first_arg(a) {
        RVal::DataFrame(df) => df,
        _ => return Err(R2Err { msg: "select() needs data.frame".into(), kind: ErrKind::Runtime }),
    };

    let mut col_names: Vec<String> = Vec::new();
    for i in 1..10 {
        match nth_arg(a, i) {
            RVal::Character(v, _) => {
                for c in &v { if let Some(s) = c { col_names.push(s.to_string()); } }
            }
            RVal::Null => break,
            _ => break,
        }
    }

    if col_names.is_empty() { return Ok(RVal::DataFrame(df)); }

    let columns: Vec<(Arc<str>, RVal)> = col_names.iter().filter_map(|name| {
        df.columns.iter().find(|(n, _)| n.as_ref() == name.as_str()).cloned()
    }).collect();

    if columns.is_empty() {
        return Err(R2Err { msg: "select: no matching columns found".into(), kind: ErrKind::Runtime });
    }
    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
}

/// `mutate(df, new_col = values, ...)` — add or replace columns.
/// Each named argument becomes a column: existing names are replaced,
/// new names are appended. (NSE for expression-style args is the engine
/// preprocessor's job; this function sees pre-evaluated values.)
pub fn bi_mutate(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let mut df = match first_arg(a) {
        RVal::DataFrame(df) => df,
        _ => return Err(R2Err { msg: "mutate() needs data.frame".into(), kind: ErrKind::Runtime }),
    };
    // Process named args (skip the first positional which is the df).
    for ea in a.iter().skip(1) {
        if let Some(name) = &ea.name {
            let val = ea.value.clone();
            if let Some(pos) = df.columns.iter().position(|(n, _)| n.as_ref() == name.as_ref()) {
                df.columns[pos].1 = val;
            } else {
                df.columns.push((name.clone(), val));
            }
        }
    }
    Ok(RVal::DataFrame(df))
}

/// `arrange(df, col_values, decreasing = FALSE)` — sort df by given values.
pub fn bi_arrange(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let df = match first_arg(a) {
        RVal::DataFrame(df) => df,
        _ => return Err(R2Err { msg: "arrange() needs data.frame".into(), kind: ErrKind::Runtime }),
    };
    let sort_vals = nth_arg(a, 1).as_reals()?;
    let decreasing = arg_named(a, "decreasing")
        .and_then(|v| v.as_logicals().ok())
        .map(|v| v[0] == Some(true))
        .unwrap_or(false);

    let nrow = df.nrow();
    let mut indices: Vec<usize> = (0..nrow).collect();
    indices.sort_by(|&i, &j| {
        let vi = sort_vals.get(i).and_then(|x| *x).unwrap_or(f64::NAN);
        let vj = sort_vals.get(j).and_then(|x| *x).unwrap_or(f64::NAN);
        if decreasing { vj.partial_cmp(&vi).unwrap_or(std::cmp::Ordering::Equal) }
        else          { vi.partial_cmp(&vj).unwrap_or(std::cmp::Ordering::Equal) }
    });

    let columns: Vec<(Arc<str>, RVal)> = df.columns.iter().map(|(name, col)| {
        let new_col = match col {
            RVal::Numeric(v, _)   => RVal::Numeric(indices.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Integer(v, _)   => RVal::Integer(indices.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Character(v, _) => RVal::Character(indices.iter().map(|&r| v.get(r).cloned().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Logical(v, _)   => RVal::Logical(indices.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            _ => col.clone(),
        };
        (name.clone(), new_col)
    }).collect();

    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn df_2col() -> DataFrame {
        DataFrame {
            columns: vec![
                (Arc::from("x"), RVal::Numeric(vec![Some(1.0), Some(2.0), Some(3.0)].into(), Attrs::default())),
                (Arc::from("y"), RVal::Character(vec![Some(Arc::from("a")), Some(Arc::from("b")), Some(Arc::from("c"))], Attrs::default())).into(),
            ],
            row_names: None,
        }
    }

    #[test]
    fn filter_keeps_true_rows() {
        let args = vec![
            EvalArg { name: None, value: RVal::DataFrame(df_2col()) },
            EvalArg { name: None, value: RVal::Logical(vec![Some(true), Some(false), Some(true)].into(), Attrs::default()) },
        ];
        match bi_filter(&args).unwrap() {
            RVal::DataFrame(d) => {
                assert_eq!(d.nrow(), 2);
                if let RVal::Numeric(v, _) = &d.columns[0].1 {
                    assert_eq!(v.as_vec(), &vec![Some(1.0), Some(3.0)]);
                } else { panic!("expected Numeric"); }
            }
            _ => panic!("expected DataFrame"),
        }
    }

    #[test]
    fn select_picks_named_columns() {
        let args = vec![
            EvalArg { name: None, value: RVal::DataFrame(df_2col()) },
            EvalArg { name: None, value: RVal::Character(vec![Some(Arc::from("y"))], Attrs::default()) },
        ];
        match bi_select(&args).unwrap() {
            RVal::DataFrame(d) => {
                assert_eq!(d.columns.len(), 1);
                assert_eq!(d.columns[0].0.as_ref(), "y");
            }
            _ => panic!("expected DataFrame"),
        }
    }

    #[test]
    fn arrange_sorts_ascending() {
        let args = vec![
            EvalArg { name: None, value: RVal::DataFrame(df_2col()) },
            EvalArg { name: None, value: RVal::Numeric(vec![Some(3.0), Some(1.0), Some(2.0)].into(), Attrs::default()) },
        ];
        match bi_arrange(&args).unwrap() {
            RVal::DataFrame(d) => {
                if let RVal::Numeric(v, _) = &d.columns[0].1 {
                    // x was [1,2,3]; sort by [3,1,2] ascending → keys [1,2,3] → indices [1,2,0] → x = [2,3,1]
                    assert_eq!(v.as_vec(), &vec![Some(2.0), Some(3.0), Some(1.0)]);
                } else { panic!("expected Numeric"); }
            }
            _ => panic!("expected DataFrame"),
        }
    }

    #[test]
    fn mutate_adds_new_column_and_replaces_existing() {
        let args = vec![
            EvalArg { name: None, value: RVal::DataFrame(df_2col()) },
            // Replace existing "x" with new values
            EvalArg { name: Some(Arc::from("x")), value: RVal::Numeric(vec![Some(10.0), Some(20.0), Some(30.0)].into(), Attrs::default()) },
            // Add new column "z"
            EvalArg { name: Some(Arc::from("z")), value: RVal::Numeric(vec![Some(0.5), Some(0.6), Some(0.7)].into(), Attrs::default()) }.into(),
        ];
        match bi_mutate(&args).unwrap() {
            RVal::DataFrame(d) => {
                assert_eq!(d.columns.len(), 3, "expected 3 columns (x replaced, y kept, z added)");
                assert_eq!(d.columns[0].0.as_ref(), "x");
                if let RVal::Numeric(v, _) = &d.columns[0].1 {
                    assert_eq!(v.as_vec(), &vec![Some(10.0), Some(20.0), Some(30.0)]);
                } else { panic!("expected Numeric"); }
                assert_eq!(d.columns[2].0.as_ref(), "z");
            }
            _ => panic!("expected DataFrame"),
        }
    }

    #[test]
    fn arrange_decreasing() {
        let args = vec![
            EvalArg { name: None, value: RVal::DataFrame(df_2col()) },
            EvalArg { name: None, value: RVal::Numeric(vec![Some(3.0), Some(1.0), Some(2.0)].into(), Attrs::default()) },
            EvalArg { name: Some(Arc::from("decreasing")), value: RVal::Logical(vec![Some(true)].into(), Attrs::default()) }.into(),
        ];
        match bi_arrange(&args).unwrap() {
            RVal::DataFrame(d) => {
                if let RVal::Numeric(v, _) = &d.columns[0].1 {
                    // sort keys [3,1,2] descending → indices [0,2,1] → x = [1,3,2]
                    assert_eq!(v.as_vec(), &vec![Some(1.0), Some(3.0), Some(2.0)]);
                } else { panic!("expected Numeric"); }
            }
            _ => panic!("expected DataFrame"),
        }
    }
}
