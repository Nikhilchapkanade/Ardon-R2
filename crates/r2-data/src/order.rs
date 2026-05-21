//! `duplicated`, `unique`, `order`, `rank`. Phase R.7.
//!
//! Vector-shaped utilities for sorting and de-duplication.

use r2_types::{Attrs, ErrKind, EvalArg, Logical, R2Err, RVal, Real};
use std::sync::Arc;

#[inline]
fn first_arg(a: &[EvalArg]) -> RVal { a.first().map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn arg_named(a: &[EvalArg], name: &str) -> Option<RVal> {
    a.iter().find(|x| x.name.as_ref().map(|n| n.as_ref()) == Some(name)).map(|x| x.value.clone())
}

#[inline]
fn rnums(v: &[f64]) -> RVal {
    RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default())
}

pub fn bi_duplicated(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &first_arg(a) {
        RVal::Numeric(v, _) => {
            let mut seen: Vec<f64> = Vec::new();
            let result: Vec<Logical> = v.iter().map(|x| match x {
                Some(n) => {
                    let dup = seen.iter().any(|s| (*s - *n).abs() < 1e-10);
                    seen.push(*n);
                    Some(dup)
                }
                None => Some(false),
            }).collect();
            Ok(RVal::Logical(result.into(), Attrs::default()))
        }
        RVal::Character(v, _) => {
            let mut seen: Vec<Arc<str>> = Vec::new();
            let result: Vec<Logical> = v.iter().map(|x| match x {
                Some(s) => {
                    let dup = seen.iter().any(|p| p == s);
                    seen.push(s.clone());
                    Some(dup)
                }
                None => Some(false),
            }).collect();
            Ok(RVal::Logical(result.into(), Attrs::default()))
        }
        _ => Err(R2Err { msg: "duplicated() not supported for this type".into(), kind: ErrKind::Type }),
    }
}

pub fn bi_unique(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = first_arg(a).as_reals()?;
    let mut seen: Vec<f64> = Vec::new();
    let mut r = Vec::new();
    for x in &v {
        if let Some(n) = x {
            if !seen.iter().any(|s| (s - n).abs() < f64::EPSILON) {
                seen.push(*n);
                r.push(Some(*n));
            }
        }
    }
    Ok(RVal::Numeric(r.into(), Attrs::default()))
}

pub fn bi_order(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let decreasing = arg_named(a, "decreasing")
        .and_then(|v| v.as_logicals().ok())
        .map(|v| v.first().copied().flatten() == Some(true))
        .unwrap_or(false);
    match &first_arg(a) {
        RVal::Numeric(v, _) => {
            let mut indices: Vec<usize> = (0..v.len()).collect();
            indices.sort_by(|&i, &j| {
                let vi = v[i].unwrap_or(f64::NAN);
                let vj = v[j].unwrap_or(f64::NAN);
                if decreasing { vj.partial_cmp(&vi).unwrap_or(std::cmp::Ordering::Equal) }
                else { vi.partial_cmp(&vj).unwrap_or(std::cmp::Ordering::Equal) }
            });
            Ok(RVal::Integer(indices.iter().map(|i| Some((*i + 1) as i32)).collect(), Attrs::default()))
        }
        RVal::Character(v, _) => {
            let mut indices: Vec<usize> = (0..v.len()).collect();
            indices.sort_by(|&i, &j| {
                let vi = v[i].as_ref().map(|s| s.as_ref()).unwrap_or("");
                let vj = v[j].as_ref().map(|s| s.as_ref()).unwrap_or("");
                if decreasing { vj.cmp(vi) } else { vi.cmp(vj) }
            });
            Ok(RVal::Integer(indices.iter().map(|i| Some((*i + 1) as i32)).collect(), Attrs::default()))
        }
        _ => Err(R2Err { msg: "order() needs numeric or character".into(), kind: ErrKind::Type }),
    }
}

pub fn bi_rank(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v: Vec<Real> = first_arg(a).as_reals()?;
    let n = v.len();
    let mut indexed: Vec<(usize, f64)> = v.iter().enumerate()
        .map(|(i, x)| (i, x.unwrap_or(f64::NAN)))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; n];
    for (rank, (orig_idx, _)) in indexed.iter().enumerate() {
        ranks[*orig_idx] = (rank + 1) as f64;
    }
    Ok(rnums(&ranks))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nums(v: &[f64]) -> RVal { RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default()) }
    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }

    #[test]
    fn duplicated_marks_repeats() {
        let r = bi_duplicated(&[evarg(nums(&[1.0, 2.0, 1.0, 3.0, 2.0]))]).unwrap();
        match r {
            RVal::Logical(v, _) => {
                let got: Vec<bool> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![false, false, true, false, true]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn unique_keeps_first_occurrences() {
        let r = bi_unique(&[evarg(nums(&[1.0, 2.0, 1.0, 3.0, 2.0]))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                let got: Vec<f64> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn order_returns_sort_indices_1based() {
        let r = bi_order(&[evarg(nums(&[3.0, 1.0, 2.0]))]).unwrap();
        match r {
            RVal::Integer(v, _) => {
                let got: Vec<i32> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![2, 3, 1]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn rank_assigns_ascending_ranks() {
        let r = bi_rank(&[evarg(nums(&[30.0, 10.0, 20.0]))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                let got: Vec<f64> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![3.0, 1.0, 2.0]);
            }
            _ => panic!(),
        }
    }
}
