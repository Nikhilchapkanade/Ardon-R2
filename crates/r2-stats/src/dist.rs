//! Probability distributions — Phase R.9.
//!
//! Density, CDF, and quantile functions for the standard normal,
//! plus the small numerical helpers (`erf`, `qnorm_approx`, `phi`) that
//! are also used by other r2 builtins (t-test, chisq-test, etc.).
//!
//! Random-variate generators (`rnorm`, `runif`, `sample`, `rbinom`,
//! `rpois`) are NOT in this module. They share an RNG state with
//! `r2_ml::tree::SEED_STATE` and currently live in r2-engine pending
//! a separate decision on where the RNG primitive should live.

use r2_types::{Attrs, EvalArg, R2Err, RVal, Real};

#[inline]
fn first(a: &[EvalArg]) -> RVal { a.first().map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn arg_named(a: &[EvalArg], name: &str) -> Option<RVal> {
    a.iter().find(|x| x.name.as_ref().map(|n| n.as_ref()) == Some(name)).map(|x| x.value.clone())
}

// ─────────────────────────────────────────────────────────────────────
// Numerical primitives — used by dnorm/pnorm/qnorm AND by t-test /
// chisq-test in r2-engine. Re-exported from r2-stats so engine helpers
// can drop the duplicated definitions.
// ─────────────────────────────────────────────────────────────────────

/// Abramowitz & Stegun 7.1.26 polynomial approximation. Max error ≈ 1.5e-7.
pub fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 { result } else { -result }
}

/// Standard normal CDF: P(Z ≤ x).
#[inline]
pub fn phi(x: f64) -> f64 { 0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2)) }

/// Inverse standard-normal CDF (Beasley-Springer-Moro). Max error ≈ 4.5e-4
/// across (0, 1); good enough for everyday quantile lookups, NOT for
/// extreme-tail work (use the rational approx in dlmf or a Halley
/// refinement step if more precision needed).
pub fn qnorm_approx(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if (p - 0.5).abs() < 1e-15 { return 0.0; }
    let pp = if p < 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * pp.ln()).sqrt();
    let c0 = 2.515517; let c1 = 0.802853; let c2 = 0.010328;
    let d1 = 1.432788; let d2 = 0.189269; let d3 = 0.001308;
    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if p < 0.5 { -z } else { z }
}

// ─────────────────────────────────────────────────────────────────────
// Builtins
// ─────────────────────────────────────────────────────────────────────

fn mean_sd(a: &[EvalArg]) -> (f64, f64) {
    let mean = arg_named(a, "mean").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(0.0);
    let sd = arg_named(a, "sd").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(1.0);
    (mean, sd)
}

pub fn bi_dnorm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x = first(a).as_reals()?;
    let (mean, sd) = mean_sd(a);
    let result: Vec<Real> = x.iter().map(|v| v.map(|x| {
        let z = (x - mean) / sd;
        (1.0 / (sd * (2.0 * std::f64::consts::PI).sqrt())) * (-0.5 * z * z).exp()
    })).collect();
    Ok(RVal::Numeric(result.into(), Attrs::default()))
}

pub fn bi_pnorm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x = first(a).as_reals()?;
    let (mean, sd) = mean_sd(a);
    let result: Vec<Real> = x.iter().map(|v| v.map(|x| {
        let z = (x - mean) / sd;
        0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
    })).collect();
    Ok(RVal::Numeric(result.into(), Attrs::default()))
}

pub fn bi_qnorm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let p = first(a).as_reals()?;
    let (mean, sd) = mean_sd(a);
    let result: Vec<Real> = p.iter().map(|v| v.map(|p| mean + sd * qnorm_approx(p))).collect();
    Ok(RVal::Numeric(result.into(), Attrs::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nums(v: &[f64]) -> RVal {
        RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default())
    }
    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }

    #[test]
    fn dnorm_at_zero_is_one_over_sqrt_2pi() {
        let r = bi_dnorm(&[evarg(nums(&[0.0]))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                let got = v[0].unwrap();
                let want = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
                assert!((got - want).abs() < 1e-12, "got {} want {}", got, want);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn pnorm_zero_is_half() {
        let r = bi_pnorm(&[evarg(nums(&[0.0]))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                assert!((v[0].unwrap() - 0.5).abs() < 1e-7);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn qnorm_half_is_zero() {
        let r = bi_qnorm(&[evarg(nums(&[0.5]))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                assert!(v[0].unwrap().abs() < 1e-12);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn pnorm_qnorm_round_trip_at_975() {
        let p = bi_qnorm(&[evarg(nums(&[0.975]))]).unwrap();
        let q = match p { RVal::Numeric(v, _) => v[0].unwrap(), _ => panic!() };
        // Beasley-Springer-Moro qnorm(0.975) ≈ 1.96 ± 5e-4.
        assert!((q - 1.96).abs() < 0.01, "qnorm(0.975) = {}", q);
        let r = bi_pnorm(&[evarg(nums(&[q]))]).unwrap();
        match r {
            RVal::Numeric(v, _) => assert!((v[0].unwrap() - 0.975).abs() < 0.01),
            _ => panic!(),
        }
    }
}
