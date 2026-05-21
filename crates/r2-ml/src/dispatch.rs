//! ML builtin dispatch — Phase R.1 step 4.
//!
//! Each `bi_*` here is a registry entry point: takes `&[EvalArg]`, returns
//! `Result<RVal, R2Err>`. No engine reference. r2-engine wraps these with
//! a 1-line adapter that satisfies its `BuiltinFn` signature (which carries
//! `&mut Engine` and `&EnvRef` for stateful builtins like `lm`).
//!
//! The adapter is FFI glue, not bloat. The function below is the actual
//! definition of `rpart` — language dispatch happens here.

use crate::data::extract_ml_data;
use crate::tree::{build_tree, tree_predict_one, serialize_tree, print_tree, count_splits, parallel_random, next_random, SEED_STATE, TreeNode};
use r2_types::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

#[inline]
fn gn(args: &[EvalArg], name: &str) -> Option<RVal> {
    args.iter().find(|a| a.name.as_ref().map(|n| n.as_ref()) == Some(name))
        .map(|a| a.value.clone())
}

/// Inline single-value-to-string converter (was `val_to_str` in engine).
fn rval_to_string(v: &RVal) -> String {
    match v {
        RVal::Character(v, _) => v.iter()
            .map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or("NA".into()))
            .collect::<Vec<_>>().join(" "),
        RVal::Numeric(v, _) => v.iter()
            .map(|x| x.map(fmt_num).unwrap_or("NA".into()))
            .collect::<Vec<_>>().join(" "),
        _ => format!("<{}>", v.type_name()),
    }
}

/// rpart(x, y) or rpart(y ~ ., data = df) — fits a CART decision tree.
pub fn bi_rpart(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (y, mat, col_names) = extract_ml_data(a)?;
    let max_depth = gn(a, "max_depth").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(5.0) as usize;
    let min_samples = gn(a, "min_samples").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(5.0) as usize;
    let user_type = gn(a, "type").map(|v| rval_to_string(&v));

    let (m, n) = (mat.nrow, mat.ncol);
    if y.len() != m {
        return Err(R2Err { msg: "rpart: y length must match x rows".into(), kind: ErrKind::Runtime });
    }

    // Auto-detect: if y has > 10 unique values, regression; else classification.
    let tree_type = user_type.unwrap_or_else(|| {
        let mut uniq: Vec<i64> = y.iter().map(|v| (*v * 1000.0) as i64).collect();
        uniq.sort(); uniq.dedup();
        if uniq.len() > 10 { "regression".into() } else { "classification".into() }
    });
    let is_class = tree_type != "regression";
    let mask = vec![true; m];
    let tree = build_tree(&mat.data, &y, m, n, &mask, max_depth, min_samples, 0, is_class);

    println!("\nDecision tree ({}, depth={}):", tree_type, max_depth);
    print_tree(&tree, 0, &col_names);

    // Training accuracy / MSE
    let preds: Vec<f64> = (0..m).map(|i| tree_predict_one(&tree, &mat.data, m, i)).collect();
    if is_class {
        let correct = preds.iter().zip(y.iter())
            .filter(|(p, y)| (**p as i64) == (**y as i64)).count();
        println!("\nTraining accuracy: {}/{} ({}%)", correct, m,
            fmt_num(correct as f64 / m as f64 * 100.0));
    } else {
        let mse: f64 = preds.iter().zip(y.iter())
            .map(|(p, y)| (p - y).powi(2)).sum::<f64>() / m as f64;
        println!("\nTraining MSE: {}", fmt_num(mse));
    }

    // Build the rpart TypeInstance with serialized tree fields
    let pred_vals: Vec<Real> = preds.iter().map(|p| Some(*p)).collect();
    let mut fields = HashMap::new();
    fields.insert(Arc::from("predictions"), RVal::Numeric(pred_vals.into(), Attrs::default()));
    fields.insert(Arc::from("type"), rstr(&tree_type));
    fields.insert(Arc::from("max_depth"), rnum(max_depth as f64));

    let mut feat_list = Vec::new();
    let mut thresh_list = Vec::new();
    let mut pred_list = Vec::new();
    let mut leaf_list = Vec::new();
    let mut left_list = Vec::new();
    let mut right_list = Vec::new();
    serialize_tree(&tree, &mut feat_list, &mut thresh_list, &mut pred_list,
        &mut leaf_list, &mut left_list, &mut right_list);
    fields.insert(Arc::from("_tree_feat"),
        rnums(&feat_list.iter().map(|&x| x as f64).collect::<Vec<_>>()));
    fields.insert(Arc::from("_tree_thresh"), rnums(&thresh_list));
    fields.insert(Arc::from("_tree_pred"), rnums(&pred_list));
    fields.insert(Arc::from("_tree_leaf"),
        rnums(&leaf_list.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect::<Vec<_>>()));
    fields.insert(Arc::from("_tree_left"),
        rnums(&left_list.iter().map(|&x| x as f64).collect::<Vec<_>>()));
    fields.insert(Arc::from("_tree_right"),
        rnums(&right_list.iter().map(|&x| x as f64).collect::<Vec<_>>()));

    Ok(RVal::TypeInstance(TypeInstance {
        type_name: Arc::from("rpart"),
        fields,
    }))
}

/// Random Forest fit. Trees are built independently — parallel-for via the
/// kernel (`r2_kernel::par_for(Op::TreeBuild, ...)`). No `par_iter` here;
/// Rayon stays below the kernel layer per §4.9.
pub fn bi_rf(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (y, mat, col_names) = extract_ml_data(a)?;
    let ntrees     = gn(a, "ntrees").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(100.0) as usize;
    let max_depth  = gn(a, "max_depth").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(10.0) as usize;
    let tree_type  = gn(a, "type").map(|v| rval_to_string(&v)).unwrap_or("classification".into());
    let is_class   = tree_type != "regression";
    let (m, n)     = (mat.nrow, mat.ncol);

    println!("Random Forest: {} trees, max_depth={}, features={}", ntrees, max_depth, n);

    // Heap-allocated locals captured by the per-tree closure. Cloned once so
    // the closure is `Sync` (required by kernel::par_for's parallel path).
    let mat_data = mat.data.clone();
    let y_clone  = y.clone();

    let build_one = move |t: usize| -> (crate::tree::TreeNode, Vec<f64>) {
        let mut seed = SEED_STATE.load(Ordering::Relaxed).wrapping_add(t as u64 * 999983);
        let mut mask = vec![false; m];
        for _ in 0..m {
            let idx = (parallel_random(&mut seed) * m as f64) as usize % m;
            mask[idx] = true;
        }
        let tree = build_tree(&mat_data, &y_clone, m, n, &mask, max_depth, 2, 0, is_class);
        let preds: Vec<f64> = (0..m).map(|i| tree_predict_one(&tree, &mat_data, m, i)).collect();
        (tree, preds)
    };

    // Parallelism decision lives in the kernel — `bi_rf` no longer imports
    // Rayon or calls Oracle directly.
    let tree_results: Vec<(crate::tree::TreeNode, Vec<f64>)> =
        r2_kernel::par_for(r2_oracle::Op::TreeBuild, ntrees, build_one);

    // Aggregate (sequential — fast)
    let mut all_preds = vec![vec![0.0; m]; ntrees];
    let mut importance = vec![0usize; n];
    for (t, (tree, preds)) in tree_results.iter().enumerate() {
        all_preds[t] = preds.clone();
        count_splits(tree, &mut importance);
    }

    let mut final_preds = vec![0.0; m];
    if is_class {
        for i in 0..m {
            let mut votes: HashMap<i64, usize> = HashMap::new();
            for t in 0..ntrees {
                *votes.entry(all_preds[t][i] as i64).or_insert(0) += 1;
            }
            final_preds[i] = votes.into_iter().max_by_key(|(_, c)| *c).map(|(k, _)| k as f64).unwrap_or(0.0);
        }
        let correct = final_preds.iter().zip(y.iter())
            .filter(|(p, y)| (**p as i64) == (**y as i64)).count();
        println!("Training accuracy: {}/{} ({}%)", correct, m,
            fmt_num(correct as f64 / m as f64 * 100.0));
    } else {
        for i in 0..m {
            final_preds[i] = all_preds.iter().map(|p| p[i]).sum::<f64>() / ntrees as f64;
        }
        let mse: f64 = final_preds.iter().zip(y.iter())
            .map(|(p, y)| (p - y).powi(2)).sum::<f64>() / m as f64;
        println!("Training MSE: {}", fmt_num(mse));
    }

    let pred_vals: Vec<Real> = final_preds.iter().map(|p| Some(*p)).collect();
    let total_splits: usize = importance.iter().sum();
    if total_splits > 0 {
        println!("\nFeature importance:");
        let mut imp_idx: Vec<(usize, f64)> = importance.iter().enumerate()
            .map(|(i, &c)| (i, c as f64 / total_splits as f64 * 100.0)).collect();
        imp_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (i, pct) in imp_idx.iter().take(n.min(10)) {
            if *pct > 0.0 {
                let label = col_names.get(*i).map(|s| s.as_str()).unwrap_or("?");
                println!("  {}: {}%", label, fmt_num(*pct));
            }
        }
    }
    let imp_vals: Vec<Real> = importance.iter()
        .map(|&c| Some(c as f64 / total_splits.max(1) as f64)).collect();
    let mut imp_attrs = Attrs::default();
    imp_attrs.names = Some(col_names.iter().map(|s| Arc::from(s.as_str())).collect());
    let mut fields = HashMap::new();
    fields.insert(Arc::from("predictions"), RVal::Numeric(pred_vals.into(), Attrs::default()));
    fields.insert(Arc::from("importance"), RVal::Numeric(imp_vals.into(), imp_attrs));
    fields.insert(Arc::from("ntrees"), rnum(ntrees as f64));
    fields.insert(Arc::from("type"), rstr(&tree_type));
    fields.insert(Arc::from("xnames"),
        RVal::Character(col_names.iter().map(|s| Some(Arc::from(s.as_str()))).collect(), Attrs::default()));

    Ok(RVal::TypeInstance(TypeInstance {
        type_name: Arc::from("rf"),
        fields,
    }))
}

/// Gradient Boosted Trees fit. The boosting loop is sequential (each
/// iteration depends on previous predictions). Per-iteration row work
/// (residuals, prediction updates, loss reduction) is parallelized via
/// `kernel::par_for(Op::PerElementMap, m, ...)` — no Rayon import.
pub fn bi_gbm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (y, mat, col_names) = extract_ml_data(a)?;

    let ntrees       = gn(a, "ntrees").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(100.0) as usize;
    let lr           = gn(a, "learning_rate").or(gn(a, "eta")).and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(0.1);
    let max_depth    = gn(a, "max_depth").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(3.0) as usize;
    let min_samples  = gn(a, "min_samples").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(5.0) as usize;
    let subsample    = gn(a, "subsample").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(0.8);
    let loss_type    = gn(a, "loss").map(|v| rval_to_string(&v)).unwrap_or("squared".into());

    let (m, n) = (mat.nrow, mat.ncol);
    let is_classification = loss_type == "logistic" || loss_type == "bernoulli";

    // Initialize predictions
    let mut f_vals = vec![0.0; m];
    if is_classification {
        let pos = y.iter().filter(|&&v| v > 0.5).count() as f64;
        let neg = (m as f64 - pos).max(1.0);
        let init = (pos / neg).max(1e-10).ln();
        for v in f_vals.iter_mut() { *v = init; }
    } else {
        let mean = y.iter().sum::<f64>() / m as f64;
        for v in f_vals.iter_mut() { *v = mean; }
    }

    let mut trees: Vec<TreeNode> = Vec::new();
    let mut train_losses: Vec<f64> = Vec::new();

    println!("Gradient Boosted Trees: {} trees, lr={}, depth={}, loss={}",
        ntrees, lr, max_depth, loss_type);

    for t in 0..ntrees {
        // Compute pseudo-residuals (negative gradient) — kernel-dispatched.
        let f_snap = f_vals.clone();
        let y_snap = y.clone();
        let loss = loss_type.clone();
        let residuals: Vec<f64> = r2_kernel::par_for(r2_oracle::Op::PerElementMap, m, move |i| {
            match loss.as_str() {
                "logistic" | "bernoulli" => {
                    let p = 1.0 / (1.0 + (-f_snap[i]).exp());
                    y_snap[i] - p
                }
                "huber" => {
                    let r = y_snap[i] - f_snap[i];
                    let delta = 1.0;
                    if r.abs() <= delta { r } else { delta * r.signum() }
                }
                // "squared" | "ls" | _
                _ => y_snap[i] - f_snap[i],
            }
        });

        // Stochastic subsample (sequential — RNG-bound)
        let mut mask = vec![false; m];
        let n_sample = (m as f64 * subsample) as usize;
        for _ in 0..n_sample {
            let idx = (next_random() * m as f64) as usize % m;
            mask[idx] = true;
        }

        let tree = build_tree(&mat.data, &residuals, m, n, &mask, max_depth, min_samples, 0, false);

        // Update predictions: F += lr * tree(x). Per-row, kernel-dispatched.
        let mat_snap = mat.data.clone();
        let tree_snap = tree.clone();
        let updates: Vec<f64> = r2_kernel::par_for(r2_oracle::Op::PerElementMap, m, move |i| {
            lr * tree_predict_one(&tree_snap, &mat_snap, m, i)
        });
        for i in 0..m { f_vals[i] += updates[i]; }

        trees.push(tree);

        // Loss reduction — collect per-element contributions, then sum.
        let f_snap2 = f_vals.clone();
        let y_snap2 = y.clone();
        let loss2 = loss_type.clone();
        let contribs: Vec<f64> = r2_kernel::par_for(r2_oracle::Op::PerElementMap, m, move |i| {
            match loss2.as_str() {
                "logistic" | "bernoulli" => {
                    let p = (1.0 / (1.0 + (-f_snap2[i]).exp())).max(1e-15).min(1.0 - 1e-15);
                    -(y_snap2[i] * p.ln() + (1.0 - y_snap2[i]) * (1.0 - p).ln())
                }
                _ => (y_snap2[i] - f_snap2[i]).powi(2),
            }
        });
        let loss = contribs.iter().sum::<f64>() / m as f64;
        train_losses.push(loss);

        if (t + 1) % 25 == 0 || t == ntrees - 1 {
            print!("\r  Iter {}/{}: loss = {}", t + 1, ntrees, fmt_num(loss));
        }
    }
    println!();

    // Final predictions
    let final_preds: Vec<f64> = if is_classification {
        f_vals.iter().map(|f| if 1.0 / (1.0 + (-f).exp()) > 0.5 { 1.0 } else { 0.0 }).collect()
    } else {
        f_vals.clone()
    };

    if is_classification {
        let correct = final_preds.iter().zip(y.iter()).filter(|(p, y)| (**p as i64) == (**y as i64)).count();
        println!("Training accuracy: {}/{} ({}%)", correct, m, fmt_num(correct as f64 / m as f64 * 100.0));
    } else {
        let mse = final_preds.iter().zip(y.iter()).map(|(p, y)| (p - y).powi(2)).sum::<f64>() / m as f64;
        let y_mean = y.iter().sum::<f64>() / m as f64;
        let ss_tot = y.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>();
        let r2 = 1.0 - (mse * m as f64) / ss_tot;
        println!("Training MSE: {},  R²: {}", fmt_num(mse), fmt_num(r2));
    }

    // Feature importance
    let mut importance = vec![0usize; n];
    for tree in &trees { count_splits(tree, &mut importance); }
    let total_splits: usize = importance.iter().sum();
    if total_splits > 0 {
        println!("\nFeature importance:");
        let mut imp_idx: Vec<(usize, f64)> = importance.iter().enumerate()
            .map(|(i, &c)| (i, c as f64 / total_splits as f64 * 100.0)).collect();
        imp_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (i, pct) in imp_idx.iter().take(n.min(10)) {
            if *pct > 0.0 {
                let label = col_names.get(*i).map(|s| s.as_str()).unwrap_or("?");
                println!("  {}: {}%", label, fmt_num(*pct));
            }
        }
    }

    // Serialize all trees for predict()
    let mut all_feat = Vec::new();
    let mut all_thresh = Vec::new();
    let mut all_pred = Vec::new();
    let mut all_leaf = Vec::new();
    let mut all_left = Vec::new();
    let mut all_right = Vec::new();
    let mut tree_offsets = Vec::new();
    for tree in &trees {
        tree_offsets.push(all_feat.len() as f64);
        serialize_tree(tree, &mut all_feat, &mut all_thresh, &mut all_pred, &mut all_leaf, &mut all_left, &mut all_right);
    }

    let pred_vals: Vec<Real> = final_preds.iter().map(|p| Some(*p)).collect();
    let imp_vals: Vec<Real> = importance.iter().map(|&c| Some(c as f64 / total_splits.max(1) as f64)).collect();
    let mut imp_attrs = Attrs::default();
    imp_attrs.names = Some(col_names.iter().map(|s| Arc::from(s.as_str())).collect());

    let mut fields = HashMap::new();
    fields.insert(Arc::from("predictions"), RVal::Numeric(pred_vals.into(), Attrs::default()));
    fields.insert(Arc::from("f.values"), rnums(&f_vals));
    fields.insert(Arc::from("ntrees"), rnum(ntrees as f64));
    fields.insert(Arc::from("learning_rate"), rnum(lr));
    fields.insert(Arc::from("loss"), rstr(&loss_type));
    fields.insert(Arc::from("train.loss"), rnums(&train_losses));
    fields.insert(Arc::from("importance"), RVal::Numeric(imp_vals.into(), imp_attrs));
    fields.insert(Arc::from("xnames"),
        RVal::Character(col_names.iter().map(|s| Some(Arc::from(s.as_str()))).collect(), Attrs::default()));
    fields.insert(Arc::from("_tree_offsets"), rnums(&tree_offsets));
    fields.insert(Arc::from("_tree_feat"), rnums(&all_feat.iter().map(|&x| x as f64).collect::<Vec<_>>()));
    fields.insert(Arc::from("_tree_thresh"), rnums(&all_thresh));
    fields.insert(Arc::from("_tree_pred"), rnums(&all_pred));
    fields.insert(Arc::from("_tree_leaf"),
        rnums(&all_leaf.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect::<Vec<_>>()));
    fields.insert(Arc::from("_tree_left"),
        rnums(&all_left.iter().map(|&x| x as f64).collect::<Vec<_>>()));
    fields.insert(Arc::from("_tree_right"),
        rnums(&all_right.iter().map(|&x| x as f64).collect::<Vec<_>>()));

    Ok(RVal::TypeInstance(TypeInstance {
        type_name: Arc::from("gbm"),
        fields,
    }))
}

/// K-means clustering. Per-point centroid assignment is parallelized via
/// `kernel::par_for(Op::PerPointDistance, m, ...)`. Centroid recompute and
/// convergence check stay sequential — they're cheap O(m).
pub fn bi_kmeans(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let arg0 = a.first().map(|e| &e.value).unwrap_or(&RVal::Null);
    let mat = match arg0 {
        RVal::Matrix(m) => m.clone(),
        RVal::DataFrame(df) => {
            let nrow = df.nrow();
            let mut data = Vec::new();
            let mut names: Vec<Arc<str>> = Vec::new();
            let mut ncol = 0;
            for (n, col) in &df.columns {
                if let Ok(vals) = col.as_reals() {
                    let nums: Vec<f64> = vals.into_iter().filter_map(|x| x).collect();
                    if nums.len() == nrow { data.extend(nums); names.push(n.clone()); ncol += 1; }
                }
            }
            let mut m = Matrix::new(data, nrow, ncol);
            m.col_names = Some(names);
            m
        }
        _ => return Err(R2Err { msg: "kmeans() needs a matrix or data.frame".into(), kind: ErrKind::Runtime }),
    };
    let col_names: Vec<String> = match &mat.col_names {
        Some(ns) if ns.len() == mat.ncol => ns.iter().map(|s| s.to_string()).collect(),
        _ => (0..mat.ncol).map(|i| format!("X{}", i + 1)).collect(),
    };

    let k_arg = gn(a, "centers").or_else(|| Some(a.get(1).map(|e| e.value.clone()).unwrap_or(RVal::Null)));
    let k = k_arg.and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(2.0) as usize;
    let max_iter = gn(a, "iter.max").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(100.0) as usize;

    let (m, n) = (mat.nrow, mat.ncol);
    if k == 0 || k > m {
        return Err(R2Err { msg: "kmeans: k must be between 1 and nrow".into(), kind: ErrKind::Runtime });
    }

    // Initialize centroids with **evenly-spaced** rows rather than the
    // first k. Previously rows 0..k were used; when those rows happened
    // to be in the same true cluster (e.g. the canonical 6-point demo
    // `(1,1) (1.2,0.8) (0.8,1.1) (5,5) (5.2,4.8) (4.9,5.1)`), every
    // point assigned to cluster 0 on iteration 1, matched the initial
    // all-zero `cluster` vector, the loop broke before sizes/centroids
    // were recomputed, and the printout showed `sizes 0, 0`.
    let mut centroids = vec![0.0; k * n];
    for c in 0..k {
        let row = (c * m / k).min(m - 1);
        for j in 0..n { centroids[c * n + j] = mat.get(row, j); }
    }

    // Initialise `cluster` to a value the first iteration cannot reach
    // (usize::MAX) so the convergence check never fires on a trivial
    // first-iteration match that wasn't a real fixed point.
    let mut cluster = vec![usize::MAX; m];
    let mut sizes = vec![0usize; k];
    let nrow = mat.nrow;

    for _iter in 0..max_iter {
        let old_cluster = cluster.clone();

        // Per-point centroid assignment — kernel-dispatched.
        let mat_data = mat.data.clone();
        let centroids_snap = centroids.clone();
        let new_assignments: Vec<usize> = r2_kernel::par_for(
            r2_oracle::Op::PerPointDistance, m, move |i| {
                let mut best_c = 0;
                let mut best_dist = f64::INFINITY;
                for c in 0..k {
                    let mut dist = 0.0;
                    for j in 0..n {
                        let d = mat_data[j * nrow + i] - centroids_snap[c * n + j];
                        dist += d * d;
                    }
                    if dist < best_dist { best_dist = dist; best_c = c; }
                }
                best_c
            }
        );
        for i in 0..m { cluster[i] = new_assignments[i]; }

        let converged = cluster == old_cluster;

        // Recompute centroids + sizes (sequential — O(m), cheap).
        // Done unconditionally so the final `sizes` and `centroids`
        // always reflect the converged assignment, even when the loop
        // breaks on the very first iteration.
        for v in centroids.iter_mut() { *v = 0.0; }
        for s in sizes.iter_mut() { *s = 0; }
        for i in 0..m {
            let c = cluster[i];
            sizes[c] += 1;
            for j in 0..n { centroids[c * n + j] += mat.get(i, j); }
        }
        for c in 0..k {
            if sizes[c] > 0 {
                for j in 0..n { centroids[c * n + j] /= sizes[c] as f64; }
            }
        }

        if converged { break; }
    }

    // Within-cluster + total sum of squares
    let mut withinss = vec![0.0; k];
    let mut totss = 0.0;
    let global_mean: Vec<f64> = (0..n)
        .map(|j| (0..m).map(|i| mat.get(i, j)).sum::<f64>() / m as f64).collect();
    for i in 0..m {
        let c = cluster[i];
        for j in 0..n {
            let d = mat.get(i, j) - centroids[c * n + j];
            withinss[c] += d * d;
            let dg = mat.get(i, j) - global_mean[j];
            totss += dg * dg;
        }
    }
    let total_withinss: f64 = withinss.iter().sum();
    let betweenss = totss - total_withinss;

    println!("K-means clustering with {} clusters of sizes {}", k,
        sizes.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "));
    println!("\nCluster means:");
    print!("     ");
    for j in 0..n { print!("  {:>10}", col_names.get(j).map(|s| s.as_str()).unwrap_or("?")); }
    println!();
    for c in 0..k {
        print!("  [{}]", c + 1);
        for j in 0..n { print!("  {:>10}", fmt_num(centroids[c * n + j])); }
        println!();
    }
    println!("\nWithin cluster sum of squares by cluster:");
    for w in &withinss { print!("  {}", fmt_num(*w)); }
    println!("\n(between_SS / total_SS = {}%)", fmt_num(betweenss / totss * 100.0));

    let cluster_vals: Vec<Integer> = cluster.iter().map(|c| Some((*c + 1) as i32)).collect();
    let mut centers_mat = Matrix::new(centroids, k, n);
    centers_mat.col_names = Some(col_names.iter().map(|s| Arc::from(s.as_str())).collect());

    let mut fields = HashMap::new();
    fields.insert(Arc::from("cluster"), RVal::Integer(cluster_vals.into(), Attrs::default()));
    fields.insert(Arc::from("centers"), RVal::Matrix(centers_mat));
    fields.insert(Arc::from("withinss"), rnums(&withinss));
    fields.insert(Arc::from("tot.withinss"), rnum(total_withinss));
    fields.insert(Arc::from("betweenss"), rnum(betweenss));
    fields.insert(Arc::from("totss"), rnum(totss));
    fields.insert(Arc::from("size"),
        RVal::Integer(sizes.iter().map(|s| Some(*s as i32)).collect(), Attrs::default()));

    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("kmeans"), fields }))
}

/// k-fold cross-validation. Folds run in parallel via
/// `kernel::par_for(Op::KFoldCV, k, ...)`. Each fold trains a model on
/// k-1 partitions, evaluates on the held-out fold, returns MSE.
pub fn bi_cv(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let arg0 = a.first().map(|e| &e.value).unwrap_or(&RVal::Null);
    let mat = match arg0 {
        RVal::Matrix(m) => m.clone(),
        _ => return Err(R2Err { msg: "cv: x must be matrix".into(), kind: ErrKind::Runtime }),
    };
    let y_arg = a.get(1).map(|e| e.value.clone()).unwrap_or(RVal::Null);
    let y: Vec<f64> = y_arg.as_reals()?.into_iter().filter_map(|x| x).collect();
    let model_type = gn(a, "model").map(|v| rval_to_string(&v)).unwrap_or("lm".into());
    let k = gn(a, "k").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(5.0) as usize;

    let (m, n) = (mat.nrow, mat.ncol);
    if y.len() != m {
        return Err(R2Err { msg: "cv: y length must match x rows".into(), kind: ErrKind::Runtime });
    }
    if k < 2 || k > m {
        return Err(R2Err { msg: "cv: k must be between 2 and nrow".into(), kind: ErrKind::Runtime });
    }

    println!("{}-fold cross-validation (model: {})", k, model_type);
    let fold_size = m / k;

    // Per-fold worker — pure function over its inputs, no shared mutable state.
    // Captured by clone for Send+Sync compliance in the parallel path.
    let mat_data = mat.data.clone();
    let mat_nrow = mat.nrow;
    let y_clone = y.clone();
    let model = model_type.clone();

    let run_fold = move |fold: usize| -> Result<f64, R2Err> {
        let test_start = fold * fold_size;
        let test_end = if fold == k - 1 { m } else { test_start + fold_size };

        // Split into train/test (row-major intermediate)
        let mut train_x = Vec::new();
        let mut train_y = Vec::new();
        let mut test_x = Vec::new();
        let mut test_y = Vec::new();
        let mut train_rows = 0;
        let mut test_rows = 0;

        for i in 0..m {
            // mat is column-major: mat.get(i,j) = mat_data[j*nrow + i]
            if i >= test_start && i < test_end {
                for c in 0..n { test_x.push(mat_data[c * mat_nrow + i]); }
                test_y.push(y_clone[i]);
                test_rows += 1;
            } else {
                for c in 0..n { train_x.push(mat_data[c * mat_nrow + i]); }
                train_y.push(y_clone[i]);
                train_rows += 1;
            }
        }

        // Rebuild matrices column-major
        let mut train_cm = vec![0.0; train_rows * n];
        let mut test_cm = vec![0.0; test_rows * n];
        for c in 0..n {
            for r in 0..train_rows { train_cm[c * train_rows + r] = train_x[r * n + c]; }
            for r in 0..test_rows { test_cm[c * test_rows + r] = test_x[r * n + c]; }
        }

        let metric = match model.as_str() {
            "lm" => {
                let mut x_int = vec![1.0; train_rows];
                x_int.extend(&train_cm);
                let p = n + 1;
                let coeffs = r2_linalg::dlsq_fused(train_rows, p, &x_int, &train_y)
                    .map_err(|e| R2Err {
                        msg: format!("cv lm failed: {}", e),
                        kind: ErrKind::Runtime,
                    })?;
                let mut preds = vec![0.0; test_rows];
                for i in 0..test_rows {
                    preds[i] = coeffs[0];
                    for j in 0..n { preds[i] += coeffs[j + 1] * test_cm[j * test_rows + i]; }
                }
                preds.iter().zip(test_y.iter())
                    .map(|(p, y)| (p - y).powi(2)).sum::<f64>() / test_rows as f64
            }
            "rf" => {
                let train_mat_data = train_cm.clone();
                let mut all_preds = vec![vec![0.0; test_rows]; 50];
                for t in 0..50 {
                    let mut bmask = vec![false; train_rows];
                    for _ in 0..train_rows {
                        bmask[(next_random() * train_rows as f64) as usize % train_rows] = true;
                    }
                    let tree = build_tree(&train_mat_data, &train_y, train_rows, n, &bmask, 5, 2, 0, false);
                    for i in 0..test_rows {
                        all_preds[t][i] = tree_predict_one(&tree, &test_cm, test_rows, i);
                    }
                }
                let mut preds = vec![0.0; test_rows];
                for i in 0..test_rows {
                    preds[i] = all_preds.iter().map(|p| p[i]).sum::<f64>() / 50.0;
                }
                preds.iter().zip(test_y.iter())
                    .map(|(p, y)| (p - y).powi(2)).sum::<f64>() / test_rows as f64
            }
            other => {
                return Err(R2Err {
                    msg: format!("cv: model '{}' not supported (use lm, rf)", other),
                    kind: ErrKind::Runtime,
                });
            }
        };

        println!("  Fold {}: MSE = {}", fold + 1, fmt_num(metric));
        Ok(metric)
    };

    // Kernel-dispatched parallel folds. Each fold returns Result; we
    // collect, then surface the first error if any.
    let fold_results: Vec<Result<f64, R2Err>> =
        r2_kernel::par_for(r2_oracle::Op::KFoldCV, k, run_fold);

    let mut fold_metrics = Vec::with_capacity(k);
    for r in fold_results {
        fold_metrics.push(r?);
    }

    let avg = fold_metrics.iter().sum::<f64>() / k as f64;
    let sd = (fold_metrics.iter().map(|x| (x - avg).powi(2)).sum::<f64>() / (k - 1) as f64).sqrt();
    println!("\nAverage MSE: {} (±{})", fmt_num(avg), fmt_num(sd));

    let mut fields = HashMap::new();
    fields.insert(Arc::from("mse"), rnums(&fold_metrics));
    fields.insert(Arc::from("mean.mse"), rnum(avg));
    fields.insert(Arc::from("sd.mse"), rnum(sd));
    fields.insert(Arc::from("k"), rnum(k as f64));
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("cv"), fields }))
}

/// k-nearest-neighbours classification. No parallelism — direct migration.
/// (Future K.5 could par_for over test points; not now.)
pub fn bi_knn(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let arg0 = a.first().map(|e| &e.value).unwrap_or(&RVal::Null);
    let train = match arg0 {
        RVal::Matrix(m) => m.clone(),
        _ => return Err(R2Err { msg: "knn: train must be matrix".into(), kind: ErrKind::Runtime }),
    };
    let arg1 = a.get(1).map(|e| &e.value).unwrap_or(&RVal::Null);
    let test = match arg1 {
        RVal::Matrix(m) => m.clone(),
        _ => return Err(R2Err { msg: "knn: test must be matrix".into(), kind: ErrKind::Runtime }),
    };
    let labels: Vec<f64> = a.get(2).map(|e| e.value.clone()).unwrap_or(RVal::Null)
        .as_reals()?.into_iter().filter_map(|x| x).collect();
    let k = gn(a, "k").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(3.0) as usize;

    let (n_train, p) = (train.nrow, train.ncol);
    let n_test = test.nrow;
    if labels.len() != n_train {
        return Err(R2Err { msg: "knn: labels length must match train rows".into(), kind: ErrKind::Runtime });
    }

    let mut predictions = Vec::with_capacity(n_test);
    for i in 0..n_test {
        let mut dists: Vec<(f64, usize)> = (0..n_train).map(|j| {
            let mut d = 0.0;
            for col in 0..p { let diff = test.get(i, col) - train.get(j, col); d += diff * diff; }
            (d, j)
        }).collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut votes: HashMap<i64, usize> = HashMap::new();
        for idx in 0..k.min(n_train) {
            let label = labels[dists[idx].1] as i64;
            *votes.entry(label).or_insert(0) += 1;
        }
        let best = votes.into_iter().max_by_key(|(_, c)| *c).map(|(l, _)| l).unwrap_or(0);
        predictions.push(Some(best as f64));
    }

    println!("KNN: classified {} points using k={}", n_test, k);
    Ok(RVal::Numeric(predictions.into(), Attrs::default()))
}

/// Gaussian Naive Bayes classifier. No parallelism — direct migration.
pub fn bi_naive_bayes(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let arg0 = a.first().map(|e| &e.value).unwrap_or(&RVal::Null);
    let mat = match arg0 {
        RVal::Matrix(m) => m.clone(),
        _ => return Err(R2Err { msg: "naive.bayes: x must be matrix".into(), kind: ErrKind::Runtime }),
    };
    let labels: Vec<f64> = a.get(1).map(|e| e.value.clone()).unwrap_or(RVal::Null)
        .as_reals()?.into_iter().filter_map(|x| x).collect();

    let (m, n) = (mat.nrow, mat.ncol);
    if labels.len() != m {
        return Err(R2Err { msg: "naive.bayes: labels length must match rows".into(), kind: ErrKind::Runtime });
    }

    let mut classes: Vec<f64> = labels.clone();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    classes.dedup();

    let k = classes.len();
    let mut class_means = vec![0.0; k * n];
    let mut class_vars = vec![0.0; k * n];
    let mut class_counts = vec![0usize; k];
    let mut class_priors = vec![0.0; k];

    for i in 0..m {
        let ci = classes.iter().position(|&c| (c - labels[i]).abs() < 1e-10).unwrap_or(0);
        class_counts[ci] += 1;
        for j in 0..n { class_means[ci * n + j] += mat.get(i, j); }
    }
    for ci in 0..k {
        class_priors[ci] = class_counts[ci] as f64 / m as f64;
        for j in 0..n { class_means[ci * n + j] /= class_counts[ci] as f64; }
    }
    for i in 0..m {
        let ci = classes.iter().position(|&c| (c - labels[i]).abs() < 1e-10).unwrap_or(0);
        for j in 0..n {
            let d = mat.get(i, j) - class_means[ci * n + j];
            class_vars[ci * n + j] += d * d;
        }
    }
    for ci in 0..k {
        for j in 0..n { class_vars[ci * n + j] /= (class_counts[ci] - 1).max(1) as f64; }
    }

    println!("Naive Bayes classifier: {} classes, {} features", k, n);
    for ci in 0..k {
        println!("  Class {}: prior={}, n={}", classes[ci], fmt_num(class_priors[ci]), class_counts[ci]);
    }

    let mut fields = HashMap::new();
    fields.insert(Arc::from("classes"), rnums(&classes));
    fields.insert(Arc::from("priors"), rnums(&class_priors));
    fields.insert(Arc::from("means"), RVal::Matrix(Matrix::new(class_means, k, n)));
    fields.insert(Arc::from("vars"), RVal::Matrix(Matrix::new(class_vars, k, n)));

    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("naive.bayes"), fields }))
}

/// Principal Component Analysis. No parallelism — uses r2-linalg's
/// symmetric-eigendecomposition kernel for the heavy math.
pub fn bi_prcomp(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let arg0 = a.first().map(|e| &e.value).unwrap_or(&RVal::Null);
    let mat = match arg0 {
        RVal::Matrix(m) => m.clone(),
        RVal::DataFrame(df) => {
            let nrow = df.nrow();
            let mut data = Vec::new();
            let mut col_names: Vec<Arc<str>> = Vec::new();
            let mut ncol_num = 0;
            for (name, col) in &df.columns {
                if let Ok(vals) = col.as_reals() {
                    let nums: Vec<f64> = vals.into_iter().filter_map(|x| x).collect();
                    if nums.len() == nrow {
                        data.extend(nums);
                        col_names.push(name.clone());
                        ncol_num += 1;
                    }
                }
            }
            let mut m = Matrix::new(data, nrow, ncol_num);
            m.col_names = Some(col_names);
            m
        }
        _ => return Err(R2Err { msg: "prcomp() needs a matrix or data.frame".into(), kind: ErrKind::Runtime }),
    };
    let _col_labels: Vec<String> = match &mat.col_names {
        Some(ns) if ns.len() == mat.ncol => ns.iter().map(|s| s.to_string()).collect(),
        _ => (0..mat.ncol).map(|i| format!("X{}", i + 1)).collect(),
    };

    let center = gn(a, "center").and_then(|v| v.as_logicals().ok())
        .map(|v| v[0] == Some(true)).unwrap_or(true);
    let scale = gn(a, "scale.").or_else(|| gn(a, "scale"))
        .and_then(|v| v.as_logicals().ok()).map(|v| v[0] == Some(true)).unwrap_or(false);

    let (m, n) = (mat.nrow, mat.ncol);

    let mut x = mat.data.clone();
    let means = mat.col_means();
    let mut sds = vec![1.0; n];
    if scale {
        for c in 0..n {
            let col_start = c * m;
            let mean = means[c];
            let mut ss = 0.0;
            for r in 0..m { ss += (x[col_start + r] - mean).powi(2); }
            sds[c] = (ss / (m - 1) as f64).sqrt();
            if sds[c] < 1e-15 { sds[c] = 1.0; }
        }
    }
    if center {
        for c in 0..n {
            let col_start = c * m;
            for r in 0..m {
                x[col_start + r] -= means[c];
                if scale { x[col_start + r] /= sds[c]; }
            }
        }
    }

    // Covariance via Kahan summation
    let mut cov_data = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            let mut sum = 0.0;
            let mut comp = 0.0;
            for r in 0..m {
                let prod = x[i * m + r] * x[j * m + r];
                let y_k = prod - comp;
                let t = sum + y_k;
                comp = (t - sum) - y_k;
                sum = t;
            }
            let cov = sum / (m - 1) as f64;
            cov_data[j * n + i] = cov;
            cov_data[i * n + j] = cov;
        }
    }

    // Tier 1: eigenvectors of the covariance matrix ARE the rotation matrix.
    // Switch to dsyev_full so `$rotation` is now populated.
    let (eigenvalues, rotation_data) = r2_linalg::dsyev_full(n, &cov_data)
        .map_err(|e| R2Err { msg: format!("PCA failed: {}", e), kind: ErrKind::Runtime })?;

    let eigenvalues: Vec<f64> = eigenvalues.iter()
        .map(|v| if *v < 0.0 && v.abs() < 1e-10 { 0.0 } else { *v }).collect();
    let sdev: Vec<f64> = eigenvalues.iter().map(|v| v.max(0.0).sqrt()).collect();
    let total_var: f64 = eigenvalues.iter().filter(|v| **v > 0.0).sum();
    let prop_var: Vec<f64> = eigenvalues.iter()
        .map(|v| if total_var > 0.0 { v.max(0.0) / total_var } else { 0.0 }).collect();

    // No auto-print: R's `p <- prcomp(X)` is invisible. Users call
    // `summary(p)` for the formatted importance-of-components table,
    // or access `p$sdev` / `p$rotation` directly.
    let _ = &sdev;
    let _ = &prop_var;

    let mut rotation_mat = Matrix::new(rotation_data, n, n);
    rotation_mat.col_names = Some(
        (0..n).map(|i| Arc::from(format!("PC{}", i + 1).as_str())).collect()
    );

    let mut fields = HashMap::new();
    fields.insert(Arc::from("sdev"), rnums(&sdev));
    fields.insert(Arc::from("eigenvalues"), rnums(&eigenvalues));
    fields.insert(Arc::from("prop.variance"), rnums(&prop_var));
    fields.insert(Arc::from("center"), rnums(&means));
    fields.insert(Arc::from("rotation"), RVal::Matrix(rotation_mat));
    if scale { fields.insert(Arc::from("scale"), rnums(&sds)); }

    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("prcomp"), fields }))
}
