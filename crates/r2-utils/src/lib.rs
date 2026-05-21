// R2 Utils Library — standalone utility functions
// IO, inspection, and helper functions.
// Engine builtins handle these at runtime; this crate
// provides them for use by other crates.

use r2_types::*;
use std::sync::Arc;

pub fn read_csv_to_dataframe(content: &str, header: bool) -> Result<DataFrame, String> {
    let mut lines = content.lines();
    let col_names: Vec<String> = if header {
        lines.next().map(|l| l.split(',').map(|s| s.trim().trim_matches('"').to_string()).collect())
            .unwrap_or_default()
    } else { Vec::new() };

    let mut raw_cols: Vec<Vec<String>> = vec![Vec::new(); col_names.len().max(1)];
    for line in lines {
        if line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split(',').collect();
        if raw_cols.len() < fields.len() { raw_cols.resize(fields.len(), Vec::new()); }
        for (i, field) in fields.iter().enumerate() {
            if i < raw_cols.len() { raw_cols[i].push(field.trim().trim_matches('"').to_string()); }
        }
    }

    let mut columns = Vec::new();
    for (i, col_data) in raw_cols.iter().enumerate() {
        let name = if i < col_names.len() { Arc::from(col_names[i].as_str()) }
                   else { Arc::from(format!("V{}", i + 1).as_str()) };
        // Try numeric first
        let all_numeric = col_data.iter().all(|s| s.is_empty() || s == "NA" || s.parse::<f64>().is_ok());
        let has_any_num = col_data.iter().any(|s| s.parse::<f64>().is_ok());
        if all_numeric && has_any_num {
            let nums: Vec<Real> = col_data.iter().map(|s| {
                if s.is_empty() || s == "NA" { None } else { s.parse().ok() }
            }).collect();
            columns.push((name, RVal::Numeric(nums.into(), Attrs::default())));
        } else {
            let strs: Vec<Character> = col_data.iter().map(|s| {
                if s == "NA" { None } else { Some(Arc::from(s.as_str())) }
            }).collect();
            columns.push((name, RVal::Character(strs, Attrs::default())));
        }
    }
    Ok(DataFrame { columns, row_names: None })
}
