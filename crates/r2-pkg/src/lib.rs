//! Ardon-R2 addon package system — v0.3.0.
//!
//! This crate handles the file/manifest/path side of R2's package
//! system. The actual loading (parsing + evaluating .r2 source files)
//! is orchestrated by `r2-engine` because it requires the Parser and
//! Engine — keeping that dependency direction one-way avoids cycles.
//!
//! Design (deliberately minimal, no online registry):
//!
//! * Packages live under `~/.r2/packages/<name>/` after installation.
//! * Each package has a `package.r2` manifest at its root with
//!   metadata in R2's own syntax:
//!     `package_name <- "mymath"`
//!     `package_version <- "0.1.0"`
//!     `package_exports <- c("add_one", "double_it")`
//! * R2 source files live under `<pkg_dir>/R/` and are sourced in
//!   alphabetical order when `library("name")` is called.
//! * Installation paths: local directory, zip file, GitHub clone.
//!   Only the directory path is implemented here; zip/git are thin
//!   shell-outs that live in r2-engine to keep this crate dep-free.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

// Kept for back-compat with the old stub layer that the engine
// references; new code should use `Manifest`.
pub struct PackageInfo {
    pub name: String,
    pub version: String,
    pub exports: Vec<String>,
    pub depends: Vec<String>,
    pub tier: String,
}

/// Errors that can arise during package operations.
#[derive(Debug)]
pub enum PkgError {
    NotFound(String),
    Io(io::Error),
    BadManifest(String),
    BadArg(String),
}

impl std::fmt::Display for PkgError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PkgError::NotFound(s)    => write!(f, "package not found: {}", s),
            PkgError::Io(e)          => write!(f, "io error: {}", e),
            PkgError::BadManifest(s) => write!(f, "bad manifest: {}", s),
            PkgError::BadArg(s)      => write!(f, "{}", s),
        }
    }
}

impl From<io::Error> for PkgError {
    fn from(e: io::Error) -> Self { PkgError::Io(e) }
}

/// Returns the absolute path to the user-level package directory
/// (`~/.r2/packages/`). Creates it if missing. Honors `R2_PKG_DIR`
/// env-var override.
pub fn pkg_root() -> Result<PathBuf, PkgError> {
    if let Ok(custom) = std::env::var("R2_PKG_DIR") {
        let p = PathBuf::from(custom);
        fs::create_dir_all(&p)?;
        return Ok(p);
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .ok_or_else(|| PkgError::BadArg(
            "could not locate the user home directory (HOME/USERPROFILE unset). \
             Set R2_PKG_DIR to override.".into()
        ))?;
    let dir = PathBuf::from(home).join(".r2").join("packages");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Absolute path to one installed package's root directory.
pub fn pkg_dir(name: &str) -> Result<PathBuf, PkgError> {
    validate_name(name)?;
    Ok(pkg_root()?.join(name))
}

fn validate_name(name: &str) -> Result<(), PkgError> {
    if name.is_empty() {
        return Err(PkgError::BadArg("package name is empty".into()));
    }
    if name.len() > 100 {
        return Err(PkgError::BadArg("package name too long (max 100 chars)".into()));
    }
    for ch in name.chars() {
        if !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.') {
            return Err(PkgError::BadArg(format!(
                "package name '{}' contains invalid character '{}'. \
                 Use letters, digits, '_', '-', '.' only.", name, ch
            )));
        }
    }
    if name.starts_with('.') || name == ".." {
        return Err(PkgError::BadArg(
            format!("package name '{}' is reserved", name)
        ));
    }
    Ok(())
}

/// List all installed packages — returns a sorted vector of names.
pub fn list_installed() -> Result<Vec<String>, PkgError> {
    let root = pkg_root()?;
    let mut names: Vec<String> = Vec::new();
    if !root.exists() { return Ok(names); }
    for entry in fs::read_dir(&root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if let Some(n) = entry.file_name().to_str() {
                if !n.starts_with('.') {
                    names.push(n.to_string());
                }
            }
        }
    }
    names.sort();
    Ok(names)
}

/// Parsed metadata from a package's manifest.
#[derive(Debug, Clone, Default)]
pub struct Manifest {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub exports: Vec<String>,
}

/// Read a manifest from a package directory (looks for `package.r2`).
pub fn read_manifest(dir: &Path) -> Result<Manifest, PkgError> {
    let manifest_path = dir.join("package.r2");
    if !manifest_path.exists() {
        return Err(PkgError::BadManifest(format!(
            "no package.r2 found in {} — every R2 package must have a root manifest",
            dir.display()
        )));
    }
    let source = fs::read_to_string(&manifest_path)?;
    parse_manifest(&source)
}

/// Pure parser for the manifest fields. Recognizes a small subset of
/// R2 syntax: scalar `package_<key> <- "..."` assignments and
/// `package_exports <- c("a", "b")`.
pub fn parse_manifest(source: &str) -> Result<Manifest, PkgError> {
    let mut m = Manifest::default();
    for raw in source.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') { continue; }

        // Scalar string fields.
        let scalars: [(&str, fn(&mut Manifest, String)); 5] = [
            ("package_name",        |m, s| m.name = s),
            ("package_version",     |m, s| m.version = s),
            ("package_description", |m, s| m.description = s),
            ("package_author",      |m, s| m.author = s),
            ("package_license",     |m, s| m.license = s),
        ];
        let mut matched = false;
        for (key, setter) in &scalars {
            if let Some(rest) = strip_assign(line, key) {
                if let Some(s) = parse_string_literal(rest.trim()) {
                    setter(&mut m, s);
                }
                matched = true;
                break;
            }
        }
        if matched { continue; }

        if let Some(rest) = strip_assign(line, "package_exports") {
            let r = rest.trim();
            if let Some(inner) = r.strip_prefix("c(").and_then(|s| s.strip_suffix(')')) {
                let mut names = Vec::new();
                for raw_n in inner.split(',') {
                    if let Some(s) = parse_string_literal(raw_n.trim()) {
                        names.push(s);
                    }
                }
                m.exports = names;
            } else if let Some(s) = parse_string_literal(r) {
                m.exports = vec![s];
            }
        }
    }
    if m.name.is_empty() {
        return Err(PkgError::BadManifest(
            "manifest is missing 'package_name <- \"...\"'".into()
        ));
    }
    Ok(m)
}

fn strip_assign<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let rest = line.strip_prefix(key)?.trim_start();
    if let Some(r) = rest.strip_prefix("<-") {
        return Some(r.trim_start());
    }
    if let Some(r) = rest.strip_prefix('=') {
        return Some(r.trim_start());
    }
    None
}

fn parse_string_literal(s: &str) -> Option<String> {
    let s = s.trim();
    let bytes = s.as_bytes();
    if bytes.len() < 2 { return None; }
    let quote = bytes[0];
    if quote != b'"' && quote != b'\'' { return None; }
    let mut out = String::new();
    let mut i = 1;
    while i < bytes.len() {
        let b = bytes[i];
        if b == quote { return Some(out); }
        if b == b'\\' && i + 1 < bytes.len() {
            let esc = bytes[i + 1];
            out.push(match esc {
                b'n' => '\n', b't' => '\t', b'r' => '\r',
                b'\\' => '\\', b'"' => '"', b'\'' => '\'',
                other => other as char,
            });
            i += 2;
        } else {
            out.push(b as char);
            i += 1;
        }
    }
    None
}

/// Install a package from a local directory. Validates the manifest,
/// then recursively copies into `~/.r2/packages/<name>/`. Overwrites
/// an existing install with the same name.
pub fn install_from_dir(src_dir: &Path) -> Result<Manifest, PkgError> {
    if !src_dir.exists() || !src_dir.is_dir() {
        return Err(PkgError::NotFound(format!(
            "source directory {} does not exist or is not a directory",
            src_dir.display()
        )));
    }
    let manifest = read_manifest(src_dir)?;
    let dest = pkg_dir(&manifest.name)?;
    if dest.exists() {
        fs::remove_dir_all(&dest)?;
    }
    copy_dir_recursive(src_dir, &dest)?;
    Ok(manifest)
}

/// List the R2 source files (`*.r2`) under `<pkg_dir>/R/` in
/// alphabetical order.
pub fn package_source_files(pkg: &str) -> Result<Vec<PathBuf>, PkgError> {
    let dir = pkg_dir(pkg)?;
    let src_dir = dir.join("R");
    if !src_dir.exists() {
        return Ok(Vec::new());
    }
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(&src_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("r2") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

/// Remove an installed package's files.
pub fn uninstall(name: &str) -> Result<(), PkgError> {
    let dir = pkg_dir(name)?;
    if !dir.exists() {
        return Err(PkgError::NotFound(format!("'{}' is not installed", name)));
    }
    fs::remove_dir_all(&dir)?;
    Ok(())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dst_path = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_recursive(&entry.path(), &dst_path)?;
        } else if file_type.is_file() {
            fs::copy(entry.path(), &dst_path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_parses_basic_fields() {
        let src = r#"
# This is the example package manifest.
package_name <- "mymath"
package_version <- "0.1.0"
package_description <- "Toy helpers."
package_author <- "Devendra"
package_license <- "MIT"
package_exports <- c("add_one", "double_it")
"#;
        let m = parse_manifest(src).unwrap();
        assert_eq!(m.name, "mymath");
        assert_eq!(m.version, "0.1.0");
        assert_eq!(m.description, "Toy helpers.");
        assert_eq!(m.author, "Devendra");
        assert_eq!(m.license, "MIT");
        assert_eq!(m.exports, vec!["add_one", "double_it"]);
    }

    #[test]
    fn manifest_handles_equals_and_single_quotes() {
        let src = "package_name = 'foo'\npackage_exports = c('bar')\n";
        let m = parse_manifest(src).unwrap();
        assert_eq!(m.name, "foo");
        assert_eq!(m.exports, vec!["bar"]);
    }

    #[test]
    fn manifest_without_name_field_errors() {
        let src = "package_version <- \"0.1.0\"\n";
        assert!(parse_manifest(src).is_err());
    }

    #[test]
    fn validate_name_rejects_path_traversal() {
        assert!(validate_name("..").is_err());
        assert!(validate_name("foo/bar").is_err());
        assert!(validate_name("foo\\bar").is_err());
        assert!(validate_name(".hidden").is_err());
        assert!(validate_name("").is_err());
    }

    #[test]
    fn validate_name_accepts_typical_names() {
        for n in &["mymath", "r2-survival", "my_pkg", "Hotelling2"] {
            assert!(validate_name(n).is_ok(), "should accept '{}'", n);
        }
    }
}
