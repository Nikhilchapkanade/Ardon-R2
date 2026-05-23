// R2 Package Manager — stub for standalone crate
// Full package management is wired into the engine's FunctionRegistry.
// This crate provides types for package metadata.


pub struct PackageInfo {
    pub name: String,
    pub version: String,
    pub exports: Vec<String>,
    pub depends: Vec<String>,
    pub tier: String,
}
