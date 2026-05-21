// Regenerates the `.r2d` binary dataset files from the current
// (inline or already-loaded) dataset functions in r2-base.
//
// Run with:
//   cargo test -p r2-base --release --test gen_datasets -- --ignored
//
// The test is `#[ignore]` so it does not run by default; it is a
// one-shot generator, not part of the regular test suite.

use r2_base::{iris, mtcars, airquality, tooth_growth, faithful};
use r2_base::r2d::write_r2d;
use r2_types::RVal;

fn dump(name: &str, v: RVal) {
    let df = match v {
        RVal::DataFrame(d) => d,
        _ => panic!("{}: not a data.frame", name),
    };
    let bytes = write_r2d(&df);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join(format!("{}.r2d", name));
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    eprintln!("wrote {} ({} bytes)", path.display(), bytes.len());
}

#[test]
#[ignore]
fn generate_all() {
    dump("iris", iris());
    dump("mtcars", mtcars());
    dump("airquality", airquality());
    dump("tooth_growth", tooth_growth());
    dump("faithful", faithful());
}
