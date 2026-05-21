// `.r2d` v1 — R2 native binary dataset format.
//
// Hand-rolled, no external deps. Little-endian throughout.
//
// Layout
//   0..4    Magic "R2D1"
//   4..6    u16  version (=1)
//   6..10   u32  n_cols
//   10..14  u32  n_rows
//   14..15  u8   has_row_names
//   then n_cols column blocks:
//     u16 name_len, name_len bytes UTF-8 column name,
//     u8 dtype (0=Numeric f64, 1=Integer i32, 2=Logical i8, 3=Character),
//     type-specific payload (see below).
//   then if has_row_names: a Character payload for the row names.
//
// Numeric payload : n_rows * f64 LE, then ceil(n_rows/8) byte validity bitmap.
// Integer payload : n_rows * i32 LE, then ceil(n_rows/8) byte validity bitmap.
// Logical payload : ceil(n_rows/8) byte value bitmap, then ceil(n_rows/8) byte validity bitmap.
// Character payload :
//     u32 total_bytes (string blob size)
//     n_rows * (u32 offset, u32 length)   -- length = 0xFFFFFFFF means NA
//     total_bytes UTF-8 string blob

use r2_types::*;
use std::sync::Arc;

const MAGIC: &[u8; 4] = b"R2D1";
const VERSION: u16 = 1;

const DTYPE_NUMERIC: u8 = 0;
const DTYPE_INTEGER: u8 = 1;
const DTYPE_LOGICAL: u8 = 2;
const DTYPE_CHARACTER: u8 = 3;

// ── Writer ───────────────────────────────────────────────────────────

pub fn write_r2d(df: &DataFrame) -> Vec<u8> {
    let mut out = Vec::with_capacity(64);
    let n_cols = df.columns.len() as u32;
    let n_rows = df.nrow() as u32;
    let has_row_names: u8 = if df.row_names.is_some() { 1 } else { 0 };

    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&n_cols.to_le_bytes());
    out.extend_from_slice(&n_rows.to_le_bytes());
    out.push(has_row_names);

    for (name, col) in &df.columns {
        write_name(&mut out, name);
        write_column(&mut out, col, n_rows as usize);
    }

    if let Some(rn) = &df.row_names {
        let opts: Vec<Character> = rn.iter().map(|a| Some(a.clone())).collect();
        write_character_payload(&mut out, &opts);
    }

    out
}

fn write_name(out: &mut Vec<u8>, name: &str) {
    let bytes = name.as_bytes();
    let len = bytes.len() as u16;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
}

fn write_column(out: &mut Vec<u8>, col: &RVal, n_rows: usize) {
    match col {
        RVal::Numeric(reals, _) => {
            out.push(DTYPE_NUMERIC);
            let v = reals.as_vec();
            assert_eq!(v.len(), n_rows, "numeric column row count mismatch");
            for r in v {
                let f = r.unwrap_or(f64::NAN);
                out.extend_from_slice(&f.to_le_bytes());
            }
            write_validity(out, v.iter().map(|x| x.is_some()), n_rows);
        }
        RVal::Integer(ints, _) => {
            out.push(DTYPE_INTEGER);
            let v = ints.as_vec();
            assert_eq!(v.len(), n_rows, "integer column row count mismatch");
            for r in v {
                let i = r.unwrap_or(0);
                out.extend_from_slice(&i.to_le_bytes());
            }
            write_validity(out, v.iter().map(|x| x.is_some()), n_rows);
        }
        RVal::Logical(logs, _) => {
            out.push(DTYPE_LOGICAL);
            let v = logs.as_vec();
            assert_eq!(v.len(), n_rows, "logical column row count mismatch");
            write_validity(out, v.iter().map(|x| x.unwrap_or(false)), n_rows);
            write_validity(out, v.iter().map(|x| x.is_some()), n_rows);
        }
        RVal::Character(chars, _) => {
            out.push(DTYPE_CHARACTER);
            assert_eq!(chars.len(), n_rows, "character column row count mismatch");
            write_character_payload(out, chars);
        }
        other => panic!("write_r2d: unsupported column type {:?}", std::mem::discriminant(other)),
    }
}

fn write_validity<I: Iterator<Item = bool>>(out: &mut Vec<u8>, bits: I, n_rows: usize) {
    let bytes_len = (n_rows + 7) / 8;
    let mut buf = vec![0u8; bytes_len];
    for (i, b) in bits.enumerate() {
        if b {
            buf[i / 8] |= 1 << (i % 8);
        }
    }
    out.extend_from_slice(&buf);
}

fn write_character_payload(out: &mut Vec<u8>, chars: &[Character]) {
    // First compute total blob bytes
    let total: usize = chars.iter().filter_map(|c| c.as_ref().map(|s| s.len())).sum();
    out.extend_from_slice(&(total as u32).to_le_bytes());
    let mut blob = Vec::with_capacity(total);
    for c in chars {
        match c {
            Some(s) => {
                let offset = blob.len() as u32;
                let len = s.len() as u32;
                out.extend_from_slice(&offset.to_le_bytes());
                out.extend_from_slice(&len.to_le_bytes());
                blob.extend_from_slice(s.as_bytes());
            }
            None => {
                out.extend_from_slice(&0u32.to_le_bytes());
                out.extend_from_slice(&u32::MAX.to_le_bytes());
            }
        }
    }
    out.extend_from_slice(&blob);
}

// ── Reader ───────────────────────────────────────────────────────────

struct Cur<'a> { buf: &'a [u8], pos: usize }

impl<'a> Cur<'a> {
    fn new(buf: &'a [u8]) -> Self { Cur { buf, pos: 0 } }
    fn need(&self, n: usize) -> Result<(), R2Err> {
        if self.pos + n > self.buf.len() {
            Err(err(format!("r2d: unexpected EOF at {} (need {} more, have {})",
                self.pos, n, self.buf.len() - self.pos)))
        } else { Ok(()) }
    }
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], R2Err> {
        self.need(n)?;
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn read_u8(&mut self) -> Result<u8, R2Err> { Ok(self.read_bytes(1)?[0]) }
    fn read_u16(&mut self) -> Result<u16, R2Err> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn read_u32(&mut self) -> Result<u32, R2Err> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_i32(&mut self) -> Result<i32, R2Err> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_f64(&mut self) -> Result<f64, R2Err> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }
}

fn err(msg: String) -> R2Err { R2Err { msg, kind: ErrKind::Runtime } }

pub fn read_r2d(bytes: &[u8]) -> Result<RVal, R2Err> {
    let mut c = Cur::new(bytes);
    let magic = c.read_bytes(4)?;
    if magic != MAGIC {
        return Err(err(format!("r2d: bad magic {:?}, expected R2D1", magic)));
    }
    let version = c.read_u16()?;
    if version != VERSION {
        return Err(err(format!("r2d: unsupported version {}", version)));
    }
    let n_cols = c.read_u32()? as usize;
    let n_rows = c.read_u32()? as usize;
    let has_row_names = c.read_u8()? != 0;

    let mut columns: Vec<(Arc<str>, RVal)> = Vec::with_capacity(n_cols);
    for _ in 0..n_cols {
        let name_len = c.read_u16()? as usize;
        let name_bytes = c.read_bytes(name_len)?;
        let name: Arc<str> = Arc::from(std::str::from_utf8(name_bytes)
            .map_err(|e| err(format!("r2d: bad UTF-8 in column name: {}", e)))?);
        let dtype = c.read_u8()?;
        let col = read_column(&mut c, dtype, n_rows)?;
        columns.push((name, col));
    }

    let row_names = if has_row_names {
        let rn = read_character_payload(&mut c, n_rows)?;
        Some(rn.into_iter().map(|o| o.unwrap_or_else(|| Arc::from(""))).collect())
    } else { None };

    Ok(RVal::DataFrame(DataFrame { columns, row_names }))
}

fn read_column(c: &mut Cur, dtype: u8, n_rows: usize) -> Result<RVal, R2Err> {
    match dtype {
        DTYPE_NUMERIC => {
            let mut vals = Vec::with_capacity(n_rows);
            for _ in 0..n_rows { vals.push(c.read_f64()?); }
            let valid = read_bitmap(c, n_rows)?;
            let out: Vec<Real> = vals.into_iter().enumerate()
                .map(|(i, v)| if valid[i] { Some(v) } else { None })
                .collect();
            Ok(RVal::Numeric(Reals::new(out), Attrs::default()))
        }
        DTYPE_INTEGER => {
            let mut vals = Vec::with_capacity(n_rows);
            for _ in 0..n_rows { vals.push(c.read_i32()?); }
            let valid = read_bitmap(c, n_rows)?;
            let out: Vec<Integer> = vals.into_iter().enumerate()
                .map(|(i, v)| if valid[i] { Some(v) } else { None })
                .collect();
            Ok(RVal::Integer(Ints::new(out), Attrs::default()))
        }
        DTYPE_LOGICAL => {
            let vals = read_bitmap(c, n_rows)?;
            let valid = read_bitmap(c, n_rows)?;
            let out: Vec<Logical> = (0..n_rows)
                .map(|i| if valid[i] { Some(vals[i]) } else { None })
                .collect();
            Ok(RVal::Logical(Logicals::new(out), Attrs::default()))
        }
        DTYPE_CHARACTER => {
            let chars = read_character_payload(c, n_rows)?;
            Ok(RVal::Character(chars, Attrs::default()))
        }
        other => Err(err(format!("r2d: unknown dtype {}", other))),
    }
}

fn read_bitmap(c: &mut Cur, n_rows: usize) -> Result<Vec<bool>, R2Err> {
    let bytes_len = (n_rows + 7) / 8;
    let bytes = c.read_bytes(bytes_len)?;
    let mut out = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        out.push((bytes[i / 8] >> (i % 8)) & 1 == 1);
    }
    Ok(out)
}

fn read_character_payload(c: &mut Cur, n_rows: usize) -> Result<Vec<Character>, R2Err> {
    let total_bytes = c.read_u32()? as usize;
    // Read offset/length table.
    let mut entries: Vec<(u32, u32)> = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let off = c.read_u32()?;
        let len = c.read_u32()?;
        entries.push((off, len));
    }
    let blob = c.read_bytes(total_bytes)?;
    let mut out: Vec<Character> = Vec::with_capacity(n_rows);
    for (off, len) in entries {
        if len == u32::MAX {
            out.push(None);
        } else {
            let off = off as usize; let len = len as usize;
            if off + len > blob.len() {
                return Err(err(format!("r2d: string out of bounds (off={} len={} blob={})",
                    off, len, blob.len())));
            }
            let s = std::str::from_utf8(&blob[off..off + len])
                .map_err(|e| err(format!("r2d: bad UTF-8 in string blob: {}", e)))?;
            out.push(Some(Arc::from(s)));
        }
    }
    Ok(out)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn df_eq(a: &DataFrame, b: &DataFrame) {
        assert_eq!(a.columns.len(), b.columns.len(), "ncol");
        assert_eq!(a.row_names, b.row_names, "row_names");
        for ((na, ca), (nb, cb)) in a.columns.iter().zip(b.columns.iter()) {
            assert_eq!(na, nb, "col name");
            match (ca, cb) {
                (RVal::Numeric(x, _), RVal::Numeric(y, _)) => {
                    let xv = x.as_vec(); let yv = y.as_vec();
                    assert_eq!(xv.len(), yv.len());
                    for (xi, yi) in xv.iter().zip(yv.iter()) {
                        match (xi, yi) {
                            (Some(a), Some(b)) => assert!((a - b).abs() < 1e-12 || (a.is_nan() && b.is_nan())),
                            (None, None) => {}
                            _ => panic!("NA mismatch in {}", na),
                        }
                    }
                }
                (RVal::Integer(x, _), RVal::Integer(y, _)) => assert_eq!(x.as_vec(), y.as_vec()),
                (RVal::Logical(x, _), RVal::Logical(y, _)) => assert_eq!(x.as_vec(), y.as_vec()),
                (RVal::Character(x, _), RVal::Character(y, _)) => assert_eq!(x, y),
                _ => panic!("col type mismatch on {}", na),
            }
        }
    }

    fn unwrap_df(v: RVal) -> DataFrame {
        match v { RVal::DataFrame(d) => d, _ => panic!("not a data.frame") }
    }

    #[test]
    fn roundtrip_numeric_with_na() {
        let v: Vec<Real> = vec![Some(1.5), None, Some(3.25), Some(-7.0), None];
        let df = DataFrame {
            columns: vec![(Arc::from("x"), RVal::Numeric(Reals::new(v.clone()), Attrs::default()))],
            row_names: None,
        };
        let bytes = write_r2d(&df);
        let back = unwrap_df(read_r2d(&bytes).unwrap());
        df_eq(&df, &back);
    }

    #[test]
    fn roundtrip_integer_logical_character() {
        let ints: Vec<Integer> = vec![Some(1), Some(-2), None, Some(42)];
        let logs: Vec<Logical> = vec![Some(true), Some(false), None, Some(true)];
        let chars: Vec<Character> = vec![Some(Arc::from("hi")), None, Some(Arc::from("")), Some(Arc::from("a longer string"))];
        let df = DataFrame {
            columns: vec![
                (Arc::from("i"), RVal::Integer(Ints::new(ints), Attrs::default())),
                (Arc::from("l"), RVal::Logical(Logicals::new(logs), Attrs::default())),
                (Arc::from("s"), RVal::Character(chars, Attrs::default())),
            ],
            row_names: Some(vec![Arc::from("r1"), Arc::from("r2"), Arc::from("r3"), Arc::from("r4")]),
        };
        let bytes = write_r2d(&df);
        let back = unwrap_df(read_r2d(&bytes).unwrap());
        df_eq(&df, &back);
    }

    #[test]
    fn rejects_bad_magic() {
        let r = read_r2d(b"NOPE\x01\x00");
        assert!(r.is_err());
    }
}
