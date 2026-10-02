//! Text data files: line reading, format detection and line parsing.
//!
//! upstream: src/io/parser.cpp, src/io/parser.hpp, include/LightGBM/utils/text_reader.h,
//! include/LightGBM/utils/pipeline_reader.h, and the `Atof` / `AtofPrecise` / `Atoi`
//! helpers of include/LightGBM/utils/common.h.

use std::fs::File;
use std::io::{BufRead, BufReader, ErrorKind, Read};

use crate::consts::K_ZERO_THRESHOLD;
use crate::error::{LgbmError, Result};

/// upstream: `PipelineReader::Read` block size.
const READ_BLOCK: usize = 16 * 1024 * 1024;

fn open(filename: &str) -> Result<File> {
    File::open(filename).map_err(|_| LgbmError::InvalidData(format!("Could not open {filename}")))
}

/// Fill `buf` as far as possible (upstream reads blocks with `fread`).
fn read_block(r: &mut impl Read, buf: &mut [u8]) -> Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(k) => got += k,
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(LgbmError::InvalidData(format!("Could not read data file: {e}"))),
        }
    }
    Ok(got)
}

/// The header line and the number of bytes it occupies (terminator included).
///
/// upstream: the `TextReader` constructor with `is_skip_first_line`.
pub fn read_header(filename: &str) -> Result<(String, usize)> {
    let mut r = BufReader::new(open(filename)?);
    let mut line = Vec::new();
    let mut skip = 0usize;
    let mut byte = [0u8; 1];
    let mut last = None;
    while read_block(&mut r, &mut byte)? == 1 {
        last = Some(byte[0]);
        if byte[0] == b'\n' || byte[0] == b'\r' {
            break;
        }
        line.push(byte[0]);
        skip += 1;
    }
    if last == Some(b'\r') {
        skip += 1;
        last = (read_block(&mut r, &mut byte)? == 1).then_some(byte[0]).or(last);
    }
    if last == Some(b'\n') {
        skip += 1;
    }
    Ok((String::from_utf8_lossy(&line).into_owned(), skip))
}

/// Splits blocks into lines like upstream's `TextReader::ReadAllAndProcess`:
/// a run of `\n`/`\r` ends a line, and a block that starts with `\n` while no
/// partial line is pending skips that byte.
#[derive(Default)]
struct LineSplitter {
    pending: Vec<u8>,
}

impl LineSplitter {
    fn feed(&mut self, block: &[u8], f: &mut impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
        let n = block.len();
        let (mut i, mut last_i) = (0, 0);
        if self.pending.is_empty() && n > 0 && block[0] == b'\n' {
            i = 1;
            last_i = 1;
        }
        while i < n {
            if block[i] == b'\n' || block[i] == b'\r' {
                if !self.pending.is_empty() {
                    self.pending.extend_from_slice(&block[last_i..i]);
                    f(&self.pending)?;
                    self.pending.clear();
                } else {
                    f(&block[last_i..i])?;
                }
                i += 1;
                while i < n && (block[i] == b'\n' || block[i] == b'\r') {
                    i += 1;
                }
                last_i = i;
            } else {
                i += 1;
            }
        }
        if last_i != n {
            self.pending.extend_from_slice(&block[last_i..]);
        }
        Ok(())
    }

    fn finish(self, f: &mut impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
        if !self.pending.is_empty() {
            f(&self.pending)?;
        }
        Ok(())
    }
}

/// Call `f` on every line of `filename` after the first `skip_bytes` bytes.
/// A missing file has no lines.
///
/// upstream: `TextReader::ReadAllAndProcess` over `PipelineReader::Read`.
pub fn for_each_line(filename: &str, skip_bytes: usize, mut f: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
    let Ok(mut r) = File::open(filename) else { return Ok(()) };
    std::io::copy(&mut (&mut r).take(skip_bytes as u64), &mut std::io::sink())
        .map_err(|e| LgbmError::InvalidData(format!("Could not read data file: {e}")))?;
    let mut splitter = LineSplitter::default();
    let mut block = Vec::new();
    loop {
        block.resize(READ_BLOCK, 0);
        let got = read_block(&mut r, &mut block)?;
        if got == 0 {
            break;
        }
        splitter.feed(&block[..got], &mut f)?;
        if got < READ_BLOCK {
            break;
        }
    }
    splitter.finish(&mut f)
}

/// All lines of a text file (upstream `TextReader::ReadAllLines`).
pub fn read_all_lines(filename: &str, skip_bytes: usize) -> Result<Vec<Vec<u8>>> {
    let mut out = Vec::new();
    for_each_line(filename, skip_bytes, |l| {
        out.push(l.to_vec());
        Ok(())
    })?;
    Ok(out)
}

/// upstream: `Common::Trim` (" \f\n\r\t\v").
pub fn trim(s: &[u8]) -> &[u8] {
    let ws = |c: &u8| matches!(c, b' ' | b'\x0c' | b'\n' | b'\r' | b'\t' | b'\x0b');
    let start = s.iter().position(|c| !ws(c)).unwrap_or(s.len());
    let end = s.iter().rposition(|c| !ws(c)).map_or(start, |e| e + 1);
    &s[start..end.max(start)]
}

/// `\n`-terminated lines as `std::getline` reads them (the format sniffing
/// helpers of parser.cpp read the file this way, not through `TextReader`).
struct GetLines {
    r: BufReader<File>,
    eof: bool,
}

impl GetLines {
    fn new(filename: &str) -> Result<Self> {
        let f = File::open(filename).map_err(|_| LgbmError::InvalidData(format!("Data file {filename} doesn't exist.")))?;
        if f.metadata().map_or(0, |m| m.len()) == 0 {
            return Err(LgbmError::InvalidData(format!("Data file {filename} couldn't be read.")));
        }
        Ok(Self { r: BufReader::new(f), eof: false })
    }

    fn next_line(&mut self) -> Result<Option<Vec<u8>>> {
        if self.eof {
            return Ok(None);
        }
        let mut line = Vec::new();
        self.r
            .read_until(b'\n', &mut line)
            .map_err(|e| LgbmError::InvalidData(format!("Could not read data file: {e}")))?;
        if line.last() == Some(&b'\n') {
            line.pop();
        } else {
            self.eof = true;
        }
        Ok(Some(line))
    }
}

/// upstream: parser.cpp `ReadKLineFromFile`. Returns the lines and warnings.
fn read_k_lines(filename: &str, header: bool, k: usize, warnings: &mut Vec<String>) -> Result<Vec<Vec<u8>>> {
    let mut lines = GetLines::new(filename)?;
    if header {
        lines.next_line()?;
    }
    let mut ret = Vec::new();
    for _ in 0..k {
        match lines.next_line()? {
            Some(l) => {
                let t = trim(&l);
                if !t.is_empty() {
                    ret.push(t.to_vec());
                }
            }
            None => break,
        }
    }
    if ret.is_empty() {
        return Err(LgbmError::InvalidData(format!("Data file {filename} should have at least one line.")));
    } else if ret.len() == 1 {
        warnings.push(format!("Data file {filename} only has one line."));
    }
    Ok(ret)
}

/// upstream: parser.cpp `GetNumColFromLIBSVMFile`.
fn libsvm_num_col(filename: &str, header: bool) -> Result<i32> {
    let mut lines = GetLines::new(filename)?;
    if header {
        lines.next_line()?;
    }
    let (mut max_col_idx, mut max_line_idx) = (0i32, 0usize);
    for i in 0..(1usize << 13) {
        let Some(l) = lines.next_line()? else { break };
        let l = trim(&l);
        const NPOS: usize = usize::MAX;
        let colon = l.iter().rposition(|&c| c == b':').unwrap_or(NPOS);
        let space = l.iter().rposition(|&c| matches!(c, b' ' | b'\x0c' | b'\t' | b'\x0b')).unwrap_or(NPOS);
        // std::string::substr(space + 1, space - colon - 1) with size_t wrap-around
        let start = space.wrapping_add(1).min(l.len());
        let len = space.wrapping_sub(colon).wrapping_sub(1);
        let end = start.saturating_add(len).min(l.len());
        let (cur, _) = atoi(&l[start..end], 0);
        if cur > max_col_idx {
            max_col_idx = cur;
            max_line_idx = i;
        }
        if i - max_line_idx >= (1 << 7) {
            break;
        }
    }
    if max_col_idx <= 0 {
        return Err(LgbmError::InvalidData("Check failed: (max_col_idx) > (0)".into()));
    }
    Ok(max_col_idx)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Csv,
    Tsv,
    Libsvm,
}

/// upstream: parser.cpp `GetStatistic`.
fn statistic(s: &[u8]) -> (usize, usize, usize) {
    let (mut comma, mut tab, mut colon) = (0, 0, 0);
    for &c in s.iter().take_while(|&&c| c != 0) {
        match c {
            b',' => comma += 1,
            b'\t' => tab += 1,
            b':' => colon += 1,
            _ => {}
        }
    }
    (comma, tab, colon)
}

/// upstream: parser.cpp `GetDataType`; returns the type and the number of columns.
fn data_type(filename: &str, header: bool, lines: &[Vec<u8>]) -> Result<Option<(DataType, i32)>> {
    let (comma, tab, colon) = statistic(&lines[0]);
    let mut ty = None;
    if lines.len() == 1 {
        if colon > 0 {
            ty = Some(DataType::Libsvm);
        } else if tab > 0 {
            ty = Some(DataType::Tsv);
        } else if comma > 0 {
            ty = Some(DataType::Csv);
        }
    } else {
        let (comma2, tab2, colon2) = statistic(&lines[1]);
        if colon > 0 || colon2 > 0 {
            ty = Some(DataType::Libsvm);
        } else if tab == tab2 && tab > 0 {
            ty = Some(DataType::Tsv);
        } else if comma == comma2 && comma > 0 {
            ty = Some(DataType::Csv);
        }
        if matches!(ty, Some(DataType::Tsv | DataType::Csv)) {
            for l in &lines[2..] {
                let (c2, t2, _) = statistic(l);
                if (ty == Some(DataType::Tsv) && t2 != tab) || (ty == Some(DataType::Csv) && c2 != comma) {
                    ty = None;
                    break;
                }
            }
        }
    }
    Ok(match ty {
        Some(DataType::Libsvm) => Some((DataType::Libsvm, libsvm_num_col(filename, header)? + 1)),
        Some(DataType::Csv) => Some((DataType::Csv, comma as i32 + 1)),
        Some(DataType::Tsv) => Some((DataType::Tsv, tab as i32 + 1)),
        None => None,
    })
}

/// A CSV, TSV or LibSVM line parser.
///
/// upstream: parser.hpp `CSVParser`, `TSVParser`, `LibSVMParser`.
#[derive(Debug, Clone)]
pub struct Parser {
    pub data_type: DataType,
    label_idx: i32,
    total_columns: i32,
    precise: bool,
}

impl Parser {
    /// Detect the format of `filename` from its first 32 lines. With
    /// `num_features > 0` (prediction), a first line without a label column
    /// sets the label index to -1.
    ///
    /// upstream: `Parser::CreateParser`, `GetLabelIdxForCSV` / `TSV` / `Libsvm`.
    pub fn create(
        filename: &str,
        header: bool,
        num_features: i32,
        label_idx: i32,
        precise_float_parser: bool,
        warnings: &mut Vec<String>,
    ) -> Result<Self> {
        let lines = read_k_lines(filename, header, 32, warnings)?;
        let Some((data_type, total_columns)) = data_type(filename, header, &lines)? else {
            return Err(LgbmError::InvalidData(
                "Unknown format of training data. Only CSV, TSV, and LibSVM (zero-based) formatted text files are supported."
                    .into(),
            ));
        };
        let first = &lines[0];
        let label_idx = if num_features <= 0 {
            label_idx
        } else if data_type == DataType::Libsvm {
            let space = first.iter().position(|c| matches!(c, b' ' | b'\x0c' | b'\n' | b'\r' | b'\t' | b'\x0b'));
            let colon = first.iter().position(|&c| c == b':');
            match (space, colon) {
                (None, _) => label_idx,
                (Some(s), Some(c)) if s < c => label_idx,
                (Some(_), None) => label_idx,
                _ => -1,
            }
        } else {
            let sep = if data_type == DataType::Csv { b',' } else { b'\t' };
            let tokens = first.split(|&c| c == sep).filter(|t| !t.is_empty()).count();
            if tokens as i32 == num_features { -1 } else { label_idx }
        };
        if data_type == DataType::Libsvm && label_idx > 0 {
            return Err(LgbmError::InvalidData("Label should be the first column in a LibSVM file".into()));
        }
        Ok(Self { data_type, label_idx, total_columns, precise: precise_float_parser })
    }

    /// upstream: `Parser::NumFeatures`.
    pub fn num_features(&self) -> i32 {
        match self.data_type {
            DataType::Libsvm => self.total_columns,
            _ => self.total_columns - (self.label_idx >= 0) as i32,
        }
    }

    fn atof(&self, s: &[u8], p: usize) -> Result<(f64, usize)> {
        if self.precise { atof_precise(s, p) } else { atof(s, p) }
    }

    /// Parse one line into `(column, value)` pairs (`out` is cleared) and
    /// return the label (0 when the line has none).
    ///
    /// upstream: `ParseOneLine`. CSV/TSV keep only values with
    /// `|v| > kZeroThreshold` or NaN; LibSVM keeps every listed value.
    pub fn parse_line(&self, line: &[u8], out: &mut Vec<(i32, f64)>) -> Result<f64> {
        out.clear();
        // the line is a C string: it ends at the first NUL
        let s = &line[..line.iter().position(|&c| c == 0).unwrap_or(line.len())];
        let mut label = 0.0;
        let mut p = 0;
        match self.data_type {
            DataType::Csv | DataType::Tsv => {
                let (sep, name) = if self.data_type == DataType::Csv { (b',', "CSV") } else { (b'\t', "TSV") };
                let (mut idx, mut offset) = (0i32, 0i32);
                while p < s.len() {
                    let (val, q) = self.atof(s, p)?;
                    p = q;
                    if idx == self.label_idx {
                        label = val;
                        offset = -1;
                    } else if val.abs() > K_ZERO_THRESHOLD || val.is_nan() {
                        out.push((idx + offset, val));
                    }
                    idx += 1;
                    if p < s.len() && s[p] == sep {
                        p += 1;
                    } else if p < s.len() {
                        return Err(LgbmError::InvalidData(format!("Input format error when parsing as {name}")));
                    }
                }
            }
            DataType::Libsvm => {
                if self.label_idx == 0 {
                    let (val, q) = self.atof(s, p)?;
                    label = val;
                    p = skip_space_and_tab(s, q);
                }
                while p < s.len() {
                    let (idx, q) = atoi(s, p);
                    p = skip_space_and_tab(s, q);
                    if p < s.len() && s[p] == b':' {
                        // upstream parses LibSVM values with Common::Atof even with precise_float_parser
                        let (val, q) = atof(s, p + 1)?;
                        p = q;
                        out.push((idx, val));
                    } else {
                        return Err(LgbmError::InvalidData("Input format error when parsing as LibSVM".into()));
                    }
                    p = skip_space_and_tab(s, p);
                }
            }
        }
        Ok(label)
    }
}

/// upstream: `Common::SkipSpaceAndTab`.
fn skip_space_and_tab(s: &[u8], mut p: usize) -> usize {
    while p < s.len() && (s[p] == b' ' || s[p] == b'\t') {
        p += 1;
    }
    p
}

/// upstream: `Common::Atoi<int>` (spaces, sign, digits; wraps like the C++ code).
pub fn atoi(s: &[u8], mut p: usize) -> (i32, usize) {
    while p < s.len() && s[p] == b' ' {
        p += 1;
    }
    let mut sign = 1i32;
    if p < s.len() && s[p] == b'-' {
        sign = -1;
        p += 1;
    } else if p < s.len() && s[p] == b'+' {
        p += 1;
    }
    let mut value = 0i32;
    while p < s.len() && s[p].is_ascii_digit() {
        value = value.wrapping_mul(10).wrapping_add((s[p] - b'0') as i32);
        p += 1;
    }
    while p < s.len() && s[p] == b' ' {
        p += 1;
    }
    (sign.wrapping_mul(value), p)
}

/// upstream: `Common::Pow`.
fn pow(base: f64, power: i32) -> f64 {
    if power < 0 {
        1.0 / pow(base, -power)
    } else if power == 0 {
        1.0
    } else if power % 2 == 0 {
        pow(base * base, power / 2)
    } else if power % 3 == 0 {
        pow(base * base * base, power / 3)
    } else {
        base * pow(base, power - 1)
    }
}

/// upstream: `Common::Atof`, the legacy, not correctly rounded parser
/// (the default, `precise_float_parser=false`). Returns the value and the
/// position after it and any following spaces.
pub fn atof(s: &[u8], mut p: usize) -> Result<(f64, usize)> {
    let at = |p: usize| if p < s.len() { s[p] } else { 0 };
    let mut out = f64::NAN;
    while at(p) == b' ' {
        p += 1;
    }
    let mut sign = 1.0;
    if at(p) == b'-' {
        sign = -1.0;
        p += 1;
    } else if at(p) == b'+' {
        p += 1;
    }
    let c = at(p);
    if c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E') {
        let mut value = 0.0f64;
        while at(p).is_ascii_digit() {
            value = value * 10.0 + (at(p) - b'0') as f64;
            p += 1;
        }
        if at(p) == b'.' {
            let mut right = 0.0f64;
            let mut nn = 0;
            p += 1;
            while at(p).is_ascii_digit() {
                right = (at(p) - b'0') as f64 + right * 10.0;
                nn += 1;
                p += 1;
            }
            value += right / pow(10.0, nn);
        }
        let mut frac = false;
        let mut scale = 1.0f64;
        if matches!(at(p), b'e' | b'E') {
            p += 1;
            if at(p) == b'-' {
                frac = true;
                p += 1;
            } else if at(p) == b'+' {
                p += 1;
            }
            let mut expon: u32 = 0;
            while at(p).is_ascii_digit() {
                expon = expon.wrapping_mul(10).wrapping_add((at(p) - b'0') as u32);
                p += 1;
            }
            expon = expon.min(308);
            while expon >= 50 {
                scale *= 1e50;
                expon -= 50;
            }
            while expon >= 8 {
                scale *= 1e8;
                expon -= 8;
            }
            while expon > 0 {
                scale *= 10.0;
                expon -= 1;
            }
        }
        out = sign * if frac { value / scale } else { value * scale };
    } else {
        let cnt = s[p.min(s.len())..]
            .iter()
            .take_while(|&&c| !matches!(c, 0 | b' ' | b'\t' | b',' | b'\n' | b'\r' | b':'))
            .count();
        if cnt > 0 {
            let token = String::from_utf8_lossy(&s[p..p + cnt]).to_lowercase();
            match token.as_str() {
                "na" | "nan" | "null" => out = f64::NAN,
                "inf" | "infinity" => out = sign * 1e308,
                _ => return Err(LgbmError::InvalidData(format!("Unknown token {token} in data file"))),
            }
            p += cnt;
        }
    }
    while at(p) == b' ' {
        p += 1;
    }
    Ok((out, p))
}

/// Length of the decimal number at `s[p..]` in `-?digits(.digits)?([eE][+-]?digits)?`
/// form (any digit counts, at least one before or after the point).
fn decimal_len(s: &[u8], p: usize, allow_empty_int: bool) -> usize {
    let at = |i: usize| if i < s.len() { s[i] } else { 0 };
    let mut q = p;
    let int_digits = s[q.min(s.len())..].iter().take_while(|c| c.is_ascii_digit()).count();
    q += int_digits;
    let mut frac_digits = 0;
    if at(q) == b'.' {
        frac_digits = s[(q + 1).min(s.len())..].iter().take_while(|c| c.is_ascii_digit()).count();
        if frac_digits > 0 || (int_digits > 0 && allow_empty_int) {
            q += 1 + frac_digits;
        }
    }
    if int_digits == 0 && frac_digits == 0 {
        return 0;
    }
    if matches!(at(q), b'e' | b'E') {
        let mut e = q + 1;
        if matches!(at(e), b'+' | b'-') {
            e += 1;
        }
        let exp_digits = s[e.min(s.len())..].iter().take_while(|c| c.is_ascii_digit()).count();
        if exp_digits > 0 {
            q = e + exp_digits;
        }
    }
    q - p
}

fn parse_decimal(s: &[u8]) -> f64 {
    // a correctly rounded conversion, as fast_double_parser and glibc strtod give
    std::str::from_utf8(s).ok().and_then(|t| t.parse::<f64>().ok()).unwrap_or(f64::NAN)
}

/// upstream: `Common::AtofPrecise` (`fast_double_parser::parse_number`, then
/// `strtod`). Both are correctly rounded, so the value is Rust's parse of the
/// same digits; the accepted syntax and end position follow `strtod`.
/// Hexadecimal floats are not supported.
pub fn atof_precise(s: &[u8], p: usize) -> Result<(f64, usize)> {
    let at = |i: usize| if i < s.len() { s[i] } else { 0 };
    // strtod: leading white space (isspace), sign, then inf / nan / a decimal number
    let mut q = p;
    while matches!(at(q), b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r') {
        q += 1;
    }
    let negative = at(q) == b'-';
    if matches!(at(q), b'-' | b'+') {
        q += 1;
    }
    let sign = if negative { -1.0 } else { 1.0 };
    let rest = &s[q.min(s.len())..];
    let starts = |w: &str| rest.len() >= w.len() && rest[..w.len()].eq_ignore_ascii_case(w.as_bytes());
    if starts("infinity") {
        return Ok((sign * f64::INFINITY, q + 8));
    }
    if starts("inf") {
        return Ok((sign * f64::INFINITY, q + 3));
    }
    if starts("nan") {
        let mut e = q + 3;
        if at(e) == b'(' {
            let close = s[e..].iter().position(|&c| !(c.is_ascii_alphanumeric() || c == b'_' || c == b'('));
            if let Some(k) = close {
                if s[e + k] == b')' {
                    e += k + 1;
                }
            }
        }
        return Ok((f64::NAN, e));
    }
    if at(q) == b'0' && matches!(at(q + 1), b'x' | b'X') {
        return Err(LgbmError::Unsupported("hexadecimal floats in text data files (precise_float_parser)".into()));
    }
    let len = decimal_len(s, q, true);
    if len == 0 {
        let tail = String::from_utf8_lossy(&s[p.min(s.len())..]);
        return Err(LgbmError::InvalidData(format!("no conversion to double for: {tail}")));
    }
    Ok((sign * parse_decimal(&s[q..q + len]), q + len))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csv(label_idx: i32) -> Parser {
        Parser { data_type: DataType::Csv, label_idx, total_columns: 4, precise: false }
    }

    #[test]
    fn csv_line_skips_zeros_and_shifts_after_label() {
        let mut out = Vec::new();
        let label = csv(1).parse_line(b"0.5,3,0,,na,7", &mut out).unwrap();
        assert_eq!(label, 3.0);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], (0, 0.5));
        assert!(out[1].0 == 2 && out[1].1.is_nan());
        assert!(out[2].0 == 3 && out[2].1.is_nan());
        assert_eq!(out[3], (4, 7.0));
        assert!(csv(0).parse_line(b"1,2\t3", &mut out).is_err());
        let e = csv(0).parse_line(b"1,abc", &mut out).unwrap_err();
        assert!(e.to_string().contains("Unknown token abc in data file"), "{e}");
    }

    #[test]
    fn libsvm_line() {
        let p = Parser { data_type: DataType::Libsvm, label_idx: 0, total_columns: 10, precise: true };
        let mut out = Vec::new();
        assert_eq!(p.parse_line(b"2 1:0.5\t3:0  7:-1", &mut out).unwrap(), 2.0);
        assert_eq!(out, vec![(1, 0.5), (3, 0.0), (7, -1.0)]);
        assert!(p.parse_line(b"1 4", &mut out).is_err());
    }

    #[test]
    fn legacy_and_precise_atof() {
        assert_eq!(atof(b"  -1.5e2 ,", 0).unwrap(), (-150.0, 9));
        assert_eq!(atof(b"inf", 0).unwrap().0, 1e308);
        assert!(atof(b",", 0).unwrap().0.is_nan());
        assert_eq!(atof_precise(b"0.1", 0).unwrap(), (0.1, 3));
        assert_eq!(atof_precise(b"1.e5x", 0).unwrap(), (1e5, 4));
        assert_eq!(atof_precise(b"-inf", 0).unwrap().0, f64::NEG_INFINITY);
        assert!(atof_precise(b",1", 0).is_err());
        // the legacy parser is not correctly rounded
        assert_eq!(atof(b"9.53084e-05", 0).unwrap().0, 9.530839999999999e-05);
        assert_eq!(atof_precise(b"9.53084e-05", 0).unwrap().0, 9.53084e-05);
    }

    #[test]
    fn line_splitting_matches_text_reader() {
        let mut got: Vec<Vec<u8>> = Vec::new();
        let mut s = LineSplitter::default();
        let mut f = |l: &[u8]| {
            got.push(l.to_vec());
            Ok(())
        };
        s.feed(b"\na,b\r\n\r\nc", &mut f).unwrap();
        s.feed(b"d\n", &mut f).unwrap();
        s.finish(&mut f).unwrap();
        assert_eq!(got, vec![b"a,b".to_vec(), b"cd".to_vec()]);
    }
}
