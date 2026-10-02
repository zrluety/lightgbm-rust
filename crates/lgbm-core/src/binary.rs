//! lightgbm-rust's own binary Dataset file (`Dataset.save_binary`).
//!
//! The format is specific to lightgbm-rust and is not compatible with
//! upstream's `.bin` files in either direction: upstream files are detected
//! by their token and rejected. What is stored and the load-time checks
//! follow upstream (`Dataset::SaveBinaryFile`, `Metadata::SaveBinaryToFile`,
//! `DatasetLoader::LoadFromBinFile` / `CheckDataset`): labels, weights and
//! query boundaries are kept; `init_score` and positions are not.
//!
//! Layout (little-endian): the 16-byte [`MAGIC`], a `u32` format version,
//! then the dataset sections in the order written by [`Dataset::save_binary`].
//! The reader rejects other versions, truncated files and trailing bytes.

use std::collections::HashMap;
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use crate::binning::{BinMapper, BinType, MissingType};
use crate::config::Config;
use crate::dataset::{check_max_bin_by_feature, BinColumn, BinConstructConfig, Dataset, Metadata};
use crate::error::{LgbmError, Result};

/// First bytes of a lightgbm-rust binary dataset file.
pub const MAGIC: &[u8; 16] = b"\x89LGBMRS-DATASET\n";
/// Version written by this build. Versions 1 (without `label_idx`), 2
/// (without `max_bin_by_feature` and forced bin bounds) and 3 (without the
/// multi-value feature group) are also read.
pub const FORMAT_VERSION: u32 = 4;
/// upstream `Dataset::binary_file_token`.
pub const UPSTREAM_TOKEN: &[u8] = b"______LightGBM_Binary_File_Token______\n";

/// What [`detect_data_file`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFileKind {
    /// A lightgbm-rust binary dataset.
    Binary,
    /// An upstream LightGBM binary dataset (not readable here).
    UpstreamBinary,
    /// Anything else; upstream would parse it as a text data file.
    Text,
}

/// upstream `DatasetLoader::CheckCanLoadFromBin`: `<filename>.bin` is tried
/// before `filename`. Returns the file to read and its kind.
pub fn detect_data_file(filename: &str) -> Result<(PathBuf, DataFileKind)> {
    let with_bin = PathBuf::from(format!("{filename}.bin"));
    let (path, file) = match std::fs::File::open(&with_bin) {
        Ok(f) => (with_bin, f),
        Err(_) => match std::fs::File::open(filename) {
            Ok(f) => (PathBuf::from(filename), f),
            Err(_) => return Err(LgbmError::InvalidData(format!("Cannot open data file {filename}"))),
        },
    };
    file_kind(path, file)
}

/// Like [`detect_data_file`], for `filename` only (no `.bin` lookup).
pub fn detect_data_file_exact(filename: &str) -> Result<(PathBuf, DataFileKind)> {
    let file = std::fs::File::open(filename)
        .map_err(|_| LgbmError::InvalidData(format!("Cannot open data file {filename}")))?;
    file_kind(PathBuf::from(filename), file)
}

fn file_kind(path: PathBuf, mut file: std::fs::File) -> Result<(PathBuf, DataFileKind)> {
    let mut head = vec![0u8; UPSTREAM_TOKEN.len()];
    let mut got = 0;
    while got < head.len() {
        match file.read(&mut head[got..]) {
            Ok(0) => break,
            Ok(k) => got += k,
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(LgbmError::InvalidData(format!("Could not read binary data from {}: {e}", path.display()))),
        }
    }
    let head = &head[..got];
    let kind = if head.starts_with(MAGIC) {
        DataFileKind::Binary
    } else if head == UPSTREAM_TOKEN {
        DataFileKind::UpstreamBinary
    } else {
        DataFileKind::Text
    };
    Ok((path, kind))
}

struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn usize(&mut self, v: usize) {
        self.u64(v as u64);
    }
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bool(&mut self, v: bool) {
        self.u8(v as u8);
    }
    fn str(&mut self, s: &str) {
        self.usize(s.len());
        self.0.extend_from_slice(s.as_bytes());
    }
    fn f32s(&mut self, v: &[f32]) {
        for x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }
    fn f64_vec(&mut self, v: &[f64]) {
        self.usize(v.len());
        for &x in v {
            self.f64(x);
        }
    }
    fn i32_vec(&mut self, v: &[i32]) {
        self.usize(v.len());
        for &x in v {
            self.i32(x);
        }
    }
    fn usize_vec(&mut self, v: &[usize]) {
        self.usize(v.len());
        for &x in v {
            self.usize(x);
        }
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    path: &'a Path,
}

impl<'a> Reader<'a> {
    fn corrupt(&self, what: &str) -> LgbmError {
        LgbmError::InvalidData(format!(
            "Binary file error: {} is truncated or corrupted ({what})",
            self.path.display()
        ))
    }
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8]> {
        if self.buf.len() - self.pos < n {
            return Err(self.corrupt(what));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn array<const N: usize>(&mut self, what: &str) -> Result<[u8; N]> {
        Ok(self.take(N, what)?.try_into().expect("length checked"))
    }
    fn u8(&mut self, what: &str) -> Result<u8> {
        Ok(self.take(1, what)?[0])
    }
    fn u32(&mut self, what: &str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array(what)?))
    }
    fn i32(&mut self, what: &str) -> Result<i32> {
        Ok(i32::from_le_bytes(self.array(what)?))
    }
    fn usize(&mut self, what: &str) -> Result<usize> {
        usize::try_from(u64::from_le_bytes(self.array(what)?)).map_err(|_| self.corrupt(what))
    }
    fn f64(&mut self, what: &str) -> Result<f64> {
        Ok(f64::from_le_bytes(self.array(what)?))
    }
    fn bool(&mut self, what: &str) -> Result<bool> {
        match self.u8(what)? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(self.corrupt(what)),
        }
    }
    /// A length that must fit in the rest of the file at `elem` bytes each.
    fn len(&mut self, elem: usize, what: &str) -> Result<usize> {
        let n = self.usize(what)?;
        if n.checked_mul(elem).is_none_or(|b| b > self.buf.len() - self.pos) {
            return Err(self.corrupt(what));
        }
        Ok(n)
    }
    fn str(&mut self, what: &str) -> Result<String> {
        let n = self.len(1, what)?;
        let b = self.take(n, what)?;
        String::from_utf8(b.to_vec()).map_err(|_| self.corrupt(what))
    }
    fn f32s(&mut self, n: usize, what: &str) -> Result<Vec<f32>> {
        let b = self.take(n.checked_mul(4).ok_or_else(|| self.corrupt(what))?, what)?;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    fn f64_vec(&mut self, what: &str) -> Result<Vec<f64>> {
        let n = self.len(8, what)?;
        (0..n).map(|_| self.f64(what)).collect()
    }
    fn i32_vec(&mut self, what: &str) -> Result<Vec<i32>> {
        let n = self.len(4, what)?;
        (0..n).map(|_| self.i32(what)).collect()
    }
    fn usize_vec(&mut self, what: &str) -> Result<Vec<usize>> {
        let n = self.len(8, what)?;
        (0..n).map(|_| self.usize(what)).collect()
    }
}

fn write_bin_mapper(w: &mut Writer, m: &BinMapper) {
    w.i32(m.num_bin);
    w.u8(m.missing_type as u8);
    w.u8(match m.bin_type {
        BinType::Numerical => 0,
        BinType::Categorical => 1,
    });
    w.bool(m.is_trivial);
    w.f64(m.sparse_rate);
    w.f64(m.min_val);
    w.f64(m.max_val);
    w.u32(m.default_bin);
    w.u32(m.most_freq_bin);
    w.f64_vec(&m.bin_upper_bound);
    w.i32_vec(&m.bin_2_categorical);
    let mut pairs: Vec<(i32, u32)> = m.categorical_2_bin.iter().map(|(&c, &b)| (c, b)).collect();
    pairs.sort_unstable();
    w.usize(pairs.len());
    for (c, b) in pairs {
        w.i32(c);
        w.u32(b);
    }
}

fn read_bin_mapper(r: &mut Reader<'_>) -> Result<BinMapper> {
    let what = "bin mapper";
    let num_bin = r.i32(what)?;
    let missing_type = match r.u8(what)? {
        0 => MissingType::None,
        1 => MissingType::Zero,
        2 => MissingType::NaN,
        _ => return Err(r.corrupt(what)),
    };
    let bin_type = match r.u8(what)? {
        0 => BinType::Numerical,
        1 => BinType::Categorical,
        _ => return Err(r.corrupt(what)),
    };
    let is_trivial = r.bool(what)?;
    let sparse_rate = r.f64(what)?;
    let min_val = r.f64(what)?;
    let max_val = r.f64(what)?;
    let default_bin = r.u32(what)?;
    let most_freq_bin = r.u32(what)?;
    let bin_upper_bound = r.f64_vec(what)?;
    let bin_2_categorical = r.i32_vec(what)?;
    let npairs = r.len(8, what)?;
    let mut categorical_2_bin = HashMap::with_capacity(npairs);
    for _ in 0..npairs {
        let c = r.i32(what)?;
        let b = r.u32(what)?;
        categorical_2_bin.insert(c, b);
    }
    if num_bin < 1 {
        return Err(r.corrupt(what));
    }
    Ok(BinMapper {
        num_bin,
        missing_type,
        bin_upper_bound,
        is_trivial,
        sparse_rate,
        bin_type,
        min_val,
        max_val,
        default_bin,
        most_freq_bin,
        bin_2_categorical,
        categorical_2_bin,
    })
}

/// Write `bytes` to a new file, reporting short writes like upstream's
/// `LocalFile::Write`.
fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = std::fs::File::create(path)
        .map_err(|_| LgbmError::InvalidData(format!("Cannot write binary data to {} ", path.display())))?;
    let mut written = 0;
    while written < bytes.len() {
        match f.write(&bytes[written..]) {
            Ok(0) => break,
            Ok(k) => written += k,
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    if written < bytes.len() || f.flush().is_err() || f.sync_all().is_err() {
        return Err(LgbmError::InvalidData(format!(
            "Cannot write binary data to {}, wrote {written} of {} bytes",
            path.display(),
            bytes.len()
        )));
    }
    Ok(())
}

impl Dataset {
    /// Serialize to lightgbm-rust's binary format.
    pub fn to_binary_bytes(&self) -> Vec<u8> {
        let mut w = Writer(Vec::new());
        w.0.extend_from_slice(MAGIC);
        w.u32(FORMAT_VERSION);
        w.usize(self.num_data);
        w.usize(self.bin_mappers.len());
        let c = self.bin_config;
        w.i32(c.max_bin);
        w.i32(c.min_data_in_bin);
        w.i32(c.bin_construct_sample_cnt);
        w.bool(c.use_missing);
        w.bool(c.zero_as_missing);
        w.i32(self.label_idx);
        for name in &self.feature_names {
            w.str(name);
        }
        for m in &self.bin_mappers {
            write_bin_mapper(&mut w, m);
        }
        w.usize_vec(&self.used_features);
        w.usize_vec(&self.upstream_inner);
        w.usize(self.num_feature_groups);
        for col in &self.bins {
            match col {
                BinColumn::U8(v) => {
                    w.u8(1);
                    w.0.extend_from_slice(v);
                }
                BinColumn::U16(v) => {
                    w.u8(2);
                    for x in v {
                        w.0.extend_from_slice(&x.to_le_bytes());
                    }
                }
                BinColumn::U32(v) => {
                    w.u8(4);
                    for x in v {
                        w.0.extend_from_slice(&x.to_le_bytes());
                    }
                }
            }
        }
        let md = &self.metadata;
        w.f32s(&md.label);
        w.bool(md.weight.is_some());
        if let Some(wt) = &md.weight {
            w.f32s(wt);
        }
        // upstream writes the boundaries only for a positive query count
        let queries = md.query_boundaries.as_ref().filter(|b| b.len() > 1);
        w.bool(queries.is_some());
        if let Some(b) = queries {
            w.i32_vec(b);
        }
        w.i32_vec(&self.max_bin_by_feature);
        for b in &self.forced_bin_bounds {
            w.f64_vec(b);
        }
        w.usize_vec(&self.multi_val_group);
        w.0
    }

    /// upstream `Dataset::SaveBinaryFile`. An existing file is left
    /// untouched (with a warning), as upstream does.
    pub fn save_binary(&self, filename: &str) -> Result<()> {
        if self.data_filename.as_deref() == Some(filename) {
            crate::log::warning(&format!("Binary file {filename} already exists"));
            return Ok(());
        }
        let path = Path::new(filename);
        if path.exists() {
            crate::log::warning(&format!("File {filename} exists, cannot save binary to it"));
            return Ok(());
        }
        crate::log::info(&format!("Saving data to binary file {filename}"));
        write_file(path, &self.to_binary_bytes())?;
        if self.metadata.init_score.is_some() {
            crate::log::warning(
                "Please note that `init_score` is not saved in binary file.\n\
                 If you need it, please set it again after loading Dataset.",
            );
        }
        Ok(())
    }

    /// Parse lightgbm-rust's binary format; `path` is used in messages.
    pub fn from_binary_bytes(bytes: &[u8], path: &Path) -> Result<Dataset> {
        let mut r = Reader { buf: bytes, pos: 0, path };
        if r.take(MAGIC.len(), "header")? != MAGIC {
            return Err(LgbmError::InvalidData(format!(
                "{} is not a lightgbm-rust binary dataset file",
                path.display()
            )));
        }
        let version = r.u32("header")?;
        if !(1..=FORMAT_VERSION).contains(&version) {
            return Err(LgbmError::Unsupported(format!(
                "binary dataset format version {version} in {} (this build reads versions 1 to {FORMAT_VERSION})",
                path.display()
            )));
        }
        let num_data = r.usize("header")?;
        let ncol = r.len(1, "header")?;
        let bin_config = BinConstructConfig {
            max_bin: r.i32("header")?,
            min_data_in_bin: r.i32("header")?,
            bin_construct_sample_cnt: r.i32("header")?,
            use_missing: r.bool("header")?,
            zero_as_missing: r.bool("header")?,
        };
        let label_idx = if version >= 2 { r.i32("header")? } else { 0 };
        let feature_names = (0..ncol).map(|_| r.str("feature names")).collect::<Result<Vec<_>>>()?;
        let bin_mappers = (0..ncol).map(|_| read_bin_mapper(&mut r)).collect::<Result<Vec<_>>>()?;
        let used_features = r.usize_vec("feature map")?;
        let upstream_inner = r.usize_vec("feature map")?;
        let num_feature_groups = r.usize("feature map")?;
        let expected: Vec<usize> = (0..ncol).filter(|&c| !bin_mappers[c].is_trivial).collect();
        if used_features != expected || upstream_inner.len() != used_features.len() {
            return Err(r.corrupt("feature map"));
        }
        let mut real_to_inner = vec![None; ncol];
        for (inner, &real) in used_features.iter().enumerate() {
            real_to_inner[real] = Some(inner);
        }
        let mut bins = Vec::with_capacity(used_features.len());
        for &real in &used_features {
            let what = "feature bins";
            let num_bin = bin_mappers[real].num_bin as u32;
            let col = match r.u8(what)? {
                1 => BinColumn::U8(r.take(num_data, what)?.to_vec()),
                2 => {
                    let b = r.take(num_data.checked_mul(2).ok_or_else(|| r.corrupt(what))?, what)?;
                    BinColumn::U16(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
                }
                4 => {
                    let b = r.take(num_data.checked_mul(4).ok_or_else(|| r.corrupt(what))?, what)?;
                    BinColumn::U32(b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
                }
                _ => return Err(r.corrupt(what)),
            };
            if (0..num_data).any(|i| col.get(i) >= num_bin) {
                return Err(r.corrupt(what));
            }
            bins.push(col);
        }
        let label = r.f32s(num_data, "metadata")?;
        let weight = if r.bool("metadata")? { Some(r.f32s(num_data, "metadata")?) } else { None };
        let query_boundaries = if r.bool("metadata")? {
            let b = r.i32_vec("metadata")?;
            let ok = b.len() > 1
                && b[0] == 0
                && b.windows(2).all(|w| w[0] <= w[1])
                && b.last().copied() == i32::try_from(num_data).ok();
            if !ok {
                return Err(r.corrupt("metadata"));
            }
            Some(b)
        } else {
            None
        };
        let (max_bin_by_feature, forced_bin_bounds) = if version >= 3 {
            let m = r.i32_vec("bin parameters")?;
            if !(m.is_empty() || m.len() == ncol) {
                return Err(r.corrupt("bin parameters"));
            }
            let f = (0..ncol).map(|_| r.f64_vec("bin parameters")).collect::<Result<Vec<_>>>()?;
            (m, f)
        } else {
            (Vec::new(), vec![Vec::new(); ncol])
        };
        let multi_val_group = if version >= 4 { r.usize_vec("feature groups")? } else { Vec::new() };
        if multi_val_group.iter().any(|&c| real_to_inner.get(c).is_none_or(Option::is_none)) {
            return Err(r.corrupt("feature groups"));
        }
        if r.pos != bytes.len() {
            return Err(r.corrupt("unexpected trailing bytes"));
        }
        if num_data == 0 {
            return Err(LgbmError::InvalidData(format!("Data file {} is empty", path.display())));
        }
        let mut metadata = Metadata { label, ..Default::default() };
        metadata.set_weight(num_data, weight.as_deref())?;
        if let Some(b) = &query_boundaries {
            let counts: Vec<i32> = b.windows(2).map(|w| w[1] - w[0]).collect();
            metadata.set_query(num_data, Some(&counts))?;
        }
        Ok(Dataset {
            num_data,
            bin_mappers,
            used_features,
            upstream_inner,
            num_feature_groups,
            multi_val_group,
            real_to_inner,
            bins,
            metadata,
            feature_names,
            bin_config,
            max_bin_by_feature,
            forced_bin_bounds,
            data_filename: None,
            label_idx,
        })
    }

    /// Load a dataset from a file, as upstream's `LGBM_DatasetCreateFromFile`
    /// does for binary files. With `check = Some(cfg)` (no reference
    /// dataset), the construction parameters must match `cfg`
    /// (upstream `DatasetLoader::CheckDataset`).
    pub fn load_binary(filename: &str, check: Option<&Config>) -> Result<Dataset> {
        let (path, kind) = detect_data_file(filename)?;
        match kind {
            DataFileKind::Binary => {}
            DataFileKind::UpstreamBinary => {
                return Err(LgbmError::Unsupported(format!(
                    "{} is an upstream LightGBM binary dataset file; that format is not compatible with \
                     lightgbm-rust's. Construct the Dataset from the original data instead",
                    path.display()
                )))
            }
            DataFileKind::Text => return Err(LgbmError::Unsupported("training from files".into())),
        }
        let bytes = std::fs::read(&path)
            .map_err(|_| LgbmError::InvalidData(format!("Could not read binary data from {}", path.display())))?;
        let mut ds = Self::from_binary_bytes(&bytes, &path)?;
        ds.data_filename = Some(filename.to_string());
        if let Some(cfg) = check {
            // upstream: LoadFromBinFile takes a given max_bin_by_feature
            // over the stored one, so only an omitted one can mismatch
            if !cfg.max_bin_by_feature.is_empty() {
                check_max_bin_by_feature(
                    cfg,
                    ds.num_total_features(),
                    "static_cast<size_t>(dataset->num_total_features_)",
                )?;
                ds.max_bin_by_feature = cfg.max_bin_by_feature.clone();
            }
            check_loaded_config(&ds.bin_config, cfg)?;
            if ds.max_bin_by_feature != cfg.max_bin_by_feature {
                return Err(LgbmError::InvalidParameter(
                    "Parameter max_bin_by_feature cannot be changed when loading from binary file.".into(),
                ));
            }
        }
        // upstream: Metadata::LoadFromMemory recomputes the query weights
        if ds.metadata.weight.is_some() && ds.metadata.query_boundaries.is_some() {
            crate::log::info("Calculating query weights...");
        }
        Ok(ds)
    }
}

/// upstream `DatasetLoader::CheckDataset` (`is_load_from_binary` branch).
fn check_loaded_config(stored: &BinConstructConfig, cfg: &Config) -> Result<()> {
    let given = BinConstructConfig::from_config(cfg);
    let msg = |name: &str, a: i32, b: i32| {
        Err(LgbmError::InvalidParameter(format!(
            "Dataset was constructed with parameter {name}={a}. It cannot be changed to {b} when loading from binary file."
        )))
    };
    if stored.max_bin != given.max_bin {
        return msg("max_bin", stored.max_bin, given.max_bin);
    }
    if stored.min_data_in_bin != given.min_data_in_bin {
        return msg("min_data_in_bin", stored.min_data_in_bin, given.min_data_in_bin);
    }
    if stored.use_missing != given.use_missing {
        return msg("use_missing", stored.use_missing as i32, given.use_missing as i32);
    }
    if stored.zero_as_missing != given.zero_as_missing {
        return msg("zero_as_missing", stored.zero_as_missing as i32, given.zero_as_missing as i32);
    }
    if stored.bin_construct_sample_cnt != given.bin_construct_sample_cnt {
        return msg("bin_construct_sample_cnt", stored.bin_construct_sample_cnt, given.bin_construct_sample_cnt);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{DatasetFields, DenseMatrix};

    fn sample() -> Dataset {
        let n = 300;
        let ncol = 4;
        let mut x = vec![0.0f64; n * ncol];
        for i in 0..n {
            x[i * ncol] = (i % 37) as f64 * 0.5;
            x[i * ncol + 1] = if i % 3 == 0 { f64::NAN } else { (i % 11) as f64 };
            x[i * ncol + 2] = 1.0;
            x[i * ncol + 3] = (i % 5) as f64;
        }
        let label: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();
        let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i % 3) as f32).collect();
        let mat = DenseMatrix::from_f64_row_major(&x, n, ncol).unwrap();
        let cfg = Config::from_pairs([("categorical_feature", "3"), ("max_bin", "63")]).unwrap();
        let fields = DatasetFields { label: &label, weight: Some(&weight), ..Default::default() };
        let mut ds = Dataset::from_dense(&mat, fields, &cfg).unwrap();
        ds.metadata.set_query(n, Some(&[100, 150, 50])).unwrap();
        ds
    }

    #[test]
    fn round_trip_is_exact() {
        let ds = sample();
        let bytes = ds.to_binary_bytes();
        let back = Dataset::from_binary_bytes(&bytes, Path::new("mem")).unwrap();
        for (a, b) in back.bin_mappers.iter().zip(&ds.bin_mappers) {
            let bits = |m: &BinMapper| m.bin_upper_bound.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(a), bits(b));
            let (mut a, mut b) = (a.clone(), b.clone());
            a.bin_upper_bound.clear();
            b.bin_upper_bound.clear();
            assert_eq!(a, b);
        }
        assert_eq!(back.used_features, ds.used_features);
        assert_eq!(back.upstream_inner, ds.upstream_inner);
        assert_eq!(back.num_feature_groups, ds.num_feature_groups);
        assert_eq!(back.feature_names, ds.feature_names);
        assert_eq!(back.bin_config, ds.bin_config);
        for f in 0..ds.num_features() {
            assert_eq!(back.bin_indices(f), ds.bin_indices(f));
        }
        assert_eq!(back.metadata.label, ds.metadata.label);
        assert_eq!(back.metadata.weight, ds.metadata.weight);
        assert_eq!(back.metadata.query_boundaries, ds.metadata.query_boundaries);
        assert_eq!(back.metadata.query_weights, ds.metadata.query_weights);
        assert_eq!(back.to_binary_bytes(), bytes);
    }

    #[test]
    fn truncation_version_and_trailing_bytes_are_rejected() {
        let bytes = sample().to_binary_bytes();
        for cut in [0, 10, MAGIC.len() + 2, bytes.len() / 2, bytes.len() - 1] {
            assert!(Dataset::from_binary_bytes(&bytes[..cut], Path::new("t")).is_err(), "cut at {cut}");
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(Dataset::from_binary_bytes(&extra, Path::new("t")).is_err());
        let mut next = bytes.clone();
        next[MAGIC.len()] = FORMAT_VERSION as u8 + 1;
        let err = Dataset::from_binary_bytes(&next, Path::new("t")).unwrap_err();
        assert!(matches!(err, LgbmError::Unsupported(_)), "{err}");
        // version 3 has no multi-value group (empty in the sample)
        assert!(sample().multi_val_group.is_empty());
        let mut v3 = bytes[..bytes.len() - 8].to_vec();
        v3[MAGIC.len()] = 3;
        let back = Dataset::from_binary_bytes(&v3, Path::new("t")).unwrap();
        assert_eq!(back.to_binary_bytes(), bytes);
        // version 2 has no bin-parameter section either (an empty
        // max_bin_by_feature and an empty bound list per column)
        let tail = 8 + 8 * sample().num_total_features();
        let mut v2 = v3[..v3.len() - tail].to_vec();
        v2[MAGIC.len()] = 2;
        let back = Dataset::from_binary_bytes(&v2, Path::new("t")).unwrap();
        assert_eq!(back.to_binary_bytes(), bytes);
        // version 1 also has no label_idx (after magic, version, 2 sizes, 3 i32 and 2 bool fields)
        let at = MAGIC.len() + 4 + 8 + 8 + 12 + 2;
        let mut v1 = v2.clone();
        v1[MAGIC.len()] = 1;
        v1.drain(at..at + 4);
        let back = Dataset::from_binary_bytes(&v1, Path::new("t")).unwrap();
        assert_eq!(back.label_idx, 0);
        assert_eq!(back.metadata.label, sample().metadata.label);
    }

    #[test]
    fn max_bin_by_feature_and_forced_bins_round_trip_and_are_checked() {
        let dir = std::env::temp_dir().join(format!("lgbmrs-binary-mbf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let forced = dir.join("forced.json");
        std::fs::write(&forced, r#"[{"feature": 0, "bin_upper_bound": [3.0, 7.5, 7.5]}]"#).unwrap();
        let n = 200;
        let x: Vec<f64> = (0..n * 2).map(|i| ((i * 7) % 23) as f64 * 0.5).collect();
        let label: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();
        let mat = DenseMatrix::from_f64_row_major(&x, n, 2).unwrap();
        let pairs = |mbf: &str| {
            let f = forced.to_str().unwrap().to_string();
            let mut v = vec![("forcedbins_filename".to_string(), f)];
            if !mbf.is_empty() {
                v.push(("max_bin_by_feature".to_string(), mbf.to_string()));
            }
            Config::from_pairs(v).unwrap()
        };
        let cfg = pairs("4,6");
        let ds = Dataset::from_dense(&mat, DatasetFields { label: &label, ..Default::default() }, &cfg).unwrap();
        assert_eq!(ds.forced_bin_bounds, vec![vec![3.0, 7.5], vec![]]);
        assert_eq!(ds.max_bin_by_feature, vec![4, 6]);
        let back = Dataset::from_binary_bytes(&ds.to_binary_bytes(), Path::new("m")).unwrap();
        assert_eq!(back.forced_bin_bounds, ds.forced_bin_bounds);
        assert_eq!(back.max_bin_by_feature, ds.max_bin_by_feature);

        let path = dir.join("d.bin");
        let _ = std::fs::remove_file(&path);
        let p = path.to_str().unwrap();
        ds.save_binary(p).unwrap();
        assert_eq!(Dataset::load_binary(p, Some(&cfg)).unwrap().max_bin_by_feature, vec![4, 6]);
        // a given value replaces the stored one, as upstream
        assert_eq!(Dataset::load_binary(p, Some(&pairs("5,5"))).unwrap().max_bin_by_feature, vec![5, 5]);
        let err = Dataset::load_binary(p, Some(&pairs(""))).unwrap_err();
        assert!(err.to_string().contains("max_bin_by_feature cannot be changed"), "{err}");
        let err = Dataset::load_binary(p, Some(&pairs("5"))).unwrap_err();
        assert!(err.to_string().contains("(config_.max_bin_by_feature.size())"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_checks_construction_parameters() {
        let dir = std::env::temp_dir().join(format!("lgbmrs-binary-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("d.bin");
        let _ = std::fs::remove_file(&path);
        let p = path.to_str().unwrap();
        let (r, w) = crate::log::capture(|| sample().save_binary(p));
        r.unwrap();
        assert_eq!(w, vec![format!("Saving data to binary file {p}")]);
        let (r, w) = crate::log::capture(|| sample().save_binary(p));
        r.unwrap();
        assert_eq!(w, vec![format!("File {p} exists, cannot save binary to it")]);
        let ok = Config::from_pairs([("max_bin", "63")]).unwrap();
        assert!(Dataset::load_binary(p, Some(&ok)).is_ok());
        let err = Dataset::load_binary(p, Some(&Config::default())).unwrap_err();
        assert!(err.to_string().contains("parameter max_bin=63. It cannot be changed to 255"), "{err}");
        let upstream = dir.join("up.bin");
        std::fs::write(&upstream, UPSTREAM_TOKEN).unwrap();
        let err = Dataset::load_binary(upstream.to_str().unwrap(), None).unwrap_err();
        assert!(err.to_string().contains("upstream LightGBM binary dataset"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
