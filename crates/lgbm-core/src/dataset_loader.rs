//! Datasets from CSV, TSV and LibSVM text files.
//!
//! upstream: src/io/dataset_loader.cpp (`SetHeader`, `LoadFromFile`,
//! `LoadFromFileAlignWithOtherDataset`, `LoadTextDataToMemory`,
//! `SampleTextDataFromMemory`, `SampleTextDataFromFile`,
//! `ConstructBinMappersFromTextData`, `ExtractFeaturesFromMemory`,
//! `ExtractFeaturesFromFile`, `CheckDataset`) and src/io/metadata.cpp
//! (`Init(const char*)`, `LoadWeights`, `LoadQueryBoundaries`, `LoadPositions`,
//! `LoadInitialScore`, `CalculateQueryBoundaries`, `CheckOrPartition`).
//!
//! Parsed rows are pushed into the feature groups batch by batch, as
//! upstream's `PushOneRow` does, so only the bins and the labels, weights
//! and query ids are kept; the sample, the bin mappers, the ignored columns
//! and the metadata follow the text-file code paths.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::config::Config;
use crate::consts::K_ZERO_THRESHOLD;
use crate::dataset::{
    avoid_inf_f32, avoid_inf_f64, check_max_bin_by_feature, find_bin_mappers, sanitize_feature_names,
    train_layout, with_num_threads, Dataset, Layout,
};
use crate::error::{LgbmError, Result};
use crate::feature_groups::SampleColumn;
use crate::random::Random;
use crate::text_parser::{self, Parser};

/// Lines parsed per parallel batch when streaming a file (`two_round`).
const BATCH_LINES: usize = 1 << 16;

/// The operand upstream's `max_bin_by_feature` size check names for text files.
const NTF_EXPR: &str = "static_cast<size_t>(dataset->num_total_features_)";

/// Column roles resolved from the parameters and the header line.
///
/// upstream: `DatasetLoader::SetHeader`.
#[derive(Debug, Default)]
struct Columns {
    label_idx: i32,
    weight_idx: i32,
    group_idx: i32,
    ignore: HashSet<i32>,
    categorical: HashSet<i32>,
    /// Header names without the label column; empty without a header.
    feature_names: Vec<String>,
    header_bytes: usize,
    /// The skipped header line (`None` without a header).
    header_line: Option<String>,
}

impl Columns {
    /// upstream: the debug line of one more `TextReader` over the data file.
    fn reopen(&self, filename: &str) {
        if let Some(h) = &self.header_line {
            text_parser::log_skipped_header(filename, h);
        }
    }
}

fn not_a_number(what: &str) -> LgbmError {
    LgbmError::InvalidParameter(format!(
        "{what} is not a number,\nif you want to use a column name,\nplease add the prefix \"name:\" to the column name"
    ))
}

/// upstream: `Common::Split(str, ',')` (empty tokens dropped).
fn split_commas(s: &str) -> impl Iterator<Item = &str> {
    s.split(',').filter(|t| !t.is_empty())
}

fn set_header(filename: &str, cfg: &Config) -> Result<Columns> {
    let mut c = Columns { label_idx: 0, weight_idx: -1, group_idx: -1, ..Default::default() };
    let mut name2idx: HashMap<String, i32> = HashMap::new();
    if cfg.header {
        let (first_line, skip) = text_parser::read_header(filename)?;
        c.header_bytes = skip;
        c.header_line = Some(first_line.clone());
        // upstream: Common::Split(first_line, "\t,")
        c.feature_names = first_line.split(['\t', ',']).filter(|t| !t.is_empty()).map(str::to_string).collect();
    }
    if !cfg.label_column.is_empty() {
        if let Some(name) = cfg.label_column.strip_prefix("name:") {
            c.label_idx = c.feature_names.iter().position(|n| n == name).map_or(-1, |i| i as i32);
            if c.label_idx < 0 {
                return Err(LgbmError::InvalidParameter(format!(
                    "Could not find label column {name} in data file \nor data file doesn't contain header"
                )));
            }
            crate::log::info(&format!("Using column {name} as label"));
        } else {
            c.label_idx = text_parser::atoi_and_check(&cfg.label_column).ok_or_else(|| not_a_number("label_column"))?;
            crate::log::info(&format!("Using column number {} as label", c.label_idx));
        }
    }
    if !c.feature_names.is_empty() {
        // upstream erases without a bounds check; a label outside the header is caught by later checks
        if c.label_idx >= 0 && (c.label_idx as usize) < c.feature_names.len() {
            c.feature_names.remove(c.label_idx as usize);
        }
        for (i, n) in c.feature_names.iter().enumerate() {
            name2idx.insert(n.clone(), i as i32);
        }
    }
    if !cfg.ignore_column.is_empty() {
        if let Some(names) = cfg.ignore_column.strip_prefix("name:") {
            for name in split_commas(names) {
                let i = name2idx.get(name).ok_or_else(|| {
                    LgbmError::InvalidParameter(format!("Could not find ignore column {name} in data file"))
                })?;
                c.ignore.insert(*i);
            }
        } else {
            for t in split_commas(&cfg.ignore_column) {
                c.ignore.insert(text_parser::atoi_and_check(t).ok_or_else(|| not_a_number("ignore_column"))?);
            }
        }
    }
    let mut role = |value: &str, what: &str, missing: &str, used_as: &str| -> Result<i32> {
        let idx = match value.strip_prefix("name:") {
            Some(name) => {
                let idx = *name2idx.get(name).ok_or_else(|| {
                    LgbmError::InvalidParameter(format!("Could not find {missing} column {name} in data file"))
                })?;
                crate::log::info(&format!("Using column {name} as {used_as}"));
                idx
            }
            None => {
                let idx = text_parser::atoi_and_check(value).ok_or_else(|| not_a_number(what))?;
                crate::log::info(&format!("Using column number {idx} as {used_as}"));
                idx
            }
        };
        c.ignore.insert(idx);
        Ok(idx)
    };
    if !cfg.weight_column.is_empty() {
        c.weight_idx = role(&cfg.weight_column, "weight_column", "weight", "weight")?;
    }
    if !cfg.group_column.is_empty() {
        c.group_idx = role(&cfg.group_column, "group_column", "group/query", "group/query id")?;
    }
    c.categorical = categorical_columns(cfg, &name2idx)?;
    Ok(c)
}

/// The `categorical_feature` part of `SetHeader`, which upstream also runs
/// for binary files (with no column names).
fn categorical_columns(cfg: &Config, name2idx: &HashMap<String, i32>) -> Result<HashSet<i32>> {
    let mut out = HashSet::new();
    if let Some(names) = cfg.categorical_feature.strip_prefix("name:") {
        for name in split_commas(names) {
            let i = name2idx.get(name).ok_or_else(|| {
                LgbmError::InvalidParameter(format!("Could not find categorical_feature {name} in data file"))
            })?;
            out.insert(*i);
        }
    } else {
        for t in split_commas(&cfg.categorical_feature) {
            out.insert(text_parser::atoi_and_check(t).ok_or_else(|| not_a_number("categorical_feature"))?);
        }
    }
    Ok(out)
}

/// Metadata read from `<data>.weight`, `.query`, `.position` and `.init`.
#[derive(Default)]
struct SideFiles {
    weights: Option<Vec<f32>>,
    query_boundaries: Option<Vec<i32>>,
    positions: Option<(Vec<i32>, Vec<String>)>,
    init_score: Option<Vec<f64>>,
}

/// upstream: `Metadata::Init(const char* data_filename)`.
fn load_side_files(filename: &str) -> Result<SideFiles> {
    let mut s = SideFiles::default();
    let lines = text_parser::read_all_lines(&format!("{filename}.query"), 0)?;
    if !lines.is_empty() {
        crate::log::info("Calculating query boundaries...");
        let mut b = Vec::with_capacity(lines.len() + 1);
        b.push(0i32);
        for l in &lines {
            let cnt = text_parser::atoi(l, 0).0;
            b.push(b[b.len() - 1].wrapping_add(cnt));
        }
        s.query_boundaries = Some(b);
    }
    let lines = text_parser::read_all_lines(&format!("{filename}.weight"), 0)?;
    if !lines.is_empty() {
        crate::log::info("Loading weights...");
        s.weights = Some(
            lines.iter().map(|l| text_parser::atof(l, 0).map(|(v, _)| avoid_inf_f32(v as f32))).collect::<Result<_>>()?,
        );
    }
    let lines = text_parser::read_all_lines(&format!("{filename}.position"), 0)?;
    if !lines.is_empty() {
        crate::log::info(&format!("Loading positions from {filename}.position ..."));
        let mut ids: HashMap<&[u8], i32> = HashMap::new();
        let mut names = Vec::new();
        let dense = lines
            .iter()
            .map(|l| {
                *ids.entry(l.as_slice()).or_insert_with(|| {
                    names.push(String::from_utf8_lossy(l).into_owned());
                    names.len() as i32 - 1
                })
            })
            .collect();
        s.positions = Some((dense, names));
    }
    if s.weights.is_some() && s.query_boundaries.is_some() {
        crate::log::info("Calculating query weights...");
    }
    s.init_score = load_initial_score(filename)?;
    Ok(s)
}

/// upstream: `Metadata::LoadInitialScore` (`<data>.init`, class-major).
fn load_initial_score(filename: &str) -> Result<Option<Vec<f64>>> {
    let lines = text_parser::read_all_lines(&format!("{filename}.init"), 0)?;
    if lines.is_empty() {
        return Ok(None);
    }
    crate::log::info("Loading initial scores...");
    let split_tabs = |l: &[u8]| -> Vec<Vec<u8>> {
        l.split(|&c| c == b'\t').filter(|t| !t.is_empty()).map(<[u8]>::to_vec).collect()
    };
    let num_class = split_tabs(&lines[0]).len().max(1);
    let n = lines.len();
    let mut init = vec![0.0f64; n * num_class];
    for (i, l) in lines.iter().enumerate() {
        if num_class == 1 {
            init[i] = avoid_inf_f64(text_parser::atof(l, 0)?.0);
        } else {
            let tokens = split_tabs(l);
            if tokens.len() != num_class {
                return Err(LgbmError::InvalidData(
                    "Invalid initial score file. Redundant or insufficient columns".into(),
                ));
            }
            for (k, t) in tokens.iter().enumerate() {
                init[k * n + i] = avoid_inf_f64(text_parser::atof(t, 0)?.0);
            }
        }
    }
    Ok(Some(init))
}

/// Labels, weights and query ids of the parsed rows.
struct Rows {
    label: Vec<f32>,
    weight: Option<Vec<f32>>,
    queries: Option<Vec<i32>>,
}

impl Rows {
    fn new(c: &Columns) -> Self {
        Self { label: Vec::new(), weight: (c.weight_idx >= 0).then(Vec::new), queries: (c.group_idx >= 0).then(Vec::new) }
    }

    /// upstream: the per-row loop of `ExtractFeaturesFromMemory` /
    /// `ExtractFeaturesFromFile`: records each line's label, weight and
    /// query id, and returns its parsed `(column, value)` pairs, which the
    /// caller pushes into the feature groups.
    fn extend(&mut self, lines: &[&[u8]], parser: &Parser, c: &Columns, ntf: usize) -> Result<Vec<Vec<(i32, f64)>>> {
        let parsed: Vec<(f64, Vec<(i32, f64)>)> = lines
            .par_iter()
            .map(|l| {
                let mut out = Vec::new();
                parser.parse_line(l, &mut out).map(|label| (label, out))
            })
            .collect::<Result<_>>()?;
        let ntf = ntf as i32;
        let mut out = Vec::with_capacity(parsed.len());
        for (label, feats) in parsed {
            self.label.push(label as f32);
            let (mut w, mut q) = (0.0f32, 0i32);
            for &(f, v) in &feats {
                if f < 0 {
                    return Err(LgbmError::InvalidData(format!("negative feature index {f} in data file")));
                }
                if f >= ntf {
                    continue;
                }
                if f == c.weight_idx {
                    w = v as f32;
                } else if f == c.group_idx {
                    q = v as i32;
                }
            }
            if let Some(ws) = self.weight.as_mut() {
                ws.push(w);
            }
            if let Some(qs) = self.queries.as_mut() {
                qs.push(q);
            }
            out.push(feats);
        }
        Ok(out)
    }

    fn num_rows(&self) -> usize {
        self.label.len()
    }
}

/// upstream: `DatasetLoader::ConstructBinMappersFromTextData` (the sample
/// parsing; bins are found by [`find_bin_mappers`]).
fn sample_columns(sample: &[Vec<u8>], parser: &Parser) -> Result<Vec<SampleColumn>> {
    let parsed: Vec<Vec<(i32, f64)>> = sample
        .par_iter()
        .map(|l| {
            let mut out = Vec::new();
            parser.parse_line(l, &mut out).map(|_| out)
        })
        .collect::<Result<_>>()?;
    let mut cols: Vec<SampleColumn> = Vec::new();
    for (i, feats) in parsed.into_iter().enumerate() {
        for (f, v) in feats {
            if f < 0 {
                return Err(LgbmError::InvalidData(format!("negative feature index {f} in data file")));
            }
            let f = f as usize;
            if f >= cols.len() {
                cols.resize_with(f + 1, || SampleColumn { indices: Vec::new(), values: Vec::new() });
            }
            if v.abs() > K_ZERO_THRESHOLD || v.is_nan() {
                cols[f].indices.push(i as i32);
                cols[f].values.push(v);
            }
        }
    }
    Ok(cols)
}

/// The bin-construction sample of a training file.
struct Sampled {
    /// All data lines; `None` with `two_round` (the file is read again).
    lines: Option<Vec<Vec<u8>>>,
    sample: Vec<Vec<u8>>,
    num_data: usize,
}

/// Read all data lines, or (with `two_round`) only the reservoir sample and
/// the line count.
///
/// upstream: `LoadTextDataToMemory` + `SampleTextDataFromMemory`, or
/// `SampleTextDataFromFile` (`TextReader::SampleFromFile`).
fn read_and_sample(filename: &str, skip: usize, cfg: &Config) -> Result<Sampled> {
    let mut random = Random::new(cfg.data_random_seed);
    if !cfg.two_round {
        let lines = text_parser::read_all_lines(filename, skip)?;
        let n = lines.len();
        let cnt = if cfg.bin_construct_sample_cnt < 0 || cfg.bin_construct_sample_cnt as usize > n {
            n as i32
        } else {
            cfg.bin_construct_sample_cnt
        };
        let sample = random.sample(n as i32, cnt).into_iter().map(|i| lines[i as usize].clone()).collect();
        return Ok(Sampled { lines: Some(lines), sample, num_data: n });
    }
    let k = cfg.bin_construct_sample_cnt;
    let mut sample: Vec<Vec<u8>> = Vec::new();
    let mut n = 0usize;
    text_parser::for_each_line(filename, skip, |l| {
        if (sample.len() as i32) < k {
            sample.push(l.to_vec());
        } else {
            let idx = random.next_int(0, n as i32 + 1);
            if idx < k {
                sample[idx as usize] = l.to_vec();
            }
        }
        n += 1;
        Ok(())
    })?;
    Ok(Sampled { lines: None, sample, num_data: n })
}

/// Parse the `num_data` data lines in batches, from memory or streaming the
/// file, pushing each batch into `ds`'s feature groups (upstream
/// `PushOneRow`/`FinishOneRow` per line); returns the labels, weights and
/// query ids.
fn extract(
    ds: &mut Dataset,
    filename: &str,
    skip: usize,
    lines: Option<Vec<Vec<u8>>>,
    parser: &Parser,
    c: &Columns,
) -> Result<Rows> {
    let n = ds.num_data;
    let ntf = ds.num_total_features();
    let mut rows = Rows::new(c);
    let mut result = Ok(());
    ds.fill_groups(|b, d| {
        let mut flush = |batch: &[&[u8]], rows: &mut Rows| -> Result<()> {
            let row0 = rows.num_rows();
            if row0 + batch.len() > n {
                return Err(LgbmError::InvalidData(format!("Data file {filename} changed while it was read")));
            }
            let feats = rows.extend(batch, parser, c, ntf)?;
            b.push_rows(row0, feats.len(), &d.real_to_inner, |k, sink| {
                for &(f, v) in &feats[k] {
                    sink(f as usize, v);
                }
            });
            Ok(())
        };
        result = (|| match lines {
            Some(mut lines) => {
                let mut start = 0;
                while start < lines.len() {
                    let end = (start + BATCH_LINES).min(lines.len());
                    let batch: Vec<Vec<u8>> = lines[start..end].iter_mut().map(std::mem::take).collect();
                    let refs: Vec<&[u8]> = batch.iter().map(Vec::as_slice).collect();
                    flush(&refs, &mut rows)?;
                    start = end;
                }
                Ok(())
            }
            None => {
                let mut batch: Vec<Vec<u8>> = Vec::new();
                let mut flush_batch = |batch: &mut Vec<Vec<u8>>, rows: &mut Rows| -> Result<()> {
                    let refs: Vec<&[u8]> = batch.iter().map(Vec::as_slice).collect();
                    flush(&refs, rows)?;
                    batch.clear();
                    Ok(())
                };
                text_parser::for_each_line(filename, skip, |l| {
                    batch.push(l.to_vec());
                    if batch.len() == BATCH_LINES {
                        flush_batch(&mut batch, &mut rows)?;
                    }
                    Ok(())
                })?;
                flush_batch(&mut batch, &mut rows)
            }
        })();
    });
    result?;
    if rows.num_rows() != n {
        return Err(LgbmError::InvalidData(format!("Data file {filename} changed while it was read")));
    }
    Ok(rows)
}

/// upstream: `Metadata::CalculateQueryBoundaries` (runs of equal query ids).
fn boundaries_from_query_ids(q: &[i32]) -> Vec<i32> {
    let mut counts = Vec::new();
    let (mut last, mut cnt) = (None, 0i32);
    for &id in q {
        if last != Some(id) {
            if cnt > 0 {
                counts.push(cnt);
            }
            cnt = 0;
            last = Some(id);
        }
        cnt += 1;
    }
    counts.push(cnt);
    let mut b = vec![0i32];
    for c in counts {
        b.push(b[b.len() - 1] + c);
    }
    b
}

/// upstream: the messages of `Metadata::Init(num_data, weight_idx, query_idx)`.
fn log_metadata_init(c: &Columns, side: &SideFiles) {
    if c.weight_idx >= 0 && side.weights.is_some() {
        crate::log::info("Using weights in data file, ignoring the additional weights file");
    }
    if c.group_idx >= 0 && side.query_boundaries.is_some() {
        crate::log::info("Using query id in data file, ignoring the additional query file");
    }
}

/// Replace the metadata with the file's: labels as parsed, weight and query
/// columns or the side files, positions and initial scores from side files.
///
/// upstream: `Metadata::Init(num_data, weight_idx, query_idx)`, `FinishLoad`
/// and `CheckOrPartition` (without partitioning).
fn set_metadata(ds: &mut Dataset, filename: &str, rows: Rows, side: SideFiles) -> Result<()> {
    let n = ds.num_data;
    if rows.queries.is_some() && (rows.weight.is_some() || side.weights.is_some()) {
        crate::log::info("Calculating query weights...");
    }
    let md = &mut ds.metadata;
    md.label = rows.label;
    md.weight = rows.weight.or(side.weights);
    md.query_boundaries = match rows.queries {
        Some(q) => Some(boundaries_from_query_ids(&q)),
        None => side.query_boundaries,
    };
    md.positions = None;
    md.position_ids.clear();
    if let Some((p, ids)) = side.positions {
        md.positions = Some(p);
        md.position_ids = ids;
    }
    md.init_score = side.init_score;
    if md.weight.as_ref().is_some_and(|w| w.len() != n) {
        return Err(LgbmError::InvalidData("Weights size doesn't match data size".into()));
    }
    if let Some(p) = &md.positions {
        if p.len() != n {
            return Err(LgbmError::InvalidData(format!(
                "Positions size ({}) doesn't match data size ({n})",
                p.len()
            )));
        }
    }
    if md.query_boundaries.as_ref().is_some_and(|b| *b.last().unwrap() as i64 != n as i64) {
        return Err(LgbmError::InvalidData("Query size doesn't match data size".into()));
    }
    if md.init_score.as_ref().is_some_and(|s| s.len() % n != 0) {
        return Err(LgbmError::InvalidData("Initial score size doesn't match data size".into()));
    }
    md.query_weights = None;
    if let (Some(w), Some(b)) = (&md.weight, &md.query_boundaries) {
        md.query_weights = Some(
            b.windows(2)
                .map(|q| {
                    let mut s = 0.0f32;
                    for &x in &w[q[0] as usize..q[1] as usize] {
                        s += x;
                    }
                    s / (q[1] - q[0]) as f32
                })
                .collect(),
        );
    }
    log_num_queries(ds, filename);
    Ok(())
}

/// upstream: the end of `Metadata::CheckOrPartition` (`filename` is the
/// metadata's data file, empty for binary files).
fn log_num_queries(ds: &Dataset, filename: &str) {
    let nq = ds.metadata.query_boundaries.as_ref().map_or(0, |b| b.len().saturating_sub(1));
    if nq > 0 {
        crate::log::debug(&format!(
            "Number of queries in {filename}: {nq}. Average number of rows per query: {:.6}.",
            ds.num_data as f64 / nq as f64
        ));
    }
}

impl Dataset {
    /// Load a dataset from a binary or text file, as upstream's
    /// `LGBM_DatasetCreateFromFile`: `<filename>.bin` or `filename` if either
    /// is a binary dataset file, else `filename` as text. Training datasets
    /// (`reference = None`) loaded from binary files must have been
    /// constructed with `cfg`'s binning parameters.
    ///
    /// upstream: `DatasetLoader::LoadFromFile`, `LoadFromFileAlignWithOtherDataset`.
    pub fn load_file(filename: &str, cfg: &Config, reference: Option<&Dataset>) -> Result<Self> {
        let (path, kind) = crate::binary::detect_data_file(filename)?;
        if kind == crate::binary::DataFileKind::Text {
            return Self::load_text(filename, cfg, reference);
        }
        // upstream: SetHeader (from the DatasetLoader constructor) for a binary file
        categorical_columns(cfg, &HashMap::new())?;
        crate::log::info(&format!("Load from binary file {}", path.display()));
        let mut ds = Self::load_binary(filename, reference.is_none().then_some(cfg))?;
        if let Some(init) = load_initial_score(&path.to_string_lossy())? {
            if init.len() % ds.num_data != 0 {
                return Err(LgbmError::InvalidData("Initial score size doesn't match data size".into()));
            }
            ds.metadata.init_score = Some(init);
        }
        log_num_queries(&ds, "");
        if reference.is_none() {
            // upstream: CheckDataset (is_load_from_binary)
            for (set, name) in [
                (!cfg.label_column.is_empty(), "label_column"),
                (!cfg.weight_column.is_empty(), "weight_column"),
                (!cfg.group_column.is_empty(), "group_column"),
                (!cfg.ignore_column.is_empty(), "ignore_column"),
                (cfg.two_round, "two_round"),
                (cfg.header, "header"),
            ] {
                if set {
                    crate::log::warning(&format!(
                        "Parameter {name} works only in case of loading data directly from text file. \
                         It will be ignored when loading from binary file."
                    ));
                }
            }
        }
        Ok(ds)
    }

    /// Load a training dataset (`reference = None`) or a validation dataset
    /// aligned with `reference` from a CSV, TSV or LibSVM file. Uses
    /// `cfg.num_threads` threads (all cores when <= 0).
    ///
    /// upstream: `DatasetLoader::LoadFromFile` / `LoadFromFileAlignWithOtherDataset`
    /// for text files.
    pub fn load_text(filename: &str, cfg: &Config, reference: Option<&Dataset>) -> Result<Self> {
        with_num_threads(cfg.num_threads, || Self::load_text_impl(filename, cfg, reference))?
    }

    fn load_text_impl(filename: &str, cfg: &Config, reference: Option<&Dataset>) -> Result<Self> {
        if !cfg.explicit.get("parser_config_file").is_none_or(|v| v.is_empty()) {
            return Err(LgbmError::Unsupported("parser_config_file (custom C++ parsers)".into()));
        }
        let c = set_header(filename, cfg)?;
        let parser = Parser::create(filename, cfg.header, 0, c.label_idx, cfg.precise_float_parser)?;
        let side = load_side_files(filename)?;
        let skip = c.header_bytes;

        let Some(reference) = reference else {
            c.reopen(filename);
            let Sampled { lines, sample, num_data: n } = read_and_sample(filename, skip, cfg)?;
            if n == 0 {
                return Err(LgbmError::InvalidData(format!("Data file {filename} is empty")));
            }
            // upstream: CheckSampleSize
            if (sample.len() as f64 / n as f64) < 0.2f32 as f64 && sample.len() < 100_000 {
                crate::log::warning(
                    "Using too small ``bin_construct_sample_cnt`` may encounter unexpected errors and poor accuracy.",
                );
            }
            let mut columns = sample_columns(&sample, &parser)?;
            let found = columns.len();
            let ntf = found.max(parser.num_features().max(0) as usize);
            if !c.feature_names.is_empty() && c.feature_names.len() != ntf {
                return Err(LgbmError::InvalidData(
                    "Check failed: (dataset->num_total_features_) == (static_cast<int>(feature_names_.size()))".into(),
                ));
            }
            check_max_bin_by_feature(cfg, ntf, NTF_EXPR)?;
            if !(c.label_idx >= 0 && c.label_idx as usize <= ntf) {
                return Err(LgbmError::InvalidData(
                    "Check failed: label_idx_ >= 0 && label_idx_ <= dataset->num_total_features_".into(),
                ));
            }
            if !(c.weight_idx < 0 || (c.weight_idx as usize) < ntf) {
                return Err(LgbmError::InvalidData(
                    "Check failed: weight_idx_ < 0 || weight_idx_ < dataset->num_total_features_".into(),
                ));
            }
            if !(c.group_idx < 0 || (c.group_idx as usize) < ntf) {
                return Err(LgbmError::InvalidData(
                    "Check failed: group_idx_ < 0 || group_idx_ < dataset->num_total_features_".into(),
                ));
            }
            columns.resize_with(ntf, || SampleColumn { indices: Vec::new(), values: Vec::new() });
            let (feature_names, replaced) = if c.feature_names.is_empty() {
                ((0..ntf).map(|i| format!("Column_{i}")).collect(), false)
            } else {
                sanitize_feature_names(c.feature_names.clone())?
            };
            let skip_col: Vec<bool> = (0..ntf).map(|i| i >= found || c.ignore.contains(&(i as i32))).collect();
            let is_cat: Vec<bool> = (0..ntf).map(|i| c.categorical.contains(&(i as i32))).collect();
            if is_cat
                .iter()
                .enumerate()
                .any(|(i, &cat)| cat && cfg.monotone_constraints.get(i).is_some_and(|&m| m != 0))
            {
                return Err(LgbmError::InvalidParameter(
                    "The output cannot be monotone with respect to categorical features".into(),
                ));
            }
            let t1 = std::time::Instant::now();
            let found = find_bin_mappers(&columns, sample.len(), n, &is_cat, &skip_col, cfg, NTF_EXPR)?;
            let bin_time = t1.elapsed();
            let layout = train_layout(&found.bin_mappers, &columns, sample.len(), n, cfg);
            drop(columns);
            let mut ds = Self::unfilled(n, found.bin_mappers, feature_names, layout);
            // upstream reads the file again only after constructing the bin mappers
            let (rows, extract_log) = crate::log::defer(|| extract(&mut ds, filename, skip, lines, &parser, &c));
            let rows = match rows {
                Ok(rows) => rows,
                Err(e) => {
                    extract_log.emit();
                    return Err(e);
                }
            };
            ds.forced_bin_bounds = found.forced_bin_bounds;
            ds.finish_construct(cfg, replaced);
            crate::log::info(&format!("Construct bin mappers from text data time {:.2} seconds", bin_time.as_secs_f64()));
            log_metadata_init(&c, &side);
            if cfg.two_round {
                crate::log::info("Making second pass...");
                c.reopen(filename);
            }
            extract_log.emit();
            ds.label_idx = c.label_idx;
            set_metadata(&mut ds, filename, rows, side)?;
            ds.data_filename = Some(filename.to_string());
            return Ok(ds);
        };

        c.reopen(filename);
        let (lines, n) = if cfg.two_round {
            (None, text_parser::count_lines(filename, skip)?)
        } else {
            let lines = text_parser::read_all_lines(filename, skip)?;
            let n = lines.len();
            (Some(lines), n)
        };
        log_metadata_init(&c, &side);
        if cfg.two_round {
            c.reopen(filename);
        }
        if n == 0 {
            return Err(LgbmError::InvalidData(format!("Data file {filename} is empty")));
        }
        let mut ds = Self::unfilled(
            n,
            reference.bin_mappers.clone(),
            reference.feature_names.clone(),
            Layout::Valid(&reference.upstream_inner),
        );
        ds.bin_config = reference.bin_config;
        ds.forced_bin_bounds = reference.forced_bin_bounds.clone();
        ds.label_idx = reference.label_idx;
        let rows = extract(&mut ds, filename, skip, lines, &parser, &c)?;
        set_metadata(&mut ds, filename, rows, side)?;
        ds.data_filename = Some(filename.to_string());
        Ok(ds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::DatasetFields;

    fn write(dir: &std::path::Path, name: &str, text: &str) -> String {
        let p = dir.join(name);
        std::fs::write(&p, text).unwrap();
        p.to_string_lossy().into_owned()
    }

    fn data(sep: &str) -> String {
        let mut s = String::new();
        for i in 0..200 {
            let (a, b) = ((i % 17) as f64 * 0.5, ((i * 7) % 11) as f64);
            s.push_str(&format!("{}{sep}{a}{sep}{b}{sep}{}\n", i % 2, i / 50));
        }
        s
    }

    #[test]
    fn csv_and_tsv_load_alike_and_match_in_memory() {
        let dir = tempdir();
        let csv = write(&dir, "d.csv", &data(","));
        let tsv = write(&dir, "d.tsv", &data("\t"));
        let cfg = Config::from_pairs([("min_data_in_bin", "1"), ("num_threads", "1")]).unwrap();
        let a = Dataset::load_text(&csv, &cfg, None).unwrap();
        let b = Dataset::load_text(&tsv, &cfg, None).unwrap();
        assert_eq!(a.num_data(), 200);
        assert_eq!(a.num_total_features(), 3);
        assert!(a.same_bins_as(&b));
        assert_eq!(a.label(), b.label());
        let mut x = Vec::new();
        for i in 0..200 {
            x.extend([(i % 17) as f64 * 0.5, ((i * 7) % 11) as f64, (i / 50) as f64]);
        }
        let label: Vec<f32> = (0..200).map(|i| (i % 2) as f32).collect();
        let mat = crate::DenseMatrix::from_f64_row_major(&x, 200, 3).unwrap();
        let m = Dataset::from_dense(&mat, DatasetFields { label: &label, ..Default::default() }, &cfg).unwrap();
        assert!(a.same_bins_as(&m));
        for f in 0..a.num_features() {
            assert_eq!(a.bin_indices(f), m.bin_indices(f));
        }
    }

    #[test]
    fn header_columns_and_side_files() {
        let dir = tempdir();
        let mut s = String::from("y,a,w,q,b\n");
        for i in 0..60 {
            s.push_str(&format!("{},{},{},{},{}\n", i % 3, i % 7, 1 + i % 2, i / 20, i % 5));
        }
        let f = write(&dir, "h.csv", &s);
        write(&dir, "h.csv.init", &"0.5\n".repeat(60));
        let cfg = Config::from_pairs([
            ("header", "true"),
            ("weight_column", "name:w"),
            ("group_column", "name:q"),
            ("ignore_column", "name:b"),
            ("min_data_in_bin", "1"),
            ("min_data_in_leaf", "1"),
        ])
        .unwrap();
        let ds = Dataset::load_text(&f, &cfg, None).unwrap();
        assert_eq!(ds.feature_names(), ["a", "w", "q", "b"]);
        assert_eq!(ds.num_features(), 1);
        assert_eq!(ds.weight().unwrap()[..2], [1.0, 2.0]);
        assert_eq!(ds.metadata.query_boundaries.as_deref(), Some(&[0, 20, 40, 60][..]));
        assert_eq!(ds.init_score().unwrap().len(), 60);
        let bad = Config::from_pairs([("header", "true"), ("label_column", "name:zz")]).unwrap();
        let e = Dataset::load_text(&f, &bad, None).unwrap_err();
        assert!(e.to_string().contains("Could not find label column zz"), "{e}");
    }

    #[test]
    fn libsvm_and_two_round() {
        let dir = tempdir();
        let mut s = String::new();
        for i in 0..300 {
            s.push_str(&format!("{} 0:{} 3:{}\n", i % 2, i % 13, (i * 3) % 7));
        }
        let f = write(&dir, "d.svm", &s);
        let one = Config::from_pairs([("bin_construct_sample_cnt", "50")]).unwrap();
        let two = Config::from_pairs([("bin_construct_sample_cnt", "50"), ("two_round", "true")]).unwrap();
        let (a, log) = crate::log::capture(|| Dataset::load_text(&f, &one, None));
        let a = a.unwrap();
        let b = Dataset::load_text(&f, &two, None).unwrap();
        assert_eq!(a.num_total_features(), 4);
        assert_eq!((a.num_data(), b.num_data()), (300, 300));
        assert!(log.iter().any(|w| w.contains("bin_construct_sample_cnt")));
        let v = Dataset::load_text(&f, &one, Some(&a)).unwrap();
        assert!(v.same_bins_as(&a));
    }

    fn tempdir() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let d = std::env::temp_dir()
            .join(format!("lgbm_text_{}_{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&d).unwrap();
        d
    }
}
