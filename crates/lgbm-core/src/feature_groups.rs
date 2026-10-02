//! Upstream's feature-group order (exclusive feature bundling).
//!
//! upstream: src/io/dataset.cpp (`FastFeatureBundling`, `FindGroups`,
//! `FixSampleIndices`, `GetConflictCount`, `MarkUsed`, and the group loop of
//! `Dataset::Construct`). Upstream numbers its inner features in shuffled
//! group order; this engine keeps inner features in column order and
//! records upstream's numbering, which seeds per-feature generators (extra
//! trees uses `extra_seed + inner`).

use crate::binning::BinMapper;
use crate::random::Random;

/// Non-zero entries of one column within the bin-construction sample:
/// positions in the sample (ascending) and the values.
pub struct SampleColumn {
    pub indices: Vec<i32>,
    pub values: Vec<f64>,
}

/// upstream: `FixSampleIndices`.
fn fix_sample_indices(m: &BinMapper, total: i32, col: &SampleColumn) -> Vec<i32> {
    let mut ret = Vec::new();
    if m.default_bin == m.most_freq_bin {
        return ret;
    }
    let (mut i, mut j) = (0i32, 0usize);
    let idx = &col.indices;
    while i < total {
        if j < idx.len() && idx[j] < i {
            j += 1;
        } else if j < idx.len() && idx[j] == i {
            if m.value_to_bin(col.values[j]) != m.most_freq_bin {
                ret.push(i);
            }
            i += 1;
        } else {
            ret.push(i);
            i += 1;
        }
    }
    ret
}

/// upstream: `GetConflictCount`.
fn conflict_count(mark: &[bool], indices: &[i32], max_cnt: i64) -> i64 {
    let mut ret = 0i64;
    for &i in indices {
        if mark[i as usize] {
            ret += 1;
        }
        if ret > max_cnt {
            return -1;
        }
    }
    ret
}

fn mark_used(mark: &mut [bool], indices: &[i32]) {
    for &i in indices {
        mark[i as usize] = true;
    }
}

/// upstream: `FindGroups` (CPU: `is_use_gpu = false`): the groups and
/// whether each is a multi-value group.
fn find_groups(
    find_order: &[usize],
    sample_indices: &[&[i32]],
    total_sample_cnt: i64,
    num_data: i32,
    is_sparse: bool,
) -> (Vec<Vec<usize>>, Vec<bool>) {
    const MAX_SEARCH_GROUP: i32 = 100;
    let single_val_max_conflict_cnt = total_sample_cnt / 10000;
    let mut rand = Random::new(num_data);
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut marks: Vec<Vec<bool>> = Vec::new();
    let mut used_row_cnt: Vec<i64> = Vec::new();
    let mut total_data_cnt: Vec<i64> = Vec::new();
    for &fidx in find_order {
        let idx = sample_indices[fidx];
        let cur_non_zero_cnt = idx.len() as i64;
        let available: Vec<usize> = (0..groups.len())
            .filter(|&g| total_data_cnt[g] + cur_non_zero_cnt <= total_sample_cnt + single_val_max_conflict_cnt)
            .collect();
        let mut search = Vec::new();
        if !available.is_empty() {
            let last = available.len() as i32 - 1;
            let picked = rand.sample(last, last.min(MAX_SEARCH_GROUP - 1));
            search.push(*available.last().unwrap());
            search.extend(picked.iter().map(|&i| available[i as usize]));
        }
        let mut best: Option<(usize, i64)> = None;
        for g in search {
            let rest_max_cnt = single_val_max_conflict_cnt - total_data_cnt[g] + used_row_cnt[g];
            let cnt = conflict_count(&marks[g], idx, rest_max_cnt);
            if cnt >= 0 && cnt <= rest_max_cnt && cnt <= cur_non_zero_cnt / 2 {
                best = Some((g, cnt));
                break;
            }
        }
        match best {
            Some((g, cnt)) => {
                groups[g].push(fidx);
                total_data_cnt[g] += cur_non_zero_cnt;
                used_row_cnt[g] += cur_non_zero_cnt - cnt;
                mark_used(&mut marks[g], idx);
            }
            None => {
                groups.push(vec![fidx]);
                let mut m = vec![false; total_sample_cnt as usize];
                mark_used(&mut m, idx);
                marks.push(m);
                total_data_cnt.push(cur_non_zero_cnt);
                used_row_cnt.push(cur_non_zero_cnt);
            }
        }
    }
    if !is_sparse {
        let n = groups.len();
        return (groups, vec![false; n]);
    }
    // Second round: sparse groups are merged into one trailing group, which is
    // a multi-value group once its features conflict too often.
    const DENSE_THRESHOLD: f64 = 0.4;
    let mut kept = Vec::new();
    let mut second_round = Vec::new();
    for (g, feats) in groups.into_iter().enumerate() {
        if used_row_cnt[g] as f64 / total_sample_cnt as f64 >= DENSE_THRESHOLD {
            kept.push(feats);
        } else {
            second_round.extend(feats);
        }
    }
    let mut multi_val = vec![false; kept.len()];
    if !second_round.is_empty() {
        let mut mark = vec![false; total_sample_cnt as usize];
        let mut is_multi_val = false;
        let mut conflict_cnt = 0i64;
        for &fidx in &second_round {
            if !is_multi_val {
                let rest_max_cnt = single_val_max_conflict_cnt - conflict_cnt;
                let cnt = conflict_count(&mark, sample_indices[fidx], rest_max_cnt);
                conflict_cnt += cnt;
                if cnt < 0 || conflict_cnt > single_val_max_conflict_cnt {
                    is_multi_val = true;
                    continue;
                }
                mark_used(&mut mark, sample_indices[fidx]);
            }
        }
        kept.push(second_round);
        multi_val.push(is_multi_val);
    }
    (kept, multi_val)
}

/// One of upstream's feature groups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupSpec {
    /// Real column indices, in group order.
    pub features: Vec<usize>,
    pub is_multi_val: bool,
}

/// Upstream's feature groups of the used features (real column indices,
/// ascending), in upstream group order. `columns` holds every column's
/// sampled non-zero entries.
pub fn upstream_groups(
    bin_mappers: &[BinMapper],
    used_features: &[usize],
    columns: &[SampleColumn],
    total_sample_cnt: usize,
    num_data: usize,
    enable_bundle: bool,
    is_sparse: bool,
) -> Vec<GroupSpec> {
    if !enable_bundle || used_features.is_empty() {
        return used_features.iter().map(|&f| GroupSpec { features: vec![f], is_multi_val: false }).collect();
    }
    let total = total_sample_cnt as i64;
    // upstream FastFeatureBundling: dense features first
    let mut sorted: Vec<usize> = (0..used_features.len()).collect();
    sorted.sort_by(|&a, &b| columns[used_features[b]].indices.len().cmp(&columns[used_features[a]].indices.len()));
    let by_cnt: Vec<usize> = sorted.iter().map(|&i| used_features[i]).collect();

    let fixed: Vec<Vec<i32>> = used_features
        .iter()
        .map(|&f| fix_sample_indices(&bin_mappers[f], total as i32, &columns[f]))
        .collect();
    let mut sample_indices: Vec<&[i32]> = columns.iter().map(|c| c.indices.as_slice()).collect();
    for (k, &f) in used_features.iter().enumerate() {
        if !fixed[k].is_empty() {
            sample_indices[f] = fixed[k].as_slice();
        }
    }
    let (mut groups, mut multi_val) = find_groups(used_features, &sample_indices, total, num_data as i32, is_sparse);
    let (groups2, multi_val2) = find_groups(&by_cnt, &sample_indices, total, num_data as i32, is_sparse);
    if groups.len() > groups2.len() {
        groups = groups2;
        multi_val = multi_val2;
    }
    let num_group = groups.len() as i32;
    let mut rand = Random::new(num_data as i32);
    for i in 0..num_group - 1 {
        let j = rand.next_short(i + 1, num_group);
        groups.swap(i as usize, j as usize);
        multi_val.swap(i as usize, j as usize);
    }
    groups.into_iter().zip(multi_val).map(|(features, is_multi_val)| GroupSpec { features, is_multi_val }).collect()
}
