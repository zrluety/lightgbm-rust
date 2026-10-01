//! Discounted cumulative gain for the ranking objectives and metrics.
//!
//! upstream: src/metric/dcg_calculator.cpp. Upstream keeps the label gains
//! and discounts in static members shared by every objective and metric;
//! here each user owns a [`DcgCalculator`].

use crate::consts::K_EPSILON;
use crate::dataset::Metadata;
use crate::error::{LgbmError, Result};

/// upstream `DCGCalculator::kMaxPosition`.
pub const K_MAX_POSITION: usize = 10000;

/// upstream `DCGCalculator::DefaultEvalAt`.
pub fn default_eval_at(eval_at: &[i32]) -> Result<Vec<i32>> {
    if eval_at.is_empty() {
        return Ok((1..=5).collect());
    }
    for &k in eval_at {
        if k <= 0 {
            return Err(LgbmError::InvalidParameter("Check failed: (ref_eval_at[i]) > (0)".into()));
        }
    }
    Ok(eval_at.to_vec())
}

/// upstream `DCGCalculator::DefaultLabelGain`: `2^i - 1` for `i < 31`.
pub fn default_label_gain(label_gain: &[f64]) -> Vec<f64> {
    if !label_gain.is_empty() {
        return label_gain.to_vec();
    }
    let mut g = vec![0.0];
    g.extend((1..31).map(|i| ((1i32 << i) - 1) as f64));
    g
}

#[derive(Debug, Clone)]
pub struct DcgCalculator {
    label_gain: Vec<f64>,
    discount: Vec<f64>,
}

impl DcgCalculator {
    /// upstream `DCGCalculator::Init`.
    pub fn new(label_gain: Vec<f64>) -> Self {
        let discount = (0..K_MAX_POSITION).map(|i| 1.0 / (2.0 + i as f64).log2()).collect();
        Self { label_gain, discount }
    }

    pub fn label_gain(&self) -> &[f64] {
        &self.label_gain
    }

    #[inline]
    pub fn discount(&self, k: usize) -> f64 {
        self.discount[k]
    }

    fn label_counts(&self, label: &[f32]) -> Vec<i32> {
        let mut cnt = vec![0i32; self.label_gain.len()];
        for &l in label {
            cnt[l as i32 as usize] += 1;
        }
        cnt
    }

    /// upstream `DCGCalculator::CalMaxDCGAtK`.
    pub fn max_dcg_at_k(&self, k: usize, label: &[f32]) -> f64 {
        let mut cnt = self.label_counts(label);
        let mut top = self.label_gain.len() as i32 - 1;
        let mut ret = 0.0f64;
        for j in 0..k.min(label.len()) {
            while top > 0 && cnt[top as usize] <= 0 {
                top -= 1;
            }
            if top < 0 {
                break;
            }
            ret += self.discount[j] * self.label_gain[top as usize];
            cnt[top as usize] -= 1;
        }
        ret
    }

    /// upstream `DCGCalculator::CalMaxDCG`: max DCG at every `ks` (ascending).
    pub fn max_dcg(&self, ks: &[i32], label: &[f32], out: &mut [f64]) {
        let mut cnt = self.label_counts(label);
        let mut top = self.label_gain.len() as i32 - 1;
        let mut cur = 0.0f64;
        let mut left = 0usize;
        for (o, &k) in out.iter_mut().zip(ks) {
            let k = (k as usize).min(label.len());
            for j in left..k {
                while top > 0 && cnt[top as usize] <= 0 {
                    top -= 1;
                }
                if top < 0 {
                    break;
                }
                cur += self.discount[j] * self.label_gain[top as usize];
                cnt[top as usize] -= 1;
            }
            *o = cur;
            left = k;
        }
    }

    /// upstream `DCGCalculator::CalDCG`: DCG at every `ks` of the ranking by `score`.
    pub fn dcg(&self, ks: &[i32], label: &[f32], score: &[f64], out: &mut [f64]) {
        let sorted = sort_by_score_desc(score);
        let mut cur = 0.0f64;
        let mut left = 0usize;
        for (o, &k) in out.iter_mut().zip(ks) {
            let k = (k as usize).min(label.len());
            for (j, &idx) in sorted.iter().enumerate().take(k).skip(left) {
                cur += self.label_gain[label[idx] as i32 as usize] * self.discount[j];
            }
            *o = cur;
            left = k;
        }
    }

    /// upstream `DCGCalculator::CheckLabel`.
    pub fn check_label(&self, label: &[f32]) -> Result<()> {
        for &l in label {
            let delta = (l - l as i32 as f32).abs();
            if delta as f64 > K_EPSILON {
                return Err(LgbmError::InvalidData(format!(
                    "label should be int type (met {l:.6}) for ranking task,\n\
                     for the gain of label, please set the label_gain parameter"
                )));
            }
            if l < 0.0 {
                return Err(LgbmError::InvalidData(format!(
                    "Label should be non-negative (met {l:.6}) for ranking task"
                )));
            }
            if l as usize >= self.label_gain.len() {
                return Err(LgbmError::InvalidData(format!(
                    "Label {} is not less than the number of label mappings ({})",
                    l as usize,
                    self.label_gain.len()
                )));
            }
        }
        Ok(())
    }
}

/// upstream `DCGCalculator::CheckMetadata`.
pub fn check_metadata(meta: &Metadata) -> Result<()> {
    if let Some(b) = &meta.query_boundaries {
        for q in b.windows(2) {
            let rows = q[1] - q[0];
            if rows as usize > K_MAX_POSITION {
                return Err(LgbmError::InvalidData(format!(
                    "Number of rows {rows} exceeds upper limit of {K_MAX_POSITION} for a query"
                )));
            }
        }
    }
    Ok(())
}

/// Indices ordered by descending score, ties in index order
/// (upstream `std::stable_sort` with `score[a] > score[b]`).
pub fn sort_by_score_desc(score: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    idx.sort_by(|&a, &b| {
        if score[a] > score[b] {
            std::cmp::Ordering::Less
        } else if score[b] > score[a] {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_gains_and_eval_at() {
        let g = default_label_gain(&[]);
        assert_eq!(g.len(), 31);
        assert_eq!(&g[..4], &[0.0, 1.0, 3.0, 7.0]);
        assert_eq!(g[30], ((1i64 << 30) - 1) as f64);
        assert_eq!(default_eval_at(&[]).unwrap(), vec![1, 2, 3, 4, 5]);
        assert!(default_eval_at(&[3, 0]).is_err());
    }

    #[test]
    fn dcg_values() {
        let c = DcgCalculator::new(default_label_gain(&[]));
        let label = [0.0f32, 2.0, 1.0];
        // ideal order 2,1,0: 3/log2(2) + 1/log2(3)
        let ideal = 3.0 + 1.0 / 3f64.log2();
        assert_eq!(c.max_dcg_at_k(3, &label), ideal);
        assert_eq!(c.max_dcg_at_k(1, &label), 3.0);
        let mut out = [0.0; 2];
        c.max_dcg(&[1, 10], &label, &mut out);
        assert_eq!(out, [3.0, ideal]);
        // scores rank rows 0, 1, 2 (the tie keeps index order)
        c.dcg(&[1, 3], &label, &[1.0, 0.0, 0.0], &mut out);
        assert_eq!(out, [0.0, 3.0 / 3f64.log2() + 1.0 / 4f64.log2()]);
    }

    #[test]
    fn check_label_messages() {
        let c = DcgCalculator::new(default_label_gain(&[]));
        assert!(c.check_label(&[0.0, 3.0]).is_ok());
        let e = c.check_label(&[0.5]).unwrap_err().to_string();
        assert!(e.contains("label should be int type (met 0.500000)"), "{e}");
        let e = c.check_label(&[-1.0]).unwrap_err().to_string();
        assert!(e.contains("Label should be non-negative (met -1.000000)"), "{e}");
        let e = c.check_label(&[31.0]).unwrap_err().to_string();
        assert!(e.contains("Label 31 is not less than the number of label mappings (31)"), "{e}");
    }
}
