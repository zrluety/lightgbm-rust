//! Quantized gradients (`use_quantized_grad`).
//!
//! upstream: src/treelearner/gradient_discretizer.cpp / .hpp and the
//! quantized branches of `LeafSplits::Init` (leaf_splits.hpp).
//!
//! Gradients and hessians are rounded to small integers per row, and
//! histograms hold packed integer sums, which are exact in any order.
//! Upstream stores those sums in 8-, 16- or 32-bit cells chosen by leaf
//! size; the sizes are chosen so that the sums fit, so 64-bit cells give the
//! same values.

use crate::mt19937::Mt19937;
use crate::multi_val_bin::block_info;

/// upstream `GradientDiscretizer`.
pub(crate) struct GradientDiscretizer {
    num_grad_quant_bins: i32,
    is_constant_hessian: bool,
    stochastic_rounding: bool,
    gradient_random_values: Vec<f64>,
    hessian_random_values: Vec<f64>,
    start_eng: Mt19937,
    /// Upper end of upstream's `random_values_use_start_dist_` (`num_data` at `Init`).
    start_max: i32,
    /// `[hessian, gradient]` of each row (upstream's int8 pairs), by row.
    raw: Vec<[i8; 2]>,
    /// `(gradient << 32) | hessian` of each row, the hessian read as an
    /// unsigned byte (upstream's packing of the int8 pairs), by row.
    pub packed: Vec<i64>,
    pub grad_scale: f64,
    pub hess_scale: f64,
}

/// `static_cast<int8_t>(double)` as x86-64 compiles it (`cvttsd2si` to 32
/// bits, then the low byte).
#[inline(always)]
fn to_i8(x: f64) -> i8 {
    let t = if x.is_nan() || !(-2_147_483_648.0..2_147_483_648.0).contains(&x) { i32::MIN } else { x as i32 };
    t as i8
}

impl GradientDiscretizer {
    /// upstream: the constructor and `Init`. The random values depend on the
    /// thread count, as upstream's per-thread generators do.
    pub fn new(
        num_grad_quant_bins: i32,
        seed: i32,
        is_constant_hessian: bool,
        stochastic_rounding: bool,
        num_data: usize,
        num_threads: usize,
    ) -> Self {
        let mut gradient_random_values = vec![0.0; num_data];
        let mut hessian_random_values = vec![0.0; num_data];
        let (num_blocks, block_size) = block_info(num_threads, num_data, 512);
        for b in 0..num_blocks {
            let start = b * block_size;
            let end = (start + block_size).min(num_data);
            let mut ge = Mt19937::new(seed.wrapping_add(b as i32) as u32);
            let mut he = Mt19937::new(seed.wrapping_add(b as i32).wrapping_add(num_threads as i32) as u32);
            for i in start..end {
                gradient_random_values[i] = ge.uniform01();
                hessian_random_values[i] = he.uniform01();
            }
        }
        Self {
            num_grad_quant_bins,
            is_constant_hessian,
            stochastic_rounding,
            gradient_random_values,
            hessian_random_values,
            start_eng: Mt19937::new(seed as u32),
            start_max: num_data as i32,
            raw: vec![[0; 2]; num_data],
            packed: vec![0; num_data],
            grad_scale: 0.0,
            hess_scale: 0.0,
        }
    }

    pub fn set_constant_hessian(&mut self, is_constant_hessian: bool) {
        self.is_constant_hessian = is_constant_hessian;
    }

    /// The row count the tables were drawn for.
    pub fn num_data(&self) -> usize {
        self.raw.len()
    }

    /// upstream `DiscretizeGradients` on the learner's rows: `rows[i]` is
    /// row `i` of upstream's (bag subset) arrays, or `i` itself when `None`.
    pub fn discretize(&mut self, grad: &[f32], hess: &[f32], rows: Option<&[u32]>) {
        let n = rows.map_or(grad.len(), |r| r.len());
        if n == 0 {
            return;
        }
        let row = |i: usize| rows.map_or(i, |r| r[i] as usize);
        let mut max_gradient = (grad[row(0)] as f64).abs();
        let mut max_hessian = (hess[row(0)] as f64).abs();
        for i in 0..n {
            let (g, h) = ((grad[row(i)] as f64).abs(), (hess[row(i)] as f64).abs());
            if g > max_gradient {
                max_gradient = g;
            }
            if h > max_hessian {
                max_hessian = h;
            }
        }
        let bins = self.num_grad_quant_bins;
        self.grad_scale = max_gradient / (bins / 2) as f64;
        self.hess_scale = if self.is_constant_hessian { max_hessian } else { max_hessian / bins as f64 };
        let inv_g = 1.0 / self.grad_scale;
        let inv_h = 1.0 / self.hess_scale;
        let start = self.start_eng.uniform_int(0, self.start_max) as usize;
        for i in 0..n {
            let r = row(i);
            let g = grad[r] as f64;
            let (rg, rh) = if self.stochastic_rounding {
                let pos = (i + start) % n;
                (self.gradient_random_values[pos], self.hessian_random_values[pos])
            } else {
                (0.5, 0.5)
            };
            let gq = if g >= 0.0 { to_i8(g * inv_g + rg) } else { to_i8(g * inv_g - rg) };
            let hq = if self.is_constant_hessian { 1 } else { to_i8(hess[r] as f64 * inv_h + rh) };
            self.raw[r] = [hq, gq];
            self.packed[r] = ((gq as i64) << 32) | (hq as u8 as i64);
        }
    }

    /// upstream `LeafSplits::Init` over all of the learner's rows (`rows`,
    /// in partition order): the dequantized sums and the packed integer sum.
    pub fn root_sums(&self, rows: &[u32]) -> (f64, f64, i64) {
        let (mut sg, mut sh, mut si) = (0.0f64, 0.0f64, 0i64);
        for &r in rows {
            let [h, g] = self.raw[r as usize];
            sg += g as f64 * self.grad_scale;
            sh += h as f64 * self.hess_scale;
            si = si.wrapping_add(self.packed[r as usize]);
        }
        (sg, sh, si)
    }

    /// upstream `LeafSplits::Init(leaf, data_partition, int8 gradients,
    /// float scales)` on a bag of rows: the sums use `float` scales, and the
    /// packed sum reads rows `0..rows.len()` instead of the bag (upstream
    /// indexes it by position).
    pub fn bag_root_sums(&self, rows: &[u32]) -> (f64, f64, i64) {
        let (gs, hs) = (self.grad_scale as f32, self.hess_scale as f32);
        let (mut sg, mut sh, mut si) = (0.0f64, 0.0f64, 0i64);
        for (i, &r) in rows.iter().enumerate() {
            let [h, g] = self.raw[r as usize];
            sg += (g as f32 * gs) as f64;
            sh += (h as f32 * hs) as f64;
            si = si.wrapping_add(self.packed[i]);
        }
        (sg, sh, si)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_casts_like_x86() {
        assert_eq!(to_i8(2.7), 2);
        assert_eq!(to_i8(-2.7), -2);
        assert_eq!(to_i8(200.0), -56);
        assert_eq!(to_i8(f64::NAN), 0);
        assert_eq!(to_i8(1e300), 0);
    }

    #[test]
    fn packs_gradient_high_and_hessian_low() {
        let mut d = GradientDiscretizer::new(4, 0, false, false, 2, 1);
        d.discretize(&[1.0, -0.5], &[0.5, 0.25], None);
        // grad scale 0.5, hess scale 0.125: rows are (2, 4) and (-1, 2)
        assert_eq!(d.raw, vec![[4, 2], [2, -1]]);
        assert_eq!(d.packed[1], (-1i64 << 32) | 2);
        let (sg, sh, si) = d.root_sums(&[0, 1]);
        assert_eq!((sg, sh), (0.5, 0.75));
        assert_eq!(si, (1i64 << 32) | 6);
    }
}
