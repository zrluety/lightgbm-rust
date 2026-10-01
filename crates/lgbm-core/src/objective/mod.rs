//! Objective function interfaces.
//!
//! Two extension points:
//!
//! * [`RowObjective`]: conventional objectives whose gradient and Hessian for
//!   row `i` depend only on row `i`'s scores (upstream `ObjectiveFunction`).
//! * [`GroupedObjective`]: objectives where rows are grouped (for example all
//!   monthly observations of one loan, ordered in time). These may have
//!   several raw outputs per row and couple rows within a group. They report
//!   their full curvature structure in a [`GradHessBlock`]. If the tree
//!   learner needs a diagonal, the objective says explicitly which
//!   [`DiagonalReduction`] it uses, and that choice is recorded in the model.
//!
//! Score layout follows upstream: class-major, `scores[k * num_data + i]`.

pub mod binary;
pub mod multiclass;
pub mod percentile;
pub mod regression;

use crate::config::Config;
use crate::dataset::Metadata;
use crate::error::{LgbmError, Result};

/// Read-only view of the current raw scores.
#[derive(Debug, Clone, Copy)]
pub struct ScoreView<'a> {
    pub scores: &'a [f64],
    pub num_data: usize,
    pub num_outputs: usize,
}

impl<'a> ScoreView<'a> {
    #[inline]
    pub fn get(&self, output: usize, row: usize) -> f64 {
        self.scores[output * self.num_data + row]
    }
}

/// A conventional per-row objective.
pub trait RowObjective: Send + Sync {
    /// Name as written in the model file (`objective=` line, first token).
    fn name(&self) -> &str;
    /// Number of raw outputs (trees per iteration).
    fn num_outputs(&self) -> usize {
        1
    }
    fn init(&mut self, meta: &Metadata, num_data: usize) -> Result<()>;
    /// Upstream `Log::Warning` messages raised while constructing or
    /// initializing the objective; drained by the booster.
    fn take_warnings(&mut self) -> Vec<String> {
        Vec::new()
    }
    /// Writes `grad[k*n+i]`, `hess[k*n+i]` (f32 like upstream `score_t`).
    fn gradients(&self, scores: ScoreView<'_>, grad: &mut [f32], hess: &mut [f32]);
    /// Initial raw score when `boost_from_average` is on.
    fn boost_from_score(&self, _output: usize) -> f64 {
        0.0
    }
    fn class_need_train(&self, _output: usize) -> bool {
        true
    }
    /// upstream `IsRenewTreeOutput`: leaf values are recomputed from the
    /// residuals after each tree is grown (L1, quantile, MAPE).
    fn is_renew_tree_output(&self) -> bool {
        false
    }
    /// upstream `RenewTreeOutput` for one leaf: `indices` are the leaf's rows
    /// in partition order, `scores` the training scores before this tree.
    /// Only called when [`is_renew_tree_output`](Self::is_renew_tree_output).
    fn renew_leaf_output(&self, _indices: &[u32], _scores: &[f64]) -> f64 {
        unreachable!("renew_leaf_output without is_renew_tree_output")
    }
    /// Map one row's raw outputs to the prediction scale.
    fn convert_output(&self, raw: &[f64], out: &mut [f64]) {
        out.copy_from_slice(raw);
    }
    /// Full `objective=` string, e.g. `binary sigmoid:1`.
    fn to_model_string(&self) -> String;
}

/// Contiguous row ranges, one per group; rows within a group are ordered
/// (for panel data: by time since origination).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupIndex {
    boundaries: Vec<usize>,
}

impl GroupIndex {
    /// Build from group sizes; rows must already be sorted by group.
    pub fn from_sizes(sizes: &[usize]) -> Result<Self> {
        let mut boundaries = Vec::with_capacity(sizes.len() + 1);
        boundaries.push(0);
        for &s in sizes {
            if s == 0 {
                return Err(LgbmError::InvalidData("empty group".into()));
            }
            boundaries.push(boundaries.last().unwrap() + s);
        }
        Ok(Self { boundaries })
    }

    /// Build from a per-row group id; rows of a group must be contiguous.
    pub fn from_row_ids<T: PartialEq>(ids: &[T]) -> Result<Self> {
        let mut sizes = Vec::new();
        let mut i = 0;
        while i < ids.len() {
            let mut j = i + 1;
            while j < ids.len() && ids[j] == ids[i] {
                j += 1;
            }
            sizes.push(j - i);
            i = j;
        }
        let gi = Self::from_sizes(&sizes)?;
        // detect non-contiguous groups (same id appearing in two runs)
        for g in 1..gi.num_groups() {
            let start = gi.range(g).start;
            for h in 0..g {
                if ids[gi.range(h).start] == ids[start] {
                    return Err(LgbmError::InvalidData(
                        "rows of a group must be contiguous".into(),
                    ));
                }
            }
            if g > 2048 {
                break; // quadratic check only on small inputs; callers sort upstream of this
            }
        }
        Ok(gi)
    }

    pub fn num_groups(&self) -> usize {
        self.boundaries.len() - 1
    }

    pub fn num_rows(&self) -> usize {
        *self.boundaries.last().unwrap()
    }

    pub fn range(&self, g: usize) -> std::ops::Range<usize> {
        self.boundaries[g]..self.boundaries[g + 1]
    }
}

/// Which curvature structure an objective produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HessianMode {
    /// Only `d2L/ds_{k,i}^2`; no coupling.
    Diagonal,
    /// A `K x K` block per row (cross-output terms, e.g. competing events).
    BlockPerRow,
    /// A dense `(m K) x (m K)` block per group of `m` rows (cross-time terms).
    BlockPerGroup,
}

/// How a block Hessian is turned into the per-(row, output) diagonal that the
/// histogram tree learner consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagonalReduction {
    /// Objective is genuinely diagonal; nothing is dropped.
    Exact,
    /// Keep diagonal entries, ignore off-diagonal coupling. A Newton
    /// approximation; can overshoot when coupling is strong.
    DropOffDiagonal,
    /// Replace the diagonal with Gershgorin row sums `sum_j |H_ij|`. For a
    /// convex loss this diagonal majorizes the Hessian, so the resulting
    /// step is a conservative majorize-minimize step.
    GershgorinBound,
}

impl DiagonalReduction {
    pub fn as_str(&self) -> &'static str {
        match self {
            DiagonalReduction::Exact => "exact",
            DiagonalReduction::DropOffDiagonal => "drop_off_diagonal",
            DiagonalReduction::GershgorinBound => "gershgorin_bound",
        }
    }
}

/// Gradient plus full Hessian structure for one boosting iteration.
///
/// Index convention: entry `(k, i)` is output `k` of row `i`, flat index
/// `k * num_data + i`. Row blocks are stored row-major `K x K` per row;
/// group blocks are `(m K) x (m K)` row-major with local index `k * m + t`.
#[derive(Debug, Clone)]
pub struct GradHessBlock {
    pub num_data: usize,
    pub num_outputs: usize,
    pub mode: HessianMode,
    pub grad: Vec<f64>,
    pub hess_diag: Vec<f64>,
    pub row_blocks: Vec<f64>,
    pub group_blocks: Vec<Vec<f64>>,
}

impl GradHessBlock {
    pub fn new(num_data: usize, num_outputs: usize, mode: HessianMode, groups: &GroupIndex) -> Self {
        let nk = num_data * num_outputs;
        let row_blocks = if mode == HessianMode::BlockPerRow {
            vec![0.0; num_data * num_outputs * num_outputs]
        } else {
            Vec::new()
        };
        let group_blocks = if mode == HessianMode::BlockPerGroup {
            (0..groups.num_groups())
                .map(|g| {
                    let m = groups.range(g).len() * num_outputs;
                    vec![0.0; m * m]
                })
                .collect()
        } else {
            Vec::new()
        };
        Self {
            num_data,
            num_outputs,
            mode,
            grad: vec![0.0; nk],
            hess_diag: vec![0.0; nk],
            row_blocks,
            group_blocks,
        }
    }

    /// Reduce the stored Hessian to a per-(output,row) diagonal.
    pub fn reduce_to_diagonal(&self, how: DiagonalReduction, groups: &GroupIndex) -> Vec<f64> {
        let n = self.num_data;
        let k = self.num_outputs;
        match (self.mode, how) {
            (_, DiagonalReduction::Exact | DiagonalReduction::DropOffDiagonal)
            | (HessianMode::Diagonal, _) => self.hess_diag.clone(),
            (HessianMode::BlockPerRow, DiagonalReduction::GershgorinBound) => {
                let mut out = vec![0.0; n * k];
                for i in 0..n {
                    let b = &self.row_blocks[i * k * k..(i + 1) * k * k];
                    for a in 0..k {
                        out[a * n + i] = (0..k).map(|c| b[a * k + c].abs()).sum();
                    }
                }
                out
            }
            (HessianMode::BlockPerGroup, DiagonalReduction::GershgorinBound) => {
                let mut out = vec![0.0; n * k];
                for g in 0..groups.num_groups() {
                    let r = groups.range(g);
                    let m = r.len();
                    let dim = m * k;
                    let b = &self.group_blocks[g];
                    for a in 0..k {
                        for t in 0..m {
                            let row = a * m + t;
                            out[a * n + r.start + t] =
                                (0..dim).map(|c| b[row * dim + c].abs()).sum();
                        }
                    }
                }
                out
            }
        }
    }
}

/// An objective that needs grouped rows, multiple outputs, or cross-row terms.
pub trait GroupedObjective: Send + Sync {
    fn name(&self) -> &str;
    fn num_outputs(&self) -> usize;
    fn hessian_mode(&self) -> HessianMode;
    /// How the block Hessian is reduced for the diagonal tree learner. Must be
    /// [`DiagonalReduction::Exact`] iff `hessian_mode() == Diagonal`.
    fn diagonal_reduction(&self) -> DiagonalReduction;
    fn init(&mut self, meta: &Metadata, groups: &GroupIndex) -> Result<()>;
    fn gradients(&self, scores: ScoreView<'_>, groups: &GroupIndex, out: &mut GradHessBlock);
    /// Total (weighted) loss at `scores`; used for finite-difference checks
    /// and as the training-loss metric.
    fn loss(&self, scores: ScoreView<'_>, groups: &GroupIndex) -> f64;
    fn boost_from_score(&self, _output: usize) -> f64 {
        0.0
    }
    fn convert_output(&self, raw: &[f64], out: &mut [f64]) {
        out.copy_from_slice(raw);
    }
    fn to_model_string(&self) -> String;
}

/// The objective driving a booster.
pub enum Objective {
    Row(Box<dyn RowObjective>),
    Grouped { objective: Box<dyn GroupedObjective>, groups: GroupIndex },
}

impl Objective {
    pub fn num_outputs(&self) -> usize {
        match self {
            Objective::Row(o) => o.num_outputs(),
            Objective::Grouped { objective, .. } => objective.num_outputs(),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Objective::Row(o) => o.name(),
            Objective::Grouped { objective, .. } => objective.name(),
        }
    }

    pub fn init(&mut self, meta: &Metadata, num_data: usize) -> Result<()> {
        match self {
            Objective::Row(o) => o.init(meta, num_data),
            Objective::Grouped { objective, groups } => {
                if groups.num_rows() != num_data {
                    return Err(LgbmError::InvalidData(format!(
                        "group index covers {} rows, dataset has {num_data}",
                        groups.num_rows()
                    )));
                }
                objective.init(meta, groups)
            }
        }
    }

    pub fn take_warnings(&mut self) -> Vec<String> {
        match self {
            Objective::Row(o) => o.take_warnings(),
            Objective::Grouped { .. } => Vec::new(),
        }
    }

    pub fn boost_from_score(&self, k: usize) -> f64 {
        match self {
            Objective::Row(o) => o.boost_from_score(k),
            Objective::Grouped { objective, .. } => objective.boost_from_score(k),
        }
    }

    pub fn class_need_train(&self, k: usize) -> bool {
        match self {
            Objective::Row(o) => o.class_need_train(k),
            Objective::Grouped { .. } => true,
        }
    }

    pub fn is_renew_tree_output(&self) -> bool {
        match self {
            Objective::Row(o) => o.is_renew_tree_output(),
            Objective::Grouped { .. } => false,
        }
    }

    pub fn renew_leaf_output(&self, indices: &[u32], scores: &[f64]) -> f64 {
        match self {
            Objective::Row(o) => o.renew_leaf_output(indices, scores),
            Objective::Grouped { .. } => unreachable!("grouped objectives do not renew leaf outputs"),
        }
    }

    pub fn convert_output(&self, raw: &[f64], out: &mut [f64]) {
        match self {
            Objective::Row(o) => o.convert_output(raw, out),
            Objective::Grouped { objective, .. } => objective.convert_output(raw, out),
        }
    }

    pub fn to_model_string(&self) -> String {
        match self {
            Objective::Row(o) => o.to_model_string(),
            Objective::Grouped { objective, .. } => format!(
                "{} hessian_reduction:{}",
                objective.to_model_string(),
                objective.diagonal_reduction().as_str()
            ),
        }
    }

    /// Fill f32 gradient/Hessian buffers for the tree learner.
    pub fn gradients(&self, scores: ScoreView<'_>, grad: &mut [f32], hess: &mut [f32]) {
        match self {
            Objective::Row(o) => o.gradients(scores, grad, hess),
            Objective::Grouped { objective, groups } => {
                let mut block = GradHessBlock::new(
                    scores.num_data,
                    scores.num_outputs,
                    objective.hessian_mode(),
                    groups,
                );
                objective.gradients(scores, groups, &mut block);
                let diag = block.reduce_to_diagonal(objective.diagonal_reduction(), groups);
                for (g, v) in grad.iter_mut().zip(&block.grad) {
                    *g = *v as f32;
                }
                for (h, v) in hess.iter_mut().zip(&diag) {
                    *h = *v as f32;
                }
            }
        }
    }
}

/// Create a built-in objective from the config.
pub fn create_objective(cfg: &Config) -> Result<Option<Objective>> {
    match cfg.objective.as_str() {
        o if regression::REGRESSION_OBJECTIVES.contains(&o) => {
            Ok(Some(Objective::Row(Box::new(regression::Regression::new(cfg)?))))
        }
        "binary" => Ok(Some(Objective::Row(Box::new(binary::BinaryLogloss::new(cfg)?)))),
        "multiclass" => Ok(Some(Objective::Row(Box::new(multiclass::MulticlassSoftmax::new(cfg))))),
        "multiclassova" => Ok(Some(Objective::Row(Box::new(multiclass::MulticlassOva::new(cfg)?)))),
        "custom" => Ok(None),
        other => Err(LgbmError::Unsupported(format!("objective={other}"))),
    }
}

/// Recreate an objective (prediction-time only) from a model file `objective=` line.
pub fn objective_from_model_string(s: &str) -> Result<Option<Objective>> {
    let mut toks = s.split_whitespace();
    let name = toks.next().unwrap_or("");
    let rest: Vec<&str> = toks.collect();
    if let Some(r) = regression::Regression::for_prediction(name, &rest) {
        return Ok(Some(Objective::Row(Box::new(r))));
    }
    match name {
        "binary" => {
            let mut sigmoid = 1.0;
            for t in &rest {
                if let Some(v) = t.strip_prefix("sigmoid:") {
                    sigmoid = v.parse().map_err(|_| {
                        LgbmError::ModelFormat(format!("bad sigmoid in objective line: {s}"))
                    })?;
                }
            }
            Ok(Some(Objective::Row(Box::new(binary::BinaryLogloss::for_prediction(sigmoid)))))
        }
        "multiclass" => Ok(Some(Objective::Row(Box::new(multiclass::MulticlassSoftmax::for_prediction(&rest)?)))),
        "multiclassova" => Ok(Some(Objective::Row(Box::new(multiclass::MulticlassOva::for_prediction(&rest)?)))),
        "" | "custom" => Ok(None),
        other => Err(LgbmError::Unsupported(format!("loading model with objective={other}"))),
    }
}

#[cfg(test)]
mod tests;
