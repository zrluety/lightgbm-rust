# Discrete-time competing-risks hazard objective: draft design note

**Status: draft for review. No hazard objective is implemented or validated.** What exists today is the extension interface in `lgbm-core`: `GroupedObjective`, `GradHessBlock`, `HessianMode`, and `DiagonalReduction` (see [ARCHITECTURE.md](../ARCHITECTURE.md#objective-extension-design)). The formulas in sections 4 and 7 and the worked examples in section 6 are checked numerically by [`check_formulas.py`](check_formulas.py), which uses NumPy only. That checks the algebra, not a training objective. Every choice marked **[open]** is left for review.

Notation: loans $i = 1..n$; months since origination $t = 1, 2, \dots$; event types $k \in \{D, P\}$ (default, payoff/prepayment); $0$ denotes "continue".

## 1. Panel schema and conventions

- **Rows.** There is one row $(i, t)$ for each month $t$ in which loan $i$ is *at risk at the start of the month*: active and not yet defaulted, paid off, or censored. Covariates $x_{it}$ may be static (origination attributes), deterministic in $t$ (loan age, scheduled balance), or time-varying (delinquency status, macro series). A time-varying covariate in row $(i,t)$ may only use information available at the start of month $t$.
- **Labels.** The row label $y_{it} \in \{0, D, P\}$ is the outcome *during* month $t$. Every non-terminal row has $y = 0$.
- **Terminal events.**
  - A loan that defaults or pays off in month $\tau_i$ has its last row at $\tau_i$, labeled $D$ or $P$. There are no rows after a terminal event.
  - If default and payoff are both recorded in the same month, a precedence rule is needed **[open; suggested: the event that ends the exposure first in the servicing data, otherwise default]**.
- **Right censoring.**
  - **Administrative censoring.** A loan still active at the end of observed performance (the data cutoff) has rows through its last fully observed month $c_i$, all labeled $0$. It contributes no information about months $> c_i$, and those months are *not* treated as known non-defaults.
  - **Censoring timing within the month.** Censoring at month $c$ is assumed to happen *after* the month-$c$ outcome is observed. This is the standard discrete-time convention, so an event in month $c$ is never censored.
- **Payoff is a competing event, not censoring.** A payoff ends the risk of default: after payoff the loan *cannot* default. Treating payoff as censoring would estimate a hypothetical default rate in a world without prepayment, and would overstate cumulative default probability.
- **Censoring assumption (independent censoring).** Given the covariates used by the model, the censoring time is independent of the event time and type.
  - For administrative censoring the potential censoring time $C_i = $ (data cutoff − origination date) is known for *every* loan. The assumption then reduces to "no cohort effect beyond what the covariates capture" (exchangeability of origination cohorts). Including vintage or macro covariates weakens it, but cannot remove it.
  - Other censoring mechanisms (loan sales, servicing transfers, data gaps) are **not** guaranteed to be independent; for example, troubled loans may be sold. They must be identified and either modeled or shown to be ignorable **[open]**.

## 2. Model outputs and hazards

The model produces $K = 2$ raw scores per row, $a_D(x_{it})$ and $a_P(x_{it})$ (two trees per boosting iteration), with "continue" as the reference category ($a_0 \equiv 0$). The cause-specific discrete hazards are

$$
h_k(i,t) = \frac{e^{a_k}}{1 + e^{a_D} + e^{a_P}}, \qquad h_0(i,t) = \frac{1}{1 + e^{a_D} + e^{a_P}} = 1 - h_D - h_P .
$$

- **Validity.** For any real scores, $h_k \in (0,1)$ and $h_D + h_P < 1$, so the probabilities are always valid. Extreme logits are handled with the usual log-sum-exp shift; `check_formulas.py` shows the stable form.
- **Interpretation.** $h_k(i,t) = P(\text{event } k \text{ in month } t \mid \text{at risk at start of } t, x_{it})$.
- **Initial scores** (`boost_from_average`). Use $\log(n_D / n_0)$ and $\log(n_P / n_0)$, the row-level class counts. These are the maximum-likelihood constants.

## 3. Survival and cumulative incidence

For a loan with hazards over months $1..t$ (conditional on its covariate path):

$$
S(t) = \prod_{s=1}^{t} h_0(s), \quad S(0) = 1; \qquad
P(T = s, \text{type} = k) = S(s-1)\, h_k(s); \qquad
\mathrm{CIF}_k(t) = \sum_{s=1}^{t} S(s-1)\, h_k(s).
$$

- **Identity.** $S(t) + \mathrm{CIF}_D(t) + \mathrm{CIF}_P(t) = 1$ for every $t$.
- **Covariate paths.** The event-time distribution $\{P(T=s, k)\}$ is a distribution over (month, type), and it is defined *conditional on a covariate path*. With time-varying covariates, a loan-level distribution beyond the last observed row requires covariate forecasts or a convention. This matters for the timing-sensitive losses in section 6; see open choice 1.

## 4. Baseline: discrete-time competing-risks likelihood

Under independent censoring, the likelihood of loan $i$ factorizes over its rows. The negative log-likelihood is a per-row multinomial cross-entropy over $\{0, D, P\}$:

$$
\ell_i = -\sum_{t \le \tau_i} \log h_{y_{it}}(i,t), \qquad
\mathcal{L}_{\text{NLL}} = \sum_i w_i\, \ell_i .
$$

Censored loans contribute $-\log h_0$ for each observed month and nothing after $c_i$. That is exactly the "do not treat unobserved months as non-defaults" requirement.

**Gradient and Hessian** with respect to the raw scores of row $(i,t)$, for $k, j \in \{D, P\}$ (FD-verified):

$$
\frac{\partial \ell}{\partial a_k} = h_k - \mathbb{1}[y = k], \qquad
\frac{\partial^2 \ell}{\partial a_k \partial a_j} = h_k(\mathbb{1}[k=j] - h_j), \qquad
H = \begin{pmatrix} h_D(1-h_D) & -h_D h_P \\ -h_D h_P & h_P(1-h_P) \end{pmatrix}.
$$

- **Convexity.** $H$ is the covariance matrix of the multinomial indicator, so it is positive semi-definite and the loss is convex in $(a_D, a_P)$.
- **No cross-row terms.** The loss is row-separable, so the Hessian has no cross-month terms. Each row carries a $2\times2$ block coupling the two event types (`HessianMode::BlockPerRow`).
- **Interfaces.** The existing per-row machinery suffices. The objective is a `GroupedObjective` (or a 2-output `RowObjective`) with `BlockPerRow`. It does not need group indices for training, only for evaluation (CIFs).
- **Diagonal reduction.** One tree is grown per event type, so each tree's Newton step uses one diagonal entry. Two justified options exist:
  - `DropOffDiagonal`: use $h_k(1-h_k)$, ignoring the $-h_Dh_P$ coupling. This is what upstream LightGBM's `multiclass` does, with an extra factor $K/(K-1)$ for its over-parameterized softmax; with a reference category no such factor is needed. The coupling is small when either hazard is small, which is typical for monthly default ($h_D \ll 1$).
  - `GershgorinBound`: use $h_k(1 - h_k + h_j)$. Here $\mathrm{diag} - H = h_Dh_P \begin{pmatrix}1&1\\1&1\end{pmatrix} \succeq 0$, so this diagonal majorizes $H$, and each step is a guaranteed-descent (majorize-minimize) step for the quadratic model. It is slightly more conservative.

  The chosen reduction is written into the model file (`hessian_reduction:` in the `objective=` line). **[open: default reduction; suggested `DropOffDiagonal` for parity with upstream multiclass, with `GershgorinBound` as an option.]**
- **Conventional binary hazard baseline (for comparison).** Fit a default-vs-not hazard on the same rows, where payoff months end the loan. This estimates the *cause-specific* default hazard correctly. However, a CIF for default also needs the payoff hazard; using $1 - h_D$ as survival overstates default incidence. The binary baseline is therefore valid for hazards, but must not be turned into a CIF without a payoff model.

## 5. What the likelihood does and does not encode about timing

The log-likelihood of a loan is the **log score** of its observed outcome cell: $-\log P(T=\tau, k) = -\log S(\tau-1) - \log h_k(\tau)$ for events, and $-\log S(c)$ for censored loans.

- **It is strictly proper.** It is minimized in expectation by the true hazards, so a correctly specified hazard model *can* represent event timing, and fitting it recovers the timing distribution.
- **It is local.** It depends only on the probability of the observed cell, not on how far away the rest of the forecast mass is. Whether "default predicted at month 30, observed at 31" is penalized less than "observed at 5" therefore depends entirely on the *shape* of the forecast distribution, not on the loss. A forecast that is smooth in $t$ (adjacent months with similar hazards) gives month 31 more mass than month 5, so the likelihood prefers it. A forecast spiked at month 30 scores the two outcomes identically (section 6.3).
- **Practical consequence.** Trees with month/age as a feature produce piecewise-constant hazards in $t$, and these are usually smooth enough that the likelihood behaves sensibly. A distance-sensitive score adds an explicit preference that the likelihood lacks. Whether that preference is wanted *as a training target*, rather than as an evaluation metric only, is open choice 2.

## 6. Candidate timing-sensitive losses

All candidates score the loan-level event-time distribution, never a single predicted month.

### 6.1 Censoring-aware integrated Brier score on the CIFs (IPCW-IBS)

For horizon months $t = 1..T_h$ and event types $k$, with outcome indicator $Y_{ik}(t) = \mathbb{1}[\tau_i \le t, \delta_i = k]$:

$$
\mathcal{L}_{\text{IBS}} = \sum_i \sum_{t=1}^{T_h} \omega(t) \sum_{k \in \{D,P\}} W_i(t) \big(\mathrm{CIF}_{ik}(t) - Y_{ik}(t)\big)^2 .
$$

The inverse-probability-of-censoring weights (Graf et al. 1999; Gerds & Schumacher 2006) use $G(s) = P(C \ge s \mid \cdot)$, the probability of still being under observation in month $s$. With the convention of section 1, an event in month $\tau$ is observed iff $C \ge \tau$. The weights are:

- $W_i(t) = 1/G(\tau_i)$ if loan $i$ had an observed event at $\tau_i \le t$;
- $W_i(t) = 1/G(t)$ if loan $i$ had no event by $t$ and is observed through month $t$ ($c_i \ge t$);
- $W_i(t) = 0$ if loan $i$ was censored before $t$ ($c_i < t$, no event), so its unknown status is never imputed.

**Properties.**
- **Propriety.** The score is proper when $G$ is correct and the independent-censoring assumption holds.
- **Distance sensitivity.** The error accumulates over every month between the predicted and the observed event. Forecasting the right event a month late costs roughly one month of squared CIF error; forecasting it 25 months early costs about 25 months.
- **Event type.** Each CIF is scored separately, so a wrong type is penalized in both $\mathrm{CIF}_D$ and $\mathrm{CIF}_P$ for every month after the event.
- **Administrative censoring.** When censoring is purely administrative, $C_i$ is known for every loan. $G$ is then the empirical distribution of $C$ over loans, or it can be stratified by cohort. No model of $G$ is needed, but the assumption in section 1 still applies.
- **Other censoring.** For non-administrative censoring, $G$ must be estimated: by Kaplan–Meier (marginal, assuming censoring is independent of covariates) or by a covariate model, which makes a weaker assumption but adds model error. **[open]**
- **Horizon weights.** $\omega(t)$ weights horizons (e.g. uniform over $T_h$, or concentrated on 12/24/36 months) **[open]**. Large weights $1/G$ increase variance; truncating $T_h$ where $G$ becomes small is standard.

### 6.2 Ranked probability score on the event-time distribution with an event-type term (RPS)

Without censoring and with $\omega \equiv W \equiv 1$, IPCW-IBS *is* the ranked probability score of the joint (time, type) outcome. It is computed on the two CIFs, i.e. the cumulative distribution functions of the time of each event type, and is the discrete analog of the CRPS. Two variants are worth considering:

- **Separate time and type terms.** Write $\mathrm{RPS}_{\text{any}}$ on $1 - S(t)$ (time of *any* event) plus $\lambda \cdot$ a type score on $P(\text{type} \mid \text{event})$. This makes the trade-off between timing and type errors an explicit parameter $\lambda$ instead of a by-product of the horizon length. **[open]**
- **Truncating at censoring.** Score censored loans only up to $c_i$ (the "survival-CRPS" of Avati et al. 2020). This is simpler than IPCW, but it is **not proper**: it rewards forecasts that put mass after the censoring time. It is therefore listed only to be rejected unless a reviewer prefers it.

### 6.3 Example: timing (forecast near month 30; default observed in month 31 vs. month 5)

The setup has horizon $T_h = 36$, no censoring, and $\omega = W = 1$. The forecast puts a background mass of $0.005$ per month on default and on payoff. On top of that:
- the **spike** forecast puts $P(30, D) = 0.45$;
- the **smooth** forecast spreads about the same mass over months 29–31 ($0.155, 0.15, 0.155$).

Both forecasts have $P(\text{default by } 36) = 0.625$ and $P(\text{payoff by } 36) = 0.18$. From `check_formulas.py`:

| forecast | observed outcome | log score (= NLL) | RPS on CIFs |
|---|---|---|---|
| spike | default in month 31 | 5.298 | 1.874 |
| spike | default in month 5 | 5.298 | 22.434 |
| smooth | default in month 31 | 1.864 | 1.784 |
| smooth | default in month 5 | 5.298 | 22.344 |

- **Log score.** It penalizes "observed at 5" more than "observed at 31" only for the smooth forecast; for the spike forecast the two are tied. This is the locality described in section 5.
- **RPS.** It penalizes "observed at 5" about 12 times more than "observed at 31" for *both* forecasts, because the error accumulates over months 5–29.

### 6.4 Example: event type (default forecast; payoff observed)

The forecasts are the same as in 6.3.

| forecast | observed outcome | log score | RPS on CIFs |
|---|---|---|---|
| spike | default in month 31 (timing off by one month) | 5.298 | 1.874 |
| spike | payoff in month 30 (wrong type, right time) | 5.298 | 7.914 |
| smooth | default in month 31 | 1.864 | 1.784 |
| smooth | payoff in month 30 | 5.298 | 7.824 |

- **Log score.** For the spike forecast it cannot tell "one month late" from "wrong event type": both cells have probability 0.005.
- **RPS.** It charges the wrong type about 4 times more than the one-month timing error. The penalty comes from the default CIF being high while no default occurred, plus the payoff CIF being low while payoff did occur, over months 30–36. Note that the size of this penalty grows with the number of horizon months after the event; the type-term variant in 6.2 decouples the two.

### 6.5 Comparison

| | NLL (section 4) | IPCW-IBS / RPS (6.1, 6.2) | NLL + $\lambda$ · IBS |
|---|---|---|---|
| proper | yes | yes (with correct $G$) | yes (sum of proper scores) |
| distance-sensitive in time | only through forecast smoothness | yes | yes |
| distinguishes type errors from timing errors | no (local) | yes; relative size depends on horizon or $\lambda$ | yes |
| censoring | exact (likelihood) | through weights $W$; needs an assumption about $G$ | both |
| row-separable | yes | no: couples all months of a loan | no |
| Hessian structure | $2\times2$ per row (`BlockPerRow`) | dense $(2m)\times(2m)$ per loan (`BlockPerGroup`) | sum of both |
| convex in raw scores | yes | no | no |
| needs hazards after the last row | no | yes (open choice 1) | yes |

## 7. Derivatives for the grouped losses

The derivatives are for one loan with rows $u = 1..m$ and raw scores $a_j(u)$.

### 7.1 CIF Jacobian (FD-verified)

From $\partial h_k(u)/\partial a_j(u) = h_k(u)(\mathbb{1}[k=j] - h_j(u))$ and $\partial S(t)/\partial a_j(u) = -S(t)\,h_j(u)$ for $u \le t$:

$$
\frac{\partial\, \mathrm{CIF}_k(t)}{\partial a_j(u)} = \mathbb{1}[u \le t]\Big( \mathbb{1}[k=j]\, S(u-1)\, h_k(u) \;-\; h_j(u)\big(\mathrm{CIF}_k(t) - \mathrm{CIF}_k(u-1)\big) \Big).
$$

The score of month $u$ therefore affects the CIF at every later month: this is the cross-month coupling.

### 7.2 Gradient in $O(mK)$ per loan (FD-verified)

For one loan's term of $\mathcal{L}_{\text{IBS}}$, $\mathcal{L} = \sum_t \omega(t) W(t) \sum_k (\mathrm{CIF}_k(t) - Y_k(t))^2$, write $r_k(t) = \partial \mathcal{L} / \partial\, \mathrm{CIF}_k(t) = 2\,\omega(t) W(t) (\mathrm{CIF}_k(t) - Y_k(t))$. Define the suffix sums $R_k(u) = \sum_{t \ge u} r_k(t)$ and $Q_k(u) = \sum_{t \ge u} r_k(t)\,\mathrm{CIF}_k(t)$. Then

$$
\frac{\partial \mathcal{L}}{\partial a_j(u)} = S(u-1)\, h_j(u)\, R_j(u) \;-\; h_j(u) \sum_{k} \big( Q_k(u) - \mathrm{CIF}_k(u-1)\, R_k(u) \big).
$$

### 7.3 Hessian structure and reduction

- **Exact Hessian.** It is $\sum_t\sum_k 2\omega W\, J_{kt}^\top J_{kt}$ (the Gauss–Newton part) plus $\sum_t\sum_k r_k(t)\, \nabla^2 \mathrm{CIF}_k(t)$. It is dense across all $2m$ scores of the loan: within-row cross-type terms, and cross-month terms for $u \ne v$. The second part is **indefinite** in general, so the loss is non-convex in the raw scores.
- **Gauss–Newton (GN) Hessian.** $H^{\text{GN}} = \sum 2\omega W\, J^\top J$ is positive semi-definite and is the natural curvature for a squared-error score. Using GN instead of the exact Hessian is a documented approximation, not a silent clip; it must be recorded alongside the diagonal reduction **[open: add a `GaussNewton` marker to the model metadata]**.
- **Diagonal for the tree learner.** One tree is grown per event type with per-row scalar Hessians. The options are:
  - the GN diagonal $\sum_t\sum_k 2\omega W\, J_{k t; j u}^2$, which drops all cross-month and cross-type terms (`DropOffDiagonal`);
  - Gershgorin row sums of $|H^{\text{GN}}|$ (`GershgorinBound`). This majorizes $H^{\text{GN}}$ and is conservative. It costs $O(m^2K^2)$ per loan, which is acceptable for $m \le 360$.

  Both are supported by `GradHessBlock::reduce_to_diagonal`, and the choice is recorded in the model. A first-order alternative (a constant Hessian with line search) is also possible, but would need learner support that does not exist yet.
- **Cost.** The gradient is $O(mK)$ per loan. The GN diagonal is $O(mT_hK^2)$ naively; suffix-sum tricks similar to 7.2 can reduce it. A full GN block is $O(m^2K^2)$ memory per loan, so `BlockPerGroup` storage should be avoided unless a Gershgorin reduction is requested.

## 8. Integration plan in lightgbm-rust (not implemented)

1. **Dataset.** Add a loan/group id (rows contiguous per loan and sorted by $t$; `GroupIndex::from_row_ids` exists). Add the row label in $\{0, D, P\}$, and the potential censoring time $C_i$ or the IPCW inputs.
2. **Likelihood objective.** Add `objective="competing_risks"`: $K=2$ trees per iteration, `BlockPerRow`, and a configurable `hessian_reduction`. Prediction returns $(h_D, h_P)$ per row; a helper turns a loan's rows into $S$, $\mathrm{CIF}_D$, $\mathrm{CIF}_P$.
3. **Grouped objectives.** Add the IPCW-IBS / RPS objective as a separate, explicitly selected objective, with documented parameters: horizon $T_h$, $\omega$, the censoring-weight source, $\lambda$ for the hybrid, and the reduction. It needs `BlockPerGroup` or an on-the-fly diagonal. Conventional objectives remain unchanged; behavior without the new objective is untouched.
4. **Derivative tests.** Port `check_formulas.py` into Rust finite-difference tests on the real implementation, including the extreme-logit cases.

## 9. Calibration vs. economic preferences

- **Training and evaluation are probabilistic.** They use proper scores: NLL, or IPCW-IBS with $\omega(t)$ chosen for statistical reasons. These target calibrated hazards and CIFs.
- **Economic quantities are computed from the calibrated distribution afterwards.** Examples are expected loss and its timing:
  - $\mathrm{EL} = \sum_t \mathrm{EAD}(t)\,\mathrm{LGD}\,\mathrm{DF}(t)\,[\mathrm{CIF}_D(t) - \mathrm{CIF}_D(t-1)]$, with exposure, recovery, and discounting;
  - prepayment-driven income loss.

  These are decisions, not scoring rules.
- **Weights in the loss.** Putting exposure-, recovery-, or discount-weights *inside* the loss changes what is being estimated. Weights that depend only on information available at prediction time keep the score proper per loan. Weights that depend on the *outcome* (e.g. exposure at default) generally do not. Any such weighting would be a separate, explicitly named option with this caveat. **[open]**

## 10. Validation plan (milestones 5–6)

- **Synthetic data with known hazards.** Generate loans that cover:
  - early and late defaults;
  - adjacent-month timing errors;
  - default vs. payoff errors;
  - administrative and random censoring;
  - rare events;
  - varying durations;
  - extreme logits.
- **Checks.** Recovery of the true hazards and CIFs; validity of the probabilities; and the timing and type behavior shown in sections 6.3 and 6.4, measured on held-out loans.
- **Baselines.** Compare against the binary default hazard and the competing-risks likelihood. Metrics: calibration by horizon, CIF error, IPCW-IBS, timing sensitivity, concordance (discrimination), and the economic metrics from section 9.
- **Splits.** Split by loan and by origination cohort; no loan appears in both training and validation. Predictors must not use post-observation information.
- **Evidence standard.** Improvement is never claimed from the training loss alone.

## 11. Open modeling choices (for review)

1. **Hazards after the last observed row.** A timing-sensitive score needs $\mathrm{CIF}_k(t)$ for $t$ beyond a loan's terminal month, where no rows or time-varying covariates exist. The options are:
   - (a) score only up to each loan's last row, which loses sensitivity to forecasts that are too late;
   - (b) extend each loan with forecast rows up to $T_h$, using static and deterministic-in-$t$ covariates, with time-varying covariates frozen or forecast;
   - (c) score origination-time forecasts built from static covariates only.

   This is the main unresolved question for 6.1 and 6.2.
2. **Role of the timing-sensitive score.** Use it as the training objective, as a regularizer added to the likelihood (hybrid, $\lambda$), or only for evaluation and model selection with likelihood training. The suggested first step is likelihood training plus IPCW-IBS evaluation.
3. **Censoring weights.** Treat censoring as administrative-only, with known $C_i$ and cohort-stratified $G$, or estimate $G$ by Kaplan–Meier or a covariate model for other censoring mechanisms. Also decide on truncation of large weights.
4. **Horizon and weights.** Choose $T_h$, $\omega(t)$, and whether the type error is decoupled through $\lambda$ (section 6.2).
5. **Hessian treatment.** Choose the reduction for the likelihood (`DropOffDiagonal` vs `GershgorinBound`), and Gauss–Newton plus which diagonal for the grouped losses.
6. **Same-month events.** Choose the precedence rule for same-month default/payoff, and how to handle the month of origination ($t = 0$ vs $t = 1$).
7. **Economic weighting.** Decide whether any economic weighting enters training at all (section 9).
