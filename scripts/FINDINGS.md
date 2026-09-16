# compileml 0.4.3 — test notes (2026-09-04; upstream status as of 0.8.0 at the end)

Environment: macOS, Python 3.12.12, numpy 1.26.4, scikit-learn 1.7.2, CatBoost 1.2.10
as teacher. Data: AWS Fraud Detector samples (`data/`).

## Bugs

1. **`export_sql` emits invalid SQL for a one-band artifact.** `_band_case` in
   `compileml/export/sql.py` builds `CASE ELSE 'G01' END` with no `WHEN` clause when
   `len(edges_int) == 2`. SQLite fails with `near "ELSE": syntax error`; ANSI engines
   will too. The COBOL exporter's `EVALUATE TRUE ... WHEN OTHER` is fine. Repro: build
   an artifact from `semantic_bands()` on a small sample (it returns 1 band), export
   with `dialect="sqlite"`, execute. Suggested fix: emit `'G01'` directly when there is
   a single label, or refuse to build a one-band artifact.

2. **Quantile band builders return duplicate edges that `build_artifact` then rejects.**
   With a ~5% bad rate, 20-30% of whitebox latents clip to 0, so `quantile_bands` and
   `monotone_quantile_bands` produce edges `[0.0, 0.0(+1e-12), ...]`. `_quantile_edges`
   nudges duplicates apart by 1e-12, which survives float checks but collides at
   `scale=1000`, and `build_artifact` raises
   `band edges collide after fixed-point conversion`. `allow_merge=True` does not help:
   it merges only bad-rate inversions, not degenerate edges. Suggested fix: after the
   quantile step, drop edges that map to the same integer at the target scale (the
   builders could take `scale=` like `band_efficiency` does), or have
   `monotone_quantile_bands` merge empty bands. Workaround used in the scripts:
   step K down until the integer ladder is strictly increasing.

## Observations (not bugs, but worth a note in the docs)

3. **Distilled latents go negative on low-base-rate targets.** `train_whitebox` is a
   squared-error regressor on the teacher's probabilities, so with a 5% rate 20-30% of
   predictions are < 0 and `build_artifact` warns that latents fall outside [0, 1]. The
   warning text suggests distilling margin-space models first, which is confusing when
   the model *was* produced by `train_whitebox`. Clamping handles it and every check
   passes, but the warning should probably distinguish "teacher was in margin space"
   from "regressor overshoot on a skewed target".

4. **`semantic_bands` / `governance_bands` certify only 1 band on small or drifting
   samples.** 1,400 insurance rows with `min_band_size=100`: 1 band. 81,000 transaction
   rows with a 10% rate: 1 band, even though `sweep_bands` shows a fixed-K=10 ladder
   retains 98% of the continuous Gini with zero monotonicity violations. That may be the
   intended strictness, but the quickstart-style flow (`semantic_bands` then
   `build_artifact`) then dead-ends in bug 1. A hint in the return metadata pointing at
   `monotone_quantile_bands` when `expressible_n_bands == 1` would help.

5. **Depth-2 scorecard is almost all interaction grids.** 60 depth-2 trees on 12
   features collapse to 1 main effect and 39 pairwise grids. The scorecard is exact
   (re-sum mismatches = 0 on 600 rows) but not the "one bin per feature" table a
   validator expects. Worth setting expectations in the scorecard docs, or offering a
   `max_depth=1` example alongside.

6. **`recalibrate_artifact` lineage key.** The docstring says
   `metadata.recalibrated_from` is set; check the actual key name (the script prints
   whatever keys contain "recalibrat").

7. **xgboost extra fails to import on Apple Silicon without libomp** (`brew install
   libomp`). Not compileml's bug, but the `[xgboost]` extra could mention it.

## What worked well

- CatBoost as teacher, distilled through `train_whitebox`: 98.5% Gini retention on the
  insurance set, 96.9% on transactions at depth 2.
- All ten validation checks pass on both artifacts; check 10 (WOE-logit reference
  floor) shows the artifact 12% above the reference on insurance.
- SQL export executed in SQLite matches the Python runtime on every holdout row
  (latent_int, band, pd_ppm).
- Scorecard re-sum matches `raw_micro` on every row.
- stdlib-only runtime confirmed: no numpy/pandas/sklearn/catboost in `sys.modules`;
  0.012 ms per row score-only, 0.7 ms per row with full 12-feature explanation.
- Recalibration on a later window with a 4x base-rate drop: model and band edges
  byte-identical, zero band churn, PD table refit.
- CLI `inspect`, `verify`, `score` (single row and CSV), `scorecard`, `export` all
  work. COBOL output not compiled here (`cobc` not installed).

## xbooster 0.2.8 notes (the user's own package, from script 04)

- `import xbooster` fails outright when xgboost is installed but `libxgboost.dylib`
  cannot load (missing libomp). `_try_import` in `shap_scorecard.py` catches only
  `ImportError`/`AttributeError`; `xgboost.core.XGBoostError` escapes. Catching
  `Exception` would keep the CatBoost path usable.
- `xbooster.constructor` imports both xgb and cb constructors at module level, so the
  unified import path has the same dependency on a healthy xgboost.
- In the CatBoost rule table, `Feature`/`Sign`/`Split` describe only the last
  (deepest) split of the oblivious tree, so all 8 leaves of a depth-3 tree show the same
  feature. `DetailedSplit` carries the full AND-path. Fine once you know, but the
  three summary columns are misleading at a glance.
- Empty leaves (`Count == 0`) get `EventRate` equal to the prior and `WOE = 0`; expected,
  but worth a docstring line.
- Cross-check vs compileml on the insurance holdout: xbooster SHAP points and the
  compileml integer latent rank-correlate at 0.94; top-1 risk driver agrees on 63% of
  rows, compileml's top-1 is in xbooster's top-3 on 72%. Baselines differ (SHAP
  expected value vs. median row), so full agreement is not expected.

## Upstream status (compileml 0.8.0, checked 2026-09-15)

The notes above were written against 0.4.3. In the following week the project went
0.5.0 → 0.8.0 and cited this evaluation ([README "Independent evaluation"](https://github.com/orgoca/CompileML#independent-evaluation),
`docs/concepts/where-it-fits.md`, changelog 0.5.2). Status of the seven items:

| # | item | status |
| --- | --- | --- |
| 1 | `export_sql` invalid SQL for a one-band artifact | fixed in 0.5.0 ([#40](https://github.com/orgoca/CompileML/issues/40), PR by @tote10): emits the label directly |
| 2 | quantile band builders emit colliding edges | fixed in 0.5.0 ([#41](https://github.com/orgoca/CompileML/issues/41)): builders take `scale=`, drop edges that would collide, return fewer bands and record `metadata["requested_n_bands"]`. The step-K-down workaround in `02` is no longer needed. |
| 3 | circular "distill margin-space models" warning | fixed in 0.5.2: reads the range, names overshoot vs margin scale |
| 4 | `semantic_bands` dead-ends silently on one band | fixed in 0.5.2: warns, points at `monotone_quantile_bands` / `min_band_size`; `flags.no_discrete_classes` stays the machine signal |
| 5 | depth-2 scorecard is mostly interaction grids | 0.5.2 docs: README and tuning guide set the expectation; `max_depth=1` is the classic form |
| 6 | `recalibrate_artifact` lineage key | 0.5.2: docstring now says `metadata.recalibration.recalibrated_from`; key unchanged |
| 7 | xgboost extra needs libomp on Apple Silicon | 0.5.2: README note |

Also shipped since: per-tree exact attribution (0.5.0), `compileml.fairness` (0.5.0),
`compileml.monitor` (0.6.0), reason codes and PD in the SQL and COBOL exports with
`explain=True` (0.7.0), `retention_by_segment` and `sample_weight` (0.8.0).

Rerun of scripts 01–04 on 0.8.0 (this repo bumped to `compileml>=0.8.0`): every check
passes, continuous Gini and retention unchanged, scorecard and SQL parity still 0
mismatches. The fixes do change the analyses that went through the workarounds:

- **Insurance band ladder (bug 2).** On 0.4.3 the step-K-down workaround landed on
  4 bands (edges `[0, 0.0018, 0.0137, 0.0363, 1]`, band-ordinal Gini 0.798, banding gap
  16.5%) because every wider ladder collided at the zero edge. On 0.8.0 the builder drops
  that edge itself and the widest clean ladder is 6 bands (edges
  `[0, 0.0018, 0.0076, 0.0212, 0.0363, 0.0746, 1]`, band-ordinal Gini 0.876, gap 8.3%).
  Band labels shift accordingly: the top band is now `G06`, not `G04`, and `03`'s counts
  are `{G01: 502, G02: 227, G03: 519, G04: 252, G05: 234, G06: 266}` instead of four near
  quartiles. Latent, PD and reason codes per row are unchanged (row 0 is still latent
  1000, PD 1.0, `VEHICLES|VEHICLE_AMOUNT|INJURY_AMOUNT`).
- **Transaction ladder.** `semantic_bands` still certifies one band and now says so; the
  fixed-K=10 monotone-quantile ladder is unchanged (band-ordinal 0.807, gap 2.1%), and
  recalibration on the later window still leaves model bytes and band edges identical.
- The pre-executed notebooks were re-run on 0.8.0 so their saved outputs match.

Other differences:

- artifact hash: `compileml_version`, `bands.requested_n_bands` and `bands.scale` now enter
  the hashed document, so the rebuilt `insurance_fraud.json` hashes `8a9bae69bb72…` instead
  of `266752582d53…`; model bytes and band edges are unchanged.
- full-explanation timing in `03`: 0.25 ms median per row, from 0.7 ms on 0.4.3 (per-tree
  attribution). Score-only still 0.012 ms.
- API: every function the scripts call keeps its signature; new keyword arguments only.
  Artifact schema still 2, so 0.4.3 artifacts load and score under 0.8.0.
- the BYOC image (`sagemaker-ai/container`) rebuilt on 0.8.0 returns rows identical to
  `03`'s decisions CSV.
