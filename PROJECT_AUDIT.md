# SmartDine project audit

Audit date: 28 October 2026 (user timezone Asia/Calcutta). This is an implementation and verification record. Completion is partial only where original data/history is missing; no raw-to-end-to-end success is claimed.

## Initial inventory and root

The starting workspace contained only `node_modules/` and a nonfunctional `.git/` directory. `git status --short`, `git branch --show-current`, `git ls-files` and `git rev-parse --show-toplevel` all returned **fatal: not a git repository**. No branch, tracked/untracked status or user working-tree delta could be established. These were not invented. The ZIP contained no `.git`, tests, CI configuration or model metadata. There were no AGENTS.md instructions in the extracted project. No commit, push, reset, history rewrite or branch deletion was performed.

The ZIP was inspected before extraction/editing. It contained the redundant wrappers `smart-dine-main/smart-dine-main/`, then the outer app/server/README/requirements and nested `SmartDine/` code, data and artifacts. Extraction created a new `smart-dine-main/` in the workspace without overwriting existing project files. The original uploaded ZIP remains unchanged. A second recoverable snapshot is stored locally at `.local-backup/original-snapshot.zip`, excluded from Git and the submission archive. Every original file's path, size, classification and SHA-256 is in `docs/initial_inventory.json`.

Canonical root: **the extracted outer `smart-dine-main/` containing app.py, server.py and pipeline.py**. The outer launchers already depended on nested data/src paths, so the nested directory was retained. Source UI is the outer app; nested app delegates to it. Nested requirements and README delegate to the parent. No structural deletion was needed. The final directory tree is in README, and per-file changes/inventory are in `docs/FILE_CHANGES.md` and `docs/final_inventory.json`.

Inventory classes: Python/frontend/notebooks are source; feature CSV, synthetic profiles and model/schema/metadata are small operational artifacts; cleaned CSV and matrix/train/test are regenerable local artifacts; absent raw input is private/local-only; figures/metrics/scenarios are generated outputs; historical named scripts became compatibility modules; cache files are temporary. Existing cache files were not known to be tracked and were not destructively removed. Ignore rules cover them.

## Main repairs

- Added one root-aware stage runner and shared pathlib paths/schema. Imports no longer execute long pipeline jobs or silently write datasets.
- Repaired cleaning validation, preserved unknown numeric/boolean values, counted all source records (including missing ratings), retained branch addresses and recorded provenance. This is fixture-tested; original-data cleaning is blocked.
- Shared feature generation between batch training, Streamlit and FastAPI. Fixed rating normalization, cuisine/type/location weighting, cuisine consensus, booking/online conventions, missing inputs and named inference schema drift. Feature contract is version 2.0.
- Preserved the corrected target weights and noise scale; removed the earlier overwritten target. Added input/content hashes, target/version manifests and group separation assertions to reject stale artifacts.
- Selected among four regressors on an inner **group validation** split, then evaluated the selected/refitted estimator on the untouched outer group holdout. Replaced the old unsupported-version model with a newly trained model under the pinned installed environment, preserving the original in backup.
- Added actual metadata with model hash, feature schema, target, dataset/split provenance, validation IDs, metrics, baseline, generation timestamp and Python/scikit-learn versions. Missing/corrupt/stale metadata returns unavailable. Cross-version joblib warnings fail loading rather than silently continuing in normal serving.
- Removed automatic all-catalogue fallback when filters yield no results. Added deterministic ties, validated budgets/ratings/weights/top-N and enforced requested booking/rating filters. Unknown observations do not satisfy hard filters.
- Replaced dietary safety and probability-like claims with an unverified heuristic and compatibility index. Explanations compare real inputs; a small cost overrun is no longer described as within budget. Flexible location preferences generate no invented location reason.
- Preserved Streamlit styling, repaired preset/session reruns, exposed all catalogue neighborhood choices, escaped dynamic HTML fields and added clear artifact/empty-state messages.
- Replaced independent FastAPI scoring with the same engine. Health metrics load dynamically. Startup errors return controlled 503 responses without exposing local paths. Existing dashboard was repaired: root HTML now loads `/static/app.js` and `/static/index.css`; before this fix it was stuck at placeholders. Dashboard consensus now matches pairwise Jaccard, initial performance placeholders were removed, and grid sizing was adjusted to stop result cards overflowing the viewport.
- Repaired notebook machine paths and cleared stale notebook outputs, retaining originals in backup. Notebooks were inspected but not executed.
- Added 9 reproducible EDA plots and 3 evaluation plots, plus deterministic tests and documentation. Old root-level output images/G00x CSV/txt remain clearly labeled legacy.

## Pipeline execution status

| Stage | Status and evidence |
| --- | --- |
| Raw validation | **Blocked**: original `SmartDine/data/raw/zomato.csv` absent; actual runner exit 2 recorded in docs/raw-validation.json |
| Raw cleaning/entity consolidation | **Not run on original data**; deterministic cleaning/provenance tests passed |
| Restaurant features | Completed from supplied 12,499-row cleaned CSV; original processed provenance cannot be independently validated |
| Group profiles | Loaded/preserved 401 existing synthetic members in 100 groups; no fabricated restaurant data |
| Shared matrix | Generated 1,249,900 group–restaurant pairs, manifest/hash saved |
| Target and split | Completed: 999,920 training rows / 80 groups; 249,980 test rows / 20 groups; zero group overlap |
| Validation/model comparison | All four completed: Linear Regression, Decision Tree, Random Forest, Gradient Boosting; 64 fit groups / 16 validation groups within training partition |
| Training/test evaluation | Completed; selected Linear Regression refitted on 80 training groups, then measured on 20 test groups |
| Metadata | Generated from real model/split/predictions; independent evaluation agrees |
| EDA | **9 images generated**, listed in outputs/figures/eda_manifest.json |
| Evaluation plots | **3 images generated**, actual/predicted, residuals, coefficients; total current PNG count 12 |
| Scenarios | Saved recommendations for actual synthetic G001–G005 in outputs/scenarios.json; 5 results each, 925 eligible candidates after member booking/rating requirements |
| Full original-data pipeline | **Blocked** pending original CSV; not claimed complete |

The matrix is 266,061,341 bytes, training CSV 231,374,300 bytes and test CSV 57,305,457 bytes. They remain local/ignored. Manifest hashes prevent accidental reuse with different upstream feature/profile inputs. Restaurants overlap intentionally across partitions because this experiment tests new synthetic groups on a known catalogue, not new restaurants. No learned catalogue-wide imputation is fitted on test rows; fixed model-input conventions are documented in README.

## Model/target integrity and historical discrepancy

The saved original is a LinearRegression with 15 named features, serialized with scikit-learn **1.7.2**; installed **1.3.1** emitted InconsistentVersionWarning. The active model was regenerated with installed 1.3.1 and loads without that warning. Python 3.12.4 and pinned runtime requirements were used. `pip check` reports no broken requirements. The environment reused installed packages through `--system-site-packages`; a separate clean dependency installation was not performed.

Version 2.0 target is the corrected weighted sum in schema.py: cuisine .20, budget .20, dietary .15, type .10, location .10, agreement .10, rating .05, booking .03, online .03, popularity .04; additive Normal(0,.03), RandomState(42), clipped to [0,1]. Weights/noise were not adjusted to improve metrics. The original matrix builder used different target weights/no agreement, then preparation overwrote it. Now there is exactly one target generation stage.

Measured final test results: MAE **0.023990342322099**, RMSE **0.030067818908677**, R² **0.807314823708621**. Evaluation reads the intended saved model and hashed split and reproduces these values. Validation results are separately labeled in model_comparison.csv; they must not be presented as test scores.

Fair constant training-target-mean baseline: MAE 0.054356026118884, RMSE 0.070893274368585, R² -0.071160438148488. Negative R² here means squared error exceeds the test-mean constant reference; it does not establish a general claim about other recommendation methods.

Original CSVs report R² 0.7641078524673492. Forensic replay of the original corrected target and legacy builder with the supplied model's linear coefficients yielded MAE 0.023990239346000208, RMSE 0.03006778339925726, R² 0.7640651554365689. The unsupported-version object was used only to extract the linear equation for this replay; this is not a supported cross-version deployment claim. Evidence and warning are recorded in outputs/historical_replay.json. The result is close, **not an exact reproduction**. No artifact for ~0.7577 was supplied, so the exact historical discrepancy remains unresolved. Original matrix/splits or an immutable generation manifest are required to attribute it. Feature-weighting/missingness/normalization changes in version 2.0 are documented and make its result a separate benchmark. Do not interpret higher R² with essentially unchanged noise-scale MAE/RMSE as a demonstrated human-preference improvement.

The target is directly constructed from the features, so predictive agreement with it is expected. No actual multi-member human preferences, satisfaction measurements, surveys, fairness guarantees or real-user validation were obtained. Group agreement is a cuisine-set statistic; time/day are context labels only. Cuisine strings cannot establish vegetarian/vegan kitchen safety, allergen handling or certification. Neighborhood equality is not measured distance.

## Verified commands and outcomes

Executed from canonical root unless stated otherwise:

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install httpx==0.28.1
python -u pipeline.py features groups matrix split train evaluate eda
python -u pipeline.py train evaluate eda scenarios
python -u tools/replay_historical.py
.\.venv\Scripts\python.exe -m pytest -q --tb=short --junitxml=docs/test-results.xml
node --check SmartDine/frontend/app.js
node tests/test_frontend.cjs
.\.venv\Scripts\python.exe tools/verify_repository.py
.\.venv\Scripts\python.exe tools/smoke_apps.py
```

The first data run completed features/groups/matrix/split, then Windows sandbox restrictions blocked Random Forest worker pipe creation after Linear Regression and Decision Tree validation. With approved execution outside the sandbox, the second command completed all four models, test evaluation, EDA and scenarios. The initial sandbox venv bootstrap and package lookup also failed; approved reruns succeeded. No failure was counted as a success.

First pytest attempt: **26 passed, 42 setup errors**, caused by denied temporary-directory access. Approved rerun: 68 passed. After additional matrix/split/train integrity tests: 73 passed. Final dashboard asset regression run: **74 passed, 0 failed, 0 errors, 0 skipped**, 13 dependency deprecation warnings, 19.15 seconds. Exact JUnit result: docs/test-results.xml. Warnings concern dateutil UTC, Starlette's httpx adapter, and joblib AST compatibility; they were not hidden by rewriting assertions.

Frontend Node checks: **5 assertions passed**, covering consensus calculation, per-member budget overrun, unknown cost, dietary uncertainty and empty/singleton cuisine consensus. Syntax checks and all source-module imports passed. The repository check also runs `pip check`, launches evaluation from the parent directory, scans active source/notebooks for machine paths and verifies all 12 generated PNGs decode. Its initial path scanner falsely matched encoded notebook newlines and https schemes; the scanner was corrected to parse notebook cell sources and require a drive-letter boundary, and then passed.

Live HTTP smoke checks returned **200** for API health, recommendation and dashboard; strict unknown-neighborhood request returned an empty list without fallback. Streamlit health returned **200/ok**. Programmatic Streamlit AppTest passed initial render, Recommend, preset change, member-count change and another Recommend. An early AppTest harness selected a main-panel diet control instead of the sidebar preset, producing a test-selector error; this was fixed by explicitly selecting the sidebar. No app failure was concealed.

Real browser review: opened Streamlit, clicked Recommend, inspected rendered cards/cuisine-budget-dietary-location breakdowns and cautions. Opened the API dashboard, discovered/fixed missing static asset URLs, reloaded, saw dynamic R² 0.8073 and 12,499 venues, clicked Compute and inspected generated cards. Screenshots are in docs/browser-recommendations.png and docs/browser-dashboard.png. This is limited smoke verification, not an exhaustive visual/usability audit. No claim of browser testing is inferred from HTTP checks alone.

## Remaining work and manual checklist

1. Supply the original raw CSV and provenance/license information. Run `pipeline.py all`; inspect cleaning provenance, entity merging and unknown-value counts against original source. Without it, this audit cannot verify original row removals, raw schemas or end-to-end raw reproducibility.
2. Obtain the historical train/test files or exact run manifests for both ~0.7641 and ~0.7577 if an exact historical attribution is required. Do not cite a guessed cause.
3. Reproduce setup in a fresh Python 3.12 environment before institutional handoff; this run reused existing pinned packages. Confirm package availability on that machine.
4. Manual UI: launch each app at normal desktop and mobile widths; switch all demo presets; edit each field; compare contrasting cuisine/budget groups; set impossible constraints; confirm the empty message and no automatic fallback; check unknown costs/flags; confirm index/dietary labels and booking filters; restart after model regeneration.
5. Optional future research: obtain consented real group feedback and evaluate ranking/utility on it. Implementation regression tests are not substitutes for empirical recommendation-quality evaluation.

The slim submission ZIP contains source, small operational artifacts, current plots, documentation and evidence. It intentionally excludes the original backup, raw/cleaned/large matrix/split CSVs, local environment, caches, logs and legacy output images/profile CSVs. Since the raw file is absent, the slim package launches from included feature/model artifacts or can retrain starting at `groups matrix split train evaluate eda scenarios`. No Git LFS assumption is made and no raw/private data is uploaded. GitHub file limits were checked against official documentation (linked in README). The workspace original `.git` and all original user files remain untouched.
