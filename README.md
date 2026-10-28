# SmartDine

Context-Aware Group Restaurant Recommendation System Using Machine Learning.

SmartDine combines group members' cuisine choices, budgets for two, neighborhood names and service preferences to rank a known Bangalore restaurant catalogue. It is an academic implementation of preference aggregation and explainable ranking. Its model learns a **constructed weighted target with synthetic noise**, using synthetic group profiles. It has no real-user satisfaction labels, survey evidence or validated dietary safety information. Time/day inputs are displayed context only; the trained features do not model opening hours or a time/day effect.

## Canonical root and architecture

Run commands from this directory, the outer `smart-dine-main` folder containing `pipeline.py`, `app.py` and `server.py`. The ZIP contained two additional wrapper folders; these were flattened on extraction. `SmartDine/` remains the Python package and artifact directory, not a second independent project. The nested app is a compatibility launcher, and its README/requirements point to the parent. No original Git repository was present in the ZIP. The workspace's unusable `.git` was not changed; nothing was committed or pushed.

```text
smart-dine-main/
  app.py                         canonical Streamlit UI
  server.py                      optional FastAPI/dashboard launcher
  pipeline.py                    explicit, root-aware stages
  requirements.txt, pytest.ini, .gitignore
  README.md, PROJECT_AUDIT.md
  SmartDine/
    src/                         cleaning, shared features, ranking, pipeline, API
    data/raw/zomato.csv           REQUIRED for raw validation/cleaning; absent
    data/processed/               supplied small inputs; local generated matrix/splits
    models/                      current model, feature order, verified metadata
    frontend/                    preserved dashboard styling and assets
    notebooks/                   preserved investigations, paths repaired, stale outputs cleared
    outputs/figures/              9 EDA + 3 evaluation figures from this run
  tests/                         deterministic fixtures; no raw data or network required
  tools/                         smoke checks, repository checks, historical replay
  docs/                          original inventory, change list, test and smoke evidence
  .local-backup/                  ignored recoverable original snapshot (local only)
```

Data flow: original CSV → validate/clean/consolidate → restaurant features → load/generate synthetic groups → shared group–restaurant features → one target generation and group split → validation comparison → selected model refit on all training groups → untouched test evaluation → metadata/figures → the same feature builder at inference. Streamlit loads the model directly; it does not require the API. The optional API serves the existing HTML/CSS/JS dashboard.

## Data, provenance and unknowns

The supplied snapshot contains **12,499 processed restaurant rows** and **401 synthetic members in 100 groups**. These were inspected directly. The original raw CSV is absent; earlier claims about 51,717 original records or a particular raw-file size have not been independently verified. No raw replacement was created. The snapshot calls its source the Bangalore Zomato dataset, but supplies no authoritative source URL, version or license. Obtain the original and check provenance and redistribution rights before sharing its data.

Expected original location: `SmartDine/data/raw/zomato.csv`. Required fields: `name`, `address`, `location`, `cuisines`, `rate`, `votes`, `approx_cost(for two people)`, `online_order`, `book_table`. Headers are stripped/lowercased. Optional listing/type/menu fields may be absent. The raw file is never overwritten.

Cleaning groups by whitespace/case-normalized **name + full address**, retaining different addresses as different branches. It removes rows missing identity and records their zero-based source indexes. Rating and cost aggregate by median, votes by maximum (avoiding duplicated vote totals), descriptive fields by unique union. Boolean disagreement remains unknown; unknown input flags are never silently confirmed. A provenance JSON records removal/consolidation counts, missingness, aggregation policy and raw SHA-256. This key is conservative, not verified entity resolution: co-located identically named businesses can merge; address variants can remain separate.

The supplied feature CSV has 2,992 unknown ratings, 55 unknown costs, 9 unknown locations, 18 unknown cuisines and 62 unknown restaurant types. Unknown fields remain unknown in returned results and fail relevant hard filters. For finite model inputs only, rating defaults to 3.5, cost to INR 800, votes to zero and unknown service flags to 0.5. These are explicit fixed conventions, not learned facts or safety claims. No full-data median is fitted. Popularity uses a fixed reference of 20,000 votes, avoiding catalogue-wide fitting across the holdout.

## Setup (Windows PowerShell)

Use Python **3.12**. The verified machine used Python 3.12.4 and the exact runtime versions in `requirements.txt`. Scikit-learn is pinned to **1.3.1**, the version used to regenerate the active model. The original model was saved by 1.7.2 and raised a version warning here; it was preserved in the ignored backup. Cross-version persistence is unsupported by [scikit-learn](https://scikit-learn.org/stable/model_persistence.html). Retrain after changing that version.

```powershell
Set-Location 'path/to/smart-dine-main'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Activation is optional; all examples invoke the environment directly. The audit created its local environment with `python -m venv --system-site-packages .venv` to reuse the installed scientific stack, installed httpx locally, and checked dependency consistency. A separate clean-environment installation was not performed.

## Pipeline commands

Each command uses paths relative to the source, independent of the shell's working directory. Use an absolute path to `pipeline.py` when invoking it from elsewhere. Exit codes: 0 success, 2 missing required input, 1 invalid arguments/data or stage failure. `all` stops at the first blocked stage; it never silently skips a missing raw input.

```powershell
.\.venv\Scripts\python.exe pipeline.py validate
.\.venv\Scripts\python.exe pipeline.py clean
.\.venv\Scripts\python.exe pipeline.py features
.\.venv\Scripts\python.exe pipeline.py groups
.\.venv\Scripts\python.exe pipeline.py matrix
.\.venv\Scripts\python.exe pipeline.py split
.\.venv\Scripts\python.exe pipeline.py train
.\.venv\Scripts\python.exe pipeline.py evaluate
.\.venv\Scripts\python.exe pipeline.py eda
.\.venv\Scripts\python.exe pipeline.py scenarios
```

With the original raw file supplied, run `pipeline.py all`. With this snapshot, the verified sequence starts at `features`:

```powershell
.\.venv\Scripts\python.exe pipeline.py features groups matrix split train evaluate eda scenarios
```

`groups` preserves and loads an existing profile CSV. If absent, it generates seeded synthetic profiles from available catalogue categories (100 groups, seed 42). It does not create restaurant records. Matrix construction uses the same feature implementation as serving, writes groups in stable order and saves input/content hashes. `split` refuses a stale/missing matrix manifest and generates the target exactly once. `train` rejects modified split hashes, source drift and overlapping groups. Regenerate upstream stages when a source changes; do not reuse unrelated historical CSVs.

The matrix has 1,249,900 pairs; train has 999,920 rows/80 groups and test 249,980 rows/20 groups. Restaurants intentionally occur in both partitions: evaluation addresses **new synthetic groups on a known restaurant catalogue**, not unseen restaurants. Validation further holds out 16 of the 80 training groups. Model comparison uses validation only; selection is by R² then MAE, followed by a refit on all 80 training groups and one test evaluation. Full matrix/splits require substantial local disk and memory (about 555 MB CSV combined); training also uses several times that size in memory.

Historical modules `clean_restaurants`, `train_models`, `evaluate_model` now delegate to their canonical implementations. Invoke package modules with `python -m SmartDine.src.preprocessing`, etc.; old direct `python SmartDine/src/...py` invocations were intentionally replaced by the root runner/package commands.

## Features and constructed target (version 2.0)

The exact model order is in `SmartDine/models/feature_columns.txt` and checked against the estimator's `feature_names_in_`. Prediction passes a named DataFrame, not an unnamed array.

| Feature | Meaning |
| --- | --- |
| group_size | Number of members |
| cuisine_match | Importance-weighted mean of member preferred-cuisine coverage; empty preference gives neutral 0.5 |
| budget_match | Importance-weighted mean: cost/budget <=1 → 1; <=1.25 → .75; <=1.5 → .5; <=2 → .25; otherwise 0; unknown cost .5 |
| dietary_compatibility | Weak cuisine-string heuristic; restricted diets remain unverified, never a hard safety guarantee |
| restaurant_type_match | Importance-weighted exact type-label membership; absent preference .5 |
| location_match | Importance-weighted exact normalized neighborhood-name equality; absent preference .5; no distance |
| booking_match | Mean member booking preference match; no requirement gives 1; unknown required flag .5 |
| online_order_match | Mean online preference match; no preference gives 1; unknown preferred flag .5 |
| group_agreement | Mean pairwise Jaccard between members' cuisine sets; empty union and singleton give 0 |
| rating_score | Observed rating/5; unknown uses 3.5/5 for the model only |
| popularity_score | clip(log1p(votes)/log1p(20000),0,1) |
| cost_for_two, votes, online_order, book_table | Catalogue values with the fixed input conventions above |

For restricted diets, absence of nonvegetarian cuisine keywords yields only a neutral 0.5; indicative keywords give vegetarian .5 or vegan .25. No restriction/nonvegetarian yields 1 (no exclusion requested), not proof of menu contents. Importance weights must be finite/nonnegative; all-zero weights fall back to equal weighting. Budget units are INR **for two people**, not a per-person or total-group bill. Weighted budget is a summary, not a promise each member's limit is met. Group rating floor is the 75th percentile of member requested minimum ratings, combined with any explicit minimum-rating filter.

```text
y = clip(0.20*cuisine_match + 0.20*budget_match
       + 0.15*dietary_compatibility + 0.10*restaurant_type_match
       + 0.10*location_match + 0.10*group_agreement
       + 0.05*rating_score + 0.03*booking_match
       + 0.03*online_order_match + 0.04*popularity_score
       + Normal(0, 0.03), 0, 1)
```

Noise uses NumPy `RandomState(42)` over the stable full matrix row order. These weights and noise scale preserve the supplied corrected target definition; they were not tuned to improve scores. The earlier builder also made a different target without agreement, which the old preparation script overwrote. That duplicate target generation has been removed. Inputs themselves define the target, so an accurate linear approximation is expected; this is not independent evidence of recommendation quality.

## Verified model results

Validation comparison from this run (not the final test set):

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Linear Regression | 0.02398345 | 0.03005407 | 0.84605963 |
| Gradient Boosting | 0.02527049 | 0.03169602 | 0.82877963 |
| Random Forest | 0.02722899 | 0.03415010 | 0.80123949 |
| Decision Tree | 0.02851396 | 0.03582086 | 0.78131543 |

Final held-out group test, selected Linear Regression:

- MAE: **0.023990342322**
- RMSE: **0.030067818909**
- R²: **0.807314823709**
- Constant training-target-mean baseline: MAE 0.054356026119, RMSE 0.070893274369, R² -0.071160438148.

The negative baseline R² means its squared error exceeds that of predicting the test-target mean. This says nothing universal about other recommendation systems. It is an honest constant baseline fitted on training labels only.

Metadata contains the actual model name, feature version, target weights, source hashes, group split, validation membership, measured metrics, baseline, Python/scikit-learn versions, timestamp and model hash. API metrics load dynamically. Missing/corrupt/stale metadata reports unavailable, not a historical constant.

Historical evidence: archived CSVs report R² 0.7641078524673492. Replaying original feature/target formulas and the supplied linear equation produced MAE 0.023990239346000208, RMSE 0.03006778339925726, R² 0.7640651554365689 (`outputs/historical_replay.json`). This is close but not exact. The later ~0.7577 has no corresponding supplied split/matrix/run manifest. Its exact cause remains unresolved; neither dataset nor noise/version changes can be attributed definitively. Version 2.0 changes weighting consistency, missing-value treatment, normalization and heuristic semantics, so its R² is a separate experiment. Similar MAE/RMSE with different R² can reflect different target variance; do not claim an empirical gain. Replay uses the original local backup and is unavailable in the slim submission ZIP unless that backup is supplied.

## Applications and filters

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
# Optional dashboard and API, in another terminal:
.\.venv\Scripts\python.exe server.py
Invoke-RestMethod http://127.0.0.1:8000/api/health
$body = @{ members = @(@{ budget = 800; preferred_cuisines = @('Italian') }); top_n = 3 } | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8000/api/recommend -Method Post -ContentType 'application/json' -Body $body
```

Streamlit: `http://localhost:8501`. Dashboard: `http://localhost:8000`. API docs: `http://localhost:8000/docs`. Nested `streamlit run SmartDine/app.py` remains a compatibility launch.

Required for immediate app operation: `restaurant_features.csv`, current `smartdine_best_model.joblib` and its matching feature schema; include `model_metadata.json` to report verified metrics. `group_members.csv` supports pipeline/scenarios but manual UI inputs do not require it. Raw/cleaned/matrix/train/test CSVs are not needed to launch an already-trained app.

Top-N is 1–50 at the engine/API (UI 5–20). Location and cuisine filters use exact normalized names/labels. Maximum cost, rating, requested table booking and online-order filters exclude unknown values. Any member requiring booking makes booking a hard requirement; online preference remains soft unless the online filter is set. Dietary requests are **not verifiable hard constraints** in this dataset. There is no automatic filter relaxation. Empty results explicitly ask users to adjust filters. Stable ties sort by restaurant ID. Scores displayed on 0–100 are clipped ranking indices, not calibrated probabilities. Explanations check actual member budgets and preferences; they do not infer a location preference when none was provided.

## Tests, figures and evidence

```powershell
.\.venv\Scripts\python.exe -m pytest -q --tb=short --junitxml=docs/test-results.xml
node tests/test_frontend.cjs
.\.venv\Scripts\python.exe tools/verify_repository.py
.\.venv\Scripts\python.exe tools/smoke_apps.py
```

Python suite: **74 passed, 0 failed, 0 errors, 0 skipped**; 13 dependency deprecation warnings (dateutil, Starlette/httpx and joblib AST compatibility). Node: 5 frontend regression assertions passed. Tests use clearly marked tiny fixtures, never a replacement dataset. App smoke checks require free ports 8000/8501; stop manually running servers first. They start and stop their own servers and exercise Streamlit widgets programmatically. The audit additionally clicked Recommend in a real browser and inspected rendered cards; this is a limited smoke review, not exhaustive usability testing.

Regenerate 9 EDA figures with `pipeline.py eda` and 3 evaluation figures with `pipeline.py evaluate`. The current source is the processed catalogue, not missing raw listings. EDA unknowns are shown explicitly; observed-only distributions do not impute values for plotting. Correlations use pairwise observed values. Group budgets are labeled synthetic. `outputs/figures/eda_manifest.json` lists actual files and source hash. Older images and G00x CSV/txt files in the parent `outputs/` were retained as unverified legacy outputs; do not cite them as this run's results.

## Troubleshooting and submission limits

- Missing raw CSV: supply the original at the documented path, then `pipeline.py all`. Raw validation currently exits 2; full raw-to-app reproducibility is not claimed.
- Missing cleaned CSV in the slim ZIP: use supplied feature/model artifacts for the apps or restore raw input and regenerate. To retrain from the supplied features, run `groups matrix split train evaluate eda scenarios` (omit `features`).
- Stale hash or feature schema: regenerate matrix/split/train in order. Do not copy unrelated train/test files into place.
- Model version mismatch: use the pinned environment or retrain under the new version. The app catches loading failures and shows a useful artifact error rather than a path/traceback.
- No matches: lower the explicit filters or member rating requirements; no fallback is hidden.
- Windows permission errors: the audit needed permission outside the sandbox for venv bootstrap, model worker pipes and pytest temp directories. This is recorded in the audit, not counted as model failure.
- Runtime cache after retraining: restart Streamlit/API so cached engines reload the new model.
- Notebook investigations of raw data remain blocked until raw input is supplied. Notebook paths were repaired and outputs cleared; notebooks were not executed in this audit.

`.gitignore` excludes raw/private data, cleaned data, large generated matrix/splits, caches, environments, backup ZIPs and secrets; essential small feature/model/metadata files remain includable. GitHub warns above 50 MiB and blocks files above 100 MiB in regular Git; see [GitHub file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). No Git LFS configuration is assumed. No raw upload, commit or push was performed.

See `PROJECT_AUDIT.md` for stage-by-stage outcomes and remaining manual checks.
