# What was wrong, what changed, and how to verify it

## Running the pipeline

```bat
:: one-time environment (see requirements.txt for the Smart App Control note)
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements.txt

:: rebuild everything, in this order
.venv\Scripts\python build_graph.py            :: graph + preprocessor  (~2 min)
.venv\Scripts\python save_edge_attr_lookup.py  :: inference attachment index
.venv\Scripts\python save_node_meta.py         :: real columns for the dashboard
.venv\Scripts\python train.py                  :: ~20 min on an RTX 4050
.venv\Scripts\python evaluate.py
.venv\Scripts\python baselines.py
.venv\Scripts\python visualize.py
```

`build_graph.py --split random` reproduces the old (over-optimistic) protocol
for comparison. The default is `--split temporal`.

## Verifying it

| Command | Checks | Needs server |
| --- | --- | --- |
| `python validate_artifacts.py` | Model / preprocessor / graph / lookups all agree; splits don't leak; serving reproduces training features | no |
| `python validate_api.py` | Every endpoint returns real data, not the old fabricated fallbacks | yes |
| `python verify_serving.py` | Upload path scored on held-out transactions | no |

Start the server with:

```bat
.venv\Scripts\python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Login for `validate_api.py` is `admin` / `fraud123` (seeded in `database.py`).

## The bugs that mattered

**Prediction was broken by a threshold mismatch.** `evaluate.py` tuned a
decision threshold on validation, printed a confusion matrix at it, and threw
it away. The server hardcoded `0.5`. The trained model's probabilities topped
out at 0.65 with a median of 0.19, so at 0.5 the app flagged almost nothing.
The threshold is now tuned, saved into `best_model.pt`, and read everywhere.

**The model was barely trained.** The loop was full-batch: one optimiser step
per epoch, so `EPOCHS = 200` meant 200 gradient updates. The training log shows
validation AUC still climbing at the final epoch with a train/val loss gap of
5e-5 — underfitting, not overfitting. Mini-batch training via `graph_sampler.py`
now gives ~200 steps per epoch.

**Class imbalance was never handled.** `FocalLoss(alpha=0.25)` multiplied
positives and negatives by the same constant, which is a rescale, not
balancing. `compute_pos_weight()` was defined and never called, while the
module docstring claimed it was in use.

**The graph deleted its own signal.** `GROUP_CAP = 30` dropped every attribute
group with more than 30 members — exactly what a fraud ring looks like. On a
2,500-row upload it produced zero intra-batch edges. Edges now link each
transaction to its 3 most recent peers per attribute: 2.28M edges instead of
481K, average degree 7.7 instead of 1.6.

**Edge weights were decorative.** `build_edges` accumulated weights (card1=3.0,
DeviceType=1.0, …) into `data.edge_attr`, and `SAGEConv` never received them.
The model now uses `GraphConv`, which takes `edge_weight`.

**Evaluation was optimistic.** Preprocessing was `fit_transform`-ed on all
590,540 rows, so scaler and encoder statistics came partly from the test set;
and a time-ordered dataset was split at random with raw `TransactionDT` left in
as a feature. The split is now chronological and preprocessing is fit on
training rows only. Reported numbers drop as a result — that is the measurement
being corrected, not the model getting worse. Random Forest falls from 0.567 to
0.506 F1 on unchanged code.

**The dashboard invented data.** `/api/metrics` and `/api/dashboard` returned
hardcoded numbers (accuracy 0.974, precision 0.82, recall 0.71) from a bare
`except`, indistinguishable from real ones. Analytics drew its fraud-by-device,
by-card and by-email breakdowns from `rng.uniform()`, over device categories
("Tablet", "POS Terminal", "ATM") that do not exist in IEEE-CIS. The
transaction table showed a random card type and a random 2019 date per row, and
reconstructed "amount" as `abs(x[0]) * 500 + 10` from a standardised feature.
All of it now reads `outputs/models/node_meta.pkl`, and failures raise instead
of substituting fiction.

**The Settings threshold slider did nothing.** It saved to the database and no
prediction code read it.

**The sample CSV was unusable.** `/api/upload-predict/sample` generated 19
V-columns of `rng.uniform(0, 1)` where the real data has 339 structured ones.
Anyone who downloaded it and uploaded it got meaningless predictions.

## Known-bad files in this repo

`synthetic_corrected_2500.csv` (12 MB) is **fabricated data**, almost certainly
the output of the old sample generator: its `V1` is `0.9957` where the real
`V1` is exactly `1.0`, and all 339 V-columns are random floats uncorrelated
with the label. The AUC of 0.504 recorded against it is the expected result for
random data and says nothing about the model. Delete it or keep it clearly
labelled — do not evaluate against it.

Thirteen diagnostic scripts carry a `SUPERSEDED` banner explaining why their
output cannot be trusted (most score rows the model trained on). `verify_serving.py`
replaces them.

## Where the model stands

Test split, chronological, 88,581 rows, 3.48% fraud. Thresholds tuned on
validation for every model.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Flag nothing | 0.9652 | — | 0.000 | 0.000 | — | — |
| Logistic Regression | 0.9389 | 0.253 | 0.389 | 0.307 | 0.825 | 0.164 |
| Random Forest | 0.9694 | 0.575 | 0.465 | 0.515 | 0.906 | 0.531 |
| XGBoost | 0.9707 | 0.607 | 0.448 | 0.516 | 0.892 | 0.524 |
| LightGBM | 0.9706 | 0.604 | 0.447 | 0.514 | 0.891 | 0.520 |
| GraphSAGE | 0.9645 | 0.488 | 0.405 | 0.442 | 0.826 | 0.415 |

**Do not quote accuracy.** Flagging nothing scores 96.5% — higher than the GNN.
Accuracy on a 3.5%-positive problem measures the base rate. Use F1 and PR-AUC,
which have a meaningful floor (0.0348 for random guessing).

Gradient boosting still beats the GNN. That is the expected result on 400+
heterogeneous tabular columns with a thin graph over them (average degree 7.7,
13% of nodes isolated), not a remaining bug. The realistic way to clear it is
to blend the GNN with LightGBM, whose errors are decorrelated from it.

### Upload path

Measured on 4,000 held-out transactions (`verify_serving.py`):

| Scenario | ROC-AUC | PR-AUC | Recall |
| --- | --- | --- | --- |
| Full schema | 0.779 | 0.339 | 31% |
| Reduced schema (17 of 434 columns) | 0.734 | 0.104 | 0.7% |
| Same rows scored in-graph (ceiling) | 0.824 | 0.367 | 34% |

**A thin CSV destroys the model.** Sending only transaction-level columns
drops recall to 0.7% — one fraud caught out of 141. The V/C/D columns are not
optional. Uploads are now warned about in the log; consider rejecting them
outright in the UI.
