# FreshSave - AI-assisted markdown and waste control for ONE supermarket store (proof of concept)

Each evening the store manager uploads the closing-stock file. FreshSave predicts which fresh batches will **not sell
before they expire**, suggests a markdown / front-display / discard action, records what the manager decides, and after
two weeks shows waste, estimated savings and how accurate the predictions were. **The system suggests, the manager decides.**

## The flow (explain this first)

```
 Kaggle history file ──► train.py ──► 2 models (Decision Tree, KNN) + test results      [done once, at "head office"]
 Kaggle demo store   ──► make_daily_files.py ──► Day01 ... Day14 CSVs                    [stands in for the store's
                                                                                           nightly inventory export]
 ONCE:           upload Master_Fortnight_Batches.csv (the fortnight's batch list) + choose KNN or Decision Tree   [Tab 1]
 EVERY EVENING:  upload DayNN.csv ─► check + clean ─► score risk ─► rules ─► manager list ─► Approve / Change / Reject
                                                                                      │
                                              powerbi_data/*.csv  ◄───────────────────┘  (rewritten after every change)
 AFTER DAY 14:   Manager dashboard opens ─► waste so far + recommended order for the next 15 days
```

A horizontal **step bar** at the top shows where you are: 1 Setup, 2 Upload day files, 3 Model and recommendations,
4 Manager dashboard. Each step turns green when it is done.

| Tab | Who | What |
|---|---|---|
| 1. Setup | you / the store | upload the **master file** (every batch of the fortnight) and **choose the model** (KNN or Decision Tree). The other tabs wait for the master file |
| 2. Evening upload | store system / you in the demo | upload the day's file; checks, cleaning, comparison with the master file, upload history |
| 3. Daily decisions | store manager | top 10-15 batches by money at stake, reason, suggested price, Approve / Change / Reject |
| 4. Manager dashboard | manager, head office | opens after all 14 days: waste, **recommended order for the next 15 days**, predicted-vs-actual, override rate |
| 5. Model comparison | analyst, sir | full model comparison, plus "How the models work": the tree's printed rules and the neighbours behind a KNN prediction |

## Quick start (Windows + VS Code)

```powershell
cd FreshSave
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py             # a few minutes: trains and tests both models (Decision Tree and KNN)
python src/make_daily_files.py  # creates the 14 daily files + the master file (also in C:\Users\18ani\Downloads\DailyFiles-Retail)
streamlit run app.py            # opens http://localhost:8501
python -m pytest tests -q       # optional: 22 automated checks
python tests/app_smoke.py       # optional: runs every tab's code without a browser
```
The Kaggle file must be at `data/raw/perishable_goods_management.csv` (Kaggle: "Managing Perishable Inventory Data").

## Demo script (about 10 minutes)
1. Tab 1 (Setup): upload `Master_Fortnight_Batches.csv` (or click "Load the sample master file"), choose **Decision Tree** or **KNN**. The step bar turns green.
2. Tab 2: upload `Day01_Sun.csv` -> checks and scoring happen. Tab 3: show the action list, approve some, change one, reject one, **Save**.
3. Upload `Day02_Mon.csv` -> risks have moved (stock sold, days left fell). Scorecard appears.
4. Click **Load ALL remaining days + auto-approve** (demo shortcut; those decisions are tagged `auto-demo`, say so).
5. Tab 4 (Manager dashboard) is now open: waste, and the recommended order for the next 15 days.
6. Go back to Tab 1, switch the model, and watch Tabs 3-4 change (a different model flags different products as high risk).
7. Tab 5: model comparison. Open "How the models work": the decision tree's printed rules, and the neighbours behind one KNN prediction.
8. Power BI: refresh and show the same numbers (see `POWERBI_GUIDE.md`).

## Where to look in the code (walk-through order)
| File | One-line purpose |
|---|---|
| `config.py` | every setting in one place (store, thresholds, paths, assumptions) |
| `src/features.py` | cleans the Kaggle data, builds the label, builds the model inputs (log transform for skewed columns) |
| `src/simulate.py` | the day-by-day sales simulator (the main assumption) |
| `src/models/decision_tree_model.py` | the Decision Tree: its settings, how it is built, its printed rules |
| `src/models/knn_model.py` | KNN: scaling + neighbours, and the table of neighbours behind one prediction |
| `src/models/__init__.py` | the list of models (`REGISTRY`). Add a model here and train.py, scoring and the app pick it up |
| `src/train.py` | loops over the models: tunes and trains each, tests on 2024, saves every metric |
| `src/make_daily_files.py` | creates the 14 daily files and the master file for the demo store |
| `src/master.py` | checks, saves and uses the master file (batch list of the fortnight) |
| `src/scoring.py` | validates/cleans an uploaded file, predicts risk, writes the plain-English reason |
| `src/rules.py` | risk -> action (the rule layer is NOT a model) + the markdown effect estimate |
| `src/order_plan.py` | how much to order per product for the next 15 days, and in how many deliveries |
| `src/store.py` | the app's memory (saved days, decisions) and the Power BI exports |
| `app.py` | the traffic controller: sidebar, step bar, five tabs. It only displays; the logic is in `src/` |
| `ui/` | one file per tab (`tab_setup.py`, `tab_upload.py`, `tab_decisions.py`, `tab_dashboard.py`, `tab_models.py`), `step_bar.py`, `helpers.py` |
| `tests/` | `test_pipeline.py` (22 checks) and `app_smoke.py` (runs the app logic without a browser) |

The model files, `master.py`, `order_plan.py` and the `ui/` files follow one pattern: a STEP header above each block and a
"WHAT WE DID" note below it, so each can be walked through in order.

## What is real and what is simulated (please say this out loud to sir)
| | Status |
|---|---|
| The Kaggle dataset | **Synthetic** (not a real supermarket's records) |
| Product, category, quantity received, prices, storage and handling scores, final units sold | Real values from that file |
| Delivery day (Sunday / Wednesday top-up) | **Simulated** |
| Shelf life | **Compressed** to 2-10 days (and capped so every batch finishes inside the 14 days) |
| Stock left on each evening | **Simulated**: the real final sales are spread over the shelf life with random weights; the path ends exactly on the real final waste |
| Effect of a markdown on sales | **Assumption** (default: 20% off gives +15% sales). Only used for the "estimated" figures; adjustable in the sidebar |
| Master file | **Derived** from the 14 daily files by `make_daily_files.py` (a real store would export it from its ordering system) |
| 15-day order plan | **Assumption**: sales over the next 15 days continue at the speed of the last fortnight (no seasonality). All numbers in `config.py` |

## Results (tested on 2024 data from all 50 stores, trained on 2023 only)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time |
|---|---|---|---|---|---|---|
| Decision Tree | 85.6% | 84.7% | 79.9% | 82.3% | 0.937 | 2 s |
| KNN | run `python src/train.py` | | | | | |

The Decision Tree row was measured before KNN replaced the ANN and the tree's code did not change. **KNN has not been measured on
the real Kaggle file yet**: run `python src/train.py` and the numbers appear in Tab 5 (and in `powerbi_data/model_comparison.csv`).
The default model is whichever of the two has the best average of recall and F1 on the 2024 test data; `train.py` decides, and
Tab 1 shows the result and lets you switch.

## Honest limits
1. **Accuracy grows through the shelf life partly by construction.** The simulated path is built from each batch's real final sales, so later evenings reveal most of the answer. The honest number is the **first evening**: about **73% (Decision Tree)** accuracy on all stores. (The earlier ANN figure was about 74%; the same number for KNN is in Tab 5, "Accuracy by evening", once you have trained it.) On the demo store's 848 batches the first-evening result is lower, and it is a small sample.
2. **Markdowns reduce waste but, under the default assumption, lose money.** If a 20% discount lifts sales by less than about 25%, the discount costs more than it recovers; in this data (where the stock cap limits extra sales) it only breaks even at roughly 3x response (20% off giving ~+60% sales). The dashboard shows waste avoided, revenue recovered, discount given away and the net, so this is visible rather than hidden. Move the sidebar slider to explore.
3. The demo's "decisions" on later days come from the auto-approve shortcut, not a manager.
4. Synthetic data, one store, 14 simulated days: this is a proof of concept of the **workflow**, not evidence of real-world savings.
5. The app was logic-tested without a browser (a stand-in replaced Streamlit and Plotly in the test); the first real browser run is on your machine.
6. **KNN, the Setup tab, the master file, the step bar and the 15-day order plan were added without access to the Kaggle file.** They were tested
   with a synthetic stand-in dataset (train.py ran end to end, 22 checks and the smoke test passed), so the code paths are proven but no real KNN
   accuracy exists yet. KNN predicts slower than the tree: `config.py` caps its stored training rows (`KNN_MAX_TRAIN_ROWS`) to keep training and testing reasonable.
7. The order plan is per product and assumes the last fortnight's sales speed continues. It is a starting point for the manager, not a forecast.

## Git
```powershell
git init            # already done if you got the zip with .git
git add .
git commit -m "FreshSave proof of concept"
git branch -M main
git remote add origin https://github.com/<your-user>/freshsave.git
git push -u origin main
```
`.gitignore` keeps the 21 MB Kaggle file, the trained models and the app's saved state out of the repository; anyone who clones it re-creates them with the three commands above.

## Working with Git (VS Code <-> GitHub)
- Before you start working: **Sync Changes** in VS Code (or `git pull`) to get the newest code.
- After you edit: Source Control icon > write a message > **Commit** > **Sync Changes** (or `git add .`, `git commit -m "message"`, `git push`).
- Files kept out of Git on purpose (see `.gitignore`): the Kaggle CSV, trained models and the app's saved state. After a fresh clone, put the Kaggle file in `data/raw/` and run `python src/train.py`.
