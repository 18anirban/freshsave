# FreshSave - AI-assisted markdown and waste control for ONE supermarket store (proof of concept)

Each evening the store manager uploads the closing-stock file. FreshSave predicts which fresh batches will **not sell
before they expire**, suggests a markdown / front-display / discard action, records what the manager decides, and after
two weeks shows waste, estimated savings and how accurate the predictions were. **The system suggests, the manager decides.**

## The flow (explain this first)

```
 Kaggle history file ──► train.py ──► 2 models (Decision Tree, ANN) + test results      [done once, at "head office"]
 Kaggle demo store   ──► make_daily_files.py ──► Day01 ... Day14 CSVs                    [stands in for the store's
                                                                                           nightly inventory export]
 EVERY EVENING:  upload DayNN.csv ─► check + clean ─► score risk ─► rules ─► manager list ─► Approve / Change / Reject
                                                                                      │
                                              powerbi_data/*.csv  ◄───────────────────┘  (rewritten after every change)
```

| Tab | Who | What |
|---|---|---|
| 1. Evening upload | store system / you in the demo | upload the day's file; checks, cleaning, upload history |
| 2. Daily decisions | store manager | top 10-15 batches by money at stake, reason, suggested price, Approve / Change / Reject |
| 3. Two-week dashboard | manager, head office | waste, estimated savings, predicted-vs-actual, override rate |
| 4. Model comparison | analyst, sir | full model comparison + the dropdown that picks the active model |

## Quick start (Windows + VS Code)

```powershell
cd FreshSave
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py             # ~1-2 minutes: trains and tests both models
python src/make_daily_files.py  # creates the 14 daily files (also in C:\Users\18ani\Downloads\DailyFiles-Retail)
streamlit run app.py            # opens http://localhost:8501
python -m pytest tests -q       # optional: 14 automated checks
```
The Kaggle file must be at `data/raw/perishable_goods_management.csv` (Kaggle: "Managing Perishable Inventory Data").

## Demo script (about 10 minutes)
1. Tab 1: upload `Day01_Sun.csv` -> checks and scoring happen. Tab 2: show the action list, approve some, change one, reject one, **Save**.
2. Upload `Day02_Mon.csv` -> risks have moved (stock sold, days left fell). Scorecard appears.
3. Click **Load ALL remaining days + auto-approve** (demo shortcut; those decisions are tagged `auto-demo`, say so).
4. Tab 3: the two-week dashboard. Tab 4: model comparison; switch the model and watch Tabs 2-3 change.
5. Power BI: refresh and show the same numbers (see `POWERBI_GUIDE.md`).

## Where to look in the code (walk-through order)
| File | One-line purpose |
|---|---|
| `config.py` | every setting in one place (store, thresholds, paths, assumptions) |
| `src/features.py` | cleans the Kaggle data, builds the label, builds the model inputs (log transform for skewed columns) |
| `src/simulate.py` | the day-by-day sales simulator (the main assumption) |
| `src/train.py` | trains + tunes Decision Tree and ANN, tests on 2024, saves every metric |
| `src/make_daily_files.py` | creates the 14 daily files for the demo store |
| `src/scoring.py` | validates/cleans an uploaded file, predicts risk, writes the plain-English reason |
| `src/rules.py` | risk -> action (the rule layer is NOT a model) + the markdown effect estimate |
| `src/store.py` | the app's memory (saved days, decisions) and the Power BI exports |
| `app.py` | the 4-tab screen. It only displays; the logic is in `src/` |
| `tests/` | `test_pipeline.py` (14 checks) and `app_smoke.py` (runs the app logic without a browser) |

## What is real and what is simulated (please say this out loud to sir)
| | Status |
|---|---|
| The Kaggle dataset | **Synthetic** (not a real supermarket's records) |
| Product, category, quantity received, prices, storage and handling scores, final units sold | Real values from that file |
| Delivery day (Sunday / Wednesday top-up) | **Simulated** |
| Shelf life | **Compressed** to 2-10 days (and capped so every batch finishes inside the 14 days) |
| Stock left on each evening | **Simulated**: the real final sales are spread over the shelf life with random weights; the path ends exactly on the real final waste |
| Effect of a markdown on sales | **Assumption** (default: 20% off gives +15% sales). Only used for the "estimated" figures; adjustable in the sidebar |

## Results (tested on 2024 data from all 50 stores, trained on 2023 only)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time |
|---|---|---|---|---|---|---|
| Decision Tree | 85.6% | 84.7% | 79.9% | 82.3% | 0.937 | 2 s |
| ANN | 86.3% | 84.2% | 82.7% | 83.4% | 0.944 | 6 s |

The ANN is the default (best recall + F1). The gap is small; the Decision Tree is the better choice if explainability matters more.

## Honest limits
1. **Accuracy grows through the shelf life partly by construction.** The simulated path is built from each batch's real final sales, so later evenings reveal most of the answer. The honest number is the **first evening**: about **74% (ANN) / 73% (Decision Tree)** accuracy on all stores. On the demo store's 848 batches, at the 40% rule, the first-evening result is lower: ANN accuracy about 66%, recall 77%, AUC 0.78 (a small sample).
2. **Markdowns reduce waste but, under the default assumption, lose money.** If a 20% discount lifts sales by less than about 25%, the discount costs more than it recovers; in this data (where the stock cap limits extra sales) it only breaks even at roughly 3x response (20% off giving ~+60% sales). The dashboard shows waste avoided, revenue recovered, discount given away and the net, so this is visible rather than hidden. Move the sidebar slider to explore.
3. The demo's "decisions" on later days come from the auto-approve shortcut, not a manager.
4. Synthetic data, one store, 14 simulated days: this is a proof of concept of the **workflow**, not evidence of real-world savings.
5. The app was logic-tested without a browser (a stand-in replaced Streamlit and Plotly in the test); the first real browser run is on your machine.

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
