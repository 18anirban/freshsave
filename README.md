# FreshSave - predicting which fresh-food batches will be wasted (retail, KNN + Decision Tree)

## The retail problem
Supermarkets throw away fresh food (milk, meat, bakery, vegetables) that expires before it sells. That is lost money and wasted food.
**Question:** when a batch of fresh food arrives at a store, will 20% or more of it be wasted?
If the answer is yes, the store can order less, promote it early or mark it down in time.

## What the project does
1. Prepares the Kaggle history (which batches were wasted).
2. Teaches two models - **KNN** and **Decision Tree** - to predict "at risk" from what is known on delivery day.
3. **Compares** the two models (and one simple rule) on the same unseen batches.
4. Shows the result in a **web dashboard** that runs on your own computer.

## Folder structure
```
FreshSave/
  01_prepare_data.py     load the Kaggle file, build the label and the inputs, split train/test by date
  02_knn_model.py        KNN (same 14-step pattern as the heart-attack example)
  03_decision_tree.py    Decision Tree (same steps, plus the printed questions it asks)
  04_compare_models.py   side-by-side table, waste cost caught/missed, charts, recommendation
  05_export_dashboard_data.py  writes the tables the dashboard reads
  06_dashboard.py        the web dashboard (streamlit), opens in your browser
  run_all.py             runs files 01 to 05 with one command
  data/make_store_dataset.py   makes the SIMULATED single-store dataset (one row = one batch on one evening)
  data/store/            store_train_70.csv and store_test_30.csv (13 columns, 25,555 rows, 49 product types)
  data/raw/              perishable_goods_management.csv (Kaggle: "Managing Perishable Inventory Data"), kept in Git
  output/                created by the scripts (prepared data, results, charts)
  dashboard_data/        tables for the dashboard
  requirements.txt
```
Every file is written in one pattern: a **STEP header** above each block saying what it does, and a **WHAT WE DID** note after it.
Files 02 and 03 have the same steps in the same order, so they can be read side by side.

## How to run (Windows, VS Code terminal)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python 01_prepare_data.py
python 02_knn_model.py
python 03_decision_tree.py
python 04_compare_models.py
python 05_export_dashboard_data.py
streamlit run 06_dashboard.py
```
Or run files 01 to 05 at once with `python run_all.py`, then `streamlit run 06_dashboard.py`.
Each file prints its results. Charts are saved in `output/`. The dashboard opens at http://localhost:8501.

## How to explain it (one minute)
- **Data:** each Kaggle row is one batch delivered to a store: quantity, normal sales per day, shelf life, price, storage and handling scores, and how much finally sold.
- **Answer column:** a batch is *at risk* if 20% or more of it was wasted.
- **Key input:** `cover_gap_days` = days the stock would last minus days of shelf life. Positive means more stock than can sell in time.
- **Split:** teach on 2023, test on 2024. We do not use a random split, because it would mix the future into the training and make the score look better than it is.
- **KNN:** finds the K most similar past batches and takes a vote. Needs scaling to 0-1 so no column dominates.
- **Decision Tree:** learns yes/no questions and follows them to an answer. Needs no scaling; its questions can be printed.
- **Comparison:** accuracy, precision, recall, F1, ROC-AUC and the waste cost each model caught or missed. A missed batch (false negative) is the costly mistake.

## Honest limits
1. The Kaggle data is **synthetic**. Results show the method works, not that a real store would save money.
2. The models predict on **delivery day** from real columns only. Nothing in the inputs is simulated and no outcome column is used as an input (file 1, step 7 lists what is left out and why).
3. A **simple rule** (stock lasts longer than shelf life) is included as a yardstick. If a model does not beat it, file 4 says so.
4. The recommendation is the model with the best average of recall and F1. It depends on that choice and on the 20% threshold in file 1.
5. Results are not printed here on purpose: run the files and read the numbers from your own run.

## Git
The repository has a single branch, `main`. Earlier work (the Streamlit app with five tabs) is kept in the history of `main`.
