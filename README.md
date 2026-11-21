# FreshSave

Predicts which fresh-food batches will be wasted, using **KNN** and a **Decision Tree**, and compares them.

## Structure
```
app.py                                  web app: landing page with two tabs, "Model Train Details" and "Demo"
data/store/                             simulated retail data (train 70% / test 30%, split by time and batch)
src/data_prep.py                        shared: load, features, missing values, skewness
src/evaluation.py                       shared: one scoring function for all models
src/charts.py                           charts for the app
src/demo/                               Demo tab: scoring.py (check + predict), history.py (saved days), dashboard.py, page.py
data/demo/                              8 upload files (Day 0 stock, Day 1-7 sales) + hidden answer key
src/model_deployment_train/
    knn_model.py                        model 1
    decision_tree_model.py              model 2
    model_comparison.py                 model 3: compare + train everything
```

## Run
```
pip install -r requirements.txt
python -m src.model_deployment_train.model_comparison   # trains, saves to models/
streamlit run app.py                                     # open the "Model Train Details" tab
```
(The app also has a "Train now" button if you skip the first command.)

## Demo tab
1. Click **Demo** on the landing page, pick a model (Decision Tree is the default).
2. Day 0: upload `data/demo/day_0_opening_stock.csv` (or tick "Use demo file"), look at the risk list, set the action taken, click **Save Day 0**.
3. Repeat for Day 1 to Day 7 with the matching `day_N_sales.csv`.
4. After Day 7 the final dashboard opens. **Reset demo** starts again. Saved days live in `demo_runs/` (not in git).
