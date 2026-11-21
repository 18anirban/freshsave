# FreshSave - project context (saved checkpoint)

## Goal
Predict which fresh-food batches will be wasted, using KNN and a Decision Tree, compare them, and explain it in one web page for the supervisor.

## Data (data/store/, simulated, built to behave like a real retailer)
- 103,695 rows, 25,656 batches, 4 stores, 8 categories. One row = one batch on one evening of its shelf life.
- Label `at_risk` = 1 when 20% or more of the batch is wasted at the end of its life.
- Train 70% (older batches) / test 30% (newer batches). Split by time and batch; no batch is in both.
- Outcome columns (units_wasted_final, waste_cost, at_risk) and ID columns are never model inputs.

## Code
- src/data_prep.py (shared prep), src/evaluation.py (shared scoring), src/charts.py
- src/model_deployment_train/: knn_model.py, decision_tree_model.py, model_comparison.py
- app.py: landing page with one "Model Train Details" button -> sections 1-6 (data, preparation steps, skewness, models and parameters, results, comparison + conclusion)

## Latest results (test set, nothing tuned on it)
- Decision Tree: accuracy 0.889, recall 0.871, F1 0.862, ROC-AUC 0.957 (winner)
- KNN: accuracy 0.872, recall 0.828, F1 0.837, ROC-AUC 0.948

## Run
```
git pull
pip install -r requirements.txt
python -m src.model_deployment_train.model_comparison
streamlit run app.py
```

## Rules we follow
- Code in the STEP / "WHAT WE DID" snippet style, simple and modular.
- Honest results: no changing data or labels to hit an accuracy target; settings chosen by cross-validation on train only; test used once.

## Still to do
- Project not finished yet; user will add more. Questions-and-answers prep for the supervisor to be written after the project is complete.
