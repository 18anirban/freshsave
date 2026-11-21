# FreshSave

Predicts which fresh-food batches will be wasted, using **KNN** and a **Decision Tree**, and compares them.

## Structure
```
app.py                                  one-tab web app: "Model Train Details"
data/store/                             simulated retail data (train 70% / test 30%, split by time and batch)
src/data_prep.py                        shared: load, features, missing values, skewness
src/evaluation.py                       shared: one scoring function for all models
src/charts.py                           charts for the app
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
