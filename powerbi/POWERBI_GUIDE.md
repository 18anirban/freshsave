# Power BI guide - FreshSave

**You need:** Power BI Desktop (free, Windows only) - https://aka.ms/pbidesktop
**Before you start:** run the five Python files in order (`01` to `05`). File 5 writes the tables into this `powerbi` folder.

## 1. Load the tables
Home > Get data > Text/CSV, and load these six files from this folder:

| File | One row = | Use it for |
|---|---|---|
| `batches_scored.csv` | one batch (2024 test data) | everything at batch level; both models' risk and whether each was right |
| `category_summary.csv` | one category | waste cost and how much each model would have caught |
| `product_summary.csv` | one product | the products that waste the most |
| `model_comparison.csv` | one model (KNN, Decision Tree, Simple rule) | accuracy, precision, recall, F1, ROC-AUC |
| `waste_by_model.csv` | one model | waste cost caught and missed |
| `feature_importance.csv` | one input | what the Decision Tree relied on |

In *Transform data*: set `transaction_date` to Date, prices and costs to Decimal, counts to Whole number. Close & Apply.

For the model-comparison chart, in Power Query select `model_comparison`, select the `model` column, and use
**Unpivot other columns** on `accuracy, precision, recall, f1, roc_auc` (keep a copy of the original table for the numbers).

## 2. Measures (Modeling > New measure)
```DAX
Total Waste Cost   = SUM ( batches_scored[waste_cost] )
Batches            = COUNTROWS ( batches_scored )
At Risk Batches    = CALCULATE ( COUNTROWS ( batches_scored ), batches_scored[at_risk] = 1 )
At Risk Share      = DIVIDE ( [At Risk Batches], [Batches] )

KNN Recall         = DIVIDE ( CALCULATE ( COUNTROWS ( batches_scored ), batches_scored[knn_result] = "TP caught" ), [At Risk Batches] )
Tree Recall        = DIVIDE ( CALCULATE ( COUNTROWS ( batches_scored ), batches_scored[tree_result] = "TP caught" ), [At Risk Batches] )
Missed Waste Cost (Tree) = CALCULATE ( [Total Waste Cost], batches_scored[tree_result] = "FN missed" )
Missed Waste Cost (KNN)  = CALCULATE ( [Total Waste Cost], batches_scored[knn_result] = "FN missed" )
```

## 3. Three pages
1. **Waste overview** - cards: Total Waste Cost, Batches, At Risk Share. Bar chart: Total Waste Cost by `category`. Line chart: Total Waste Cost by `month`. Slicers: `category`, `store_id`.
2. **Model comparison** - clustered column chart from the unpivoted `model_comparison` (metric on the axis, model as legend). Cards: KNN Recall, Tree Recall, Missed Waste Cost (KNN), Missed Waste Cost (Tree). Two matrix visuals: rows `knn_result` / `tree_result`, values `Batches` (this is the confusion matrix). Bar chart of `feature_importance` (top 10 inputs).
3. **Products to watch** - table from `product_summary`: category, product, batches, at_risk_share, waste_cost, sorted by waste_cost; slicer on `category`.

## 4. Say this on the page
A text box: *"Kaggle data is synthetic. Results show the method, not savings in a real store. Test period: 2024."*

## 5. Refresh
Re-run files 02 to 05 after any change, then Home > Refresh in Power BI. If you move the project, use Transform data > Data source settings > Change source.
