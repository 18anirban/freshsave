# Demo files (one store: STORE_01, large urban)

Cut from the hidden test data (batches the models never saw). Simulated data.

| File | Date | Use |
|---|---|---|
| day_0_opening_stock.csv | Sun 3 Nov 2024 | Upload on Day 0: stock on the shelf |
| day_1_sales.csv ... day_7_sales.csv | Mon 4 - Sun 10 Nov 2024 | Upload one per day |
| answer_key_hidden.csv | - | Real outcome per batch. Used ONLY by the final dashboard's prediction check, never to predict |

Upload files have the same columns as training, without the outcome columns (units_wasted_final, waste_cost, at_risk).
Two or three rows per day have a blank handling_score on purpose (real files have blanks).
Rebuild with: `python data/make_demo_files.py`
