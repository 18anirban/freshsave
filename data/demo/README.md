# Demo files (one store: STORE_01, large urban)

Cut from the hidden test data (batches the models never saw). Simulated data.

**Day 0 = Saturday night: inventory loaded. Day 1 = Sunday evening closing ... Day 7 = Saturday evening closing.**

| File | Date | What it is |
|---|---|---|
| day_0_opening_stock.csv | Sat 2 Nov 2024 | Shelf at Saturday night, before Sunday's sales. Batches already on the shelf show their Saturday closing count; newly delivered batches show the full delivery (units_left = units_received, units_sold_today = 0, days_left = shelf_life_days) |
| day_1_sales.csv | Sun 3 Nov | Sunday evening closing |
| day_2_sales.csv ... day_6_sales.csv | Mon 4 - Fri 8 Nov | Daily evening closing |
| day_7_sales.csv | Sat 9 Nov | Saturday evening closing |
| answer_key_hidden.csv | - | Real outcome per batch. Used ONLY by the final dashboard's prediction check, never to predict |

Upload files have the same 15 columns as training, without the outcome columns (units_wasted_final, waste_cost, at_risk).
One or two rows per day have a blank handling_score on purpose (real files have blanks).
Rebuild with: `python data/make_demo_files.py`
