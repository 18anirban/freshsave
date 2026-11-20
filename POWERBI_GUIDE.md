# Power BI guide - FreshSave

**You need:** Power BI Desktop (free, **Windows only**) - https://aka.ms/pbidesktop . Nothing else.
**Data:** the CSVs in `powerbi_data/`. The folder ships with a **sample 14-day run** so you can build the report before using the app;
the app rewrites these files after every upload, decision or model switch (the Reset button deletes them).

## 1. Files (tables)
| File | One row = | Key columns |
|---|---|---|
| `batch_scores.csv` | one batch on one evening | day_no, batch_id, product_name, category, qty_left, days_left, risk, action, suggested_markdown, reposition, new_price, money_at_stake, discard_cost, reason, model_used |
| `outcomes.csv` | one finished batch | batch_id, closed_day, units_wasted, waste_pct, waste_cost, actually_at_risk, risk_first_evening, risk_last_evening |
| `decisions_log.csv` | one manager decision | day_no, batch_id, risk, suggested_markdown, decision (Approve/Change/Reject), final_markdown, decided_by (manager/auto-demo) |
| `decision_effects.csv` | one decision on a finished batch (ESTIMATED) | batch_id, waste_cost_avoided, revenue_recovered, margin_given_away, net_cash_effect |
| `daily_summary.csv` | one day | batches_on_shelf, at_risk_flagged, money_at_stake, discard_tonight, discard_cost, closed_batches, waste_cost_closed, approved/changed/rejected |
| `calendar.csv` | one day | day_no, date, weekday, week, delivery |
| `model_comparison.csv` | one model x one test view | accuracy, precision, recall, f1, roc_auc, specificity, mcc, log_loss, brier, tp/fp/tn/fn, train_seconds |

## 2. Load and clean (Power Query)
1. Home > Get data > **Text/CSV**, load each file (or Get data > Folder).
2. For each table: *Transform data* and set types: dates -> Date, `risk`, prices, costs -> Decimal, counts -> Whole number.
3. `batch_scores`: `reason`/`action` stay text. Keep rows as they are (days_left = 0 rows are the discard list, risk is empty there on purpose).
4. Create a small **Batches** table: duplicate `batch_scores`, keep batch_id, product_name, category, delivered_date, expiry_date, qty_received, cost_price, then *Remove duplicates* on batch_id.
5. Create a what-if parameter **Risk Threshold** (Modeling > New parameter, 0.1 to 0.9, step 0.05, default 0.4).
6. Close & Apply.

## 3. Relationships (Model view)
```
calendar[day_no] 1 ──► * batch_scores[day_no]      calendar[day_no] 1 ──► * daily_summary[day_no]
calendar[day_no] 1 ──► * decisions_log[day_no]     calendar[day_no] 1 ──► * outcomes[closed_day]
Batches[batch_id] 1 ──► * batch_scores / outcomes / decisions_log / decision_effects  (by batch_id)
```
Leave `model_comparison` unrelated (a stand-alone table).

## 4. DAX measures
```DAX
Total Waste Cost      = SUM ( outcomes[waste_cost] )
Waste %               = DIVIDE ( SUM ( outcomes[units_wasted] ), SUM ( outcomes[qty_received] ) )
Batches Finished      = COUNTROWS ( outcomes )
Money At Stake        = CALCULATE ( SUM ( batch_scores[money_at_stake] ), batch_scores[days_left] > 0 )
Discard Cost          = SUM ( batch_scores[discard_cost] )

-- Estimated effect of the manager's decisions (ASSUMPTION-based: label it on the page)
Est Waste Avoided     = SUM ( decision_effects[waste_cost_avoided] )
Est Revenue Recovered = SUM ( decision_effects[revenue_recovered] )
Est Discount Given    = SUM ( decision_effects[margin_given_away] )
Est Net Cash Effect   = [Est Revenue Recovered] - [Est Discount Given]
Savings (waste cost)  = [Est Waste Avoided]

-- Predicted vs actual (first evening = the honest test)
Actual At Risk        = CALCULATE ( COUNTROWS ( outcomes ), outcomes[actually_at_risk] = 1 )
Caught                = CALCULATE ( COUNTROWS ( outcomes ), outcomes[actually_at_risk] = 1,
                                    outcomes[risk_first_evening] >= 'Risk Threshold'[Risk Threshold Value] )
False Alarms          = CALCULATE ( COUNTROWS ( outcomes ), outcomes[actually_at_risk] = 0,
                                    outcomes[risk_first_evening] >= 'Risk Threshold'[Risk Threshold Value] )
Recall                = DIVIDE ( [Caught], [Actual At Risk] )
Precision             = DIVIDE ( [Caught], [Caught] + [False Alarms] )
Avg Predicted Risk    = AVERAGE ( outcomes[risk_first_evening] )
Avg Actual Risk       = AVERAGE ( outcomes[actually_at_risk] )

-- Manager behaviour
Override Rate         = DIVIDE ( CALCULATE ( COUNTROWS ( decisions_log ), decisions_log[decision] IN { "Change", "Reject" } ),
                                 COUNTROWS ( decisions_log ) )
Auto-approved Share   = DIVIDE ( CALCULATE ( COUNTROWS ( decisions_log ), decisions_log[decided_by] = "auto-demo" ),
                                 COUNTROWS ( decisions_log ) )
```
(If your parameter table has a different name, use the name Power BI created: the measure is `[<ParameterName> Value]`.)

## 5. Four pages
1. **Executive Summary** - cards: Total Waste Cost, Waste %, Batches Finished, Est Waste Avoided, Est Net Cash Effect, Override Rate. Line: waste cost by calendar[date]. Text box: *"Estimated figures assume a sales response to markdowns; the daily sales path is simulated."*
2. **Daily Action** - slicer on calendar[day_no]; table of batch_scores (product, category, qty_left, days_left, risk, action, new_price, money_at_stake, reason) filtered to `days_left > 0` and `risk >= threshold`, sorted by money_at_stake; card: Discard Cost; second table of the discard list (`days_left = 0`).
3. **Waste and Savings Trend** - clustered column: Total Waste Cost vs (Total Waste Cost - Est Waste Avoided) by day; bar by category; stacked bar of Est Revenue Recovered vs Est Discount Given (shows markdowns cost margin); actions per day (daily_summary).
4. **Model Comparison** - clustered bar of accuracy / recall / f1 / roc_auc by model (slicer on `view`); table of model_comparison; calibration-style column chart: Avg Predicted Risk vs Avg Actual Risk by risk band (add a calculated column that bins outcomes[risk_first_evening]); cards Recall and Precision with the Risk Threshold slider.

## 6. Refresh
Run the demo in the app, then in Power BI click **Home > Refresh**. (Keep `powerbi_data` where it is; if you move the project, Home > Transform data > Data source settings > Change source.) For a published report, a gateway would schedule the same refresh nightly.
