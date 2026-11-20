"""
ui - THE SCREEN, ONE FILE PER TAB
==================================
    helpers.py         small shared helpers (tables, charts, money format, file reading)
    step_bar.py        the horizontal step strip at the top
    tab_setup.py       Tab 1  Setup               master file + model choice
    tab_upload.py      Tab 2  Evening upload      day files, checks, upload history
    tab_decisions.py   Tab 3  Daily decisions     the manager's action list, scorecard
    tab_dashboard.py   Tab 4  Manager dashboard   waste, predictions, decisions, 15-day order plan
    tab_models.py      Tab 5  Model comparison    how KNN and the Decision Tree were tested

These files only DISPLAY things. The real work is in src/ (scoring, rules, store, order_plan ...).
"""
