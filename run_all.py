# ============================================================
# FRESHSAVE - RUN EVERYTHING WITH ONE COMMAND
# ------------------------------------------------------------
# THIS FILE: runs files 01 to 05 in order. If one of them fails it stops and tells you which.
# RUN:       python run_all.py
# THEN:      in Power BI click Home > Refresh. The dashboard shows the new numbers.
# ============================================================


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------
# - subprocess : starts each Python file, the same as typing "python 01_prepare_data.py"
# - sys        : tells us which Python to use (the one running this file, so your .venv is kept)
# ============================================================

import subprocess
import sys
import time
from pathlib import Path

# WHAT WE DID IN STEP 1:
# - Loaded the tools to start the other files.



# ============================================================
# STEP 2: RUN THE FIVE FILES IN ORDER
# ------------------------------------------------------------
# The order matters: each file reads what the one before it saved.
# ============================================================

FOLDER = Path(__file__).resolve().parent
FILES = ["01_prepare_data.py", "02_knn_model.py", "03_decision_tree.py", "04_compare_models.py", "05_export_for_bi.py"]

start = time.time()
for number, name in enumerate(FILES, start=1):
    print(f"\n########## {number} of {len(FILES)}: {name} ##########")
    done = subprocess.run([sys.executable, str(FOLDER / name)], cwd=FOLDER)
    if done.returncode != 0:
        raise SystemExit(f"\nSTOPPED: {name} failed (see the red text above). Fix it and run again.")

# WHAT WE DID IN STEP 2:
# - Ran every file. Each one printed its own results above.



# ============================================================
# STEP 3: TELL THE USER WHAT TO DO NEXT
# ============================================================

print(f"\nALL DONE in {time.time() - start:.0f} seconds.")
print("The Power BI tables are in:", FOLDER / "powerbi")
print("Now open Power BI and click Home > Refresh.")

# WHAT WE DID IN STEP 3:
# - Printed where the files are and the next click.
