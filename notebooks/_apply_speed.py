"""Speed up the Day 4-5 cells without changing any result.

1. n_jobs=2 -> n_jobs=-1 on the remaining heavy learners (box has 8 cores).
2. Cells 21 (calibration) and 23 (top-K) re-trained the exact same estimator
   on the exact same features as cell 20 (stats) -- same params, same seed,
   same data. Re-training reproduces the identical predictions, just slowly.
   They now reuse cell 20's out-of-fold predictions directly.
3. Cell 20's memory cleanup must NOT delete esm_scores / y_tests, which
   cells 21 and 23 now depend on.

Run once, then delete.
"""
import json

NB_PATH = 'rabies_antibody_ml.ipynb'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)


def src_of(i):
    return ''.join(nb['cells'][i]['source'])


def set_src(i, text):
    nb['cells'][i]['source'] = text.splitlines(keepends=True)


# ---------------------------------------------------------------------
# 1. n_jobs=2 -> n_jobs=-1 (cells 21/23 are handled by the rewrite below)
# ---------------------------------------------------------------------
for ci in [15, 16, 17, 20, 28]:
    s = src_of(ci)
    n = s.count('n_jobs=2')
    assert n >= 1, f'cell {ci}: no n_jobs=2 found'
    set_src(ci, s.replace('n_jobs=2', 'n_jobs=-1'))
    print(f'cell {ci}: {n} learner(s) -> n_jobs=-1')

# ---------------------------------------------------------------------
# 2a. Cell 21 (calibration) -- reuse cell 20's out-of-fold predictions
# ---------------------------------------------------------------------
MARK21 = '# --- Bin predictions into 10 deciles ---'
s = src_of(21)
head, sep, tail = s.partition(MARK21)
assert sep, 'cell 21: binning marker not found'

PREFIX21 = '''# ============================================================
# DAY 5 — Exp 3: Reliability Diagram (Calibration)
# ============================================================
# Question: When the model says "80% escape probability",
# does escape actually happen 80% of the time?
#
# Method:
#   1. Take the out-of-fold predictions from Exp 1 (cell above)
#   2. Bin them into deciles (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
#   3. Compute observed positive rate per bin
#   4. Plot predicted vs observed (diagonal = perfect calibration)
#   5. Compute Expected Calibration Error (ECE)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# --- Out-of-fold predictions from Day 5 Exp 1 ---
# Exp 1 trains this exact estimator (same features, params, seed) per held-out
# antibody, so re-training here would reproduce these predictions exactly --
# just far more slowly. Reuse them instead.
antibodies = sorted(esm_scores)
all_scores = np.concatenate([esm_scores[ab] for ab in antibodies])
all_labels = np.concatenate([y_tests[ab] for ab in antibodies])

'''
set_src(21, PREFIX21 + sep + tail)
print('cell 21: rewired to reuse Exp 1 predictions')

# ---------------------------------------------------------------------
# 2b. Cell 23 (top-K) -- reuse cell 20's out-of-fold predictions
# ---------------------------------------------------------------------
MARK23 = '# --- Sort by predicted score (descending) ---'
s = src_of(23)
head, sep, tail = s.partition(MARK23)
assert sep, 'cell 23: sort marker not found'

PREFIX23 = '''# ============================================================
# DAY 5 — Exp 4: Top-K Enrichment
# ============================================================
# Question: If we rank all mutations by predicted escape
# probability, do the top-K actually escape more than chance?
#
# Method:
#   1. Take the out-of-fold predictions from Exp 1 (cell above)
#   2. For K = 10, 25, 50, 100, 250, 500, 1000:
#        - Take top-K mutations by predicted score
#        - Compute precision (fraction that truly escape)
#        - Compute enrichment = precision / base_rate
#   3. Plot precision@K curve
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# --- Out-of-fold predictions from Day 5 Exp 1 ---
antibodies = sorted(esm_scores)
all_scores = np.concatenate([esm_scores[ab] for ab in antibodies])
all_labels = np.concatenate([y_tests[ab] for ab in antibodies])

'''
set_src(23, PREFIX23 + sep + tail)
print('cell 23: rewired to reuse Exp 1 predictions')

# ---------------------------------------------------------------------
# 3. Cell 20 cleanup must keep esm_scores / y_tests alive
# ---------------------------------------------------------------------
s = src_of(20)
old = "              'all_scores', 'all_labels', 'base_scores', 'esm_scores', 'y_tests']:"
new = ("              'all_scores', 'all_labels', 'base_scores']:\n"
       "# NOTE: esm_scores / y_tests are deliberately kept -- cells 21 and 23\n"
       "# (calibration, top-K) reuse these out-of-fold predictions.")
assert s.count(old) == 1, 'cell 20: cleanup list not found as expected'
set_src(20, s.replace(old, new))
print('cell 20: cleanup now preserves esm_scores / y_tests')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print('speed patch applied')
