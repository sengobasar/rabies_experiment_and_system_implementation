"""Rewire the notebook so result JSONs and figures are computed from the run
instead of hardcoded literals transcribed from an earlier (buggy) session.

Analysis cells now bundle their results into variables; the JSON/figure cells
consume those variables.

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


def sub(i, old, new, count=1):
    """Replace inside cell i, asserting the match count."""
    s = src_of(i)
    n = s.count(old)
    assert n == count, f'cell {i}: expected {count} match(es), found {n} for:\n{old[:160]}'
    set_src(i, s.replace(old, new))


# =====================================================================
# CELL 4 (Day 2) — bundle baseline_results
# =====================================================================
sub(4,
    'print(f\'{"XGBoost (mean)":<25} {np.mean([a for _, a, _ in aucs]):<10.4f} '
    '{np.mean([a for _, _, a in aucs]):<10.4f}\')',
    'print(f\'{"XGBoost (mean)":<25} {np.mean([a for _, a, _ in aucs]):<10.4f} '
    '{np.mean([a for _, _, a in aucs]):<10.4f}\')\n'
    '\n'
    '# ------------------------------------------------------------\n'
    '# STEP 7: BUNDLE RESULTS FOR EXPORT\n'
    '# ------------------------------------------------------------\n'
    '# The next cell writes this to ../data/baseline_results.json.\n'
    '# Later days read these numbers instead of hardcoding them.\n'
    'baseline_results = {\n'
    "    'random':  {'auroc': float(results[0][1]), 'auprc': float(results[0][2])},\n"
    "    'seqsim':  {'auroc': float(results[1][1]), 'auprc': float(results[1][2])},\n"
    "    'xgboost': {'auroc': float(np.mean([a for _, a, _ in aucs])),\n"
    "                'auprc': float(np.mean([a for _, _, a in aucs]))},\n"
    '}\n'
    "print('\\nbaseline_results:', baseline_results)")

# =====================================================================
# CELL 5 — dump the live dict
# =====================================================================
set_src(5, '''import json

# Values come from the Day 2 run above (no literals).
with open('../data/baseline_results.json', 'w') as f:
    json.dump(baseline_results, f, indent=2)
print('\\u2705 Baseline results saved:', baseline_results)
''')

# =====================================================================
# CELL 11 (Day 3) — per-antibody capture + live comparisons + bundle
# =====================================================================
sub(11, 'aucs, auprcs = [], []', 'aucs, auprcs, per_ab = [], [], {}')
sub(11,
    '    aucs.append(auroc); auprcs.append(auprc)\n'
    '    print(f"  {ab:8s}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")',
    '    aucs.append(auroc); auprcs.append(auprc)\n'
    "    per_ab[ab] = {'auroc': float(auroc), 'auprc': float(auprc)}\n"
    '    print(f"  {ab:8s}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")')
sub(11,
    'print(f"  Day 2 XGBoost:     AUROC=0.7031   AUPRC=0.1373")\n'
    'print(f"  Day 3 XGBoost+ESM: AUROC={np.mean(aucs):.4f}   AUPRC={np.mean(auprcs):.4f}")\n'
    'print(f"  Delta AUROC: {np.mean(aucs) - 0.7031:+.4f}")\n'
    'print(f"  Delta AUPRC: {np.mean(auprcs) - 0.1373:+.4f}")',
    'print(f"  Day 2 XGBoost:     AUROC={baseline_results[\'xgboost\'][\'auroc\']:.4f}   '
    'AUPRC={baseline_results[\'xgboost\'][\'auprc\']:.4f}")\n'
    'print(f"  Day 3 XGBoost+ESM: AUROC={np.mean(aucs):.4f}   AUPRC={np.mean(auprcs):.4f}")\n'
    'print(f"  Delta AUROC: {np.mean(aucs) - baseline_results[\'xgboost\'][\'auroc\']:+.4f}")\n'
    'print(f"  Delta AUPRC: {np.mean(auprcs) - baseline_results[\'xgboost\'][\'auprc\']:+.4f}")\n'
    '\n'
    '# Bundle for ../data/day3_results.json (next cell)\n'
    'day3_results = {\n'
    "    'xgboost_esm': {'auroc': float(np.mean(aucs)), 'auprc': float(np.mean(auprcs))},\n"
    "    'baseline_xgboost': dict(baseline_results['xgboost']),\n"
    "    'delta': {'auroc': float(np.mean(aucs) - baseline_results['xgboost']['auroc']),\n"
    "              'auprc': float(np.mean(auprcs) - baseline_results['xgboost']['auprc'])},\n"
    "    'per_antibody': per_ab,\n"
    '}')

# =====================================================================
# CELL 12 — dump the live dict
# =====================================================================
set_src(12, '''import json

# Values come from the Day 3 run above (no literals).
with open('../data/day3_results.json', 'w') as f:
    json.dump(day3_results, f, indent=2)
print("\\u2705 Day 3 results saved.", day3_results['xgboost_esm'])
''')

# =====================================================================
# CELL 14 (Day 4 EXP 1b) — per-antibody capture + live comparisons + bundle
# =====================================================================
sub(14, 'aucs, auprcs = [], []', 'aucs, auprcs, per_ab = [], [], {}')
sub(14,
    '    aucs.append(auroc); auprcs.append(auprc)\n'
    '    print(f"  {ab:8s}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")',
    '    aucs.append(auroc); auprcs.append(auprc)\n'
    "    per_ab[ab] = {'auroc': float(auroc), 'auprc': float(auprc)}\n"
    '    print(f"  {ab:8s}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")')
sub(14,
    'print(f"  Day 2 XGBoost:       AUROC=0.7031   AUPRC=0.1373")\n'
    'print(f"  Day 3 ESM-35M:       AUROC=0.7625   AUPRC=0.1591")\n'
    'print(f"  Day 4 ESM-650M:      AUROC={np.mean(aucs):.4f}   AUPRC={np.mean(auprcs):.4f}")\n'
    'print(f"  Delta vs Day 3:      {np.mean(aucs) - 0.7625:+.4f}")',
    'print(f"  Day 2 XGBoost:       AUROC={baseline_results[\'xgboost\'][\'auroc\']:.4f}   '
    'AUPRC={baseline_results[\'xgboost\'][\'auprc\']:.4f}")\n'
    'print(f"  Day 3 ESM-35M:       AUROC={day3_results[\'xgboost_esm\'][\'auroc\']:.4f}   '
    'AUPRC={day3_results[\'xgboost_esm\'][\'auprc\']:.4f}")\n'
    'print(f"  Day 4 ESM-650M:      AUROC={np.mean(aucs):.4f}   AUPRC={np.mean(auprcs):.4f}")\n'
    'print(f"  Delta vs Day 3:      {np.mean(aucs) - day3_results[\'xgboost_esm\'][\'auroc\']:+.4f}")\n'
    '\n'
    '# Bundle for the family-holdout comparison below\n'
    'day4_results = {\n'
    "    'xgboost_esm_650m': {'auroc': float(np.mean(aucs)), 'auprc': float(np.mean(auprcs))},\n"
    "    'per_antibody': per_ab,\n"
    '}')

# =====================================================================
# CELL 15 (ablation) — return metrics, bundle ablation_results
# =====================================================================
sub(15,
    '    print(f"  {name:40s}  AUROC={np.mean(aucs):.4f}  AUPRC={np.mean(auprcs):.4f}")\n'
    '    del X; gc.collect()',
    '    result = {\'auroc\': float(np.mean(aucs)), \'auprc\': float(np.mean(auprcs))}\n'
    '    print(f"  {name:40s}  AUROC={result[\'auroc\']:.4f}  AUPRC={result[\'auprc\']:.4f}")\n'
    '    del X; gc.collect()\n'
    '    return result')
sub(15,
    'run_cv([X_cat, X_pos],                  "A) One-hot + position")\n'
    'run_cv([X_cat, X_pos, X_g],             "B) A + G-ESM")\n'
    'run_cv([X_cat, X_pos, X_ab],            "C) A + Antibody-ESM")\n'
    'run_cv([X_cat, X_pos, X_g, X_ab],       "D) A + G-ESM + Antibody-ESM (full)")',
    'ablation_results = {}\n'
    'ablation_results[\'A_onehot_pos\']  = run_cv([X_cat, X_pos],            "A) One-hot + position")\n'
    'ablation_results[\'B_plus_G_esm\']  = run_cv([X_cat, X_pos, X_g],       "B) A + G-ESM")\n'
    'ablation_results[\'C_plus_ab_esm\'] = run_cv([X_cat, X_pos, X_ab],      "C) A + Antibody-ESM")\n'
    'ablation_results[\'D_plus_both\']   = run_cv([X_cat, X_pos, X_g, X_ab], '
    '"D) A + G-ESM + Antibody-ESM (full)")')

# =====================================================================
# CELL 17 (family holdout) — return metrics, bundle generalization_results
# =====================================================================
sub(17,
    '        print(f"  {name}: skipped (single class)")\n'
    '        return',
    '        print(f"  {name}: skipped (single class)")\n'
    '        return None')
sub(17,
    '    print(f"  {name:35s}  AUROC={roc_auc_score(yte, score):.4f}  '
    'AUPRC={average_precision_score(yte, score):.4f}")\n'
    '    del Xtr, Xte; gc.collect()',
    '    auroc = roc_auc_score(yte, score)\n'
    '    auprc = average_precision_score(yte, score)\n'
    '    print(f"  {name:35s}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")\n'
    '    del Xtr, Xte; gc.collect()\n'
    "    return {'auroc': float(auroc), 'auprc': float(auprc)}")
sub(17,
    'print("Reference: Day 4 single-antibody holdout  AUROC=0.7674")\n'
    'print()\n'
    'for fam_name, abs_in_fam in families.items():\n'
    '    family_holdout(abs_in_fam, fam_name)',
    'print(f"Reference: Day 4 single-antibody holdout  '
    'AUROC={day4_results[\'xgboost_esm_650m\'][\'auroc\']:.4f}")\n'
    'print()\n'
    'family_keys = {\n'
    "    'Crucell (CR57 + CR4098)':    'crucell_family',\n"
    "    'RVC-series (RVC20+58+68)':   'rvc_series_family',\n"
    "    'Murine (17C7 alone)':        'murine_17C7',\n"
    "    'Distinct (CTB012 + RVA122)': 'distinct_family',\n"
    '}\n'
    'generalization_results = {\n'
    "    'single_antibody': {'auroc': float(day4_results['xgboost_esm_650m']['auroc'])},\n"
    '}\n'
    'for fam_name, abs_in_fam in families.items():\n'
    '    res = family_holdout(abs_in_fam, fam_name)\n'
    '    if res is not None:\n'
    '        generalization_results[family_keys[fam_name]] = res')

# =====================================================================
# CELL 18 — dump the live dict
# =====================================================================
set_src(18, '''import json

# Values come from the family-holdout run above (no literals).
with open('../data/generalization_results.json', 'w') as f:
    json.dump(generalization_results, f, indent=2)
print("\\u2705 Generalization results saved.", generalization_results)
''')

# =====================================================================
# CELL 20 (Day 5 Exp 1) — bundle stats_results
# =====================================================================
sub(20,
    'print(f"  significant (p<0.05): {p_val < 0.05}")',
    'print(f"  significant (p<0.05): {p_val < 0.05}")\n'
    '\n'
    '# Bundle for the Day 5 figures and the proxy comparison (later cells)\n'
    'stats_results = {\n'
    "    'mean_baseline_auroc': float(np.mean(base_aucs)),\n"
    "    'mean_esm_auroc':      float(np.mean(esm_aucs)),\n"
    "    'mean_delta':          float(np.mean(esm_aucs) - np.mean(base_aucs)),\n"
    "    't_statistic':         float(t_stat),\n"
    "    'p_value':             float(p_val),\n"
    "    'significant':         bool(p_val < 0.05),\n"
    "    'per_antibody_base':   {ab: float(v) for ab, v in zip(antibodies, base_aucs)},\n"
    "    'per_antibody_esm':    {ab: float(v) for ab, v in zip(antibodies, esm_aucs)},\n"
    '}')

# =====================================================================
# CELL 21 (calibration) — bundle calibration_results
# =====================================================================
sub(21,
    'print(f"Overall positive rate: {all_labels.mean():.4f}")',
    'print(f"Overall positive rate: {all_labels.mean():.4f}")\n'
    '\n'
    '# Bundle for ../data/calibration_results.json (next cell)\n'
    'calibration_results = {\n'
    "    'ece': float(ece),\n"
    "    'total_predictions': int(len(all_scores)),\n"
    "    'overall_positive_rate': float(all_labels.mean()),\n"
    "    'bins': [\n"
    "        {'bin': i + 1, 'predicted': float(p), 'observed': float(o), 'count': int(c)}\n"
    '        for i, (p, o, c) in enumerate(zip(bin_centers, obs_freq, counts))\n'
    '    ],\n'
    '}')

# =====================================================================
# CELL 22 — dump the live dict
# =====================================================================
set_src(22, '''import json

# Values come from the calibration run above (no literals).
with open('../data/calibration_results.json', 'w') as f:
    json.dump(calibration_results, f, indent=2)
print("\\u2705 Calibration results saved.", {'ece': calibration_results['ece']})
''')

# =====================================================================
# CELL 24 — re-dump the live topk dict (was clobbering cell 23 with literals)
# =====================================================================
set_src(24, '''import json

# `topk` is the live dict computed in the cell above. Re-dump it so the file
# always matches this run. Do NOT reintroduce literals here.
with open('../data/topk_results.json', 'w') as f:
    json.dump(topk, f, indent=2)
print("\\u2705 Top-K results saved.", {'base_rate': topk['base_rate']})
''')

# =====================================================================
# CELL 25 — per-antibody figure from stats_results
# =====================================================================
set_src(25, '''# ============================================================
# DAY 5 - Exp 5: Per-Antibody Bar Chart
# ============================================================
# Visual: AUROC per antibody, baseline vs ESM, with delta labels.
# Values come from the bootstrap in Exp 1 (cell above) - no literals.
# ============================================================

import json
import numpy as np
import matplotlib.pyplot as plt

antibodies = sorted(stats_results['per_antibody_base'].keys())
base = [stats_results['per_antibody_base'][ab] for ab in antibodies]
esm  = [stats_results['per_antibody_esm'][ab]  for ab in antibodies]

with open('../data/stats_results.json', 'w') as f:
    json.dump(stats_results, f, indent=2)
print("\\u2705 stats_results.json saved.")

# --- Plot ---
x = np.arange(len(antibodies))
width = 0.38

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width/2, base, width, label='Baseline (one-hot + position)', color='gray', edgecolor='black')
ax.bar(x + width/2, esm,  width, label='ESM-650M',                      color='steelblue', edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(antibodies)
ax.set_ylabel('AUROC')
ax.set_ylim(max(0.0, min(base + esm) - 0.05), max(base + esm) + 0.06)
ax.set_title('Per-Antibody AUROC: Baseline vs ESM-650M')
ax.axhline(0.5, color='red', linestyle=':', alpha=0.5, label='Random (0.50)')
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)

# Annotate delta above each pair
for i, (b, e) in enumerate(zip(base, esm)):
    ax.text(i, max(b, e) + 0.01, f"{e-b:+.3f}", ha='center', fontsize=9,
            color='darkgreen' if e >= b else 'firebrick')

plt.tight_layout()
plt.savefig('../data/fig_per_antibody.png', dpi=200)
plt.show()
print("\\u2705 Saved: ../data/fig_per_antibody.png")
''')

# =====================================================================
# CELL 26 — ablation figure from ablation_results
# =====================================================================
set_src(26, '''# ============================================================
# DAY 5 - Exp 6: Ablation Bar Chart
# ============================================================
# Visual: 4 feature-set configurations from Day 4 Exp 2 (cell above).
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import json

keys = ['A_onehot_pos', 'B_plus_G_esm', 'C_plus_ab_esm', 'D_plus_both']
configs = [
    'A) One-hot\\n+ position',
    'B) + G-ESM',
    'C) + Antibody-ESM',
    'D) + Both\\n(full)',
]
auroc = [ablation_results[k]['auroc'] for k in keys]
auprc = [ablation_results[k]['auprc'] for k in keys]

with open('../data/ablation_results.json', 'w') as f:
    json.dump(ablation_results, f, indent=2)
print("\\u2705 ablation_results.json saved.")

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['gray', 'steelblue', 'darkorange', 'seagreen']
bars = ax.bar(configs, auroc, color=colors, edgecolor='black')

for bar, val in zip(bars, auroc):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
            f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('AUROC')
ax.set_ylim(min(auroc) - 0.04, max(auroc) + 0.045)
ax.set_title('Ablation Study: Which Features Drive AUROC?')
ax.axhline(0.50, color='red', linestyle=':', alpha=0.5, label='Random (0.50)')
ax.grid(axis='y', alpha=0.3)
ax.legend(loc='lower right')

# Highlight the G-ESM gain over the plain baseline
gain = auroc[1] - auroc[0]
ax.annotate(f'G-ESM gain over A\\n({gain:+.3f})',
            xy=(1, auroc[1]), xytext=(1.5, (auroc[1] + auroc[0]) / 2),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=10, ha='left')

plt.tight_layout()
plt.savefig('../data/fig_ablation.png', dpi=200)
plt.show()
print("\\u2705 Saved: ../data/fig_ablation.png")
''')

# =====================================================================
# CELL 27 — interpretability figure from cell 16's consensus
# =====================================================================
set_src(27, '''# ============================================================
# DAY 5 - Exp 7: Interpretability Plot
# ============================================================
# Visual: consensus escape-prone positions on the G protein,
# with known antigenic sites highlighted.
# `consensus` comes from Day 4 Exp 3 (grouped by reference site).
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import json

consensus_sites = sorted(int(s) for s in consensus)

# Known antigenic regions (rabies literature; same reference numbering as 'site')
antigenic_regions = {
    'Site I (lateral loop)':  (223, 231),
    'Site II':                (34, 42),
    'Site III':               (330, 338),
    'Site IV':                (251, 255),
    'Fusion loop':            (143, 145),
}

# Which consensus sites fall inside a known antigenic region?
known_matches = {
    name: [s for s in consensus_sites if lo <= s <= hi]
    for name, (lo, hi) in antigenic_regions.items()
}
known_matches = {k: v for k, v in known_matches.items() if v}
matched = {s for v in known_matches.values() for s in v}
novel_candidates = [s for s in consensus_sites if s not in matched]

with open('../data/interpretability_results.json', 'w') as f:
    json.dump({
        'consensus_sites': consensus_sites,
        'known_matches': known_matches,
        'novel_candidates': novel_candidates,
    }, f, indent=2)
print("\\u2705 interpretability_results.json saved.")

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 3))

# G protein backbone
ax.plot([1, 433], [0, 0], color='black', linewidth=6, solid_capstyle='round', zorder=1)

region_colors = {
    'Site I (lateral loop)':  'gold',
    'Site II':                'lightgreen',
    'Site III':               'salmon',
    'Site IV':                'plum',
    'Fusion loop':            'skyblue',
}
for name, (start, end) in antigenic_regions.items():
    ax.axvspan(start, end, color=region_colors[name], alpha=0.5, zorder=0, label=name)

for s in consensus_sites:
    ax.scatter(s, 0, s=120, color='red', edgecolor='black', zorder=3)
    ax.annotate(str(s), xy=(s, 0), xytext=(s, 0.15),
                ha='center', fontsize=8, color='darkred')

ax.set_xlim(0, 440)
ax.set_ylim(-0.3, 0.4)
ax.set_xlabel('RABV-G residue (DMS reference numbering)')
ax.set_yticks([])
ax.set_title('Consensus Escape-Prone Positions on RABV-G\\n(red = top sites across \\u22653 antibodies)')
ax.legend(loc='upper right', fontsize=8, ncol=3)
plt.tight_layout()
plt.savefig('../data/fig_interpretability.png', dpi=200)
plt.show()
print("\\u2705 Saved: ../data/fig_interpretability.png")
''')

# =====================================================================
# CELL 28 — use live per-antibody AUROC; bundle comparison_result
# =====================================================================
sub(28,
    "# --- Your model's per-antibody AUROC (from Day 5 stats) ---\n"
    'your_auroc = {\n'
    "    '17C7': 0.7471, 'CR4098': 0.7904, 'CR57': 0.7418, 'CTB012': 0.7110,\n"
    "    'RVA122': 0.8201, 'RVC20': 0.7915, 'RVC58': 0.7985, 'RVC68': 0.7413,\n"
    '}',
    "# --- Your model's per-antibody AUROC (bootstrap means from Day 5 Exp 1) ---\n"
    "your_auroc = dict(stats_results['per_antibody_esm'])")
sub(28,
    "result_df.loc['Mean'] = ['Mean', result_df[baseline_name].mean(),\n"
    "                         result_df['Your Model'].mean(),\n"
    "                         (result_df['Your Model'] - result_df[baseline_name]).mean()]",
    '# Capture means BEFORE adding the Mean row (otherwise it averages itself in)\n'
    'mean_proxy = float(result_df[baseline_name].mean())\n'
    "mean_yours = float(result_df['Your Model'].mean())\n"
    "delta_series = result_df['Your Model'] - result_df[baseline_name]\n"
    'mean_delta = float(delta_series.mean())\n'
    'delta_min, delta_max = float(delta_series.min()), float(delta_series.max())\n'
    '\n'
    "result_df.loc['Mean'] = ['Mean', mean_proxy, mean_yours, mean_delta]")
sub(28,
    'print("\\n\u2705 Saved: ../data/comparison_antigencentric_vs_yours.csv")',
    'print("\\n\u2705 Saved: ../data/comparison_antigencentric_vs_yours.csv")\n'
    '\n'
    '# Bundle for ../data/proxy_comparison.json (next cell)\n'
    'comparison_result = {\n'
    "    'baseline': baseline_name,\n"
    "    'proxy_mean_auroc': mean_proxy,\n"
    "    'your_model_mean_auroc': mean_yours,\n"
    "    'mean_delta': mean_delta,\n"
    "    'per_antibody_delta_range': [delta_min, delta_max],\n"
    "    'conclusion': ('Antibody embeddings add no predictive value; model is effectively "
    "antigen-centric'\n"
    '                   if abs(mean_delta) < 0.01 else\n'
    "                   'Antibody embeddings change predictions beyond the antigen-centric "
    "proxy'),\n"
    '}\n'
    'print("comparison_result:", comparison_result)')

# =====================================================================
# CELL 29 — dump the live dict
# =====================================================================
set_src(29, '''import json

# Values come from the comparison run above (no literals).
with open('../data/proxy_comparison.json', 'w') as f:
    json.dump(comparison_result, f, indent=2)
print("\\u2705 Saved.", comparison_result)
''')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print('Rewire applied. Cells now:', len(nb['cells']))
