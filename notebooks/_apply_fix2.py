"""One-shot patcher: fix ESM indexing (df['site'] -> df['sequential_site'])
and append the final verification cell to rabies_antibody_ml.ipynb.

Run once, then delete.
"""
import json

NB_PATH = 'rabies_antibody_ml.ipynb'

OLD_INLINE = "np.clip(df['site'].values.astype(int) - 1, 0, len(G_emb) - 1)"
NEW_INLINE = "np.clip(df['sequential_site'].values.astype(int) - 1, 0, len(G_emb) - 1)"

OLD_SITES = "sites = df['site'].values"
NEW_SITES = "sites = df['sequential_site'].values"

OLD_LABEL = 'print(f"Site range after filter: {sites.min()} → {sites.max()}")'
NEW_LABEL = 'print(f"Sequential site range after filter: {sites.min()} → {sites.max()}")'

INLINE_CELLS = [15, 16, 17, 20, 21, 23, 28]
SITES_CELLS = [11, 14]

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

changed = {}

# --- Pattern A: inline np.clip(df['site'].values...) ---
for ci in INLINE_CELLS:
    cell = nb['cells'][ci]
    assert cell['cell_type'] == 'code', f'cell {ci} not code'
    src = ''.join(cell['source'])
    n = src.count(OLD_INLINE)
    assert n == 1, f'cell {ci}: expected 1 inline match, found {n}'
    cell['source'] = (src.replace(OLD_INLINE, NEW_INLINE)).splitlines(keepends=True)
    changed[ci] = 'inline'

# --- Pattern B: sites = df['site'].values (cells 11, 14) ---
for ci in SITES_CELLS:
    cell = nb['cells'][ci]
    src = ''.join(cell['source'])
    n = src.count(OLD_SITES)
    assert n == 1, f'cell {ci}: expected 1 sites match, found {n}'
    src = src.replace(OLD_SITES, NEW_SITES)
    # keep the printed range label honest
    if ci == 11:
        assert src.count(OLD_LABEL) == 1, f'cell {ci}: label not found'
        src = src.replace(OLD_LABEL, NEW_LABEL)
    cell['source'] = src.splitlines(keepends=True)
    changed[ci] = 'sites'

# --- Sanity: no stale G_emb indexing remains anywhere ---
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    assert OLD_INLINE not in src, f'cell {i}: stale inline index remains'
    assert OLD_SITES not in src, f'cell {i}: stale sites assignment remains'
    # X_pos must NOT have been touched (per instruction: only ESM indexing)
    if "X_pos = df[['site']]" in src:
        assert "X_pos = df[['sequential_site']]" not in src

# --- Append the verification cell (code cell with the required check) ---
VERIFY_SRC = '''# Final verification: pipeline indexing matches wildtype
from Bio import SeqIO
import numpy as np
import pandas as pd

g_path = "../data/RABV_Pasteur_G_DMS/data/gene_sequence/protein.fasta"
g_seq = str(next(SeqIO.parse(g_path, "fasta")).seq)

df = pd.read_csv("../data/feature_matrix.csv")
df = df[df['region'] != 'signal_peptide'].copy()

site_idx = np.clip(df['sequential_site'].astype(int) - 1, 0, len(g_seq) - 1)
match = (np.array(list(g_seq))[site_idx] == df['wildtype'].values).mean()
print(f"Final check — pipeline indexing matches wildtype: {match:.4f}")
'''

nb['cells'].append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': VERIFY_SRC.splitlines(keepends=True),
})

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print('Patched cells:', changed)
print('Appended verification cell at index', len(nb['cells']) - 1)
print('Total cells now:', len(nb['cells']))
