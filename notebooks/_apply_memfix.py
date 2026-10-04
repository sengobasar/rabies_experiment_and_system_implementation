"""Append explicit cleanup to the memory-heavy cells.

This machine has ~7 GB RAM. Without freeing the feature matrices between cells,
the kernel gets OOM-killed partway through (observed: died loading ESM-650M
while cell 11's ~1 GB X matrix was still alive).

Only large arrays / model objects are dropped; every result variable that a
later cell consumes (day3_results, ablation_results, stats_results, ...) is
kept.

Run once, then delete.
"""
import json

NB_PATH = 'rabies_antibody_ml.ipynb'

CLEANUP = '''

# --- Free the large arrays before the next cell ---
# This box has ~7 GB RAM; without this the kernel gets OOM-killed later.
# Result variables (per_ab, *_results, stats_results, consensus, ...) are kept.
import gc
for _name in ['X', 'X_cat', 'X_pos', 'X_g', 'X_ab', 'X_base', 'X_esm', 'Xtr', 'Xte',
              'G_feat', 'ab_feat', 'G_emb', 'df', 'y', 'model', 'enc',
              'all_scores', 'all_labels', 'base_scores', 'esm_scores', 'y_tests']:
    globals().pop(_name, None)
gc.collect()
'''

# cell 13 additionally has to release the 650M model from GPU memory
CLEANUP_650M = '''

# --- Release the 650M model and its activations ---
# This box has ~7 GB RAM / 4 GB VRAM; the analysis cells that follow need
# the room. Embeddings are already saved to ../data/esm_cache/.
import gc
for _name in ['model', 'batch_converter', 'tokens', 'out', 'G_emb',
              'ab_seqs', 'antibody_embeddings', 'embed_sequence']:
    globals().pop(_name, None)
try:
    import torch
    torch.cuda.empty_cache()
except Exception:
    pass
gc.collect()
print("freed ESM-650M from memory")
'''

GENERIC_CELLS = [11, 14, 15, 16, 17, 20, 21, 23, 28]

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)


def append(i, text):
    src = ''.join(nb['cells'][i]['source'])
    assert 'Free the large arrays' not in src and 'Release the 650M model' not in src, \
        f'cell {i} already patched'
    nb['cells'][i]['source'] = (src.rstrip('\n') + '\n' + text).splitlines(keepends=True)


for ci in GENERIC_CELLS:
    append(ci, CLEANUP)

append(13, CLEANUP_650M)

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print('memory cleanup added to cells:', GENERIC_CELLS + [13])
