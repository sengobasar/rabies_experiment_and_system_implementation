"""Execute selected cells of rabies_antibody_ml.ipynb in a live kernel,
write their outputs back into the notebook, and log all stdout/stderr.

Usage:  python _run_cells.py 3,4,5            # run only these cell indices
        python _run_cells.py 7,8,9,10,11,12   # etc.

Cells not listed are stubbed with `pass` in the execution copy, so their
stored outputs in the real notebook are left untouched.

The notebook is checkpointed after EVERY completed cell, so a kernel crash
(this box has ~7 GB RAM) cannot lose the outputs of cells that already ran.

NOTE: we drive the per-cell loop ourselves instead of using client.execute().
nbclient's `on_cell_complete` hook fires right after the execute *request* is
sent (client.py:987), i.e. BEFORE the results come back -- using it would
checkpoint the notebook's stale outputs. Awaiting async_execute_cell returns
only once the cell has genuinely finished.
"""
import asyncio
import json
import sys
import time

import nbformat
from nbclient import NotebookClient

NB = 'rabies_antibody_ml.ipynb'
LOG = sys.argv[2] if len(sys.argv) > 2 else '_run.log'
PROGRESS = '_progress.log'
targets = [int(x) for x in sys.argv[1].split(',')]


def _note(msg):
    with open(PROGRESS, 'a', encoding='utf-8') as fh:
        fh.write(f'{time.strftime("%H:%M:%S")}  {msg}\n')


nb = nbformat.read(NB, as_version=4)

# Work on a copy: stub out non-target code cells so they are not executed.
exec_nb = nbformat.from_dict(json.loads(json.dumps(nb)))
for i, c in enumerate(exec_nb.cells):
    if c.cell_type == 'code' and i not in targets:
        c.source = 'pass'

open(PROGRESS, 'w', encoding='utf-8').close()
print(f'Executing cells {targets} ...', flush=True)


def checkpoint(i):
    """Copy cell i's outputs into the real notebook and persist immediately."""
    src_cell = exec_nb.cells[i]
    nb.cells[i]['outputs'] = src_cell.get('outputs', [])
    nb.cells[i]['execution_count'] = src_cell.get('execution_count')
    nbformat.write(nb, NB)


async def main():
    client = NotebookClient(
        exec_nb,
        timeout=7200,
        kernel_name='python3',
        resources={'metadata': {'path': '.'}},
        allow_errors=False,
    )
    client.reset_execution_trackers()
    t_all = time.time()
    async with client.async_setup_kernel():
        for index, cell in enumerate(exec_nb.cells):
            t_cell = time.time()
            if index in targets:
                _note(f'START  cell {index}')
            await client.async_execute_cell(
                cell, index, execution_count=client.code_cells_executed + 1
            )
            if index in targets:
                dt = time.time() - t_cell
                _note(f'DONE   cell {index}  ({dt:.1f}s)')
                checkpoint(index)
    return time.time() - t_all


elapsed = asyncio.run(main())
print(f'Kernel run finished in {elapsed/60:.1f} min', flush=True)
_note(f'RUN COMPLETE ({elapsed/60:.1f} min)')

with open(LOG, 'w', encoding='utf-8') as log:
    for i in targets:
        src_cell = exec_nb.cells[i]
        log.write(f"\n{'='*78}\nCELL {i}  (execution_count={src_cell.get('execution_count')})\n{'='*78}\n")
        for o in src_cell.get('outputs', []):
            if o.get('output_type') == 'stream':
                log.write(o.get('text', ''))
            elif o.get('output_type') == 'error':
                log.write('ERROR: ' + o.get('ename', '') + ': ' + o.get('evalue', '') + '\n')
                log.write('\n'.join(o.get('traceback', [])) + '\n')
            elif o.get('output_type') == 'execute_result':
                log.write(str(o.get('data', {}).get('text/plain', '')) + '\n')
            elif o.get('output_type') == 'display_data':
                keys = ','.join(o.get('data', {}).keys())
                log.write(f'[display_data: {keys}]\n')
        log.write('\n')

nbformat.write(nb, NB)
print(f'Wrote outputs back to {NB}; log -> {LOG}', flush=True)
