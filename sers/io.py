"""Load the frozen final inputs. Paths are relative to this repository."""
from pathlib import Path
import gzip
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load_json(name):
    return json.loads((ROOT / name).read_text())

def tasks(split):
    if split not in ('validation', 'test', 'window_size'):
        raise ValueError(split)
    with gzip.open(ROOT / 'data' / (split + '_tasks.jsonl.gz'), 'rt') as handle:
        return [json.loads(line) for line in handle]

def grid():
    return load_json('data/grid.json')

def task_seed(index, base_seed=710, namespace=0):
    return int(np.random.SeedSequence([base_seed, index, namespace]).generate_state(1, dtype=np.uint32)[0])

def training_input(task):
    """Remove the reference before any fit or loss-based selection."""
    return {**task, 'gt_f': None}

def sliding_windows(x_axis, mixtures, ordered_levels, backgrounds, size=3):
    """Construct stride-one windows from already-processed ordered spectra.

    This convenience function does not recreate upstream experimental
    preprocessing or replicate holdouts. Use the frozen inputs for exact replay.
    """
    if size < 2 or size > len(mixtures) or len(mixtures) != len(ordered_levels):
        raise ValueError('Invalid window size or mismatched levels')
    for start in range(len(mixtures) - size + 1):
        yield dict(x_axis=np.asarray(x_axis).tolist(),
                   x_mix=np.asarray(mixtures[start:start + size]).tolist(),
                   concentrations=np.asarray(ordered_levels[start:start + size]).tolist(),
                   bg=np.asarray(backgrounds).tolist(), gt_f=None,
                   dataset_id='user_data', task_id=f'window_{start:02d}',
                   metadata={'preprocess_mode':'none','task_start_index':start,
                             'task_end_index':start + size - 1,'task_window':size})
