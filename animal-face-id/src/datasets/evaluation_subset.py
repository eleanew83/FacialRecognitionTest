"""Select a frozen evaluation cohort without changing training or class indices."""
import hashlib
import json
from pathlib import Path


def select_evaluation_subset(samples, split, benchmark, class_names, expected_sha256=None):
    raw = Path(benchmark).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Frozen evaluation benchmark changed; its SHA-256 no longer matches the config.')
    spec = json.loads(raw)
    if spec.get('schema_version') != 1 or spec['class_names'] != list(class_names):
        raise ValueError('Frozen benchmark taxonomy does not match this dataset.')
    requested = spec['splits'].get(split)
    if not requested:
        raise ValueError('Frozen benchmark has no samples for ' + split)
    available = {}
    for row in samples:
        token = (row['id'], row['sha256'], row['pixels_sha256'])
        if token in available:
            raise ValueError('Duplicate content in evaluation split.')
        available[token] = row
    selected, seen = [], set()
    for row in requested:
        token = (row['id'], row['sha256'], row['pixels_sha256'])
        if token in seen or token not in available:
            raise ValueError('Frozen benchmark sample is repeated, changed, or missing from ' + split)
        seen.add(token)
        selected.append(available[token])
    if {r['id'] for r in selected} != set(class_names):
        raise ValueError('Frozen benchmark must cover every identity.')
    return selected, {'name': spec['name'], 'path': str(Path(benchmark).resolve()), 'sha256': digest}
