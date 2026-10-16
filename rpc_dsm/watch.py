"""Watch the active DSM pipeline and exit when it completes or fails."""

import argparse
import json
import time
from pathlib import Path

from rpc_dsm.pipeline import STAGES


def bar(done, total, width=36):
    total = max(int(total), 1)
    count = min(width, max(0, round(int(done) / total * width)))
    return '[' + '█' * count + '░' * (width - count) + '] %d/%d' % (done, total)


def snapshot(root):
    progress = root / 'progress.json'
    if not progress.exists():
        return 'Waiting for the pipeline to start', False
    try:
        state = json.loads(progress.read_text())
    except (OSError, ValueError):
        return 'Writing progress state…', False
    stage = state.get('stage', 'UNKNOWN')
    if stage == 'DISPARITY':
        tiles = root / 'tiles'
        shards = []
        for path in tiles.glob('progress_disp_shard_*.json'):
            try:
                shards.append(json.loads(path.read_text()))
            except (OSError, ValueError):
                continue
        if shards:
            done = sum(item.get('done', 0) for item in shards)
            total = sum(item.get('total', 0) for item in shards)
            return 'DISPARITY ' + bar(done, total), False
    if stage == 'DSM':
        path = root / 'dsm' / 'progress.json'
        if path.exists():
            try:
                state = json.loads(path.read_text())
                return 'DSM ' + bar(state.get('done_tiles', 0), state.get('total_tiles', 1)), False
            except (OSError, ValueError):
                pass
    if stage == 'CROP':
        path = root / 'preprocess' / 'crop_status.json'
        if path.exists():
            try:
                current = json.loads(path.read_text())
                return stage + ' ' + bar(current.get('done', 0), current.get('total', 1)), False
            except (OSError, ValueError):
                pass
    if stage == 'GEOLOCATE':
        path = root / 'geolocation' / 'progress.json'
        if path.exists():
            try:
                current = json.loads(path.read_text())
                return stage + ' ' + bar(current.get('done_rows', 0), current.get('total_rows', 1)), False
            except (OSError, ValueError):
                pass
    if stage in ('DONE', 'FAILED'):
        return stage + (': ' + str(state.get('error')) if stage == 'FAILED' else ''), True
    name = (stage[:-5] if stage.endswith('_DONE') else stage).lower()
    done = STAGES.index(name) + (1 if stage.endswith('_DONE') else 0) if name in STAGES else 0
    return stage + ' ' + bar(done, len(STAGES)), False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=2.0)
    args = parser.parse_args()
    while True:
        message, finished = snapshot(args.output_root)
        print('\033[37m\r' + message + '\033[0m', end='\n' if finished else '', flush=True)
        if finished:
            if message.startswith('FAILED'):
                raise SystemExit(1)
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
