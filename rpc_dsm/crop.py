import argparse
import contextlib
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
import rasterio
from rasterio.windows import Window


SCENES = ('038', '039', '049', '050', '057', '058', '060', '061', '062')


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def windows(width, height):
    for row in range(0, height, 1024):
        for col in range(0, width, 1024):
            yield Window(col, row, min(1024, width-col), min(1024, height-row))


def read_pair(left, right, window):
    reference = left.read(1, window=window, masked=True)
    secondary = right.read(1, window=window, masked=True)
    valid = (~np.ma.getmaskarray(reference) & ~np.ma.getmaskarray(secondary) &
             np.isfinite(reference.data) & np.isfinite(secondary.data))
    return reference.data, secondary.data, valid


def progress(root, scene, stage, done, total, **extra):
    state = dict(pid=os.getpid(), scene=scene, stage=stage, done=done, total=total,
                 updated=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    save(root / 'crop_status.json', state)
    print('%s %s %s %d/%d %s' % (state['updated'], scene, stage, done, total, extra), flush=True)


def process_scene(root, scene, source_folder=None):
    folder = root / 'crop' / scene
    folder.mkdir(parents=True, exist_ok=True)
    if source_folder is not None:
        source = root / 'epipolar' / source_folder
        if not source.is_dir():
            raise RuntimeError('Missing source for ' + scene + ': ' + str(source))
    else:
        suffix = 'full_scene_test_20260910_058' if scene == '058' else 'full_scene_unified_20260910_' + scene
        candidates = list((root / 'epipolar').glob('*_' + scene + '_L1A_01/' + suffix))
        if len(candidates) != 1:
            raise RuntimeError('Expected one source for ' + scene)
        source = candidates[0]
    paths = [source / ('rectified_' + side + '_global.tif') for side in ('ref', 'sec')]
    if source_folder is not None and (source / 'rectified_sec_global_corrected.tif').exists():
        paths[1] = source / 'rectified_sec_global_corrected.tif'
    fingerprint = [[str(path), path.stat().st_size, path.stat().st_mtime_ns] for path in paths]
    names = ['rectified_ref_common.tif', 'rectified_sec_common.tif', 'common_valid_mask.tif']
    report_path = folder / 'common_crop.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get('verified') and report.get('fingerprint') == fingerprint and all((folder / name).exists() for name in names):
            progress(root, scene, 'SKIPPED_VERIFIED', 1, 1)
            return report
    bbox_path = folder / 'common_bbox.json'
    with rasterio.open(paths[0]) as left, rasterio.open(paths[1]) as right:
        if left.shape != right.shape:
            raise RuntimeError('Source shape mismatch')
        bbox = json.loads(bbox_path.read_text()) if bbox_path.exists() else {}
        if bbox.get('fingerprint') != fingerprint:
            xmin, ymin, xmax, ymax, count = left.width, left.height, 0, 0, 0
            total = math.ceil(left.width/1024) * math.ceil(left.height/1024)
            progress(root, scene, 'SCAN', 0, total)
            for index, window in enumerate(windows(left.width, left.height), 1):
                _, _, valid = read_pair(left, right, window)
                if valid.any():
                    rows = np.flatnonzero(valid.any(axis=1))
                    cols = np.flatnonzero(valid.any(axis=0))
                    xmin = min(xmin, int(window.col_off)+int(cols[0]))
                    ymin = min(ymin, int(window.row_off)+int(rows[0]))
                    xmax = max(xmax, int(window.col_off)+int(cols[-1])+1)
                    ymax = max(ymax, int(window.row_off)+int(rows[-1])+1)
                    count += int(valid.sum())
                if index % 25 == 0 or index == total:
                    progress(root, scene, 'SCAN', index, total)
            if not count:
                raise RuntimeError('No common valid pixels')
            bbox = dict(scene=scene, source=str(source), fingerprint=fingerprint,
                        window=[xmin, ymin, xmax-xmin, ymax-ymin], common_valid_pixels=count,
                        source_shape=[left.width, left.height],
                        validity='finite pixels valid in both raster masks; zero is not invalid unless nodata')
            save(bbox_path, bbox)
        xmin, ymin, width, height = bbox['window']
        print('BBOX %s x=%d y=%d width=%d height=%d' % (scene, xmin, ymin, width, height), flush=True)
        total = math.ceil(width/1024) * math.ceil(height/1024)
        temporary_paths = [folder / (name + '.partial.tif') for name in names]
        profile = dict(driver='GTiff', width=width, height=height, count=1, dtype='float32',
                       tiled=True, blockxsize=256, blockysize=256, compress='deflate',
                       BIGTIFF='YES', nodata=float('nan'),
                       transform=left.window_transform(Window(xmin, ymin, width, height)))
        progress(root, scene, 'WRITE', 0, total, window=bbox['window'])
        with contextlib.ExitStack() as stack:
            outputs = [stack.enter_context(rasterio.open(path, 'w', **profile)) for path in temporary_paths[:2]]
            mask = stack.enter_context(rasterio.open(temporary_paths[2], 'w', **{**profile, 'dtype':'uint8', 'nodata':0}))
            for index, window in enumerate(windows(width, height), 1):
                original = Window(window.col_off+xmin, window.row_off+ymin, window.width, window.height)
                reference, secondary, valid = read_pair(left, right, original)
                for output, values in zip(outputs, (reference, secondary)):
                    output.write(np.where(valid, values, np.nan).astype('float32'), 1, window=window)
                mask.write(valid.astype('uint8'), 1, window=window)
                if index % 25 == 0 or index == total:
                    progress(root, scene, 'WRITE', index, total, window=bbox['window'])
        progress(root, scene, 'VERIFY', 0, total)
        checked_pixels = 0
        with contextlib.ExitStack() as stack:
            outputs = [stack.enter_context(rasterio.open(path)) for path in temporary_paths]
            for index, window in enumerate(windows(width, height), 1):
                original = Window(window.col_off+xmin, window.row_off+ymin, window.width, window.height)
                reference, secondary, valid = read_pair(left, right, original)
                for output, expected in zip(outputs[:2], (reference, secondary)):
                    actual = output.read(1, window=window)
                    if not np.array_equal(np.isfinite(actual), valid) or not np.array_equal(actual[valid], expected[valid]):
                        raise RuntimeError('Pixel verification failed at ' + str(window))
                if not np.array_equal(outputs[2].read(1, window=window), valid.astype('uint8')):
                    raise RuntimeError('Mask mismatch')
                checked_pixels += int(valid.sum())
                if index % 25 == 0 or index == total:
                    progress(root, scene, 'VERIFY', index, total)
        if checked_pixels != bbox['common_valid_pixels']:
            raise RuntimeError('Common area count mismatch')
    translation = np.array([[1., 0., -xmin], [0., 1., -ymin], [0., 0., 1.]])
    for side in ('ref', 'sec'):
        matrix = source / ('global_H_' + side + '.txt')
        if side == 'sec' and paths[1].name == 'rectified_sec_global_corrected.tif':
            matrix = source / 'global_H_sec_corrected.txt'
        np.savetxt(folder / ('global_H_' + side + '_crop.txt'), translation @ np.loadtxt(matrix))
    for temporary, name in zip(temporary_paths, names):
        os.replace(temporary, folder / name)
    correction_report = source / 'vertical_pts_correction.json'
    correction_applied = False
    if correction_report.exists():
        try:
            correction_applied = bool(json.loads(correction_report.read_text()).get('applied'))
        except (OSError, json.JSONDecodeError):
            correction_applied = False
    report = {**bbox, 'verified': True, 'mask_valid_fraction':checked_pixels/(width*height),
              'crop_coordinates':'subtract [xmin,ymin] from both original rectified coordinates',
              'geometric_correction_applied':correction_applied}
    save(report_path, report)
    progress(root, scene, 'DONE', 1, 1, window=bbox['window'])
    return report


def watch(root):
    labels = {'SCAN':'扫描交集', 'WRITE':'分块写出', 'VERIFY':'逐像素验证', 'DONE':'单景完成', 'ALL_DONE':'全部完成', 'FAILED':'异常退出', 'STOPPED':'已停止'}
    while True:
        path = root / 'crop_status.json'
        if not path.exists():
            print('等待启动', flush=True)
        else:
            state = json.loads(path.read_text())
            done, total = state['done'], max(1, state['total'])
            filled = min(30, int(30*done/total))
            line = '%s %s [%s%s] %.1f%% %d/%d' % (state['scene'], labels.get(state['stage'],state['stage']), '█'*filled, '░'*(30-filled), 100*done/total, done, total)
            terminal = state['stage'] in ('ALL_DONE', 'FAILED', 'STOPPED')
            if not terminal:
                try:
                    args = Path('/proc/%d/cmdline' % state['pid']).read_bytes()
                    terminal = b'crop_gf7_common.py' not in args
                except OSError:
                    terminal = True
                if terminal:
                    line += ' 进程已退出，尚未确认完成'
            if sys.stdout.isatty():
                print('\r\033[37m' + line + '\033[0m\033[K', end='', flush=True)
            else:
                print(line, flush=True)
            if terminal:
                print('\n' + state.get('error', ''))
                return
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs/rpc_dsm/preprocess'))
    parser.add_argument('--km', action='store_true', help='Process Kunming scenes using their full scene names')
    parser.add_argument('--scenes', nargs='+', help='Only process the named scenes')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.watch:
        return watch(args.root)
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'crop.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        try:
            with rasterio.Env(GDAL_CACHEMAX=128*1024*1024):
                reports = []
                scenes = sorted(path.name for path in (args.root / 'epipolar').glob('KM_GF7_*')) if args.km else SCENES
                if args.scenes:
                    scenes = args.scenes
                for scene in scenes:
                    reports.append(process_scene(args.root, scene, scene if args.km else None))
                    save(args.root / 'crop_summary.json', reports)
            progress(args.root, '%d/%d' % (len(scenes), len(scenes)), 'ALL_DONE', len(scenes), len(scenes))
        except BaseException as error:
            progress(args.root, 'batch', 'STOPPED' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'FAILED', 0, 1, error=str(error))
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
