"""Resumable command-line GF-7 stereo, DSM, geolocation and label pipeline."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import rasterio
import numpy as np

from rpc_dsm.inputs import discover, prepare
from rpc_dsm.naming import city_name


PROJECT = Path(__file__).resolve().parents[1]
# Each stage writes to the selected output root; checkpoint paths are supplied by the caller.
STAGES = ('prepare', 'rectify', 'crop', 'disparity', 'dsm', 'geolocate', 'project', 'labels', 'heights')


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(temporary, path)


def validate_args(args):
    if args.from_stage not in STAGES or args.until_stage not in STAGES:
        raise ValueError('Unknown pipeline stage')
    if STAGES.index(args.from_stage) > STAGES.index(args.until_stage):
        raise ValueError('--from-stage must precede --until-stage')
    if not args.label.is_file():
        raise FileNotFoundError(args.label)
    if args.label.suffix.lower() not in ('.shp', '.tif', '.tiff'):
        raise ValueError('Label must be a shapefile or georeferenced TIFF')
    if args.label.suffix.lower() == '.shp':
        for suffix in ('.dbf', '.shx', '.prj'):
            if not args.label.with_suffix(suffix).exists():
                raise FileNotFoundError(args.label.with_suffix(suffix))
        from osgeo import ogr
        source = ogr.Open(str(args.label))
        if source is None or source.GetLayer().GetLayerDefn().GetFieldIndex(args.height_field) < 0:
            raise ValueError('Shapefile is unreadable or missing height field: ' + args.height_field)
        reference = source.GetLayer().GetSpatialRef()
        if reference is None or reference.GetAuthorityCode(None) != '4326':
            raise ValueError('Shapefile must be WGS84 (EPSG:4326); reproject before using the polygon workflow')
        source = None
    else:
        with rasterio.open(args.label) as image:
            if image.crs is None or image.count != 1:
                raise ValueError('Raster label requires one georeferenced height band')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.gpus and any(gpu < 0 for gpu in args.gpus):
        raise ValueError('GPU indexes must be nonnegative')
    scenes = discover(args.raw_root, args.scenes)
    if args.tie_root:
        missing = [scene for scene in scenes if not (args.tie_root / (str(int(scene)) + '.pts')).is_file()]
        if missing:
            raise FileNotFoundError('Missing original-image PTS for scenes: ' + ', '.join(missing))
    return scenes


def command(args, *segments, environment=None):
    print('$', ' '.join(str(item) for item in segments), flush=True)
    if args.dry_run:
        return
    env = dict(os.environ)
    env['PYTHONPATH'] = str(PROJECT) + os.pathsep + str(PROJECT / 'tools') + os.pathsep + env.get('PYTHONPATH', '')
    if environment:
        env.update(environment)
    subprocess.run([sys.executable, '-u', *map(str, segments)], cwd=str(PROJECT), env=env, check=True)


def check_shape(path, expected=None):
    with rasterio.open(path) as image:
        if image.width < 1024 or image.height < 1024:
            raise ValueError('Image smaller than the 1024-px matching tile: ' + str(path))
        if expected is not None and image.shape != expected:
            raise ValueError('Stereo pair dimensions differ: ' + str(path))
        return image.shape


def check_disparity_tiles(crop_root, tiles_root, scene):
    with rasterio.open(crop_root / scene / 'rectified_ref_common.tif') as image:
        count_x = (image.width - 1024 + 869) // 870 + 1
        count_y = (image.height - 1024 + 869) // 870 + 1
    city = city_name(scene)
    files = sorted((tiles_root / 'disp' / city).glob('*_disp.npy'))
    if len(files) != count_x * count_y:
        raise RuntimeError('%s: expected %d disparity tiles, found %d' %
                           (scene, count_x * count_y, len(files)))
    for index_x, index_y in ((0, 0), (count_x - 1, count_y - 1)):
        core = '%s_%d_%d' % (city, index_x, index_y)
        for kind, suffix in (('left', '.tif'), ('right', '.tif'), ('disp', '.npy')):
            path = tiles_root / kind / city / (core + ('_disp' if kind == 'disp' else '_' + kind) + suffix)
            if not path.is_file():
                raise FileNotFoundError(path)
        if np.load(tiles_root / 'disp' / city / (core + '_disp.npy'), mmap_mode='r').shape != (1024, 1024):
            raise ValueError('Unexpected disparity tile dimensions: ' + core)


def run_disparity(args, script, crop_root, tiles_root, scene):
    workers = []
    try:
        for shard, gpu in enumerate(args.gpus):
            segments = [sys.executable, '-u', str(script), '--root', str(crop_root), '--out', str(tiles_root),
                        '--checkpoint', str(args.checkpoint), '--device', 'cuda:0', '--input-mode', 'raw',
                        '--disp-subdir', 'disp', '--scenes', scene, '--num-shards', str(len(args.gpus)),
                        '--shard-index', str(shard)]
            log_path = args.output_root / 'logs' / ('disparity_%s_gpu%s.log' % (scene, gpu))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env['PYTHONPATH'] = str(PROJECT) + os.pathsep + env.get('PYTHONPATH', '')
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            print('$', ' '.join(segments), 'log=' + str(log_path), flush=True)
            stream = log_path.open('w')
            try:
                process = subprocess.Popen(segments, cwd=str(PROJECT), env=env,
                                           stdout=stream, stderr=subprocess.STDOUT)
            except BaseException:
                stream.close()
                raise
            workers.append((process, stream, log_path))
        for process, stream, log_path in workers:
            if process.wait() != 0:
                raise RuntimeError('Disparity worker failed; inspect ' + str(log_path))
    finally:
        for process, stream, _ in workers:
            if process.poll() is None:
                process.terminate()
                process.wait()
            stream.close()


def run(args):
    scenes = validate_args(args)
    output = args.output_root.resolve()
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    selected = STAGES[STAGES.index(args.from_stage):STAGES.index(args.until_stage) + 1]
    print('SCENES', ', '.join(scenes), 'STAGES', ', '.join(selected), flush=True)
    if args.dry_run:
        return {'scenes': list(scenes), 'stages': list(selected)}
    if 'prepare' in selected:
        runs = prepare(output, scenes, args.min_height, args.max_height)
    else:
        recorded = json.loads((output / 'scenes.json').read_text())
        if recorded != scenes:
            raise ValueError('Raw scenes changed since preparation; use a new --output-root')
        runs = json.loads((output / 'run_map.json').read_text())
    ids = list(scenes)
    provenance = {'raw_scenes': scenes, 'label': str(args.label.resolve()),
                  'checkpoint': str(args.checkpoint.resolve()),
                  'tie_root': str(args.tie_root.resolve()) if args.tie_root else None}
    provenance_path = output / 'inputs.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError('Input data or weights changed; use a new --output-root')
    atomic_json(provenance_path, provenance)
    crop_root = output / 'preprocess' / 'crop'
    tiles_root = output / 'tiles'
    dsm_root = output / 'dsm'
    geo_root = output / 'geolocation'
    progress = output / 'progress.json'
    for stage in selected:
        atomic_json(progress, {'stage': stage.upper(), 'scenes': ids, 'timestamp': time.time()})
        if stage == 'prepare':
            continue
        if stage == 'rectify':
            for scene, folder in runs.items():
                left, right = (Path(folder) / ('rectified_' + side + '_global.tif') for side in ('ref', 'sec'))
                if not (left.is_file() and right.is_file()):
                    command(args, PROJECT / 'rpc_dsm' / 'rectify.py', Path(folder) / 'input_config.json', '--skip-tiles')
                check_shape(right, check_shape(left))
                for matrix in ('global_H_ref.txt', 'global_H_sec.txt'):
                    values = np.loadtxt(Path(folder) / matrix)
                    if values.shape != (3, 3) or not np.isfinite(values).all():
                        raise ValueError('Invalid global rectification matrix: ' + str(Path(folder) / matrix))
        elif stage == 'crop':
            command(args, PROJECT / 'rpc_dsm' / 'crop.py', '--root', output / 'preprocess', '--km', '--scenes', *ids)
            for scene in ids:
                folder = crop_root / scene
                check_shape(folder / 'rectified_sec_common.tif', check_shape(folder / 'rectified_ref_common.tif'))
                if not json.loads((folder / 'common_crop.json').read_text()).get('verified'):
                    raise ValueError('Unverified common crop for scene ' + scene)
        elif stage == 'disparity':
            for scene in ids:
                run_disparity(args, PROJECT / 'rpc_dsm' / 'disparity.py', crop_root, tiles_root, scene)
                check_disparity_tiles(crop_root, tiles_root, scene)
        elif stage == 'dsm':
            pending = [scene for scene in ids if not (dsm_root / city_name(scene) / (city_name(scene) + '_DSM.tif')).is_file()]
            if pending:
                arguments = [PROJECT / 'rpc_dsm' / 'triangulate.py', '--out', dsm_root, '--scenes', *pending,
                             '--disp-root', tiles_root / 'disp', '--crop-root', crop_root,
                             '--epi-root', output / 'preprocess' / 'epipolar', '--run-map', output / 'run_map.json',
                             '--batch-size', args.rpc_batch_size, '--residual-threshold', args.rpc_residual_threshold]
                if args.tie_root:
                    arguments.extend(('--tie-root', args.tie_root))
                else:
                    arguments.append('--rpc-only')
                command(args, *arguments)
            for scene in ids:
                city = city_name(scene)
                check_shape(dsm_root / city / (city + '_DSM.tif'))
        elif stage == 'geolocate':
            command(args, PROJECT / 'rpc_dsm' / 'geolocate.py', '--dsm-root', dsm_root, '--crop-root', crop_root,
                    '--epi-root', output / 'preprocess' / 'epipolar', '--run-map', output / 'run_map.json',
                    '--out', geo_root, '--scenes', *ids, '--stride', args.geo_stride)
            for scene in ids:
                city = city_name(scene)
                path = geo_root / city / (city + '_lonlat_stride%d.tif' % args.geo_stride)
                if not path.is_file():
                    raise FileNotFoundError(path)
        elif stage == 'project':
            command(args, PROJECT / 'rpc_dsm' / 'project.py', '--dsm-root', dsm_root,
                    '--geo-root', geo_root, '--output-root', output / 'projected_dsm',
                    '--scenes', *ids, '--stride', args.geo_stride, '--resolution', args.target_resolution)
            for scene in ids:
                city = city_name(scene)
                path = output / 'projected_dsm' / city / (city + '_DSM_projected.tif')
                with rasterio.open(path) as image:
                    if image.crs is None:
                        raise ValueError('Projected DSM has no CRS: ' + str(path))
        elif stage == 'labels':
            if args.geo_stride != 16:
                raise ValueError('Label rasterization expects --geo-stride 16')
            if args.label.suffix.lower() == '.shp':
                if args.height_field.lower() != 'height':
                    raise ValueError('Vector label rasterizer currently requires the Height field')
                scene_prefix = '' if all(str(item).startswith('KM') for item in ids) else 'GF7_'
                command(args, PROJECT / 'rpc_dsm' / 'vector_labels.py', '--input-root', tiles_root,
                        '--output-root', output / 'labels', '--shp', args.label, '--scenes', *ids,
                        '--crop-root', crop_root, '--geo-root', geo_root, '--stable-mask',
                        '--scene-city-prefix', scene_prefix)
            else:
                command(args, PROJECT / 'rpc_dsm' / 'raster_labels.py', '--raster', args.label,
                        '--tiles-root', tiles_root, '--output-root', output / 'labels', '--crop-root', crop_root,
                        '--geo-root', geo_root, '--scenes', *ids)
            for scene in ids:
                city = city_name(scene)
                label_count = len(list((output / 'labels' / 'label' / city).glob('*_label.tif')))
                mask_count = len(list((output / 'labels' / 'mask' / city).glob('*_mask.tif')))
                if not label_count or label_count != mask_count:
                    raise RuntimeError('No geographically aligned height labels for ' + scene)
        elif stage == 'heights':
            command(args, PROJECT / 'rpc_dsm' / 'building_heights.py', '--dsm-root', dsm_root,
                    '--labels-root', output / 'labels', '--crop-root', crop_root,
                    '--geo-root', geo_root, '--output-root', output / 'heights', '--scenes', *ids,
                    '--buffer-m', args.ground_buffer_m)
            for scene in ids:
                city = city_name(scene)
                if not (output / 'heights' / (city + '.csv')).is_file():
                    raise RuntimeError('Missing building-height table for ' + scene)
        atomic_json(progress, {'stage': stage.upper() + '_DONE', 'scenes': ids, 'timestamp': time.time()})
    atomic_json(progress, {'stage': 'DONE', 'scenes': ids, 'timestamp': time.time()})
    return {'scenes': ids, 'stages': list(selected)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True, help='Directory containing unmodified GF-7 scene folders')
    parser.add_argument('--label', type=Path, required=True, help='Georeferenced height TIFF or building shapefile')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Pretrained SD2H disparity checkpoint')
    parser.add_argument('--output-root', type=Path, required=True, help='New, dedicated output directory')
    parser.add_argument('--scenes', nargs='*', help='Original folder names or numeric scene IDs; default all supported pairs')
    parser.add_argument('--height-field', default='Height')
    parser.add_argument('--tie-root', type=Path, help='Optional original-image ENVI PTS directory; otherwise RPC-only')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--min-height', type=int, default=-500)
    parser.add_argument('--max-height', type=int, default=5000)
    parser.add_argument('--geo-stride', type=int, default=16, choices=[16])
    parser.add_argument('--rpc-batch-size', type=int, default=16384)
    parser.add_argument('--rpc-residual-threshold', type=float, default=3.0)
    parser.add_argument('--ground-buffer-m', type=float, default=20.0)
    parser.add_argument('--target-resolution', type=float, default=1.0, help='Projected DSM resolution in meters')
    parser.add_argument('--from-stage', choices=STAGES, default='prepare')
    parser.add_argument('--until-stage', choices=STAGES, default='heights')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and show stages without writing data')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        if not args.dry_run:
            atomic_json(args.output_root / 'progress.json', {'stage': 'FAILED', 'error': str(error),
                         'timestamp': time.time()})
        raise


if __name__ == '__main__':
    main()
