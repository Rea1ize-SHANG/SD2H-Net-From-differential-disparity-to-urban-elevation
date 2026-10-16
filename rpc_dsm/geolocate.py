import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path('outputs/rpc_dsm')
CROP_ROOT = ROOT / 'preprocess' / 'crop'
EPI_ROOT = ROOT / 'preprocess' / 'epipolar'
DSM_ROOT = ROOT / 'dsm'
OUTPUT_ROOT = ROOT / 'geolocation'
SCENES = ('038', '039', '049', '050', '057', '058', '060', '061', '062')
RUN_MAP = {}

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tools'))
from rpc_dsm import gf7_rpc_compat  # noqa: F401
from rpc_dsm.triangulate import localize_vectorized
from rpc_dsm.naming import city_name
from s2p import initialization
from s2p.config import cfg


def atomic_json(path, value):
    temporary = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True))
    os.replace(temporary, path)


def transform_points(points, matrix):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    transformed = homogeneous @ matrix.T
    return transformed[:, :2] / transformed[:, 2:]


def scene_run(scene):
    if scene in RUN_MAP:
        return Path(RUN_MAP[scene])
    if (EPI_ROOT / scene / 'input_config.json').is_file():
        return EPI_ROOT / scene
    candidates = list(EPI_ROOT.glob('*_' + scene + '_L1A_01/full_scene_*'))
    if len(candidates) != 1:
        raise RuntimeError('Expected one epipolar run for scene %s, found %d' % (scene, len(candidates)))
    return candidates[0]


def geolocate_scene(scene, stride, progress, scene_index, scene_total):
    city = city_name(scene)
    dsm_path = DSM_ROOT / city / (city + '_DSM.tif')
    if not dsm_path.exists():
        partial = dsm_path.with_name(dsm_path.name.replace('.tif', '.partial.tif'))
        if partial.exists():
            dsm_path = partial
    crop_dir = CROP_ROOT / scene
    output_dir = OUTPUT_ROOT / city
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (city + '_lonlat_stride%d.tif' % stride)
    temporary_path = output_path.with_name(output_path.name + '.partial.tif')
    crop_info = json.loads((crop_dir / 'common_crop.json').read_text())
    offset_x, offset_y = crop_info['window'][:2]
    run = scene_run(scene)
    ref_h_inv = np.linalg.inv(np.loadtxt(run / 'global_H_ref.txt'))
    initialization.build_cfg(json.loads((run / 'input_config.json').read_text()))
    ref_rpc = cfg['images'][0]['rpcm']

    with rasterio.open(dsm_path) as source:
        width, height = source.width, source.height
        grid_width = (width + stride - 1) // stride
        grid_height = (height + stride - 1) // stride
        profile = source.profile.copy()
        profile.update(
            width=grid_width,
            height=grid_height,
            count=2,
            dtype='float64',
            nodata=np.nan,
            compress='deflate',
            BIGTIFF='YES',
            transform=rasterio.Affine.identity(),
            crs=None,
        )
        total_rows = grid_height
        completed_rows = 0
        with rasterio.open(temporary_path, 'w', **profile) as target:
            target.update_tags(
                coordinate_model='RPC inverse localization',
                band_1='longitude_degrees',
                band_2='latitude_degrees',
                source_dsm=str(dsm_path),
                source_epipolar_matrix=str(run / 'global_H_ref.txt'),
                crop_offset_pixels='[%s, %s]' % (offset_x, offset_y),
                grid_stride_pixels=stride,
                version_id=crop_info.get('version_id', ''),
                note='Coordinates sampled at DSM pixel origins; interpolate for dense DEM comparison.',
            )
            for grid_y in range(grid_height):
                source_y = min(grid_y * stride, height - 1)
                source_window = Window(0, source_y, width, 1)
                row_height = source.read(1, window=source_window)[0].astype(np.float64)
                source_x = np.arange(grid_width, dtype=np.float64) * stride
                source_x = np.minimum(source_x, width - 1)
                valid = np.isfinite(row_height[source_x.astype(np.int64)])
                longitude = np.full(grid_width, np.nan, dtype=np.float64)
                latitude = np.full(grid_width, np.nan, dtype=np.float64)
                if np.any(valid):
                    x = source_x[valid]
                    y = np.full(len(x), float(source_y))
                    height_values = row_height[source_x[valid].astype(np.int64)]
                    rectified = np.column_stack((x + offset_x, y + offset_y))
                    raw_pixels = transform_points(rectified, ref_h_inv)
                    longitude_values, latitude_values = localize_vectorized(
                        ref_rpc, raw_pixels[:, 0], raw_pixels[:, 1], height_values
                    )
                    longitude[valid] = longitude_values
                    latitude[valid] = latitude_values
                target.write(longitude[None, :], 1, window=Window(0, grid_y, grid_width, 1))
                target.write(latitude[None, :], 2, window=Window(0, grid_y, grid_width, 1))
                completed_rows = grid_y + 1
                atomic_json(progress, {
                    'stage': 'GEOLOCATION',
                    'scene': scene,
                    'scene_index': scene_index,
                    'scene_total': scene_total,
                    'done_rows': completed_rows,
                    'total_rows': total_rows,
                    'grid_width': grid_width,
                    'grid_height': grid_height,
                    'stride': stride,
                    'output': str(output_path),
                })
            target.update_tags(completed='true')
    os.replace(temporary_path, output_path)
    return output_path


def main():
    global OUTPUT_ROOT, DSM_ROOT, CROP_ROOT, EPI_ROOT, RUN_MAP
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes', nargs='+', default=list(SCENES))
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--out', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--dsm-root', type=Path, default=DSM_ROOT)
    parser.add_argument('--crop-root', type=Path, default=CROP_ROOT)
    parser.add_argument('--epi-root', type=Path, default=EPI_ROOT)
    parser.add_argument('--run-map', type=Path)
    args = parser.parse_args()
    DSM_ROOT, CROP_ROOT, EPI_ROOT = args.dsm_root, args.crop_root, args.epi_root
    if args.run_map:
        RUN_MAP = json.loads(args.run_map.read_text())
        for scene in args.scenes:
            if scene not in RUN_MAP:
                raise ValueError('Missing rectification run for ' + scene)
    OUTPUT_ROOT = args.out.resolve()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    progress = OUTPUT_ROOT / 'progress.json'
    for index, scene in enumerate(args.scenes, 1):
        city = city_name(scene)
        output = OUTPUT_ROOT / city / (city + '_lonlat_stride%d.tif' % args.stride)
        if output.exists():
            with rasterio.open(output) as image:
                if image.tags().get('completed') == 'true':
                    continue
        geolocate_scene(scene, args.stride, progress, index, len(args.scenes))
    atomic_json(progress, {
        'stage': 'DONE',
        'done_scenes': len(args.scenes),
        'total_scenes': len(args.scenes),
        'stride': args.stride,
    })


if __name__ == '__main__':
    main()
