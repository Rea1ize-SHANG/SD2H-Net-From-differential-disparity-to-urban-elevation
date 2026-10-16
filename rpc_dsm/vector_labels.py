"""Build an SD2H evaluation view for the corrected UnifiedBH_2 images."""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import rasterio
from osgeo import ogr
from rasterio.features import rasterize
from rasterio.transform import Affine

from rpc_dsm.dataset_files import copy_dataset_file, require_real_parents
from rpc_dsm.naming import city_name


SCENES = ('038', '039', '049', '050', '057', '058', '060', '061', '062')
SCENE_CITY = {
    '038': 'SZ_DG_038', '039': 'PRE_039', '049': 'HZ_049',
    '050': 'HZ_050', '057': 'HZ_057', '058': 'HZ_DG_058',
    '060': 'SZ_060', '061': 'SZ_061', '062': 'HK_062',
}
TILE_SIZE = 1024
TILE_STEP = 870
TILE_RE = re.compile(r'^(?:GF7_\d+|KM\d+)_(\d+)_(\d+)_left\.tif$', re.IGNORECASE)


def atomic_json(path, value):
    temporary = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True))
    os.replace(temporary, path)


def sample_grid(grid, x, y):
    x = float(np.clip(x, 0, grid.shape[1] - 1))
    y = float(np.clip(y, 0, grid.shape[0] - 1))
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, grid.shape[1] - 1), min(y0 + 1, grid.shape[0] - 1)
    wx, wy = x - x0, y - y0
    values = np.asarray([
        grid[y0, x0], grid[y0, x1], grid[y1, x0], grid[y1, x1]
    ], dtype=np.float64)
    weights = np.asarray([
        (1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy
    ])
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return float('nan')
    return float(np.sum(values[finite] * weights[finite]) / np.sum(weights[finite]))


def local_transform(lon_grid, lat_grid, x, y):
    corners = np.asarray([
        [sample_grid(lon_grid, x / 16.0, y / 16.0), sample_grid(lat_grid, x / 16.0, y / 16.0)],
        [sample_grid(lon_grid, (x + TILE_SIZE - 1) / 16.0, y / 16.0), sample_grid(lat_grid, (x + TILE_SIZE - 1) / 16.0, y / 16.0)],
        [sample_grid(lon_grid, x / 16.0, (y + TILE_SIZE - 1) / 16.0), sample_grid(lat_grid, x / 16.0, (y + TILE_SIZE - 1) / 16.0)],
        [sample_grid(lon_grid, (x + TILE_SIZE - 1) / 16.0, (y + TILE_SIZE - 1) / 16.0), sample_grid(lat_grid, (x + TILE_SIZE - 1) / 16.0, (y + TILE_SIZE - 1) / 16.0)],
    ])
    if not np.isfinite(corners).all():
        return None, None
    lon00, lon10, lon01, _ = corners
    ax = (lon10[0] - lon00[0]) / (TILE_SIZE - 1)
    ay = (lon01[0] - lon00[0]) / (TILE_SIZE - 1)
    bx = (lon10[1] - lon00[1]) / (TILE_SIZE - 1)
    by = (lon01[1] - lon00[1]) / (TILE_SIZE - 1)
    transform = Affine(ax, ay, lon00[0] - 0.5 * (ax + ay), bx, by, lon00[1] - 0.5 * (bx + by))
    return transform, corners


def load_selected_ids(path, scene):
    if path is None:
        return None
    payload = json.loads(path.read_text())
    scene_data = payload.get('scenes', {}).get(scene, {})
    return set(int(value) for value in scene_data.get('building_ids', []))


def feature_building_id(feature):
    if feature.GetFieldIndex('id') < 0:
        return int(feature.GetFID()) + 1
    value = feature.GetField('id')
    if value is None:
        return int(feature.GetFID()) + 1
    return int(round(float(value)))


def build_scene(scene, input_root, output_root, shp_path, progress, scene_index, total, selected_ids=None, stable_mask=False,
                crop_root=Path('outputs/rpc_dsm/preprocess/crop'),
                geo_root=Path('outputs/rpc_dsm/geolocation'),
                city_prefix=None):
    source_city = city_name(scene)
    target_city = city_prefix + scene if city_prefix is not None else SCENE_CITY.get(scene, 'GF7_' + scene)
    source_dir = input_root / 'left' / source_city
    source_right_dir = input_root / 'right' / source_city
    target_left = output_root / 'left' / target_city
    target_right = output_root / 'right' / target_city
    target_label = output_root / 'label' / target_city
    target_mask = output_root / 'mask' / target_city
    for path in (target_left, target_right, target_label, target_mask):
        require_real_parents(path / 'dataset-file')
        path.mkdir(parents=True, exist_ok=True)

    crop = json.loads((crop_root / scene / 'common_crop.json').read_text())
    width, height = crop['window'][2], crop['window'][3]
    geo_city = city_name(scene)
    geo_path = geo_root / geo_city / (geo_city + '_lonlat_stride16.tif')
    with rasterio.open(geo_path) as geo:
        lon_grid, lat_grid = geo.read(1), geo.read(2)

    source = ogr.Open(str(shp_path), 0)
    layer = source.GetLayer(0)
    files = sorted(source_dir.glob('*_left.tif'))
    generated = 0
    buildings = 0
    for tile_index, left_path in enumerate(files):
        match = TILE_RE.match(left_path.name)
        if not match:
            continue
        ix, iy = int(match.group(1)), int(match.group(2))
        x = min(ix * TILE_STEP, width - TILE_SIZE)
        y = min(iy * TILE_STEP, height - TILE_SIZE)
        transform, corners = local_transform(lon_grid, lat_grid, x, y)
        if transform is None:
            continue
        bounds = (
            float(np.min(corners[:, 0])) - 0.00005,
            float(np.min(corners[:, 1])) - 0.00005,
            float(np.max(corners[:, 0])) + 0.00005,
            float(np.max(corners[:, 1])) + 0.00005,
        )
        layer.SetSpatialFilterRect(*bounds)
        shapes = []
        mask_shapes = []
        next_instance_id = 1
        for feature in layer:
            if selected_ids is not None and feature_building_id(feature) not in selected_ids:
                continue
            height_value = feature.GetField('Height')
            geometry = feature.GetGeometryRef()
            if height_value is None or geometry is None:
                continue
            geometry_json = json.loads(geometry.ExportToJson())
            shapes.append((geometry_json, float(height_value)))
            mask_shapes.append((geometry_json, feature_building_id(feature) if stable_mask else next_instance_id))
            next_instance_id += 1
        layer.ResetReading()
        if not shapes:
            continue

        label = rasterize(
            shapes,
            out_shape=(TILE_SIZE, TILE_SIZE),
            transform=transform,
            fill=0.0,
            all_touched=True,
            dtype='float32',
        )
        instance_mask = rasterize(
            mask_shapes,
            out_shape=(TILE_SIZE, TILE_SIZE),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype='int32',
        )
        if not np.any(label > 0):
            continue
        label_name = left_path.name.replace('_left.tif', '_label.tif')
        label_path = target_label / label_name
        if label_path.is_symlink():
            copy_dataset_file(label_path, label_path)
        if not label_path.exists():
            with rasterio.open(
                label_path, 'w', driver='GTiff', width=TILE_SIZE, height=TILE_SIZE,
                count=1, dtype='float32', nodata=0.0, compress='deflate',
            ) as dst:
                dst.write(label.astype(np.float32), 1)
        mask_path = target_mask / label_name.replace('_label.tif', '_mask.tif')
        if mask_path.is_symlink():
            copy_dataset_file(mask_path, mask_path)
        if not mask_path.exists():
            with rasterio.open(
                mask_path, 'w', driver='GTiff', width=TILE_SIZE, height=TILE_SIZE,
                count=1, dtype='int32', nodata=0, compress='deflate',
            ) as dst:
                dst.write(instance_mask.astype(np.int32), 1)
        left_target = target_left / left_path.name
        right_target = target_right / left_path.name.replace('_left.tif', '_right.tif')
        copy_dataset_file(left_path, left_target)
        source_right = source_right_dir / right_target.name
        copy_dataset_file(source_right, right_target)
        generated += 1
        buildings += len(shapes)
        if generated % 100 == 0:
            atomic_json(progress, {
                'stage': 'LABELS', 'scene': scene, 'scene_index': scene_index,
                'scene_total': total, 'done_tiles': generated,
                'source_tiles': len(files), 'building_features': buildings,
            })

    atomic_json(progress, {
        'stage': 'LABELS', 'scene': scene, 'scene_index': scene_index,
        'scene_total': total, 'done_tiles': generated,
        'source_tiles': len(files), 'building_features': buildings,
    })
    return generated, buildings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, default=Path('outputs/rpc_dsm/tiles'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--shp', type=Path, default=Path('data/buildings.shp'))
    parser.add_argument('--scenes', nargs='+', default=list(SCENES))
    parser.add_argument('--selected-ids', type=Path, default=None)
    parser.add_argument('--stable-mask', action='store_true', help='Burn original shapefile building IDs into mask rasters.')
    parser.add_argument('--crop-root', type=Path, default=Path('outputs/rpc_dsm/preprocess/crop'))
    parser.add_argument('--geo-root', type=Path, default=Path('outputs/rpc_dsm/geolocation'))
    parser.add_argument('--scene-city-prefix', default=None, help='Use the same target city names as input image tiles')
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / 'progress.json'
    total_tiles = total_buildings = 0
    for index, scene in enumerate(args.scenes, 1):
        selected_ids = load_selected_ids(args.selected_ids, scene)
        tiles, buildings = build_scene(
            scene, args.input_root, args.output_root, args.shp, progress, index, len(args.scenes), selected_ids,
            args.stable_mask,
            args.crop_root, args.geo_root, args.scene_city_prefix,
        )
        total_tiles += tiles
        total_buildings += buildings
    atomic_json(progress, {
        'stage': 'DONE', 'scenes': args.scenes,
        'label_tiles': total_tiles, 'building_features': total_buildings,
    })
    print(f'label_tiles={total_tiles} building_features={total_buildings}')


if __name__ == '__main__':
    main()
