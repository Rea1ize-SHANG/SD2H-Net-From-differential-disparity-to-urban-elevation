"""Project a georeferenced full-scene height raster onto corrected stereo tiles."""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rpc_dsm.naming import city_name
from rasterio.warp import Resampling, reproject
from skimage.measure import label as connected_components

from rpc_dsm.vector_labels import local_transform
from rpc_dsm.dataset_files import copy_dataset_file, require_real_parents


def tile_grid(scene, crop_root, geo_root):
    crop = json.loads((crop_root / scene / 'common_crop.json').read_text())
    width, height = crop['window'][2:]
    city = city_name(scene)
    path = geo_root / city / (city + '_lonlat_stride16.tif')
    with rasterio.open(path) as geolocation:
        lon, lat = geolocation.read(1), geolocation.read(2)
    return width, height, lon, lat


def build_scene(scene, source, tiles_root, output_root, crop_root, geo_root):
    city = city_name(scene)
    width, height, lon, lat = tile_grid(scene, crop_root, geo_root)
    count = 0
    for left_path in sorted((tiles_root / 'left' / city).glob('*_left.tif')):
        name = left_path.stem[:-5]
        suffix = name[len(city) + 1:].split('_')
        if len(suffix) != 2 or not all(value.isdigit() for value in suffix):
            raise ValueError('Unexpected corrected tile filename: ' + left_path.name)
        offset_x = min(int(suffix[0]) * 870, width - 1024)
        offset_y = min(int(suffix[1]) * 870, height - 1024)
        transform, corners = local_transform(lon, lat, offset_x, offset_y)
        if transform is None or not np.isfinite(corners).all():
            continue
        label = np.zeros((1024, 1024), dtype=np.float32)
        reproject(source=rasterio.band(source, 1), destination=label,
                  src_transform=source.transform, src_crs=source.crs,
                  src_nodata=source.nodata, dst_transform=transform,
                  dst_crs='EPSG:4326', dst_nodata=0, resampling=Resampling.nearest)
        mask = connected_components(np.isfinite(label) & (label > 0), connectivity=2).astype(np.int32)
        if mask.max() == 0:
            continue
        for kind, values in (('label', label), ('mask', mask)):
            path = output_root / kind / city / (name + '_' + kind + '.tif')
            require_real_parents(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                copy_dataset_file(path, path)
            if not path.exists():
                with rasterio.open(path, 'w', driver='GTiff', width=1024, height=1024,
                                   count=1, dtype=values.dtype, nodata=0, compress='deflate') as target:
                    target.write(values, 1)
        count += 1
        if count % 100 == 0:
            print(scene, 'aligned raster label tiles', count, flush=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raster', type=Path, required=True)
    parser.add_argument('--tiles-root', type=Path, required=True)
    parser.add_argument('--crop-root', type=Path, required=True)
    parser.add_argument('--geo-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+', required=True)
    args = parser.parse_args()
    with rasterio.open(args.raster) as source:
        if source.crs is None or source.count != 1:
            raise ValueError('Label raster requires one georeferenced height band')
        for scene in args.scenes:
            print(scene, 'labeled tiles', build_scene(scene, source, args.tiles_root,
                  args.output_root, args.crop_root, args.geo_root), flush=True)


if __name__ == '__main__':
    main()
