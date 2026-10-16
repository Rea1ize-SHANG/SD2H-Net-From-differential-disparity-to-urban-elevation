"""Warp a rectified DSM to an actual projected GeoTIFF using RPC lon/lat grids."""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import rasterio
from osgeo import gdal
from rpc_dsm.naming import city_name


def target_crs(grid_path):
    with rasterio.open(grid_path) as grid:
        center = grid.read(window=((max(0, grid.height // 2 - 8), min(grid.height, grid.height // 2 + 8)),
                                   (max(0, grid.width // 2 - 8), min(grid.width, grid.width // 2 + 8))))
    valid = np.isfinite(center[0]) & np.isfinite(center[1])
    if not valid.any():
        raise ValueError('No valid geographic coordinates near the image center: ' + str(grid_path))
    lon = float(np.median(center[0][valid]))
    lat = float(np.median(center[1][valid]))
    zone = min(60, max(1, math.floor((lon + 180) / 6) + 1))
    return 'EPSG:%d' % ((32600 if lat >= 0 else 32700) + zone)


def warp_scene(dsm_path, grid_path, output_path, stride=16, resolution=1.0, crs=None):
    if resolution <= 0 or stride <= 0:
        raise ValueError('Resolution and geographic-grid stride must be positive')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dsm_path) as dsm, rasterio.open(grid_path) as grid:
        if grid.count != 2 or grid.width != math.ceil(dsm.width / stride) or grid.height != math.ceil(dsm.height / stride):
            raise ValueError('RPC longitude/latitude grid dimensions do not match DSM')
    target = crs or target_crs(grid_path)
    vrt_path = output_path.with_suffix('.source.vrt')
    if gdal.Translate(str(vrt_path), str(dsm_path), format='VRT') is None:
        raise RuntimeError('Cannot construct DSM VRT: ' + str(dsm_path))
    vrt = gdal.Open(str(vrt_path), gdal.GA_Update)
    if vrt is None:
        raise RuntimeError('Cannot open DSM VRT: ' + str(vrt_path))
    vrt.SetMetadata({
        'SRS': 'EPSG:4326', 'X_DATASET': str(grid_path.resolve()),
        'Y_DATASET': str(grid_path.resolve()), 'X_BAND': '1', 'Y_BAND': '2',
        'PIXEL_OFFSET': '0', 'LINE_OFFSET': '0',
        'PIXEL_STEP': str(stride), 'LINE_STEP': str(stride),
    }, 'GEOLOCATION')
    vrt = None
    options = gdal.WarpOptions(format='GTiff', dstSRS=target, xRes=resolution, yRes=resolution,
                               geoloc=True, outputType=gdal.GDT_Float32, dstNodata=-9999,
                               multithread=True, creationOptions=['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=YES'])
    temporary = output_path.with_name(output_path.stem + '.partial.tif')
    if temporary.exists():
        temporary.unlink()
    result = gdal.Warp(str(temporary), str(vrt_path), options=options)
    if result is None:
        raise RuntimeError('Geographic DSM warp failed for ' + str(dsm_path))
    result = None
    with rasterio.open(temporary) as projected:
        if projected.crs is None or projected.width == 0 or projected.height == 0:
            raise ValueError('Projected DSM has invalid CRS or dimensions: ' + str(temporary))
        sample = projected.read(1, out_shape=(min(128, projected.height), min(128, projected.width)), masked=True)
        if not np.isfinite(sample.compressed()).any():
            raise ValueError('Projected DSM has no sampled valid heights: ' + str(temporary))
        print(output_path, projected.crs, projected.width, projected.height, flush=True)
    os.replace(temporary, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsm-root', type=Path, required=True)
    parser.add_argument('--geo-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+', required=True)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--resolution', type=float, default=1.0)
    parser.add_argument('--crs', default=None, help='Optional target EPSG code; default is each scene UTM zone')
    args = parser.parse_args()
    for scene in args.scenes:
        city = city_name(scene)
        output = args.output_root / city / (city + '_DSM_projected.tif')
        if output.exists():
            with rasterio.open(output) as image:
                if image.crs is None:
                    raise ValueError('Existing projected DSM has no CRS: ' + str(output))
                sample = image.read(1, out_shape=(min(128, image.height), min(128, image.width)), masked=True)
                if not np.isfinite(sample.compressed()).any():
                    raise ValueError('Existing projected DSM has no valid sampled height: ' + str(output))
            continue
        warp_scene(args.dsm_root / city / (city + '_DSM.tif'),
                   args.geo_root / city / (city + '_lonlat_stride%d.tif' % args.stride),
                   output, args.stride, args.resolution, args.crs)


if __name__ == '__main__':
    main()
