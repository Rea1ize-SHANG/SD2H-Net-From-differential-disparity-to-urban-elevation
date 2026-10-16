"""Summarize roof minus local DSM ground by labeled building instance."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt

from rpc_dsm.vector_labels import sample_grid
from rpc_dsm.naming import city_name


def estimate_instance_heights(dsm, mask, reference, pixel_meters, buffer_m):
    if buffer_m <= 0 or pixel_meters <= 0:
        raise ValueError('Local ground buffer and pixel size must be positive')
    radius = buffer_m / pixel_meters
    if radius <= 2:
        raise ValueError('Ground buffer is smaller than two image pixels')
    valid = np.isfinite(dsm)
    building = (mask > 0) & valid
    if not building.any():
        return []
    distances, nearest = distance_transform_edt(mask == 0, return_indices=True)
    ground = (mask == 0) & valid & (distances >= 2) & (distances <= radius)
    ground_ids = mask[nearest[0][ground], nearest[1][ground]]
    ground_values = dsm[ground]
    rows = []
    for identifier in np.unique(mask[building]):
        inside = building & (mask == identifier)
        roof = dsm[inside]
        if roof.size < 16:
            continue
        local_ground = ground_values[ground_ids == identifier]
        if local_ground.size < 4:
            continue
        # Remove only upper-tail roof outliers; height is accepted roof maximum minus exterior DSM minimum.
        lower, upper = np.percentile(roof, [25, 75])
        accepted = roof[roof <= upper + 1.5 * (upper - lower)]
        if not accepted.size:
            continue
        values = reference[inside]
        values = values[np.isfinite(values) & (values > 0)]
        rows.append((int(identifier), float(np.max(accepted) - np.min(local_ground)),
                     float(np.median(values)) if values.size else float('nan'), int(inside.sum())))
    return rows


def pixel_size_meters(lon, lat, x, y):
    x_grid, y_grid = x / 16.0, y / 16.0
    mid_lat = sample_grid(lat, x_grid, y_grid)
    dx = (sample_grid(lon, x_grid + 1, y_grid) - sample_grid(lon, x_grid, y_grid)) / 16.0
    dy = (sample_grid(lat, x_grid, y_grid + 1) - sample_grid(lat, x_grid, y_grid)) / 16.0
    return float((abs(dx * np.cos(np.deg2rad(mid_lat)) * 111320.0) + abs(dy * 110540.0)) / 2)


def process_scene(scene, args):
    city = city_name(scene)
    with rasterio.open(args.dsm_root / city / (city + '_DSM.tif')) as surface, rasterio.open(
            args.geo_root / city / (city + '_lonlat_stride16.tif')) as geolocation:
        longitude, latitude = geolocation.read(1), geolocation.read(2)
        detections = defaultdict(list)
        for mask_path in sorted((args.labels_root / 'mask' / city).glob('*_mask.tif')):
            core = mask_path.stem[:-5]
            parts = core[len(city) + 1:].split('_')
            if len(parts) != 2:
                raise ValueError('Unexpected mask name: ' + mask_path.name)
            x = min(int(parts[0]) * 870, surface.width - 1024)
            y = min(int(parts[1]) * 870, surface.height - 1024)
            with rasterio.open(mask_path) as mask_image, rasterio.open(
                    args.labels_root / 'label' / city / (core + '_label.tif')) as label_image:
                mask = mask_image.read(1)
                reference = label_image.read(1)
            dsm = surface.read(1, window=Window(x, y, 1024, 1024))
            meters = pixel_size_meters(longitude, latitude, x + 512, y + 512)
            if not np.isfinite(meters) or meters <= 0:
                continue
            for identifier, predicted, target, pixels in estimate_instance_heights(
                    dsm, mask, reference, meters, args.buffer_m):
                detections[identifier].append((predicted, target, pixels))
    output_path = args.output_root / (city + '.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(('scene', 'building_id', 'pred_height_m', 'label_height_m', 'tile_observations', 'sampled_roof_pixels'))
        for identifier, measurements in sorted(detections.items()):
            heights = np.asarray(measurements, dtype=np.float64)
            writer.writerow((scene, identifier, np.median(heights[:, 0]),
                             np.nanmedian(heights[:, 1]) if np.isfinite(heights[:, 1]).any() else '',
                             len(measurements), int(np.sum(heights[:, 2]))))
    print(scene, 'buildings with roof and local ground', len(detections), flush=True)
    return len(detections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dsm-root', 'labels-root', 'crop-root', 'geo-root', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--scenes', nargs='+', required=True)
    parser.add_argument('--buffer-m', type=float, default=20.0)
    args = parser.parse_args()
    counts = {scene: process_scene(scene, args) for scene in args.scenes}
    (args.output_root / 'summary.json').write_text(json.dumps(counts, indent=2) + '\n')


if __name__ == '__main__':
    main()
