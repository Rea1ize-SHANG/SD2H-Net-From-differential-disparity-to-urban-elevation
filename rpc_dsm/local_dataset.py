"""Portable, model-free GF-7 rectification and labelled tile preparation."""

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

from rpc_dsm.inputs import discover, prepare, rpc_image


CITY_NAMES = {
    '038': 'SZ_DG_038', '039': 'PRE_039', '049': 'HZ_049',
    '050': 'HZ_050', '057': 'HZ_057', '058': 'HZ_DG_058',
    '060': 'SZ_060', '061': 'SZ_061', '062': 'HK_062',
}
SIZE = 1024
STEP = 870


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def status(root, stage, scene='', done=0, total=1, **extra):
    atomic_json(root / 'progress.json', dict(
        stage=stage, scene=scene, done=done, total=total, pid=os.getpid(),
        updated=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra))


def origins(length):
    if length < SIZE:
        raise ValueError('Common crop is smaller than 1024 pixels')
    return sorted(set(min(index * STEP, length - SIZE)
                      for index in range(math.ceil((length - SIZE) / STEP) + 1)))


def homography_points(matrix, columns, rows):
    columns, rows = np.broadcast_arrays(columns, rows)
    projected = matrix @ np.stack((columns.ravel(), rows.ravel(), np.ones(columns.size)))
    if np.any(np.abs(projected[2]) < 1e-12):
        raise ValueError('Singular point in homography')
    return (projected[:2] / projected[2]).T.reshape(columns.shape + (2,))


def write_tif(path, values, tags=None, origin=None):
    temporary = path.with_name(path.name + '.partial.tif')
    with rasterio.open(temporary, 'w', driver='GTiff', count=1,
                       width=values.shape[1], height=values.shape[0], dtype=values.dtype,
                       compress='deflate', tiled=True, blockxsize=256, blockysize=256,
                       transform=rasterio.Affine.translation(*(origin or (0, 0)))) as image:
        image.write(values, 1)
        if tags:
            image.update_tags(**tags)
    os.replace(temporary, path)


def projected_buildings(label, field, rpc, matrix, config, width, height, terrain=None):
    from osgeo import ogr, osr

    source = ogr.Open(str(label), 0)
    if source is None:
        raise ValueError('Cannot open building shapefile: ' + str(label))
    layer = source.GetLayer(0)
    if layer.GetLayerDefn().GetFieldIndex(field) < 0:
        raise ValueError('Missing height field: ' + field)
    original_crs = layer.GetSpatialRef()
    if original_crs is None:
        raise ValueError('Shapefile has no CRS; do not guess its coordinate system')
    original_crs = original_crs.Clone()
    geographic = osr.SpatialReference()
    geographic.ImportFromEPSG(4326)
    for crs in (original_crs, geographic):
        crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_geo = osr.CoordinateTransformation(original_crs, geographic)
    to_source = osr.CoordinateTransformation(geographic, original_crs)
    ground = float(config['ground_altitude_m'])
    pixel_corners = homography_points(np.linalg.inv(matrix),
                                    np.array([0, width - 1, width - 1, 0]),
                                    np.array([0, 0, height - 1, height - 1]))
    from rpc_dsm import gf7_rpc_compat
    longitude, latitude = rpc.localization(pixel_corners[:, 0], pixel_corners[:, 1], ground)
    longitude_margin = max(np.ptp(longitude) * 0.1, 0.002)
    latitude_margin = max(np.ptp(latitude) * 0.1, 0.002)
    bounds = [min(longitude) - longitude_margin, min(latitude) - latitude_margin,
              max(longitude) + longitude_margin, max(latitude) + latitude_margin]
    boundary = ogr.CreateGeometryFromJson(json.dumps({'type': 'Polygon', 'coordinates': [[
        [bounds[0], bounds[1]], [bounds[2], bounds[1]], [bounds[2], bounds[3]],
        [bounds[0], bounds[3]], [bounds[0], bounds[1]]]]}))
    boundary.Transform(to_source)
    layer.SetSpatialFilter(boundary)
    memory = ogr.GetDriverByName('Memory').CreateDataSource('projected')
    projected_layer = memory.CreateLayer('buildings', geom_type=ogr.wkbUnknown)
    projected_layer.CreateField(ogr.FieldDefn('Height', ogr.OFTReal))
    projected_layer.CreateField(ogr.FieldDefn('building_id', ogr.OFTInteger64))
    identifiers = set()
    id_field = config.get('id_field')
    if id_field and layer.GetLayerDefn().GetFieldIndex(id_field) < 0:
        raise ValueError('Missing requested building ID field: ' + id_field)
    for feature in layer:
        value = feature.GetField(field)
        geometry = feature.GetGeometryRef()
        if geometry is None or value is None or not np.isfinite(float(value)) or float(value) <= 0:
            continue
        identifier = int(feature.GetField(id_field)) if id_field else int(feature.GetFID()) + 1
        if identifier <= 0 or identifier > np.iinfo(np.int32).max or identifier in identifiers:
            raise ValueError('Building IDs must be unique positive int32 values; use FID+1 instead')
        identifiers.add(identifier)
        geometry = geometry.Clone()
        geometry.Transform(to_geo)
        geometry.Segmentize(float(config.get('edge_step_degrees', 0.00001)))
        payload = json.loads(geometry.ExportToJson())
        roof_offset = float(value) if config.get('label_surface', 'ground') == 'roof' else 0

        def project_coordinates(coordinates):
            if not coordinates:
                raise ValueError('Empty polygon coordinates')
            if isinstance(coordinates[0][0], (int, float)):
                vertices = np.asarray(coordinates)
                altitude = ground
                if terrain is not None:
                    from rasterio.warp import transform
                    columns, rows = transform('EPSG:4326', terrain.crs,
                                              vertices[:, 0].tolist(), vertices[:, 1].tolist())
                    bounds = terrain.bounds
                    if any(not (bounds.left <= column < bounds.right and
                                bounds.bottom < row <= bounds.top)
                           for column, row in zip(columns, rows)):
                        raise ValueError('Terrain DEM does not cover a building vertex')
                    samples = list(terrain.sample(zip(columns, rows), indexes=1, masked=True))
                    if any(np.ma.getmaskarray(sample).any() for sample in samples):
                        raise ValueError('Terrain DEM has no valid ground elevation at a building vertex')
                    altitude = np.array([float(sample[0]) for sample in samples])
                    if not np.isfinite(altitude).all():
                        raise ValueError('Nonfinite terrain elevation at a building vertex')
                    altitude = altitude + float(config.get('terrain_to_ellipsoid_offset_m', 0))
                column, row = rpc.projection(vertices[:, 0], vertices[:, 1], altitude + roof_offset)
                pixels = homography_points(matrix, column, row)
                if not np.isfinite(pixels).all():
                    raise ValueError('Nonfinite RPC-projected building vertices')
                return pixels.tolist()
            return [project_coordinates(child) for child in coordinates]

        if payload['type'] not in ('Polygon', 'MultiPolygon'):
            raise ValueError('Building label must contain polygons or multipolygons')
        payload['coordinates'] = project_coordinates(payload['coordinates'])
        output = ogr.Feature(projected_layer.GetLayerDefn())
        output.SetField('Height', float(value))
        output.SetField('building_id', identifier)
        output.SetGeometry(ogr.CreateGeometryFromJson(json.dumps(payload)))
        projected_layer.CreateFeature(output)
    source = None
    return memory, projected_layer


def make_tiles(root, scene, paths, config):
    crop = Path(config.get('crop_root_override', str(root / 'preprocess' / 'crop'))) / scene
    dataset = Path(config.get('tiles_root_override', str(root / 'UnifiedBH_2')))
    city = config.get('city_names', {}).get(scene, CITY_NAMES.get(scene, 'GF7_' + scene))
    if not re.fullmatch(r'[A-Za-z0-9_-]+', city):
        raise ValueError('Unsafe city folder name: ' + city)
    label = config.get('label')
    matrix = np.loadtxt(crop / 'global_H_ref_crop.txt')
    label_memory = label_layer = None
    with contextlib.ExitStack() as resources, \
            rasterio.open(crop / 'rectified_ref_common.tif') as left, \
            rasterio.open(crop / 'rectified_sec_common.tif') as right:
        if left.shape != right.shape:
            raise ValueError('Left/right common crop dimensions differ')
        columns, rows = origins(left.width), origins(left.height)
        if label:
            _, _, rpc = rpc_image(paths['left'])
            terrain = None
            if config.get('terrain_dem'):
                terrain = resources.enter_context(rasterio.open(config['terrain_dem']))
                if terrain.crs is None or terrain.count != 1:
                    raise ValueError('Terrain DEM must be a single-band geographic raster with CRS')
            status(root, 'LABEL_PROJECTION', scene)
            label_memory, label_layer = projected_buildings(
                Path(label), config.get('height_field', 'Height'), rpc, matrix,
                config, left.width, left.height, terrain)
        kinds = ['left', 'right', 'valid'] + (['label', 'mask'] if label else [])
        for kind in kinds:
            (dataset / kind / city).mkdir(parents=True, exist_ok=True)
        manifest = dataset / ('manifest_' + scene + '.jsonl')
        temporary = manifest.with_name(manifest.name + '.tmp')
        kept = 0
        total = len(columns) * len(rows)
        with temporary.open('w') as records:
            for iy, top in enumerate(rows):
                for ix, col in enumerate(columns):
                    window = Window(col, top, SIZE, SIZE)
                    reference = left.read(1, window=window, masked=True)
                    secondary = right.read(1, window=window, masked=True)
                    valid = (~np.ma.getmaskarray(reference) & ~np.ma.getmaskarray(secondary) &
                             np.isfinite(reference.data) & np.isfinite(secondary.data))
                    fraction = float(valid.mean())
                    accepted = fraction >= float(config.get('min_valid_fraction', 1.0))
                    name = 'GF7_%s_%d_%d' % (scene, ix, iy)
                    record = dict(scene=scene, city=city, tile=name, x=col, y=top,
                                  width=SIZE, height=SIZE, valid_fraction=fraction,
                                  kept=accepted, paths={})
                    if accepted:
                        values = {'left': np.where(valid, reference.data, 0).astype('float32'),
                                  'right': np.where(valid, secondary.data, 0).astype('float32'),
                                  'valid': valid.astype('uint8')}
                        if label:
                            label_layer.SetSpatialFilterRect(col - 0.5, top - 0.5,
                                                           col + SIZE - 0.5, top + SIZE - 0.5)
                            height_shapes, mask_shapes = [], []
                            for feature in label_layer:
                                geometry = json.loads(feature.GetGeometryRef().ExportToJson())
                                height_shapes.append((geometry, feature.GetField('Height')))
                                mask_shapes.append((geometry, feature.GetField('building_id')))
                            transform = rasterio.Affine.translation(col - 0.5, top - 0.5)
                            for kind, shapes, dtype in [('label', height_shapes, 'float32'),
                                                        ('mask', mask_shapes, 'int32')]:
                                values[kind] = (rasterize(shapes, out_shape=(SIZE, SIZE),
                                                         transform=transform, dtype=dtype)
                                                if shapes else np.zeros((SIZE, SIZE), dtype=dtype))
                                values[kind][~valid] = 0
                        for kind, array in values.items():
                            relative = Path(kind) / city / (name + '_' + kind + '.tif')
                            write_tif(dataset / relative, array,
                                      tags=dict(version_id=config.get('version_id', ''),
                                                source_crop=str(crop), source_x=col, source_y=top,
                                                coordinate_space='rectified cropped pixel coordinates'),
                                      origin=(col, top))
                            record['paths'][kind] = str(relative)
                        kept += 1
                    records.write(json.dumps(record) + '\n')
                    status(root, 'TILES', scene, iy * len(columns) + ix + 1, total,
                           kept_tiles=kept)
        os.replace(temporary, manifest)
        if kept == 0:
            raise ValueError('No tiles met common-valid-pixel threshold; inspect crop before changing it')
    label_memory = None
    summary = dict(scene=scene, city=city, crop_shape=[left.width, left.height],
                   candidate_tiles=total, kept_tiles=kept, tile_size=SIZE, stride=STEP,
                   actual_overlap=1 - STEP / SIZE, edge_policy='anchor last tile to image edge',
                   geographic_label_method='RPC projection then global_H_ref_crop',
                   label_surface=config.get('label_surface', 'ground'),
                   ground_altitude_m=config.get('ground_altitude_m'),
                   terrain_dem=config.get('terrain_dem'),
                   label_geolocation_is_height_prior_dependent=bool(label))
    atomic_json(dataset / ('summary_' + scene + '.json'), summary)
    return summary


def tie_report(root, scene, tie_path):
    pairs = np.loadtxt(tie_path, comments=';', ndmin=2)
    if pairs.shape[1] != 4 or not np.isfinite(pairs).all():
        raise ValueError('ENVI tie points require four finite columns')
    folder = root / 'preprocess' / 'crop' / scene
    left = homography_points(np.loadtxt(folder / 'global_H_ref_crop.txt'), pairs[:, 0], pairs[:, 1])
    right = homography_points(np.loadtxt(folder / 'global_H_sec_crop.txt'), pairs[:, 2], pairs[:, 3])
    delta = left[:, 1] - right[:, 1]
    atomic_json(folder / 'tie_validation.json', dict(
        points=len(delta), all_points_mae_px=float(np.mean(np.abs(delta))),
        all_points_rmse_px=float(np.sqrt(np.mean(delta ** 2))),
        max_abs_px=float(np.max(np.abs(delta))), used_to_fit=False,
        pixel_origin='as provided; must be same convention as RPC projection'))
    np.savetxt(folder / 'tie_mapped.csv', np.column_stack((left, right, delta)),
               delimiter=',', header='left_x,left_y,right_x,right_y,vertical_residual', comments='')


def rectify(root, scene, config_path):
    status(root, 'RECTIFY', scene)
    command = [sys.executable, '-m', 'rpc_dsm.rectify', str(config_path), '--skip-tiles']
    with (root / ('rectify_' + scene + '.log')).open('a') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   universal_newlines=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line.rstrip(), flush=True)
                match = re.search(r'GLOBAL_BLOCK (\S+) (\d+)/(\d+)', line)
                if match:
                    status(root, 'RECTIFY_WRITE', scene, int(match[2]), int(match[3]), image=match[1])
            if process.wait():
                raise RuntimeError('Rectification failed; read ' + log.name)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


def run(config_path, until='tiles'):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    for key in ('raw_root', 'output_root', 'label', 'tie_root', 'terrain_dem'):
        if config.get(key):
            candidate = Path(config[key]).expanduser()
            config[key] = str((config_path.parent / candidate).resolve())
    raw, root = Path(config['raw_root']), Path(config['output_root'])
    if raw == root or raw in root.parents or root in raw.parents:
        raise ValueError('Raw and output trees must be separate, not nested')
    if config.get('label'):
        if not Path(config['label']).is_file() or Path(config['label']).suffix.lower() != '.shp':
            raise ValueError('This entry requires a geographic building .shp; include all sidecars')
        if 'ground_altitude_m' not in config or config.get('altitude_datum') != 'ellipsoidal':
            raise ValueError('Label alignment requires explicit ground_altitude_m and altitude_datum=ellipsoidal')
        if config.get('label_surface', 'ground') not in ('ground', 'roof'):
            raise ValueError('label_surface must be ground or roof')
        if config.get('terrain_dem') and config.get('terrain_datum') != 'ellipsoidal':
            if 'terrain_to_ellipsoid_offset_m' not in config:
                raise ValueError('Nonellipsoidal DEM needs an explicitly justified conversion offset')
    if not 0 < float(config.get('min_valid_fraction', 1)) <= 1:
        raise ValueError('min_valid_fraction must be in (0,1]')
    for key in ('raw_root', 'label', 'tie_root', 'terrain_dem'):
        if config.get(key) and (root == Path(config[key]) or root in Path(config[key]).parents):
            raise ValueError('Output must not contain input data')
    scenes = discover(raw, config.get('scenes'))
    fingerprint = dict(config=config, inputs=[])
    for item in list(scenes.values()):
        for path in (Path(item['left']), Path(item['right'])):
            fingerprint['inputs'].append([str(path), path.stat().st_size, path.stat().st_mtime_ns])
            for sidecar in path.parent.glob(path.stem + '.*'):
                if sidecar != path:
                    fingerprint['inputs'].append([str(sidecar), sidecar.stat().st_size, sidecar.stat().st_mtime_ns])
    if config.get('label'):
        for path in Path(config['label']).parent.glob(Path(config['label']).stem + '.*'):
            fingerprint['inputs'].append([str(path), path.stat().st_size, path.stat().st_mtime_ns])
    if config.get('terrain_dem'):
        path = Path(config['terrain_dem'])
        fingerprint['inputs'].append([str(path), path.stat().st_size, path.stat().st_mtime_ns])
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.running.lock'
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Run lock exists; check old process before removing ' + str(lock))
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        provenance = root / 'local_config.json'
        if provenance.exists() and json.loads(provenance.read_text())['signature'] != signature:
            raise ValueError('Inputs/config changed; use a fresh output folder')
        atomic_json(provenance, dict(signature=signature, **fingerprint))
        status(root, 'PREPARE')
        runs = prepare(root, scenes, config.get('min_height', -500), config.get('max_height', 5000))
        summaries = []
        if until != 'prepare':
            from rpc_dsm.crop import process_scene
            for scene, paths in scenes.items():
                rectify(root, scene, Path(runs[scene]) / 'input_config.json')
                if until == 'rectify':
                    continue
                status(root, 'CROP', scene)
                process_scene(root / 'preprocess', scene, source_folder=scene)
                if config.get('tie_root'):
                    candidates = [Path(config['tie_root']) / (scene + '.pts'),
                                  Path(config['tie_root']) / (str(int(scene)) + '.pts')]
                    available = next((path for path in candidates if path.exists()), None)
                    if available:
                        tie_report(root, scene, available)
                if until == 'tiles':
                    summaries.append(make_tiles(root, scene, paths, config))
        atomic_json(root / 'results.json', summaries)
        status(root, 'DONE', done=len(scenes), total=len(scenes), completed_through=until)
    except BaseException as error:
        status(root, 'STOPPED' if isinstance(error, KeyboardInterrupt) else 'FAILED', error=str(error))
        raise
    finally:
        lock.unlink()


def watch(root):
    while True:
        try:
            state = json.loads((root / 'progress.json').read_text())
        except (OSError, ValueError):
            print('\r等待任务启动...', end='', flush=True)
            time.sleep(1)
            continue
        if state['stage'] == 'CROP':
            try:
                state.update(json.loads((root / 'preprocess' / 'crop_status.json').read_text()))
                state['stage'] = 'CROP_' + state['stage']
            except (OSError, ValueError):
                pass
        fraction = min(1, state['done'] / max(1, state['total']))
        filled = int(36 * fraction)
        line = '\033[37m[%s%s]\033[0m %s scene=%s %d/%d %.2f%%' % (
            '█' * filled, '░' * (36 - filled), state['stage'], state.get('scene', ''),
            state['done'], state['total'], fraction * 100)
        print('\r' + line + '\033[K', end='', flush=True)
        if state['stage'] in ('DONE', 'FAILED', 'STOPPED'):
            print('\n' + state.get('error', '任务结束'))
            return 0 if state['stage'] == 'DONE' else 1
        try:
            os.kill(state['pid'], 0)
        except ProcessLookupError:
            print('\n任务进程已消失；检查日志和 .running.lock，确认后续跑。')
            return 1
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--until-stage', choices=('prepare', 'rectify', 'crop', 'tiles'), default='tiles')
    parser.add_argument('--watch', type=Path, help='Monitor an output root and exit on completion/failure')
    args = parser.parse_args()
    if args.watch:
        sys.exit(watch(args.watch))
    if args.config is None:
        parser.error('--config is required unless --watch is used')
    run(args.config, args.until_stage)


if __name__ == '__main__':
    main()
