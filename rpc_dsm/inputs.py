"""Discover GF-7 stereo pairs and generate s2p RPC configurations."""

import hashlib
import json
import re
from pathlib import Path

import rasterio
import rpcm


RPC_FIELDS = (
    'row_offset', 'col_offset', 'lat_offset', 'lon_offset', 'alt_offset',
    'row_scale', 'col_scale', 'lat_scale', 'lon_scale', 'alt_scale',
    'row_num', 'row_den', 'col_num', 'col_den',
)


def find_pair(folder):
    pairs = (("BPA", "FPA"), ("BWDPAN", "FWDPAN"))
    for left_band, right_band in pairs:
        left = [path for path in folder.iterdir() if path.suffix.lower() in ('.tif', '.tiff')
                and (re.search(r'(^|[_-])' + left_band + r'([_.-]|$)', path.stem, re.I))]
        right = [path for path in folder.iterdir() if path.suffix.lower() in ('.tif', '.tiff')
                 and (re.search(r'(^|[_-])' + right_band + r'([_.-]|$)', path.stem, re.I))]
        if len(left) == len(right) == 1:
            return left[0], right[0]
    raise ValueError('Expected exactly one BPA/FPA or BWDPAN/FWDPAN pair in ' + str(folder))


def scene_id(folder):
    match = re.search(r'_L1A0*(\d+)$', folder.name, re.I)
    if match and folder.name.upper().startswith('KM_GF7_'):
        return 'KM' + str(int(match.group(1)))
    match = re.search(r'_(\d{3})_L1A(?:_|$)', folder.name, re.I)
    if match:
        return match.group(1)
    return str(int(hashlib.sha256(folder.name.encode('utf-8')).hexdigest()[:10], 16))


def discover(root, names=None):
    if not root.is_dir():
        raise NotADirectoryError(root)
    children = sorted(path for path in root.iterdir() if path.is_dir())
    if names:
        selected = set(names)
        children = [path for path in children if path.name in selected or scene_id(path) in selected]
        if len(children) != len(selected):
            raise ValueError('Some requested scene names/IDs were not found: ' + repr(selected))
    scenes = {}
    for folder in children:
        if not folder.name.startswith(('GF07_', 'KM_GF7_')):
            continue
        left, right = find_pair(folder)
        scene = scene_id(folder)
        if scene in scenes:
            raise ValueError('Duplicate scene ID %s for %s and %s' % (scene, scenes[scene]['folder'], folder))
        scenes[scene] = {'folder': str(folder.resolve()), 'left': str(left.resolve()), 'right': str(right.resolve())}
    if not scenes:
        raise ValueError('No GF-7 panchromatic stereo pairs found in ' + str(root))
    return scenes


def rpc_image(path):
    with rasterio.open(path) as image:
        tags = image.tags(ns='RPC')
        if len(tags) < 16:
            raise ValueError('Missing or incomplete RPC metadata for ' + str(path))
        rpc = rpcm.RPCModel(tags)
        center_x, center_y = rpc.projection(rpc.lon_offset, rpc.lat_offset, rpc.alt_offset)
        if not 0 <= center_x < image.width or not 0 <= center_y < image.height:
            raise ValueError('RPC center outside image: ' + str(path))
        values = {name: getattr(rpc, name) for name in RPC_FIELDS}
        return {'img': str(path), 'rpc': values}, (image.width, image.height), rpc


def prepare(root, scenes, min_height, max_height):
    if min_height >= max_height:
        raise ValueError('Minimum altitude must be lower than maximum altitude')
    epipolar = root / 'preprocess' / 'epipolar'
    runs = {}
    for scene, paths in scenes.items():
        folder = epipolar / scene
        left, left_shape, _ = rpc_image(paths['left'])
        right, right_shape, _ = rpc_image(paths['right'])
        if min(left_shape + right_shape) < 1024:
            raise ValueError('Stereo inputs must have dimensions >=1024 px: ' + scene)
        config = {
            'out_dir': str(folder), 'temporary_dir': str(folder / 'tmp'),
            'full_img': True, 'clean_tmp': False, 'debug': False,
            'rectification_method': 'rpc', 'disp_range_method': 'fixed_altitude_range',
            'alt_min': min_height, 'alt_max': max_height, 'use_srtm': False,
            'max_processes': 1, 'tile_size': 1024, 'images': [left, right],
        }
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'input_config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('Existing s2p configuration differs; use a fresh --output-root: ' + str(path))
        path.write_text(json.dumps(config, indent=2) + '\n')
        runs[scene] = str(folder)
    (root / 'run_map.json').write_text(json.dumps(runs, indent=2) + '\n')
    (root / 'scenes.json').write_text(json.dumps(scenes, indent=2) + '\n')
    return runs
