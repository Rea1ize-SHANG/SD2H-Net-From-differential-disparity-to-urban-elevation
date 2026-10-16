"""Version-bound stereo inference; refuse stale TIFF and NPY reuse."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.windows import Window

from rpc_dsm.dataset_files import file_hash, require_real_parents
from rpc_dsm.local_dataset import atomic_json, write_tif
from rpc_dsm.naming import city_name


SCENES = ('038', '039', '049', '050', '057', '058', '060', '061', '062')


def parse_right_shifts(values):
    result = {}
    for value in values:
        scene, shift = value.rsplit('=', 1)
        result[scene] = int(shift)
    return result


def load_shift_map(path):
    if not path:
        return {}
    payload = json.loads(Path(path).read_text())
    return {scene: {key: int(value) for key, value in data['shifts'].items()}
            for scene, data in payload.items()}


def prepare_input(array, mode):
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype('float32')
    if mode == 'raw':
        return array
    low, high = np.percentile(array, [1, 99])
    stretched = np.clip((array - low) / (high - low + 1e-6), 0, 1)
    return (255 * stretched if mode == 'percentile255' else stretched).astype('float32')


def source_record(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def bind_version(folder, payload):
    identifier = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'source_version.json'
    record = {'version_id': identifier, 'inputs': payload}
    with (folder / '.version.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError('Disparity sources or settings changed; use a fresh output folder: ' + str(folder))
        atomic_json(path, record)
    return identifier


def main():
    import torch
    from core.utils.utils import InputPadder
    from precompute_unifiedbh_disp_delta import _load_sd2h

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/rpc_dsm/preprocess/crop'))
    parser.add_argument('--out', type=Path, default=Path('outputs/rpc_dsm/tiles'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--valid_iters', type=int, default=16)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--scenes', nargs='+', default=list(SCENES))
    parser.add_argument('--disp-subdir', default='disp')
    parser.add_argument('--input-mode', choices=('raw', 'percentile255', 'legacy01'), default='raw')
    parser.add_argument('--right-shifts', nargs='*', default=[])
    parser.add_argument('--shift-map', default='')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--existing-tiles-only', action='store_true')
    parser.add_argument('--source-version', default='')
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Invalid shard index')
    shifts = parse_right_shifts(args.right_shifts)
    shift_map = load_shift_map(args.shift_map)
    checkpoint_hash = file_hash(args.checkpoint)
    tasks = []
    for scene in args.scenes:
        crop = args.root / scene
        reference = crop / 'rectified_ref_common.tif'
        secondary = crop / 'rectified_sec_common.tif'
        with rasterio.open(reference) as image:
            width, height = image.width, image.height
        city = city_name(scene)
        disparity_folder = args.out / args.disp_subdir / city
        inputs = dict(crop_files=[source_record(reference), source_record(secondary)],
                      matrix_hashes={role: file_hash(crop / ('global_H_' + role + '_crop.txt'))
                                     for role in ('ref', 'sec')},
                      checkpoint_sha256=checkpoint_hash, source_version=args.source_version,
                      mode=args.input_mode, iterations=args.valid_iters,
                      implementation_sha256=file_hash(Path(__file__)), min_disp=-320, max_disp=64,
                      shifts=shift_map.get(scene, {}), default_shift=shifts.get(scene, 0))
        inference_version = bind_version(disparity_folder, inputs)
        nx = (width - 1024 + 869) // 870 + 1
        ny = (height - 1024 + 869) // 870 + 1
        for iy in range(ny):
            for ix in range(nx):
                if (iy * nx + ix) % args.num_shards != args.shard_index:
                    continue
                stem = '%s_%d_%d' % (city, ix, iy)
                left = args.out / 'left' / city / (stem + '_left.tif')
                right = args.out / 'right' / city / (stem + '_right.tif')
                if args.existing_tiles_only and not left.exists():
                    continue
                tasks.append(dict(scene=scene, city=city, ix=ix, iy=iy,
                                  x=min(ix * 870, width - 1024), y=min(iy * 870, height - 1024),
                                  left=left, right=right, reference=reference, secondary=secondary,
                                  disparity=disparity_folder / (stem + '_disp.npy'),
                                  inference_version=inference_version))
    progress = args.out / ('progress_%s_shard_%d.json' % (args.disp_subdir, args.shard_index))
    atomic_json(progress, dict(scene=args.scenes[0], done=0, total=len(tasks), shard=args.shard_index,
                              num_shards=args.num_shards, version_id=args.source_version,
                              input_mode=args.input_mode, stage='LOAD_MODEL'))
    configuration = SimpleNamespace(restore_ckpt=str(args.checkpoint), random_initialization=False,
        random_seed=0, device=args.device, encoder='vitb', shared_backbone=True, corr_levels=2,
        hidden_dims=[128, 128, 128], n_gru_layers=3, valid_iters=args.valid_iters,
        min_disp=-320, max_disp=64, n_downsample=4, corr_implementation='reg', corr_radius=4,
        slow_fast_gru=False, use_mono=True, use_stereo=True)
    model = _load_sd2h(configuration)
    for index, task in enumerate(tasks):
        x, y, scene = task['x'], task['y'], task['scene']
        shift = shift_map.get(scene, {}).get('%d_%d' % (task['ix'], task['iy']), shifts.get(scene, 0))
        if args.existing_tiles_only and shift:
            raise ValueError('Prebuilt paired tiles must not be silently shifted during inference')
        arrays = []
        for kind, source, path, offset in [('left', task['reference'], task['left'], 0),
                                          ('right', task['secondary'], task['right'], shift)]:
            require_real_parents(path)
            if path.is_symlink():
                raise ValueError('Dataset TIFF is a link; materialize it before inference')
            if path.exists():
                with rasterio.open(path) as image:
                    tags = image.tags()
                    values = image.read(1)
                expected_version = args.source_version or task['inference_version']
                if (tags.get('version_id') != expected_version or
                        int(tags.get('source_x', -1)) != x + offset or int(tags.get('source_y', -1)) != y):
                    raise ValueError('Existing TIFF lacks matching version and offsets: ' + str(path))
            else:
                if args.existing_tiles_only:
                    raise FileNotFoundError(path)
                with rasterio.open(source) as image:
                    values = image.read(1, window=Window(x + offset, y, 1024, 1024),
                                        boundless=True, fill_value=np.nan).astype('float32')
                path.parent.mkdir(parents=True, exist_ok=True)
                write_tif(path, values, tags=dict(version_id=args.source_version or task['inference_version'],
                                                source_x=x + offset, source_y=y), origin=(x + offset, y))
            if values.shape != (1024, 1024):
                raise ValueError('Incorrect tile dimensions: ' + str(path))
            arrays.append(values)
        destination = task['disparity']
        if destination.exists():
            cached = np.load(destination, mmap_mode='r')
            if cached.shape != (1024, 1024):
                raise ValueError('Invalid version-bound disparity cache: ' + str(destination))
        else:
            tensors = [torch.from_numpy(prepare_input(values, args.input_mode)).repeat(3, 1, 1)[None].to(args.device)
                       for values in arrays]
            padder = InputPadder(tensors[0].shape)
            left, right = padder.pad(*tensors)
            with torch.inference_mode():
                prediction = model(left, right, iters=args.valid_iters, test_mode=True)
            disparity = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
            disparity = disparity[-1] if isinstance(disparity, (tuple, list)) else disparity
            values = padder.unpad(disparity).squeeze().float().cpu().numpy() - shift
            temporary = destination.with_name(destination.name + '.partial.npy')
            np.save(temporary, values.astype('float32'))
            os.replace(temporary, destination)
        atomic_json(progress, dict(scene=scene, done=index + 1, total=len(tasks), shard=args.shard_index,
                                   num_shards=args.num_shards, version_id=args.source_version,
                                   input_mode=args.input_mode))
        print(scene, 'disparity', index + 1, '/', len(tasks), flush=True)


if __name__ == '__main__':
    main()
