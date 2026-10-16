import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import rpcm
import rpcm.rpc_model as rpc_model

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rpc_dsm import gf7_rpc_compat
from s2p import initialization, pointing_accuracy
from s2p.config import cfg
from rpc_dsm.naming import city_name


SCENES = ('038', '039', '049', '050', '057', '058', '060', '061', '062')
TILE_SIZE = 1024
TILE_STEP = 870
ROOT = Path('outputs/rpc_dsm')
DISP_ROOT = ROOT / 'tiles' / 'disp'
CROP_ROOT = ROOT / 'preprocess' / 'crop'
EPI_ROOT = ROOT / 'preprocess' / 'epipolar'
TIE_ROOT = Path('data/tie_points')


def pair_projection(ref_rpc, sec_rpc, lon, lat, height):
    ref_x, ref_y = ref_rpc.projection(lon, lat, height)
    sec_x, sec_y = sec_rpc.projection(lon, lat, height)
    return np.column_stack((ref_x, ref_y, sec_x, sec_y))


def localize_vectorized(rpc, col, row, height):
    ncol = (col - rpc.col_offset) / rpc.col_scale
    nrow = (row - rpc.row_offset) / rpc.row_scale
    nalt = (height - rpc.alt_offset) / rpc.alt_scale
    lon = np.full(ncol.shape, -1.0, dtype=np.float64)
    lat = np.full(ncol.shape, -1.0, dtype=np.float64)

    for iteration in range(40):
        pred_x = rpc_model.apply_rfm(rpc.col_num, rpc.col_den, lat, lon, nalt)
        pred_y = rpc_model.apply_rfm(rpc.row_num, rpc.row_den, lat, lon, nalt)
        error_x = ncol - pred_x
        error_y = nrow - pred_y
        if max(np.max(np.abs(error_x)), np.max(np.abs(error_y))) < 1e-9:
            break

        step = 2.0 if iteration == 0 else 0.1
        pred_x_lon = rpc_model.apply_rfm(rpc.col_num, rpc.col_den, lat, lon + step, nalt)
        pred_y_lon = rpc_model.apply_rfm(rpc.row_num, rpc.row_den, lat, lon + step, nalt)
        pred_x_lat = rpc_model.apply_rfm(rpc.col_num, rpc.col_den, lat + step, lon, nalt)
        pred_y_lat = rpc_model.apply_rfm(rpc.row_num, rpc.row_den, lat + step, lon, nalt)
        j11 = (pred_x_lon - pred_x) / step
        j12 = (pred_x_lat - pred_x) / step
        j21 = (pred_y_lon - pred_y) / step
        j22 = (pred_y_lat - pred_y) / step
        determinant = j11 * j22 - j12 * j21
        good = np.isfinite(determinant) & (np.abs(determinant) > 1e-12)
        delta_lon = np.zeros_like(lon)
        delta_lat = np.zeros_like(lat)
        delta_lon[good] = (error_x[good] * j22[good] - error_y[good] * j12[good]) / determinant[good]
        delta_lat[good] = (j11[good] * error_y[good] - j21[good] * error_x[good]) / determinant[good]
        lon += delta_lon
        lat += delta_lat

    return lon * rpc.lon_scale + rpc.lon_offset, lat * rpc.lat_scale + rpc.lat_offset


# RPC inversion uses the original-image coordinates recovered from the rectification homographies.
def solve_batch(ref_rpc, sec_rpc, left_xy, right_xy, residual_threshold):
    left_x, left_y = left_xy.T
    height = np.full(left_x.shape, ref_rpc.alt_offset, dtype=np.float64)
    lon, lat = localize_vectorized(ref_rpc, left_x, left_y, height)
    target = np.column_stack((left_xy, right_xy))
    parameter_scale = np.array((0.01, 0.01, 100.0), dtype=np.float64)
    derivative_step = np.array((1e-5, 1e-5, 1.0), dtype=np.float64)

    for _ in range(12):
        value = pair_projection(ref_rpc, sec_rpc, lon, lat, height)
        error = value - target
        jacobian = np.empty((len(lon), 4, 3), dtype=np.float64)
        for axis in range(3):
            plus = [lon.copy(), lat.copy(), height.copy()]
            minus = [lon.copy(), lat.copy(), height.copy()]
            plus[axis] += derivative_step[axis]
            minus[axis] -= derivative_step[axis]
            jacobian[:, :, axis] = (
                pair_projection(ref_rpc, sec_rpc, *plus)
                - pair_projection(ref_rpc, sec_rpc, *minus)
            ) / (2.0 * derivative_step[axis])

        scaled_jacobian = jacobian * parameter_scale[None, None, :]
        normal = np.einsum('nki,nkj->nij', scaled_jacobian, scaled_jacobian)
        rhs = np.einsum('nki,nk->ni', scaled_jacobian, error)
        normal += np.eye(3, dtype=np.float64)[None, :, :] * 1e-10
        try:
            delta = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            delta = np.einsum('nij,nj->ni', np.linalg.pinv(normal), rhs)

        lon -= delta[:, 0] * parameter_scale[0]
        lat -= delta[:, 1] * parameter_scale[1]
        height -= delta[:, 2] * parameter_scale[2]
        height = np.clip(height, -1000.0, 10000.0)
        if np.max(np.abs(delta)) < 1e-5:
            break

    residual = np.max(np.abs(pair_projection(ref_rpc, sec_rpc, lon, lat, height) - target), axis=1)
    valid = np.isfinite(height) & np.isfinite(residual) & (residual <= residual_threshold)
    return height, residual, valid


def solve_disparity_block(
    run,
    city,
    disparity,
    block_origin_x,
    block_origin_y,
    ref_rpc,
    sec_rpc,
    pointing_correction,
    batch_size,
    residual_threshold,
    progress_callback,
):
    disparity = disparity.astype(np.float64, copy=False)
    height = np.full(disparity.shape, np.nan, dtype=np.float32)
    residual = np.full(disparity.shape, np.nan, dtype=np.float32)
    row, col = np.where(np.isfinite(disparity))
    if not len(col):
        return height, residual

    scene = city[4:] if city.startswith('GF7_') else city
    ref_h = np.loadtxt(CROP_ROOT / scene / 'global_H_ref_crop.txt')
    sec_h = np.loadtxt(CROP_ROOT / scene / 'global_H_sec_crop.txt')
    ref_h_inv = np.linalg.inv(ref_h)
    sec_h_inv = np.linalg.inv(sec_h)

    for begin in range(0, len(col), batch_size):
        end = min(begin + batch_size, len(col))
        current_col = col[begin:end]
        current_row = row[begin:end]
        current_disp = disparity[current_row, current_col]
        global_col = current_col + block_origin_x
        global_row = current_row + block_origin_y
        ref_point = np.column_stack((global_col, global_row, np.ones(end - begin)))
        sec_point = np.column_stack((global_col - current_disp, global_row, np.ones(end - begin)))
        ref_point = ref_point @ ref_h_inv.T
        sec_point = sec_point @ sec_h_inv.T
        ref_point = ref_point[:, :2] / ref_point[:, 2:]
        sec_point = sec_point[:, :2] / sec_point[:, 2:]
        sec_point = np.column_stack((sec_point, np.ones(end - begin))) @ pointing_correction.T
        sec_point = sec_point[:, :2] / sec_point[:, 2:]
        solved_height, solved_residual, good = solve_batch(
            ref_rpc, sec_rpc, ref_point, sec_point, residual_threshold
        )
        height[current_row[good], current_col[good]] = solved_height[good].astype(np.float32)
        residual[current_row, current_col] = solved_residual.astype(np.float32)
        progress_callback(end, len(col), float(np.nanmedian(solved_residual)))

    return height, residual


def pointing_metrics(ref_rpc, sec_rpc, points, correction):
    corrected = np.column_stack((points[:, 2:4], np.ones(len(points)))) @ correction.T
    corrected = corrected[:, :2] / corrected[:, 2:]
    _, before, _ = solve_batch(ref_rpc, sec_rpc, points[:, :2], points[:, 2:4], np.inf)
    _, after, _ = solve_batch(ref_rpc, sec_rpc, points[:, :2], corrected, np.inf)

    def summarize(values):
        return {
            'median_px': float(np.median(values)),
            'p95_px': float(np.percentile(values, 95)),
            'maximum_px': float(np.max(values)),
        }

    return {'before': summarize(before), 'after': summarize(after)}


def tile_indices(tile):
    stem = tile.stem.replace('_disp', '')
    _, tile_x, tile_y = stem.rsplit('_', 2)
    return int(tile_x), int(tile_y)


def tile_positions(tiles, width, height):
    entries = []
    for tile in tiles:
        index_x, index_y = tile_indices(tile)
        x = min(index_x * TILE_STEP, width - TILE_SIZE)
        y = min(index_y * TILE_STEP, height - TILE_SIZE)
        entries.append((tile, index_x, index_y, x, y))
    return entries


def discover_disparities(disp_root, city):
    paths = {path.name: path for path in (disp_root / city).glob('*_disp.npy')}
    dataset = disp_root.parent.parent
    manifest = dataset / 'metadata' / 'samples.json'
    if disp_root == dataset / 'tiles' / 'disp' and manifest.exists():
        for sample in json.loads(manifest.read_text()):
            if not sample['tile'].startswith(city + '_'):
                continue
            path = dataset / sample['paths']['disp']
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.name in paths:
                raise ValueError('Duplicate disparity sample: ' + path.name)
            paths[path.name] = path
    return sorted(paths.values())


def feather_axis(position, positions, size):
    weight = np.ones(size, dtype=np.float32)
    index = positions.index(position)
    if index > 0:
        overlap = positions[index - 1] + TILE_SIZE - position
        if overlap > 0:
            ramp = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
            weight[:overlap] = np.minimum(weight[:overlap], ramp)
    if index + 1 < len(positions):
        overlap = position + TILE_SIZE - positions[index + 1]
        if overlap > 0:
            ramp = (np.arange(overlap, dtype=np.float32)[::-1] + 0.5) / overlap
            weight[-overlap:] = np.minimum(weight[-overlap:], ramp)
    return weight


def fusion_weights(entries, width, height):
    x_positions = sorted({entry[3] for entry in entries})
    y_positions = sorted({entry[4] for entry in entries})
    x_weights = {x: feather_axis(x, x_positions, min(TILE_SIZE, width - x)) for x in x_positions}
    y_weights = {y: feather_axis(y, y_positions, min(TILE_SIZE, height - y)) for y in y_positions}
    return x_weights, y_weights


def fuse_block(entries, x_weights, y_weights, block_x, block_y, block_width, block_height):
    disparity_sum = np.zeros((block_height, block_width), dtype=np.float32)
    weight_sum = np.zeros((block_height, block_width), dtype=np.float32)
    block_right = block_x + block_width
    block_bottom = block_y + block_height
    for tile, _, _, tile_x, tile_y in entries:
        left = max(block_x, tile_x)
        top = max(block_y, tile_y)
        right = min(block_right, tile_x + TILE_SIZE)
        bottom = min(block_bottom, tile_y + TILE_SIZE)
        if left >= right or top >= bottom:
            continue
        source_x = slice(left - tile_x, right - tile_x)
        source_y = slice(top - tile_y, bottom - tile_y)
        target_x = slice(left - block_x, right - block_x)
        target_y = slice(top - block_y, bottom - block_y)
        disparity = np.load(tile, mmap_mode='r')[source_y, source_x]
        weight = y_weights[tile_y][source_y, None] * x_weights[tile_x][None, source_x]
        valid = np.isfinite(disparity)
        sum_window = disparity_sum[target_y, target_x]
        weight_window = weight_sum[target_y, target_x]
        sum_window[valid] += disparity[valid] * weight[valid]
        weight_window[valid] += weight[valid]
    fused = np.full((block_height, block_width), np.nan, dtype=np.float64)
    covered = weight_sum > 0
    fused[covered] = disparity_sum[covered] / weight_sum[covered]
    return fused


def atomic_progress(path, data):
    temporary = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    temporary.write_text(json.dumps(data, ensure_ascii=True))
    os.replace(temporary, path)


def read_checkpoint(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main():
    global CROP_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(ROOT / 'dsm'))
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32768)
    parser.add_argument('--residual-threshold', type=float, default=3.0)
    parser.add_argument('--disp-root', type=Path, default=DISP_ROOT)
    parser.add_argument('--crop-root', type=Path, default=CROP_ROOT)
    parser.add_argument('--epi-root', type=Path, default=EPI_ROOT)
    parser.add_argument('--run-map', type=Path, help='JSON mapping scene IDs to rectification run directories')
    parser.add_argument('--tie-root', type=Path, default=TIE_ROOT)
    parser.add_argument('--rpc-only', action='store_true', help='Use uncorrected RPC when no scene control points exist')
    parser.add_argument('--scenes', nargs='+', default=list(SCENES))
    args = parser.parse_args()
    CROP_ROOT = args.crop_root
    run_map = json.loads(args.run_map.read_text()) if args.run_map else {}
    if args.stride != 1:
        raise ValueError('Production DSM requires --stride 1 for per-pixel inversion')

    output_root = Path(args.out)
    disp_root = args.disp_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scene_work = {}
    for scene in args.scenes:
        info = json.loads((CROP_ROOT / scene / 'common_crop.json').read_text())
        width, height = info['window'][2:4]
        city = city_name(scene)
        tiles = discover_disparities(disp_root, city)
        blocks = ((width + TILE_SIZE - 1) // TILE_SIZE) * ((height + TILE_SIZE - 1) // TILE_SIZE)
        scene_work[scene] = blocks
    total_tiles = sum(scene_work.values())
    done_tiles = 0
    progress_path = output_root / 'progress.json'

    for scene in args.scenes:
        city = city_name(scene)
        crop = CROP_ROOT / scene
        if scene in run_map:
            run = Path(run_map[scene])
        elif (args.epi_root / scene / 'input_config.json').is_file():
            run = args.epi_root / scene
        elif args.rpc_only:
            run = args.epi_root / scene
        else:
            runs = list(args.epi_root.glob('*_' + scene + '_L1A_01/full_scene_*'))
            if len(runs) != 1:
                raise ValueError('Expected one rectification run for scene %s; provide --run-map' % scene)
            run = runs[0]
        initialization.build_cfg(json.loads((run / 'input_config.json').read_text()))
        ref_rpc = cfg['images'][0]['rpcm']
        sec_rpc = cfg['images'][1]['rpcm']
        if args.rpc_only:
            points = None
            pointing_correction = np.eye(3)
        else:
            points = np.loadtxt(args.tie_root / (str(int(scene)) + '.pts'), comments=';', ndmin=2)
            with rasterio.open(cfg['images'][0]['img']) as source:
                raw_width, raw_height = source.width, source.height
            pointing_correction = pointing_accuracy.local_translation(
                ref_rpc,
                sec_rpc,
                0,
                0,
                raw_width,
                raw_height,
                points,
            )
        with rasterio.open(crop / 'rectified_ref_common.tif') as source:
            profile = source.profile.copy()
            width, scene_height = source.width, source.height
        profile.update(
            width=width,
            height=scene_height,
            dtype='float32',
            count=1,
            nodata=np.nan,
            compress='deflate',
            BIGTIFF='YES',
        )
        scene_dir = output_root / city
        scene_dir.mkdir(exist_ok=True)
        control_residual = None if points is None else pointing_metrics(ref_rpc, sec_rpc, points, pointing_correction)
        pointing_report = {
            'scene': scene,
            'matrix': pointing_correction.tolist(),
            'control_residual': control_residual,
            'pass': None if points is None else control_residual['after']['p95_px'] <= 2.0,
            'mode': 'uncorrected_rpc_no_ground_control' if points is None else 'control_point_corrected',
        }
        atomic_progress(scene_dir / 'rpc_pointing_correction.json', pointing_report)
        if pointing_report['pass'] is False:
            raise RuntimeError('RPC pointing correction P95 residual exceeds 2 px')
        final = scene_dir / (city + '_DSM.tif')
        partial = scene_dir / (city + '_DSM.partial.tif')
        checkpoint_path = scene_dir / 'production_checkpoint.json'
        if final.exists():
            raise FileExistsError('Output already exists: %s' % final)
        checkpoint = read_checkpoint(checkpoint_path)
        if checkpoint is None and partial.exists():
            partial.unlink()
        if not partial.exists():
            with rasterio.open(partial, 'w', **profile):
                pass

        tiles = discover_disparities(disp_root, city)
        entries = tile_positions(tiles, width, scene_height)
        x_weights, y_weights = fusion_weights(entries, width, scene_height)
        blocks_x = (width + TILE_SIZE - 1) // TILE_SIZE
        blocks_y = (scene_height + TILE_SIZE - 1) // TILE_SIZE
        blocks_in_scene = blocks_x * blocks_y
        completed_blocks = 0
        if checkpoint is not None:
            expected = {
                'scene': scene,
                'width': width,
                'height': scene_height,
                'blocks_in_scene': blocks_in_scene,
            }
            if any(checkpoint.get(key) != value for key, value in expected.items()):
                raise ValueError('Checkpoint does not match current scene geometry')
            completed_blocks = int(checkpoint.get('completed_blocks', 0))
            if not 0 <= completed_blocks <= blocks_in_scene:
                raise ValueError('Invalid completed block count in checkpoint')
        done_tiles += completed_blocks
        block_index = 0
        for block_y in range(blocks_y):
            for block_x in range(blocks_x):
                block_index += 1
                if block_index <= completed_blocks:
                    continue
                x = block_x * TILE_SIZE
                y = block_y * TILE_SIZE
                tile_width = min(TILE_SIZE, width - x)
                tile_height = min(TILE_SIZE, scene_height - y)
                disparity = fuse_block(
                    entries,
                    x_weights,
                    y_weights,
                    x,
                    y,
                    tile_width,
                    tile_height,
                )

                def report(pixel_done, pixel_total, median_residual):
                    atomic_progress(progress_path, {
                        'scene': scene,
                        'stage': 'FUSE_AND_RPC',
                        'tile': block_index,
                        'tiles_in_scene': blocks_in_scene,
                        'done_tiles': done_tiles,
                        'total_tiles': total_tiles,
                        'tile_pixels': pixel_done,
                        'tile_valid_pixels': pixel_total,
                        'median_batch_residual_px': median_residual,
                    })

                solved_height, solved_residual = solve_disparity_block(
                    run,
                    city,
                    disparity,
                    x,
                    y,
                    ref_rpc,
                    sec_rpc,
                    pointing_correction,
                    args.batch_size,
                    args.residual_threshold,
                    report,
                )
                with rasterio.open(partial, 'r+') as target:
                    target.write(solved_height, 1, window=Window(x, y, tile_width, tile_height))
                done_tiles += 1
                atomic_progress(checkpoint_path, {
                    'scene': scene,
                    'width': width,
                    'height': scene_height,
                    'blocks_in_scene': blocks_in_scene,
                    'completed_blocks': block_index,
                })
                atomic_progress(progress_path, {
                    'scene': scene,
                    'stage': 'FUSE_AND_RPC',
                    'tile': block_index,
                    'tiles_in_scene': blocks_in_scene,
                    'done_tiles': done_tiles,
                    'total_tiles': total_tiles,
                    'tile_pixels': int(np.isfinite(solved_residual).sum()),
                    'valid_pixels': int(np.isfinite(solved_height).sum()),
                    'median_residual_px': float(np.nanmedian(solved_residual)) if np.isfinite(solved_residual).any() else None,
                })
                print(scene, block_index, blocks_in_scene, flush=True)
        os.replace(partial, final)
        checkpoint_path.unlink(missing_ok=True)
        stage = 'DSM_WRITTEN' if os.environ.get('GF7_PIPELINE') == '1' else 'DONE'
        atomic_progress(progress_path, {'scene': scene, 'stage': stage, 'done_tiles': done_tiles, 'total_tiles': total_tiles})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        failure_path = Path(os.environ.get(
            'GF7_DSM_OUTPUT',
            'outputs/rpc_dsm/dsm',
        )) / 'progress.json'
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_progress(failure_path, {
            'stage': 'FAILED',
            'error': '%s: %s' % (type(error).__name__, error),
        })
        traceback.print_exc()
        raise
