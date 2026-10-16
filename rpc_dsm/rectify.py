import argparse,datetime,json,os,sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import map_coordinates
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rpc_dsm import gf7_rpc_compat
import s2p
from s2p import common, initialization, rectification
from s2p.config import cfg

def write_global_blocks(output, source, homography, width, height):
    width, height = int(width), int(height)
    inverse = np.linalg.inv(homography)
    temporary = str(output) + '.partial.tif'
    total = ((width + 1023) // 1024) * ((height + 1023) // 1024)
    done = 0
    with rasterio.open(source) as src, rasterio.open(
        temporary, 'w', driver='GTiff', width=width, height=height,
        count=1, dtype='float32', nodata=float('nan'), tiled=True,
        blockxsize=256, blockysize=256, compress='deflate', BIGTIFF='YES'
    ) as dst:
        for row in range(0, height, 1024):
            for col in range(0, width, 1024):
                block_height, block_width = min(1024, height-row), min(1024, width-col)
                grid_row, grid_col = np.mgrid[row:row+block_height, col:col+block_width]
                projected = inverse @ np.stack((grid_col.ravel(), grid_row.ravel(), np.ones(grid_row.size)))
                source_col = projected[0] / projected[2]
                source_row = projected[1] / projected[2]
                valid = (np.isfinite(source_col) & np.isfinite(source_row) &
                         (source_col >= 0) & (source_col <= src.width-1) &
                         (source_row >= 0) & (source_row <= src.height-1))
                block = np.full(grid_row.size, np.nan, dtype=np.float32)
                if valid.any():
                    left = max(0, int(np.floor(source_col[valid].min()))-1)
                    top = max(0, int(np.floor(source_row[valid].min()))-1)
                    right = min(src.width, int(np.ceil(source_col[valid].max()))+2)
                    bottom = min(src.height, int(np.ceil(source_row[valid].max()))+2)
                    values = src.read(1, window=Window(left, top, right-left, bottom-top),
                                      masked=True).astype(np.float32).filled(np.nan)
                    block[valid] = map_coordinates(values, [source_row[valid]-top, source_col[valid]-left],
                                                   order=1, mode='constant', cval=np.nan, prefilter=False)
                dst.write(block.reshape(block_height, block_width), 1,
                          window=Window(col, row, block_width, block_height))
                done += 1
                if done == 1 or done % 25 == 0 or done == total:
                    print('GLOBAL_BLOCK %s %d/%d' % (Path(output).name, done, total), flush=True)
    os.replace(temporary, output)

def main(config_path, skip_tiles=False):
    user=json.loads(Path(config_path).read_text()); user['full_img']=True
    initialization.build_cfg(user); initialization.make_dirs()
    common.print_elapsed_time.t0=datetime.datetime.now(); common.print_elapsed_time.t1=common.print_elapsed_time.t0
    out=Path(cfg['out_dir']); out.mkdir(parents=True,exist_ok=True)
    with rasterio.open(cfg['images'][0]['img']) as ds: w,h=ds.width,ds.height
    H1p=out/'global_H_ref.txt'; H2p=out/'global_H_sec.txt'; full1=out/'rectified_ref_global.tif'; full2=out/'rectified_sec_global.tif'
    if not (H1p.exists() and H2p.exists() and full1.exists() and full2.exists()):
        print('PHASE GLOBAL_RECTIFICATION 0', flush=True)
        common.image_apply_homography = write_global_blocks
        H1,H2,disp_min,disp_max=rectification.rectify_pair(cfg['images'][0]['img'],cfg['images'][1]['img'],cfg['images'][0]['rpcm'],cfg['images'][1]['rpcm'],0,0,w,h,str(full1),str(full2),np.eye(3),None,method=cfg['rectification_method'],hmargin=cfg['horizontal_margin'],vmargin=cfg['vertical_margin'])
        np.savetxt(H1p,H1); np.savetxt(H2p,H2)
        print('PHASE GLOBAL_RECTIFICATION 100', flush=True)
    if skip_tiles:
        with rasterio.open(full1) as a, rasterio.open(full2) as b:
            if a.shape != b.shape:
                raise RuntimeError('global rectified shape mismatch')
            print('FULL_SCENE_DONE',a.width,a.height,flush=True)
        return
    with rasterio.open(full1) as a, rasterio.open(full2) as b:
        if (a.width,a.height)!=(b.width,b.height): raise RuntimeError('global rectified shape mismatch')
        prof=a.profile.copy(); tw=1024; th=1024; tiles=out/'tiles'; tiles.mkdir(exist_ok=True)
        total=((a.height+th-1)//th)*((a.width+tw-1)//tw); done=0
        print('PHASE TILE_WRITE 0/%d' % total, flush=True)
        for y in range(0,a.height,th):
            for x in range(0,a.width,tw):
                hh=min(th,a.height-y); ww=min(tw,a.width-x); td=tiles/f'row_{y:07d}_height_{hh:04d}'/f'col_{x:07d}_width_{ww:04d}'/'pair_1'; td.mkdir(parents=True,exist_ok=True)
                p=prof.copy(); p.update(width=ww,height=hh,transform=rasterio.Affine.identity(),compress='deflate')
                for side,src in [('ref',a),('sec',b)]:
                    with rasterio.open(td/f'rectified_{side}.tif','w',**p) as dst: dst.write(src.read(1,window=((y,y+hh),(x,x+ww))),1)
                done += 1
                if done == 1 or done % 25 == 0 or done == total: print('PHASE TILE_WRITE %d/%d' % (done,total), flush=True)
        print('FULL_SCENE_DONE',a.width,a.height)
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('--skip-tiles', action='store_true')
    args = parser.parse_args()
    main(args.config_path, skip_tiles=args.skip_tiles)
