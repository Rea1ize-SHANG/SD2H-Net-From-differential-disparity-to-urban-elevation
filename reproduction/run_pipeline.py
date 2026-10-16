"""Launch the repository's existing reproduction stages with explicit paths."""
from pathlib import Path
import argparse,json,os,subprocess,sys

DEFAULT_CITIES=['HZ_049','HZ_050','HZ_057','HZ_DG_058','SZ_060','SZ_061','HK_062','SZ_DG_038','PRE_039']
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--stage',required=True,choices=['pretrain','precompute','prepare','train','evaluate','plot'])
parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--python',default=sys.executable)
parser.add_argument('--data-root',type=Path)
parser.add_argument('--cache-root',type=Path)
parser.add_argument('--stereo-checkpoint',type=Path,default=Path('checkpoints/pretrain_whusside_60ep_724/final.pth'))
parser.add_argument('--height-checkpoint',type=Path,default=Path('checkpoints/city_stratified_50_50/finetune_best_rmse.pth'))
parser.add_argument('--height-config',type=Path,default=Path('config/train_city_stratified_50_50.yaml'))
parser.add_argument('--config-name',default='train_city_stratified_50_50')
parser.add_argument('--output-root',type=Path,default=Path('outputs/reproduction'))
parser.add_argument('--gpu',default='0',help='Comma-separated visible GPU IDs')
parser.add_argument('--num-processes',type=int,default=1)
parser.add_argument('--batch-size',type=int,default=2)
parser.add_argument('--valid-iters',type=int,default=16)
parser.add_argument('--cities',nargs='+',default=DEFAULT_CITIES)
parser.add_argument('--save-vis',action='store_true')
parser.add_argument('--dry-run',action='store_true',help='Print command and environment without executing a stage')
args=parser.parse_args();root=args.repo_root.resolve()
if args.num_processes<1 or args.batch_size<1:parser.error('Process and batch counts must be positive.')
if args.stage in ['precompute','prepare','evaluate','plot'] and args.data_root is None:parser.error('--data-root is required for this stage.')
def path(p):
    return str((root/p).resolve()) if not p.is_absolute() else str(p.resolve())
env={'CUDA_VISIBLE_DEVICES':args.gpu,'PY':args.python}
if args.data_root is not None:
    env['DATA_ROOT']=path(args.data_root)
    cache=args.cache_root or args.data_root/'disp/sd2h_full9_abs_groundfix_whusside724_20260622'
    env['CACHE_ROOT']=path(cache)
env['OUT_ROOT']=path(args.output_root);env['GPU_ID']=args.gpu.split(',')[0]
if args.stage=='pretrain':
    command=[args.python,'-m','accelerate.commands.launch','--num_processes',str(args.num_processes),'--mixed_precision','bf16','train_pretrain.py']
elif args.stage=='precompute':
    command=[args.python,'precompute_unifiedbh_disp_delta.py','--data_path',env['DATA_ROOT'],'--restore_ckpt',path(args.stereo_checkpoint),'--output_root',env['CACHE_ROOT'],'--cities']+args.cities+['--valid_iters',str(args.valid_iters),'--batch_size','1','--mixed_precision','--skip_existing']
elif args.stage=='prepare':
    command=['bash','run_prepare_stereo_calibration_protocols.sh']
elif args.stage=='train':
    command=[args.python,'-m','accelerate.commands.launch','--num_processes',str(args.num_processes),'train_finetune.py','--config-name',args.config_name]
elif args.stage=='evaluate':
    command=[args.python,'evaluate_height.py','--config_yaml',path(args.height_config),'--restore_ckpt',path(args.height_checkpoint),'--data_path',env['DATA_ROOT'],'--eval_cities']+args.cities+['--output_directory',env['OUT_ROOT'],'--batch_size',str(args.batch_size),'--valid_iters','0','--mixed_precision','--no_bin_metrics']
    if args.save_vis:command.append('--save_vis')
else:
    command=['bash','run_plot_stereo_calibration_protocols.sh']
print(json.dumps({'stage':args.stage,'working_directory':str(root),'command':command,'environment':env,'dry_run':args.dry_run},indent=2),flush=True)
if not args.dry_run:
    # The launcher delegates all model and dataset logic to the original entrypoints.
    actual_env=os.environ.copy();actual_env.update(env)
    subprocess.run(command,cwd=root,env=actual_env,check=True)
