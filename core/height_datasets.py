import os
import os.path as osp
from glob import glob
import hashlib
import json
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import skimage.io
from skimage.measure import label as connected_components
from tqdm import tqdm

# ==============================================================================
# GF-7 city-specific height conversion coefficients (k).
# ==============================================================================
GF7_CITY_K_MAPPING = {
    "HK_062": 1.219918,
    "HZ_049": 1.228083,
    "HZ_050": 1.228095,
    "HZ_057": 1.217361,
    "HZ_DG": 1.217375,
    "HZ_DG_058": 1.217375,
    "PRE_039": 1.311503,
    "SZ_060": 1.217382,
    "SZ_061": 1.218209,
    "SZ_DG_038": 1.311482
}

CITY_DIR_ALIASES = {
    "HZ_DG": {
        "left": "HZ_DG",
        "right": "HZ_DG_058",
        "label": "HZ_DG_058",
        "mask": "HZ_DG_058",
    },
    "HZ_DG_058": {
        "left": "HZ_DG_058",
        "right": "HZ_DG_058",
        "label": "HZ_DG_058",
        "mask": "HZ_DG_058",
    },
}


def resolve_city_dir(city, kind):
    return CITY_DIR_ALIASES.get(city, {}).get(kind, city)


# ================= Image and Label Readers =================
def _safe_float32(img, max_abs=None):
    img = np.asarray(img)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    if max_abs is not None:
        img = np.where(np.abs(img) <= max_abs, img, 0.0)
    else:
        img = np.where(np.abs(img) < np.finfo(np.float32).max, img, 0.0)
    return img.astype(np.float32, copy=False)


def read_image_as_float32(path):
    """ Read image and duplicate channels to 3 if panchromatic. """
    img = skimage.io.imread(path)
    
    if img.ndim == 2:
        img = np.tile(img[..., np.newaxis], (1, 1, 3))
    elif img.shape[2] > 3:
        img = img[..., :3]
        
    return _safe_float32(img)

def read_single_channel_as_float32(path):
    """ Read single-channel height label or mask. """
    img = skimage.io.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    
    return _safe_float32(img, max_abs=1.0e6)


def read_npy_as_float32(path):
    """Read a cached single-channel numpy map."""
    arr = np.load(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return _safe_float32(arr, max_abs=1.0e6)


def label_to_instance_mask(height_map, threshold=0.0, connectivity=2):
    """Generate an instance mask from a height label when no mask file exists."""
    building = np.isfinite(height_map) & (height_map > threshold)
    if not building.any():
        return np.zeros_like(height_map, dtype=np.float32)
    inst = connected_components(building.astype(np.uint8), connectivity=connectivity)
    return inst.astype(np.float32)


def remove_small_instances(mask, min_pixels=0):
    """Drop tiny connected components while preserving the original instance IDs."""
    min_pixels = int(min_pixels or 0)
    if min_pixels <= 1:
        return mask
    inst = np.asarray(mask)
    ids, counts = np.unique(inst[inst > 0].astype(np.int64, copy=False), return_counts=True)
    if ids.size == 0:
        return mask
    small_ids = ids[counts < min_pixels]
    if small_ids.size == 0:
        return mask
    filtered = inst.copy()
    filtered[np.isin(filtered.astype(np.int64, copy=False), small_ids)] = 0.0
    return filtered


def output_core_name(path):
    base = osp.splitext(osp.basename(path))[0]
    original = base
    for prefix in ["ImgLGF7_", "ImgRGF7_", "DispGF7_"]:
        if base.startswith(prefix):
            return base[len(prefix):]
    for token in ["_left_", "_right_", "_label_", "_mask_", "_disparity_", "_disp_"]:
        base = base.replace(token, "_")
    for suffix in ["_left", "_right", "_label", "_mask", "_disparity", "_disp"]:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    for prefix in ["left_", "right_", "label_", "mask_", "disparity_", "disp_"]:
        if base.startswith(prefix):
            base = base[len(prefix):]
    return base or original


def delta_filename_for_subdir(subdir, core_name):
    """Return the cached delta filename for the selected cache variant."""
    subdir_l = str(subdir).lower()
    if "abs" in subdir_l:
        return f"DeltaDispAbs_{core_name}.npy"
    if "reverse" in subdir_l:
        return f"DeltaDispReverse_{core_name}.npy"
    if "signed" in subdir_l:
        return f"DeltaDispSigned_{core_name}.npy"
    return f"DeltaDisp_{core_name}.npy"


# ================= Unified Building Height Dataset =================
class UnifiedBHDataset(data.Dataset):
    def __init__(
        self,
        aug_params=None,
        data_path='',
        cities=None,
        is_train=False,
        mask_from_label=True,
        mask_label_threshold=0.0,
        mask_connectivity=2,
        strict_mask=False,
        mask_min_instance_pixels=0,
        precomputed_disp_delta_root='',
        precomputed_disp_subdir='disp_npy',
        precomputed_delta_subdir='delta_relu_npy',
        precomputed_ground_subdir='',
        return_precomputed_ground=False,
        require_precomputed_disp_delta=False,
        refresh_sampler_weights=False,
        sample_weight_top_percent=5.0,
        sample_weight_height_scale=10.0,
        sample_weight_max=20.0,
        sample_weight_cache_key_override='',
        use_weighted_sampler=True,
        instance_filter_manifest='',
        drop_empty_after_instance_filter=False,
    ):
        self.data_path = data_path
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        self.is_train = is_train
        self.mask_from_label = mask_from_label
        self.mask_label_threshold = float(mask_label_threshold)
        self.mask_connectivity = int(mask_connectivity)
        self.strict_mask = bool(strict_mask)
        self.mask_min_instance_pixels = int(mask_min_instance_pixels or 0)
        self.precomputed_disp_delta_root = str(precomputed_disp_delta_root or '')
        self.precomputed_disp_subdir = str(precomputed_disp_subdir)
        self.precomputed_delta_subdir = str(precomputed_delta_subdir)
        self.precomputed_ground_subdir = str(precomputed_ground_subdir or '')
        self.return_precomputed_ground = bool(return_precomputed_ground)
        self.require_precomputed_disp_delta = bool(require_precomputed_disp_delta)
        self.refresh_sampler_weights = bool(refresh_sampler_weights)
        self.sample_weight_top_percent = float(sample_weight_top_percent)
        self.sample_weight_height_scale = float(sample_weight_height_scale)
        self.sample_weight_max = float(sample_weight_max)
        self.sample_weight_cache_key_override = str(sample_weight_cache_key_override or '')
        self.use_weighted_sampler = bool(use_weighted_sampler)
        self.instance_filter_manifest = str(instance_filter_manifest or '')
        self.instance_filter, self.valid_tiles = self._load_instance_filter(self.instance_filter_manifest)
        self.drop_empty_after_instance_filter = bool(drop_empty_after_instance_filter)
        
        _cities = cities if cities is not None and len(cities) > 0 else ['ALL']
        if 'ALL' in _cities:
            left_base_dir = osp.join(self.data_path, 'left')
            if osp.exists(left_base_dir):
                self.cities = sorted([d for d in os.listdir(left_base_dir) if osp.isdir(osp.join(left_base_dir, d))])
            else:
                self.cities = []
        else:
            self.cities = _cities

        self.image_list = []
        self.label_list = []
        self.mask_list = []
        
        for city in self.cities:
            left_dir = osp.join(self.data_path, 'left', resolve_city_dir(city, 'left'))
            right_dir = osp.join(self.data_path, 'right', resolve_city_dir(city, 'right'))
            label_dir = osp.join(self.data_path, 'label', resolve_city_dir(city, 'label')) 
            mask_dir = osp.join(self.data_path, 'mask', resolve_city_dir(city, 'mask'))

            left_files, right_files, label_files, mask_files = [], [], [], []
            
            for ext in ('*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.png'):
                left_files.extend(glob(osp.join(left_dir, ext)))
                right_files.extend(glob(osp.join(right_dir, ext)))
                label_files.extend(glob(osp.join(label_dir, ext)))
                mask_files.extend(glob(osp.join(mask_dir, ext)))

            def extract_key(filename):
                base = osp.basename(filename)
                base, _ = osp.splitext(base)
                base = base.replace('_left', '').replace('_right', '').replace('_label', '').replace('_mask', '')
                return base

            left_map = {extract_key(f): f for f in left_files}
            right_map = {extract_key(f): f for f in right_files}
            label_map = {extract_key(f): f for f in label_files}
            mask_map = {extract_key(f): f for f in mask_files} 

            base_keys = set(left_map.keys()) & set(right_map.keys()) & set(label_map.keys())
            if self.strict_mask:
                keys = base_keys & set(mask_map.keys())
                mask_source = 'file_strict' if mask_map else 'missing'
            elif self.mask_from_label and not self.strict_mask:
                keys = base_keys
                mask_source = 'file_or_label_connected_components' if mask_map else 'label_connected_components'
            elif mask_map:
                keys = base_keys & set(mask_map.keys())
                mask_source = 'file'
            else:
                keys = set()
                mask_source = 'missing'

            print(
                f"[*] City {city}: left={len(left_map)} right={len(right_map)} "
                f"label={len(label_map)} mask={len(mask_map)} matched={len(keys)} "
                f"mask_source={mask_source}"
            )

            for k in sorted(keys):
                self.image_list.append([left_map[k], right_map[k]])
                self.label_list.append(label_map[k])
                self.mask_list.append(mask_map.get(k))

        self.init_seed = False
        print(f"[*] UnifiedBHDataset | Cities: {self.cities} | Samples: {len(self.image_list)}")
        if self.precomputed_disp_delta_root:
            print(
                f"[*] Precomputed disparity/delta root: {self.precomputed_disp_delta_root} "
                f"(disp={self.precomputed_disp_subdir}, delta={self.precomputed_delta_subdir}, "
                f"required={self.require_precomputed_disp_delta})"
            )
        if self.instance_filter_manifest:
            print(f"[*] Instance filter manifest: {self.instance_filter_manifest} | tiles={len(self.instance_filter)}")
        if self.drop_empty_after_instance_filter:
            self._drop_empty_tiles_after_filter()

        # Initialize sampler weights only for the training set.
        self.sample_weights = None
        if self.is_train and self.use_weighted_sampler and len(self.label_list) > 0:
            self._compute_weights()

    def _load_instance_filter(self, manifest_path):
        if not manifest_path:
            return {}, set()
        if not osp.exists(manifest_path):
            raise FileNotFoundError(f"Instance filter manifest not found: {manifest_path}")
        with open(manifest_path, 'r') as f:
            payload = json.load(f)
        raw = payload.get('ignore_instances', payload)
        valid_tiles = {str(x) for x in payload.get('valid_tiles', [])}
        filters = {}
        for key, ids in raw.items():
            clean_ids = sorted({int(x) for x in ids if int(x) > 0})
            if clean_ids:
                filters[str(key)] = clean_ids
        return filters, valid_tiles

    def _apply_instance_filter(self, mask, left_img_path):
        if not self.instance_filter:
            return mask
        city_folder = os.path.normpath(left_img_path).split(os.sep)[-2]
        core_name = output_core_name(left_img_path)
        ignore_ids = self.instance_filter.get(f"{city_folder}/{core_name}")
        if not ignore_ids:
            ignore_ids = self.instance_filter.get(core_name)
        if not ignore_ids:
            return mask
        ignore = np.isin(mask.astype(np.int64, copy=False), np.asarray(ignore_ids, dtype=np.int64))
        if ignore.any():
            mask = mask.copy()
            mask[ignore] = 0.0
        return mask

    def _load_filtered_mask(self, index):
        height = read_single_channel_as_float32(self.label_list[index])
        mask_path = self.mask_list[index]
        if mask_path is not None:
            mask = read_single_channel_as_float32(mask_path)
        else:
            mask = label_to_instance_mask(
                height,
                threshold=self.mask_label_threshold,
                connectivity=self.mask_connectivity,
            )
        mask = remove_small_instances(mask, self.mask_min_instance_pixels)
        mask = self._apply_instance_filter(mask, self.image_list[index][0])
        return height, mask

    def _drop_empty_tiles_after_filter(self):
        if not self.image_list:
            return
        kept_images, kept_labels, kept_masks = [], [], []
        dropped = 0
        if self.valid_tiles:
            for idx in range(len(self.image_list)):
                city_folder = os.path.normpath(self.image_list[idx][0]).split(os.sep)[-2]
                core_name = output_core_name(self.image_list[idx][0])
                if f"{city_folder}/{core_name}" in self.valid_tiles or core_name in self.valid_tiles:
                    kept_images.append(self.image_list[idx])
                    kept_labels.append(self.label_list[idx])
                    kept_masks.append(self.mask_list[idx])
                else:
                    dropped += 1
            self.image_list = kept_images
            self.label_list = kept_labels
            self.mask_list = kept_masks
            print(f"[*] Dropped empty tiles via manifest valid_tiles: {dropped} | Remaining: {len(self.image_list)}")
            return

        for idx in tqdm(range(len(self.image_list)), desc="Filtering Empty Tiles"):
            _, mask = self._load_filtered_mask(idx)
            if np.any(mask > 0):
                kept_images.append(self.image_list[idx])
                kept_labels.append(self.label_list[idx])
                kept_masks.append(self.mask_list[idx])
            else:
                dropped += 1
        self.image_list = kept_images
        self.label_list = kept_labels
        self.mask_list = kept_masks
        print(f"[*] Dropped empty tiles after instance filter: {dropped} | Remaining: {len(self.image_list)}")

    def _compute_weights(self):
        """
        Compute a global sampling weight for each image.
        The default rule uses the mean height of the tallest pixels and maps it
        to W = max(1.0, H / scale), with an upper cap.
        """
        # Cache weights to avoid repeatedly scanning all label maps.
        if self.sample_weight_cache_key_override:
            cache_key = self.sample_weight_cache_key_override
        else:
            cache_key_src = '|'.join([
                '|'.join(sorted(self.cities)),
                f"top{self.sample_weight_top_percent:g}",
                f"scale{self.sample_weight_height_scale:g}",
                f"max{self.sample_weight_max:g}",
                f"manifest{osp.abspath(self.instance_filter_manifest) if self.instance_filter_manifest else 'none'}",
                f"dropempty{int(self.drop_empty_after_instance_filter)}",
            ])
            cache_key = hashlib.md5(cache_key_src.encode('utf-8')).hexdigest()[:12]
        cache_name = f"sampler_weights_{cache_key}_{len(self.label_list)}.npy"
        cache_path = osp.join(self.data_path, cache_name)

        if osp.exists(cache_path) and not self.refresh_sampler_weights:
            print(f"[*] Loading cached sampler weights: {cache_path}")
            self.sample_weights = np.load(cache_path).tolist()
            return

        print("[*] Computing image-level sampler weights for WeightedRandomSampler...")
        weights = []
        for idx in tqdm(range(len(self.label_list)), desc="Calculating Weights"):
            h_map, mask = self._load_filtered_mask(idx)
            
            # Use only building pixels that remain after instance filtering.
            valid_h = h_map[(h_map > 0) & (mask > 0)]
            
            if len(valid_h) == 0:
                # Background-only images receive the minimum weight.
                weight = 1.0
            else:
                # Use the top-height fraction, which is more robust than the maximum.
                valid_h_sorted = np.sort(valid_h)[::-1]
                top_fraction = max(0.001, self.sample_weight_top_percent / 100.0)
                top_count = max(1, int(len(valid_h_sorted) * top_fraction))
                top_h = np.mean(valid_h_sorted[:top_count])
                
                # Map height to sampling weight and apply the configured cap.
                scale = max(self.sample_weight_height_scale, 1e-6)
                weight = min(self.sample_weight_max, max(1.0, top_h / scale))
                
            weights.append(weight)
        
        self.sample_weights = weights
        # Save cache.
        np.save(cache_path, np.array(weights))
        print(f"[*] Sampler weights saved to: {cache_path}")

    def _get_k_value_from_path(self, filepath):
        path_parts = os.path.normpath(filepath).split(os.sep)
        city_folder = path_parts[-2]
        if city_folder in GF7_CITY_K_MAPPING:
            return GF7_CITY_K_MAPPING[city_folder]
        raise ValueError(
            f"[Fatal Error] Missing K value for city folder '{city_folder}' "
            f"(file path: {filepath}). Please update GF7_CITY_K_MAPPING."
        )

    def _get_precomputed_paths(self, left_img_path):
        if not self.precomputed_disp_delta_root:
            return None, None, None
        city_folder = os.path.normpath(left_img_path).split(os.sep)[-2]
        core_name = output_core_name(left_img_path)
        disp_path = osp.join(
            self.precomputed_disp_delta_root,
            self.precomputed_disp_subdir,
            city_folder,
            f"Disp_{core_name}.npy",
        )
        delta_path = osp.join(
            self.precomputed_disp_delta_root,
            self.precomputed_delta_subdir,
            city_folder,
            delta_filename_for_subdir(self.precomputed_delta_subdir, core_name),
        )
        ground_path = None
        if self.precomputed_ground_subdir:
            ground_path = osp.join(
                self.precomputed_disp_delta_root,
                self.precomputed_ground_subdir,
                city_folder,
                f"LocalGroundDisp_{core_name}.npy",
            )
        return disp_path, delta_path, ground_path

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        
        left_img_path = self.image_list[index][0]

        img1 = read_image_as_float32(left_img_path)
        img2 = read_image_as_float32(self.image_list[index][1])
        height = read_single_channel_as_float32(self.label_list[index])
        mask_path = self.mask_list[index]
        if mask_path is not None:
            mask = read_single_channel_as_float32(mask_path)
        else:
            mask = label_to_instance_mask(
                height,
                threshold=self.mask_label_threshold,
                connectivity=self.mask_connectivity,
            )
        mask = remove_small_instances(mask, self.mask_min_instance_pixels)
        mask = self._apply_instance_filter(mask, left_img_path)

        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        height = torch.from_numpy(height).unsqueeze(0)
        precomputed_disp = torch.zeros_like(mask)
        precomputed_delta = torch.zeros_like(mask)
        precomputed_ground = torch.zeros_like(mask)
        disp_path, delta_path, ground_path = self._get_precomputed_paths(left_img_path)
        if disp_path is not None and delta_path is not None:
            if osp.exists(disp_path) and osp.exists(delta_path):
                precomputed_disp = torch.from_numpy(read_npy_as_float32(disp_path)).unsqueeze(0)
                precomputed_delta = torch.from_numpy(read_npy_as_float32(delta_path)).unsqueeze(0)
            elif self.require_precomputed_disp_delta:
                missing = [p for p in (disp_path, delta_path) if not osp.exists(p)]
                raise FileNotFoundError(f"Missing precomputed disparity/delta files: {missing}")
            if self.return_precomputed_ground and ground_path is not None:
                if osp.exists(ground_path):
                    precomputed_ground = torch.from_numpy(read_npy_as_float32(ground_path)).unsqueeze(0)
                elif self.require_precomputed_disp_delta:
                    raise FileNotFoundError(f"Missing precomputed local-ground disparity file: {ground_path}")

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)
            mask = F.pad(mask, [padW]*2 + [padH]*2)
            height = F.pad(height, [padW]*2 + [padH]*2)
            precomputed_disp = F.pad(precomputed_disp, [padW]*2 + [padH]*2)
            precomputed_delta = F.pad(precomputed_delta, [padW]*2 + [padH]*2)
            precomputed_ground = F.pad(precomputed_ground, [padW]*2 + [padH]*2)
        
        info = self.image_list[index] + [self.label_list[index], mask_path if mask_path is not None else 'AUTO_MASK_FROM_LABEL']
        
        k_value = self._get_k_value_from_path(left_img_path)
        k_tensor = torch.tensor([k_value], dtype=torch.float32)
        
        if self.return_precomputed_ground:
            return info, img1, img2, precomputed_disp, precomputed_delta, precomputed_ground, mask, height, k_tensor
        return info, img1, img2, precomputed_disp, precomputed_delta, mask, height, k_tensor


# ================= Dataloader Interfaces =================
def fetch_dataloader(args):
    aug_params = {}                
    if hasattr(args, "img_pad") and args.img_pad is not None:
        aug_params["img_pad"] = args.img_pad

    train_data_path = getattr(args, 'train_data_path', '')
    train_cities = getattr(args, 'train_cities', ['ALL']) 

    # is_train=True enables sampler-weight computation.
    return UnifiedBHDataset(
        aug_params=aug_params,
        data_path=train_data_path,
        cities=train_cities,
        is_train=True,
        mask_from_label=getattr(args, 'mask_from_label', True),
        mask_label_threshold=getattr(args, 'mask_label_threshold', 0.0),
        mask_connectivity=getattr(args, 'mask_connectivity', 2),
        strict_mask=getattr(args, 'strict_mask', False),
        mask_min_instance_pixels=getattr(args, 'mask_min_instance_pixels', 0),
        precomputed_disp_delta_root=getattr(args, 'precomputed_disp_delta_root', ''),
        precomputed_disp_subdir=getattr(args, 'precomputed_disp_subdir', 'disp_npy'),
        precomputed_delta_subdir=getattr(args, 'precomputed_delta_subdir', 'delta_relu_npy'),
        precomputed_ground_subdir=getattr(args, 'precomputed_ground_subdir', ''),
        return_precomputed_ground=getattr(args, 'return_precomputed_ground', False),
        require_precomputed_disp_delta=getattr(args, 'require_precomputed_disp_delta', False),
        refresh_sampler_weights=getattr(args, 'refresh_sampler_weights', False),
        sample_weight_top_percent=getattr(args, 'sample_weight_top_percent', 5.0),
        sample_weight_height_scale=getattr(args, 'sample_weight_height_scale', 10.0),
        sample_weight_max=getattr(args, 'sample_weight_max', 20.0),
        sample_weight_cache_key_override=getattr(args, 'sample_weight_cache_key_override', ''),
        use_weighted_sampler=getattr(args, 'use_weighted_sampler', True),
        instance_filter_manifest=(
            getattr(args, 'train_instance_filter_manifest', getattr(args, 'instance_filter_manifest', ''))
            if getattr(args, 'apply_instance_filter_train', True) else ''
        ),
        drop_empty_after_instance_filter=getattr(args, 'drop_empty_after_instance_filter', False),
    )

def fetch_val_dataloader(args):
    val_data_path = getattr(args, 'val_data_path', '')
    val_cities = getattr(args, 'val_cities', ['ALL'])
    
    return UnifiedBHDataset(
        aug_params={},
        data_path=val_data_path,
        cities=val_cities,
        is_train=False,
        mask_from_label=getattr(args, 'mask_from_label', True),
        mask_label_threshold=getattr(args, 'mask_label_threshold', 0.0),
        mask_connectivity=getattr(args, 'mask_connectivity', 2),
        strict_mask=getattr(args, 'strict_mask', False),
        mask_min_instance_pixels=getattr(args, 'mask_min_instance_pixels', 0),
        precomputed_disp_delta_root=getattr(args, 'precomputed_disp_delta_root', ''),
        precomputed_disp_subdir=getattr(args, 'precomputed_disp_subdir', 'disp_npy'),
        precomputed_delta_subdir=getattr(args, 'precomputed_delta_subdir', 'delta_relu_npy'),
        precomputed_ground_subdir=getattr(args, 'precomputed_ground_subdir', ''),
        return_precomputed_ground=getattr(args, 'return_precomputed_ground', False),
        require_precomputed_disp_delta=getattr(args, 'require_precomputed_disp_delta', False),
        refresh_sampler_weights=getattr(args, 'refresh_sampler_weights', False),
        sample_weight_top_percent=getattr(args, 'sample_weight_top_percent', 5.0),
        sample_weight_height_scale=getattr(args, 'sample_weight_height_scale', 10.0),
        sample_weight_max=getattr(args, 'sample_weight_max', 20.0),
        sample_weight_cache_key_override=getattr(args, 'sample_weight_cache_key_override', ''),
        instance_filter_manifest=(
            getattr(args, 'val_instance_filter_manifest', getattr(args, 'instance_filter_manifest', ''))
            if getattr(args, 'apply_instance_filter_val', True) else ''
        ),
        drop_empty_after_instance_filter=getattr(args, 'drop_empty_after_instance_filter', False),
    )
