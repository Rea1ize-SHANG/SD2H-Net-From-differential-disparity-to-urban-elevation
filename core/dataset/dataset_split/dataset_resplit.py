import os
import random
import shutil
from pathlib import Path

def build_mixed_dataset(dst_root, whu_stereo_root, extra_sources):
    """
    Core logic to merge datasets based on provided paths and configurations.
    """
    seed = 42
    random.seed(seed)
    
    # --- Step 1: Sync Base Dataset (WHUStereo) ---
    print(f"[*] Syncing base dataset from: {whu_stereo_root}")
    modes = ["train", "val", "test"]
    sub_types = ["left", "right", "disp"]
    
    for mode in modes:
        for stype in sub_types:
            src_dir = whu_stereo_root / mode / stype
            target_dir = dst_root / mode / stype
            if src_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
                for f in src_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, target_dir / f.name)

    # --- Step 2: Collect and Standardize Extra Data ---
    all_extra_groups = []
    for src in extra_sources:
        l_dir = src["root"] / src["sub"]["left"]
        if not l_dir.exists():
            print(f"[!] Warning: Path not found: {l_dir}")
            continue

        l_files = sorted([f.name for f in l_dir.iterdir() if f.is_file()])
        count = 0
        for l_name in l_files:
            # Generate matching filenames based on source-specific prefixes
            r_name = l_name.replace(src["prefix_l"], src["prefix_r"])
            d_name = l_name.replace(src["prefix_l"], src["prefix_d"])
            
            p_l = src["root"] / src["sub"]["left"] / l_name
            p_r = src["root"] / src["sub"]["right"] / r_name
            p_d = src["root"] / src["sub"]["disp"] / d_name

            if p_l.exists() and p_r.exists() and p_d.exists():
                # Parse location and ID for uniform naming
                parts = l_name.split('_')
                if src["type"] == "sside":
                    loc = parts[1] if len(parts) > 1 else "SSIDE"
                    id_stem = Path("_".join(parts[2:])).stem
                else: # laoshan style
                    loc = parts[0]
                    id_stem = Path("_".join(parts[2:])).stem

                all_extra_groups.append({
                    "paths": {"left": p_l, "right": p_r, "disp": p_d},
                    "dest_names": {
                        "left": f"{loc}_left_{id_stem}.tiff",
                        "right": f"{loc}_right_{id_stem}.tiff",
                        "disp": f"{loc}_disparity_{id_stem}.tiff"
                    }
                })
                count += 1
        print(f"[+] Source [{src['name']}]: Found {count} valid samples.")

    # --- Step 3: Split and Merge Extra Data ---
    random.shuffle(all_extra_groups)
    total_extra = len(all_extra_groups)
    
    # Split using ratio 1220 : 122 : 415
    ratio_sum = 1220 + 122 + 415
    train_end = int(total_extra * (1220 / ratio_sum))
    val_end = train_end + int(total_extra * (122 / ratio_sum))

    extra_splits = {
        "train": all_extra_groups[:train_end],
        "val": all_extra_groups[train_end:val_end],
        "test": all_extra_groups[val_end:]
    }

    print("[*] Merging extra samples into target splits...")
    for mode, groups in extra_splits.items():
        for group in groups:
            for key in ["left", "right", "disp"]:
                src_p = group["paths"][key]
                dst_n = group["dest_names"][key]
                target_dir = dst_root / mode / key
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_p, target_dir / dst_n)
        print(f"    -> {mode:<8} added {len(groups)} samples")

    print(f"\n[Success] New dataset built at: {dst_root}")

if __name__ == "__main__":
    # --- CONFIGURATION AREA ---
    # Destination path
    TARGET_ROOT = Path("/home/shangying/workspace/dataset/WHUMixDisp")
    
    # Base dataset path
    BASE_DATASET = Path("/home/shangying/workspace/dataset/WHUStereo/withGroundTruth")
    
    # Additional sources configuration
    ADDITIONAL_SOURCES = [
        {
            "name": "Laoshan",
            "root": Path("/home/shangying/workspace/dataset/WHUStereo/Expand/Laoshan"),
            "sub": {"left": "left", "right": "right", "disp": "disp"},
            "prefix_l": "LS_left_",
            "prefix_r": "LS_right_",
            "prefix_d": "LS_disparity_",
            "type": "laoshan"
        },
        {
            "name": "WHU-SSIDE",
            "root": Path("/home/shangying/workspace/dataset/WHU-SSIDE/valtest-all/valtest-A"),
            "sub": {"left": "ImageL", "right": "ImageR", "disp": "Disp"},
            "prefix_l": "ImgLGF7",
            "prefix_r": "ImgRGF7",
            "prefix_d": "DispGF7",
            "type": "sside"
        }
    ]
    # --------------------------

    build_mixed_dataset(TARGET_ROOT, BASE_DATASET, ADDITIONAL_SOURCES)