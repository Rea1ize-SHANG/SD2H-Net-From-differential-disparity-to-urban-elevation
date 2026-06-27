import os
import glob
import numpy as np
import skimage.io
from skimage.measure import label
from tqdm import tqdm

def process_city_labels(input_base_dir, output_base_dir):
    """
    Convert height-label maps in each city subfolder into instance-label maps.
    """
    # Collect city subfolders, e.g. ['ALL', 'HZ1', 'SZ'].
    city_folders = [f for f in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, f))]
    
    if not city_folders:
        print(f"[!] No subfolders found under {input_base_dir}.")
        return

    print(f"[*] Found {len(city_folders)} city folders. Starting conversion...")

    total_processed = 0
    total_skipped = 0

    for city in city_folders:
        # Build city-specific input and output paths.
        city_input_dir = os.path.join(input_base_dir, city)
        city_output_dir = os.path.join(output_base_dir, city)

        # Collect .tif or .tiff files in the current city folder.
        tif_files = glob.glob(os.path.join(city_input_dir, '*.tif'))
        tif_files.extend(glob.glob(os.path.join(city_input_dir, '*.tiff')))

        if not tif_files:
            print(f"   Skipping {city}: no .tif files found.")
            continue

        # Create the output city folder if needed.
        os.makedirs(city_output_dir, exist_ok=True)
        print(f"\n[*] Processing city: {city} ({len(tif_files)} images)")
        print(f"   Output path: {city_output_dir}")

        # Show conversion progress.
        for path in tqdm(tif_files, desc=city, unit="image"):
            try:
                # 1. Read the height label.
                height_map = skimage.io.imread(path)
                
                # Defensively keep only the first channel if a multi-channel file is encountered.
                if height_map.ndim > 2:
                    height_map = height_map[:, :, 0]
                
                # 2. Build a binary building mask from positive heights.
                binary_mask = (height_map > 0).astype(np.uint8)
                
                # 3. Connected-component analysis assigns disconnected buildings to separate IDs.
                # connectivity=2 means 8-neighborhood connectivity.
                instance_mask = label(binary_mask, connectivity=2)
                
                # 4. Save as uint16 TIFF.
                filename = os.path.basename(path)
                save_path = os.path.join(city_output_dir, filename)
                
                # check_contrast=False suppresses low-contrast warnings for label maps.
                skimage.io.imsave(save_path, instance_mask.astype(np.uint16), check_contrast=False)
                
                total_processed += 1
                
            except Exception as e:
                print(f"\n[!] Failed to read or process {filename}: {e}")
                total_skipped += 1

    print("\n" + "="*50)
    print("[*] Instance-label generation finished.")
    print(f"[*] Successfully processed: {total_processed} images")
    if total_skipped > 0:
        print(f"[!] Failed/skipped: {total_skipped} images")
    print("="*50 + "\n")

if __name__ == '__main__':
    # Root directories.
    INPUT_LABEL_DIR = '/home/shangying/workspace/dataset/UnifiedBH/overlap15/label/'
    OUTPUT_MASK_DIR = '/home/shangying/workspace/dataset/UnifiedBH/overlap15/mask/'
    
    # Ensure the mask root exists.
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    
    # Run conversion.
    process_city_labels(INPUT_LABEL_DIR, OUTPUT_MASK_DIR)
