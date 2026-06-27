import cv2
import numpy as np
from PIL import Image, ImageEnhance
from .transform import *
from core.utils.geometry import PinholeEulerAnglesToRotationMatrix

class Augmentor:
    def __init__(
            self,
            image_height=384,
            image_width=512,
            max_disp=150,
            scale_min=0.6,
            scale_max=1.0,
            seed=0,
            camera_type='pinhole',
            albumentations_aug=True,
            white_balance_aug=True,
            rgb_noise_aug=True,
            motion_blur_aug=True,
            local_blur_aug=True,
            global_blur_aug=True,
            chromatic_aug=True,
            camera_motion_aug=True
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.max_disp = max_disp
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rng = np.random.RandomState(seed)
        self.camera_type = camera_type

        self.albumentations_aug = albumentations_aug
        self.white_balance_aug = white_balance_aug
        self.rgb_noise_aug = rgb_noise_aug
        self.motion_blur_aug = motion_blur_aug
        self.local_blur_aug = local_blur_aug
        self.global_blur_aug = global_blur_aug
        self.chromatic_aug = chromatic_aug
        self.camera_motion_aug = camera_motion_aug

        # Hardcoded Pinhole Camera Intrinsic Matrix
        intrinsic = (778, 778, 488, 681)
        self.K_mat = np.array(
            [[intrinsic[0], 0.0, intrinsic[2]],
            [0.0, intrinsic[1], intrinsic[3]],
            [0.0, 0.0, 1.0]])

    def chromatic_augmentation(self, img):
        """Adjusts brightness, contrast, and gamma of the image."""
        random_brightness = np.random.uniform(0.8, 1.2)
        random_contrast = np.random.uniform(0.8, 1.2)
        random_gamma = np.random.uniform(0.8, 1.2)

        img = Image.fromarray(img)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random_brightness)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random_contrast)

        # Gamma correction using lookup table for speed
        gamma_map = [
                        255 * 1.0 * pow(ele / 255.0, random_gamma) for ele in range(256)
                    ] * 3
        img = img.point(gamma_map)

        img_ = np.array(img)
        return img_

    def padding(self, data, pad_len):
        """Concatenates the beginning part of the data to the end (Vertical Padding)."""
        bott_ext = data[0: pad_len]
        return np.concatenate((data, bott_ext), axis=0)

    def __call__(self, dataset_name, left_img, right_img, left_disp, error=None, wire_mask=None):

        # 1. Random Crop
        h, w = left_img.shape[:2]
        ch, cw = self.image_height, self.image_width

        assert ch <= h and cw <= w, (left_img.shape, h, w, ch, cw)
        offset_x = np.random.randint(w - cw + 1)
        offset_y = np.random.randint(h - ch + 1)

        left_img = left_img[offset_y: offset_y + ch, offset_x: offset_x + cw]
        right_img = right_img[offset_y: offset_y + ch, offset_x: offset_x + cw]
        left_disp = left_disp[offset_y: offset_y + ch, offset_x: offset_x + cw]

        if error is not None:
            error = error[offset_y: offset_y + ch, offset_x: offset_x + cw]
        if wire_mask is not None:
            wire_mask = wire_mask[offset_y: offset_y + ch, offset_x: offset_x + cw]

        right_img_ori = right_img.copy()

        # 2. Disparity Mask Generation
        resize_scale = 1.0
        if error is not None:
            # Filter based on error threshold
            error_mask = error < 0.5
            disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0) & error_mask
        else:
            disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0)

        # 3. Local Blur Augmentation
        if self.local_blur_aug and self.rng.binomial(1, 0.2):
            # Left image local blur
            p_l = self.rng.binomial(1, 0.5)
            if p_l < 0.5:
                brightness = self.rng.uniform(low=-40, high=40)
                mask_l = self.rng.choice([None, 'local_mask'])
                if mask_l == 'local_mask':
                    mask_l = mask_ge(left_img.shape, self.rng, weights=[0.5, 0.5])
                left_img, _ = image_blur_mask(left_img, self.rng, mask_l, brightness)

            # Right image local blur
            p_r = self.rng.binomial(1, 0.5)
            if p_r < 0.5:
                mask_r = self.rng.choice([None, 'local_mask'])
                brightness = self.rng.uniform(low=-40, high=40)
                if mask_r == 'local_mask':
                    mask_r = mask_ge(left_img.shape, self.rng, weights=[0.5, 0.5])
                right_img, _ = image_blur_mask(right_img, self.rng, mask_r, brightness)

        # 4. RGB Noise Augmentation
        if self.rgb_noise_aug and self.rng.binomial(1, 0.5):
            sigma = self.rng.uniform(low=1, high=5)
            left_img = RGB_noise_aug(left_img, sigma, self.rng)
            right_img = RGB_noise_aug(right_img, sigma, self.rng)

        # 5. Chromatic Augmentation (Color Jitter)
        if self.chromatic_aug and self.rng.binomial(1, 0.4):
            left_img = self.chromatic_augmentation(left_img)
            right_img = self.chromatic_augmentation(right_img)

        # 6. White Balance Augmentation
        if self.white_balance_aug and self.rng.binomial(1, 0.5):
            random_ratio_L = self.rng.uniform(-0.3, 0.3)
            random_ratio_R = self.rng.uniform(-0.15, 0.15) + random_ratio_L
            left_img = white_balance_augmentation(left_img, ratio=random_ratio_L)
            right_img = white_balance_augmentation(right_img, ratio=random_ratio_R)

        # 7. Global Blur Augmentation (Defocus Simulation)
        if self.global_blur_aug and self.rng.binomial(1, 0.2):
            # Left image defocus
            p_l = self.rng.binomial(1, 0.5)
            if p_l < 0.5:
                kernel_size = self.rng.randint(2, 7) * 2 + 1
                left_img, _ = image_blur_all(left_img, (kernel_size, kernel_size))

            # Right image defocus
            p_r = self.rng.binomial(1, 0.5)
            if p_r < 0.5:
                kernel_size = self.rng.randint(2, 7) * 2 + 1
                right_img, _ = image_blur_all(right_img, (kernel_size, kernel_size))

        # 8. Spatial Augmentation (Camera Motion & Misalignment)
        if self.camera_motion_aug and self.rng.binomial(1, 0.2):
            sigma = 0.25
            mu = 0  
            ag_0, ag_1, ag_2 = np.fmod(np.random.normal(mu, sigma, size=3), 3)

            pixel = 2 
            px = self.rng.uniform(-pixel, pixel) # Vertical shift
            ag = np.deg2rad([ag_0, ag_1, ag_2])  # Random Euler angles

            # Update intrinsic matrix with vertical offset (px)
            self.K_mat_new = self.K_mat.copy()
            self.K_mat_new[1, 2] += px

            # Apply homography matrix H = K_new * R * K_inv
            R_mat = PinholeEulerAnglesToRotationMatrix(ag)
            H_mat = self.K_mat_new.dot(R_mat).dot(np.linalg.inv(self.K_mat))
            H_mat = H_mat / H_mat[2][2]
            
            # Warp right image based on homography
            right_img = cv2.warpPerspective(
                right_img, H_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )

        # 9. Random Occlusion (Simulate objects blocking the view)
        if self.rng.binomial(1, 0.5):
            sx = int(self.rng.uniform(50, 100))
            sy = int(self.rng.uniform(50, 100))
            cx = int(self.rng.uniform(sx, right_img.shape[0] - sx))
            cy = int(self.rng.uniform(sy, right_img.shape[1] - sy))
            right_img[cx - sx: cx + sx, cy - sy: cy + sy] = np.mean(
                np.mean(right_img, 0), 0
            )[np.newaxis, np.newaxis]

        # 10. Potential Color Mask (Unused in current return)
        # color_mask: filter out overexposed and underexposed pixels

        return left_img, right_img, right_img_ori, left_disp, disp_mask, wire_mask