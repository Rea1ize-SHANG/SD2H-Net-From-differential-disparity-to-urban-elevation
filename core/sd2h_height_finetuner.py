import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

# Import the stereo backbone.
from core.sd2h import SD2H


def _cfg_value(args, name, default):
    return getattr(args, name, default) if args is not None else default


def _cfg_list(args, name, default):
    value = _cfg_value(args, name, default)
    return list(value) if value is not None else list(default)


_PRECOMPUTED_DELTA_SOURCES = ('precomputed', 'cached', 'cache', 'label_ring')
_CACHED_DISP_LEARNED_SOURCES = (
    'cached_disp_learned_base_plane',
    'precomputed_disp_learned_base_plane',
    'cached_disp_learned_delta',
    'precomputed_disp_learned_delta',
    'precomputed_delta_residual',
    'cached_delta_residual',
)
_DELTA_RESIDUAL_SOURCES = (
    'cached_disp_learned_delta',
    'precomputed_disp_learned_delta',
    'precomputed_delta_residual',
    'cached_delta_residual',
)


def _bin_centers_from_edges(edges, num_bins, device, dtype):
    centers = []
    prev = 0.0
    for edge in edges:
        centers.append((prev + float(edge)) * 0.5)
        prev = float(edge)
    centers.append(prev + max(20.0, prev * 0.25))
    while len(centers) < num_bins:
        centers.append(centers[-1] + 50.0)
    return torch.tensor(centers[:num_bins], device=device, dtype=dtype)


def _activate_height(height_raw, args=None):
    mode = str(_cfg_value(args, 'height_activation', 'relu')).lower()
    if mode == 'softplus':
        beta = float(_cfg_value(args, 'height_softplus_beta', 2.0))
        return F.softplus(height_raw, beta=max(beta, 1e-3))
    if mode in ('leaky_relu', 'leaky'):
        slope = float(_cfg_value(args, 'height_leaky_relu_slope', 0.05))
        return F.leaky_relu(height_raw, negative_slope=slope)
    if mode in ('identity', 'linear', 'none'):
        return height_raw
    return F.relu(height_raw)


def _safe_gate_power(gate, power, eps=1.0e-6):
    """Avoid infinite gradients from d(x**p)/dx at x=0 when 0 < p < 1."""
    gate = torch.clamp(torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if abs(float(power) - 1.0) <= 1.0e-6:
        return gate
    powered = torch.pow(torch.clamp(gate, min=eps), float(power))
    return torch.where(gate > 0.0, powered, torch.zeros_like(powered))


def _norm_groups(channels, max_groups=8):
    for groups in (max_groups, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _last_conv2d(module):
    if isinstance(module, nn.Conv2d):
        return module
    if isinstance(module, nn.Sequential):
        for layer in reversed(module):
            if isinstance(layer, nn.Conv2d):
                return layer
    return None


def _zero_last_conv2d(module):
    conv = _last_conv2d(module)
    if conv is not None:
        nn.init.zeros_(conv.weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)


def _init_last_conv2d_bias(module, values):
    conv = _last_conv2d(module)
    if conv is None or conv.bias is None:
        return
    bias_values = torch.tensor(values, dtype=conv.bias.dtype)
    if bias_values.numel() != conv.bias.numel():
        return
    with torch.no_grad():
        conv.bias.copy_(bias_values.to(device=conv.bias.device))


class BasePlaneEstimator(nn.Module):
    """
    Large-receptive-field base-plane estimator.
    Stacked dilated convolutions expand the receptive field while GroupNorm
    avoids BatchNorm instability for small batches.
    """
    def __init__(self):
        super(BasePlaneEstimator, self).__init__()
        # 1. Basic feature extraction. Input: RGB + disparity = 4 channels.
        self.stem = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=7, padding=3),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True)
        )
        
        # 2. Dilated bottleneck for receptive-field expansion.
        self.layer1 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=2, dilation=2), nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=4, dilation=4), nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=8, dilation=8), nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=16, dilation=16), nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        
        # 3. Feature fusion and decoding.
        self.decoder = nn.Sequential(
            nn.Conv2d(64 + 32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    @staticmethod
    def _maybe_checkpoint(module, x, enabled):
        if enabled and x.is_cuda:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, img, disp, use_checkpoint=False):
        x = torch.cat([img, disp], dim=1)
        
        # Extract base features and keep a shortcut.
        feat_stem = self._maybe_checkpoint(self.stem, x, use_checkpoint)
        
        # Expand the receptive field.
        out = self._maybe_checkpoint(self.layer1, feat_stem, use_checkpoint)
        out = self._maybe_checkpoint(self.layer2, out, use_checkpoint)
        out = self._maybe_checkpoint(self.layer3, out, use_checkpoint)
        out = self._maybe_checkpoint(self.layer4, out, use_checkpoint)
        
        # Combine deep context with shallow local details.
        feat_fuse = torch.cat([out, feat_stem], dim=1)
        
        # Regress the local base-plane disparity.
        d_ground = self._maybe_checkpoint(self.decoder, feat_fuse, use_checkpoint)
        return d_ground


class HybridHeightHead(nn.Module):
    """
    Height head with lightweight nonlinearity for long-tailed residuals.
    """
    def __init__(self, args=None):
        super(HybridHeightHead, self).__init__()
        input_channels = 2
        if bool(_cfg_value(args, 'height_use_image_features', False)):
            input_channels += int(_cfg_value(args, 'height_image_feature_channels', 1))
        if bool(_cfg_value(args, 'height_use_mask_feature', False)):
            input_channels += 1
        if bool(_cfg_value(args, 'height_use_area_feature', False)):
            input_channels += 1
        if bool(_cfg_value(args, 'height_use_ground_feature', False)):
            input_channels += 1
        if bool(_cfg_value(args, 'height_use_signed_delta_feature', False)):
            input_channels += 1
        if bool(_cfg_value(args, 'height_use_physical_feature', False)):
            input_channels += 1

        head_channels = int(_cfg_value(args, 'height_head_channels', 64))
        mid_channels = int(_cfg_value(args, 'height_head_mid_channels', 32))
        head_channels = max(head_channels, 8)
        mid_channels = max(mid_channels, 8)
        head_groups = _norm_groups(head_channels)
        mid_groups = _norm_groups(mid_channels)
        self.refinement_mode = str(_cfg_value(args, 'height_refinement_mode', 'standard')).lower()
        self.use_gated_residual = self.refinement_mode in (
            'gated_residual',
            'soft_bin_gated_residual',
            'bin_gated_residual',
        )
        self.shared_trunk = bool(_cfg_value(args, 'height_shared_trunk', False))
        self.use_residual_head = bool(_cfg_value(args, 'height_use_residual_head', True))
        self.use_direct_head = bool(_cfg_value(args, 'height_use_direct_head', True))

        if self.use_gated_residual:
            blocks = max(1, int(_cfg_value(args, 'height_refinement_blocks', 4)))
            trunk_layers = [
                nn.Conv2d(input_channels, head_channels, kernel_size=3, padding=1),
                nn.GroupNorm(head_groups, head_channels),
                nn.ReLU(inplace=True),
            ]
            for idx in range(blocks):
                dilation = 1 if idx % 2 == 0 else 2
                trunk_layers.extend(
                    [
                        nn.Conv2d(
                            head_channels,
                            head_channels,
                            kernel_size=3,
                            padding=dilation,
                            dilation=dilation,
                        ),
                        nn.GroupNorm(head_groups, head_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
            trunk_layers.extend(
                [
                    nn.Conv2d(head_channels, mid_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(mid_groups, mid_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            self.trunk = nn.Sequential(*trunk_layers)
            self.class_conv = nn.Conv2d(mid_channels, 8, kernel_size=1)
            self.res_conv = nn.Conv2d(mid_channels, 8, kernel_size=1)
            self.correction_conv = None
            self.gate_conv = nn.Conv2d(mid_channels, 1, kernel_size=1)
            self.direct_conv = None
            nn.init.zeros_(self.res_conv.weight)
            nn.init.zeros_(self.res_conv.bias)
            nn.init.zeros_(self.gate_conv.weight)
            nn.init.constant_(self.gate_conv.bias, float(_cfg_value(args, 'height_residual_gate_init', -1.5)))
        elif self.shared_trunk:
            blocks = max(1, int(_cfg_value(args, 'height_refinement_blocks', 2)))
            trunk_layers = [
                nn.Conv2d(input_channels, head_channels, kernel_size=3, padding=1),
                nn.GroupNorm(head_groups, head_channels),
                nn.ReLU(inplace=True),
            ]
            for idx in range(blocks):
                dilation = 1 if idx % 2 == 0 else 2
                trunk_layers.extend(
                    [
                        nn.Conv2d(
                            head_channels,
                            head_channels,
                            kernel_size=3,
                            padding=dilation,
                            dilation=dilation,
                        ),
                        nn.GroupNorm(head_groups, head_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
            trunk_layers.extend(
                [
                    nn.Conv2d(head_channels, mid_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(mid_groups, mid_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            self.trunk = nn.Sequential(*trunk_layers)
            self.class_conv = nn.Conv2d(mid_channels, 8, kernel_size=1)
            self.res_conv = nn.Conv2d(mid_channels, 8, kernel_size=1) if self.use_residual_head else None
            self.correction_conv = nn.Conv2d(mid_channels, 1, kernel_size=1)
            self.gate_conv = None
            self.direct_conv = nn.Conv2d(mid_channels, 1, kernel_size=1) if self.use_direct_head else None
            nn.init.zeros_(self.correction_conv.weight)
            nn.init.zeros_(self.correction_conv.bias)
        else:
            # Deeper classification branch.
            self.class_conv = nn.Sequential(
                nn.Conv2d(input_channels, head_channels, kernel_size=3, padding=1),
                nn.GroupNorm(head_groups, head_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, 8, kernel_size=1)
            )
            
            # Deeper residual branch.
            self.res_conv = (
                nn.Sequential(
                    nn.Conv2d(input_channels, head_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(head_groups, head_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_channels, mid_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, 8, kernel_size=1)
                )
                if self.use_residual_head else None
            )

            # Zero-init correction branch: the model starts as the physical baseline,
            # then learns a signed height correction from noisy labels.
            self.correction_conv = nn.Sequential(
                nn.Conv2d(input_channels, head_channels, kernel_size=3, padding=1),
                nn.GroupNorm(head_groups, head_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, 1, kernel_size=1)
            )
            nn.init.zeros_(self.correction_conv[-1].weight)
            nn.init.zeros_(self.correction_conv[-1].bias)
            self.gate_conv = None

            self.direct_conv = (
                nn.Sequential(
                    nn.Conv2d(input_channels, head_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(head_groups, head_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_channels, mid_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, 1, kernel_size=1)
                )
                if self.use_direct_head else None
            )

        if bool(_cfg_value(args, 'height_zero_init_residual_head', False)) and self.res_conv is not None:
            _zero_last_conv2d(self.res_conv)
        if bool(_cfg_value(args, 'height_init_class_bias_low', False)):
            low_logit = float(_cfg_value(args, 'height_init_low_bin_logit', 3.0))
            other_logit = float(_cfg_value(args, 'height_init_other_bin_logit', -3.0))
            _init_last_conv2d_bias(self.class_conv, [low_logit] + [other_logit] * 7)

    def forward(
        self,
        disp_est,
        delta_d,
        k_tensor,
        args=None,
        extra_features=None,
        area_maps=None,
        physical_affine=None,
    ):
        k_tensor = k_tensor.view(-1, 1, 1, 1)
        
        # Physical height mapping H = K * delta_d with clipping for numerical stability.
        height_max = float(_cfg_value(args, 'height_output_clip', 320.0))
        height_phy = torch.clamp(
            torch.nan_to_num(k_tensor.float() * delta_d.float(), nan=0.0, posinf=height_max, neginf=0.0),
            min=0.0,
            max=height_max,
        ).to(dtype=delta_d.dtype)
        height_phy_safe = height_phy.float()
        if physical_affine is None:
            calibration_scale = torch.as_tensor(
                float(_cfg_value(args, 'height_physical_scale', 1.0)),
                device=height_phy.device,
                dtype=height_phy.dtype,
            )
            calibration_bias = torch.as_tensor(
                float(_cfg_value(args, 'height_physical_bias', 0.0)),
                device=height_phy.device,
                dtype=height_phy.dtype,
            )
        else:
            calibration_scale, calibration_bias = physical_affine
            calibration_scale = calibration_scale.to(device=height_phy.device, dtype=height_phy.dtype)
            calibration_bias = calibration_bias.to(device=height_phy.device, dtype=height_phy.dtype)
        height_calibrated_raw = calibration_scale * height_phy + calibration_bias
        base_mode = str(_cfg_value(args, 'height_base_mode', 'affine')).lower()
        poly_feature_base_modes = (
            'poly_area',
            'polyarea',
            'polynomial_area',
            'instance_feature_linear',
            'instance_feature_huber',
            'instance_feature_soft_blend',
            'instance_feature_blend',
            'instance_feature_dual_blend',
            'inst_feature_dual_blend',
            'inst_feature_blend',
            'inst_feature_linear',
            'inst_feature_huber',
        )
        if base_mode in poly_feature_base_modes:
            coeffs = _cfg_list(
                args,
                'height_poly_area_coeffs',
                [-0.72766, 0.04371, -0.00066, 0.39669, -15.96603, 2.10138, 0.05303, -0.51393, 0.00017, 40.2475],
            )
            if len(coeffs) != 10:
                raise ValueError("height_poly_area_coeffs must contain 10 coefficients.")
            x_clip = float(_cfg_value(args, 'height_poly_area_x_clip', 120.0))
            x_safe = height_phy_safe.float()
            x = torch.clamp(x_safe, max=x_clip) if x_clip > 0 else x_safe
            if area_maps is None:
                log_area = torch.zeros_like(x)
                sqrt_area = torch.zeros_like(x)
            else:
                log_area = area_maps[:, 0:1].to(dtype=torch.float32, device=x.device)
                sqrt_area = area_maps[:, 1:2].to(dtype=torch.float32, device=x.device)
            coeff_t = torch.as_tensor(coeffs, device=x.device, dtype=torch.float32)
            x2 = x * x
            height_calibrated_raw = (
                coeff_t[0] * x
                + coeff_t[1] * x2
                + coeff_t[2] * x2 * x
                + coeff_t[3] * torch.sqrt(torch.clamp(x, min=0.0) + 1e-6)
                + coeff_t[4] * log_area
                + coeff_t[5] * log_area * log_area
                + coeff_t[6] * x * log_area
                + coeff_t[7] * sqrt_area
                + coeff_t[8] * x * sqrt_area
                + coeff_t[9]
            )
            height_calibrated_raw = torch.nan_to_num(height_calibrated_raw, nan=0.0, posinf=320.0, neginf=0.0)
            height_calibrated_raw = torch.clamp(height_calibrated_raw, min=-20.0, max=320.0).to(dtype=height_phy.dtype)
        height_calibrated = _activate_height(torch.nan_to_num(height_calibrated_raw, nan=0.0), args=args)

        if bool(_cfg_value(args, 'height_skip_aux_head', False)):
            height = torch.clamp(
                torch.nan_to_num(height_calibrated.float(), nan=0.0, posinf=height_max, neginf=0.0),
                min=0.0,
                max=height_max,
            ).to(dtype=height_phy.dtype)
            zeros = torch.zeros_like(height, dtype=torch.float32)
            logits = zeros.expand(-1, 8, -1, -1)
            return {
                'height': height,
                'height_physical': height_phy,
                'height_base': height,
                'height_calibrated': height,
                'height_aux': height.detach() * 0.0,
                'height_direct': height.detach() * 0.0,
                'height_correction': zeros,
                'height_signed_correction': zeros,
                'height_high_boost': zeros,
                'height_high_boost_raw': zeros,
                'height_high_gate': zeros,
                'height_aux_raise': zeros,
                'height_gated_residual': zeros,
                'height_residual_gate': zeros,
                'logits': logits,
                'residuals': logits,
            }

        feat_parts = [disp_est, delta_d]
        if extra_features is not None:
            feat_parts.append(extra_features)
        feat = torch.cat(feat_parts, dim=1)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=255.0, neginf=-255.0)
        if self.shared_trunk or self.use_gated_residual:
            use_head_checkpoint = bool(_cfg_value(args, 'height_head_gradient_checkpoint', False)) and self.training
            feat = checkpoint(self.trunk, feat, use_reentrant=False) if use_head_checkpoint else self.trunk(feat)
        feat = torch.clamp(torch.nan_to_num(feat, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
        
        logits = torch.clamp(
            torch.nan_to_num(self.class_conv(feat).float(), nan=0.0, posinf=30.0, neginf=-30.0),
            -30.0,
            30.0,
        )
        residuals = self.res_conv(feat) if self.res_conv is not None else torch.zeros_like(logits)
        residuals = torch.nan_to_num(residuals.float(), nan=0.0, posinf=320.0, neginf=-320.0)
        correction = self.correction_conv(feat) if self.correction_conv is not None else torch.zeros_like(logits[:, :1])
        correction = torch.nan_to_num(correction.float(), nan=0.0, posinf=320.0, neginf=-320.0)
        num_bins = logits.shape[1]
        bin_edges = _cfg_list(args, 'height_bin_edges', [12, 24, 30, 50, 100, 150, 300])
        bin_centers = _bin_centers_from_edges(bin_edges, num_bins, logits.device, logits.dtype).view(1, num_bins, 1, 1)
        residual_clip = float(_cfg_value(args, 'height_residual_clip', 120.0))
        if residual_clip > 0:
            residuals_for_height = residual_clip * torch.tanh(residuals / residual_clip)
        else:
            residuals_for_height = residuals
        height_aux = torch.sum(torch.softmax(logits.float(), dim=1).to(residuals_for_height.dtype) * (bin_centers + residuals_for_height), dim=1, keepdim=True)
        height_aux = F.relu(torch.nan_to_num(height_aux, nan=0.0, posinf=320.0, neginf=0.0))
        soft_bin_residual = torch.sum(
            torch.softmax(logits.float(), dim=1).to(residuals_for_height.dtype) * residuals_for_height,
            dim=1,
            keepdim=True,
        )
        residual_gate = torch.ones_like(soft_bin_residual)
        if self.gate_conv is not None:
            gate_logits = torch.clamp(
                torch.nan_to_num(self.gate_conv(feat).float(), nan=0.0, posinf=20.0, neginf=-20.0),
                -20.0,
                20.0,
            )
            residual_gate = torch.sigmoid(gate_logits).to(dtype=soft_bin_residual.dtype)

        correction_clip = float(_cfg_value(args, 'height_correction_clip', 80.0))
        if correction_clip > 0:
            correction_for_height = correction_clip * torch.tanh(correction / correction_clip)
        else:
            correction_for_height = correction
        correction_scale = float(_cfg_value(args, 'height_correction_scale', 1.0))
        height_correction = correction_scale * torch.nan_to_num(correction_for_height, nan=0.0, posinf=correction_clip, neginf=-correction_clip)

        if self.direct_conv is not None:
            direct_height_raw = self.direct_conv(feat)
            direct_clip = float(_cfg_value(args, 'height_direct_clip', 180.0))
            if direct_clip > 0:
                direct_height_raw = direct_clip * torch.tanh(direct_height_raw / direct_clip)
            height_direct = _activate_height(torch.nan_to_num(direct_height_raw, nan=0.0), args=args)
        else:
            height_direct = height_calibrated.detach() * 0.0

        output_mode = str(_cfg_value(args, 'height_output_mode', 'physical')).lower()
        fusion_alpha = float(_cfg_value(args, 'height_fusion_alpha', 0.0))
        fusion_alpha = max(0.0, min(1.0, fusion_alpha))
        if output_mode in ('direct', 'direct_regression', 'learned_direct'):
            height = height_direct
        elif output_mode in ('direct_fusion', 'direct_physical_fusion'):
            height = (1.0 - fusion_alpha) * height_direct + fusion_alpha * height_phy
        elif output_mode in ('soft_residual', 'residual', 'aux'):
            height = height_aux
        elif output_mode in ('soft_residual_fusion', 'residual_fusion', 'hybrid_fusion'):
            height = (1.0 - fusion_alpha) * height_phy + fusion_alpha * height_aux
        elif output_mode in ('physical_correction', 'physics_correction', 'physical_residual', 'physics_residual'):
            height = _activate_height(height_phy + height_correction, args=args)
        elif output_mode in ('physical_correction_fusion', 'physics_correction_fusion'):
            height = _activate_height(height_phy + fusion_alpha * height_correction, args=args)
        elif output_mode in ('calibrated_correction', 'calibrated_residual', 'affine_correction', 'affine_residual'):
            height = _activate_height(height_calibrated + height_correction, args=args)
        elif output_mode in ('gated_residual', 'soft_bin_gated_residual', 'bin_gated_residual'):
            residual_scale = float(_cfg_value(args, 'height_gated_residual_scale', 1.0))
            height = _activate_height(height_calibrated + residual_scale * residual_gate * soft_bin_residual, args=args)
        elif output_mode in (
            'protected_high_correction',
            'low_protected_high_correction',
            'high_protected_correction',
            'high_boost',
            'signed_low_protected_high_correction',
            'low_protected_signed_high_correction',
        ):
            high_bin_start = int(_cfg_value(args, 'height_high_bin_start', 3))
            high_bin_start = max(0, min(high_bin_start, num_bins - 1))
            prob = torch.softmax(logits.float(), dim=1).to(dtype=height_phy.dtype)
            high_prob = torch.clamp(prob[:, high_bin_start:, :, :].sum(dim=1, keepdim=True), 0.0, 1.0)
            gate_power = max(float(_cfg_value(args, 'height_high_gate_power', 1.0)), 1e-3)
            prob_threshold = float(_cfg_value(args, 'height_high_gate_prob_threshold', -1.0))
            if prob_threshold >= 0.0:
                prob_temp = max(float(_cfg_value(args, 'height_high_gate_prob_temperature', 0.10)), 1e-3)
                high_prob_gate = torch.sigmoid((high_prob.float() - prob_threshold) / prob_temp)
            else:
                high_prob_gate = torch.clamp(high_prob.float(), 1e-6, 1.0) ** gate_power

            evidence_threshold = float(_cfg_value(args, 'height_high_evidence_height_threshold', 18.0))
            evidence_width = max(float(_cfg_value(args, 'height_high_evidence_height_width', 12.0)), 1e-3)
            phys_gate = torch.sigmoid((height_phy.float() - evidence_threshold) / evidence_width)
            base_gate = torch.sigmoid(
                (height_calibrated.float() - float(_cfg_value(args, 'height_high_evidence_base_threshold', 20.0)))
                / max(float(_cfg_value(args, 'height_high_evidence_base_width', 18.0)), 1e-3)
            )
            if area_maps is not None:
                log_area = area_maps[:, 0:1].to(dtype=torch.float32, device=height_phy.device)
                area_gate = torch.sigmoid(
                    (log_area - float(_cfg_value(args, 'height_high_evidence_log_area_threshold', 7.0)))
                    / max(float(_cfg_value(args, 'height_high_evidence_log_area_width', 4.0)), 1e-3)
                )
            else:
                log_area = torch.zeros_like(phys_gate)
                area_gate = torch.zeros_like(phys_gate)
            geom_gate = torch.maximum(phys_gate, base_gate)
            evidence_mode = str(_cfg_value(args, 'height_high_evidence_gate_mode', 'max')).lower()
            if evidence_mode in ('geom', 'geometry', 'phys_base', 'phys_base_max', 'geometry_max'):
                evidence_gate = geom_gate
            elif evidence_mode in ('geom_area_product', 'geometry_area_product', 'product'):
                evidence_gate = geom_gate * area_gate
            elif evidence_mode in ('geom_area_soft_product', 'geometry_area_soft_product'):
                area_floor = max(0.0, min(1.0, float(_cfg_value(args, 'height_high_evidence_area_floor', 0.25))))
                evidence_gate = geom_gate * (area_floor + (1.0 - area_floor) * area_gate)
            elif evidence_mode in ('all_product', 'phys_base_area_product'):
                evidence_gate = phys_gate * base_gate * area_gate
            elif evidence_mode in ('min', 'and'):
                evidence_gate = torch.minimum(geom_gate, area_gate)
            else:
                evidence_gate = torch.maximum(torch.maximum(phys_gate, base_gate), area_gate)
            min_evidence = max(0.0, min(1.0, float(_cfg_value(args, 'height_high_gate_min_evidence', 0.12))))
            high_gate = high_prob_gate * (min_evidence + (1.0 - min_evidence) * evidence_gate)
            high_gate = torch.clamp(torch.nan_to_num(high_gate, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

            aux_raise = F.relu(height_aux.float() - height_calibrated.float())
            correction_raise = F.relu(height_correction.float())
            aux_raise_weight = float(_cfg_value(args, 'height_high_aux_raise_weight', 0.75))
            correction_raise_weight = float(_cfg_value(args, 'height_high_correction_raise_weight', 0.50))
            high_boost_raw = aux_raise_weight * aux_raise + correction_raise_weight * correction_raise
            boost_clip = float(_cfg_value(args, 'height_high_boost_clip', 100.0))
            if bool(_cfg_value(args, 'height_low_protect_enabled', False)):
                low_phys_thr = float(_cfg_value(args, 'height_low_protect_physical_threshold', 10.0))
                low_phys_width = max(float(_cfg_value(args, 'height_low_protect_physical_width', 3.0)), 1e-3)
                low_base_thr = float(_cfg_value(args, 'height_low_protect_base_threshold', 10.0))
                low_base_width = max(float(_cfg_value(args, 'height_low_protect_base_width', 4.0)), 1e-3)
                low_phys_gate = torch.sigmoid((height_phy.float() - low_phys_thr) / low_phys_width)
                low_base_gate = torch.sigmoid((height_calibrated.float() - low_base_thr) / low_base_width)
                low_protect_mode = str(_cfg_value(args, 'height_low_protect_gate_mode', 'phys_base_max')).lower()
                if low_protect_mode in ('phys_base_product', 'product', 'and'):
                    nonlow_gate = low_phys_gate * low_base_gate
                else:
                    nonlow_gate = torch.maximum(low_phys_gate, low_base_gate)
                if bool(_cfg_value(args, 'height_low_protect_use_area', False)):
                    low_area_thr = float(_cfg_value(args, 'height_low_protect_log_area_threshold', 6.2))
                    low_area_width = max(float(_cfg_value(args, 'height_low_protect_log_area_width', 1.2)), 1e-3)
                    area_nonlow_gate = torch.sigmoid((log_area - low_area_thr) / low_area_width)
                    area_mode = str(_cfg_value(args, 'height_low_protect_area_mode', 'soft_product')).lower()
                    if area_mode in ('max', 'or'):
                        nonlow_gate = torch.maximum(nonlow_gate, area_nonlow_gate)
                    elif area_mode in ('product', 'and'):
                        nonlow_gate = nonlow_gate * area_nonlow_gate
                    else:
                        area_floor = max(0.0, min(1.0, float(_cfg_value(args, 'height_low_protect_area_floor', 0.35))))
                        nonlow_gate = nonlow_gate * (area_floor + (1.0 - area_floor) * area_nonlow_gate)
                nonlow_gate = torch.clamp(torch.nan_to_num(nonlow_gate, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
                low_boost_clip = max(0.0, float(_cfg_value(args, 'height_low_protect_boost_clip', 1.0)))
                high_gate = torch.clamp(high_gate * nonlow_gate, 0.0, 1.0)
                if boost_clip > 0:
                    adaptive_boost_clip = low_boost_clip + nonlow_gate * max(0.0, boost_clip - low_boost_clip)
                    high_boost_raw = torch.minimum(torch.clamp(high_boost_raw, min=0.0), adaptive_boost_clip)
                else:
                    high_boost_raw = torch.clamp(high_boost_raw, min=0.0)
            elif boost_clip > 0:
                high_boost_raw = torch.clamp(high_boost_raw, min=0.0, max=boost_clip)
            gate_scale = float(_cfg_value(args, 'height_high_gate_scale', 1.0))
            if gate_scale != 1.0:
                high_gate = torch.clamp(high_gate * gate_scale, 0.0, 1.0)
            high_boost = high_gate.to(dtype=high_boost_raw.dtype) * high_boost_raw
            signed_correction = torch.zeros_like(height_calibrated, dtype=torch.float32)
            if output_mode in ('signed_low_protected_high_correction', 'low_protected_signed_high_correction'):
                signed_correction = height_correction.float()
                signed_clip = float(_cfg_value(args, 'height_signed_correction_clip', 16.0))
                if signed_clip > 0:
                    signed_correction = signed_clip * torch.tanh(signed_correction / signed_clip)
                signed_correction = float(_cfg_value(args, 'height_signed_correction_scale', 1.0)) * signed_correction

                low_pos_clip = float(_cfg_value(args, 'height_low_signed_positive_clip', 1.5))
                low_neg_clip = float(_cfg_value(args, 'height_low_signed_negative_clip', 6.0))
                high_pos_clip = float(_cfg_value(args, 'height_high_signed_positive_clip', signed_clip))
                high_neg_clip = float(_cfg_value(args, 'height_high_signed_negative_clip', signed_clip))
                pos_clip = low_pos_clip + high_gate.float() * (high_pos_clip - low_pos_clip)
                neg_clip = low_neg_clip + high_gate.float() * (high_neg_clip - low_neg_clip)
                signed_correction = torch.minimum(signed_correction, pos_clip)
                signed_correction = torch.maximum(signed_correction, -neg_clip)
                signed_correction = torch.nan_to_num(signed_correction, nan=0.0, posinf=high_pos_clip, neginf=-high_neg_clip)

            height = _activate_height(height_calibrated.float() + signed_correction + high_boost, args=args)
        elif output_mode in ('calibrated', 'affine', 'calibrated_physical'):
            height = height_calibrated
        else:
            height = height_phy

        if 'high_boost' not in locals():
            high_boost = torch.zeros_like(height_calibrated, dtype=torch.float32)
        if 'high_boost_raw' not in locals():
            high_boost_raw = torch.zeros_like(height_calibrated, dtype=torch.float32)
        if 'high_gate' not in locals():
            high_gate = torch.zeros_like(height_calibrated, dtype=torch.float32)
        if 'aux_raise' not in locals():
            aux_raise = torch.zeros_like(height_calibrated, dtype=torch.float32)
        if 'signed_correction' not in locals():
            signed_correction = torch.zeros_like(height_calibrated, dtype=torch.float32)

        height = torch.clamp(
            torch.nan_to_num(height.float(), nan=0.0, posinf=height_max, neginf=0.0),
            min=0.0,
            max=height_max,
        ).to(dtype=height_phy.dtype)
        height_aux = torch.clamp(
            torch.nan_to_num(height_aux.float(), nan=0.0, posinf=height_max, neginf=0.0),
            min=0.0,
            max=height_max,
        ).to(dtype=height_phy.dtype)
        height_calibrated = torch.clamp(
            torch.nan_to_num(height_calibrated.float(), nan=0.0, posinf=height_max, neginf=0.0),
            min=0.0,
            max=height_max,
        ).to(dtype=height_phy.dtype)

        return {
            'height': height,
            'height_physical': height_phy,
            'height_base': height_calibrated,
            'height_calibrated': height_calibrated,
            'height_aux': height_aux,
            'height_direct': height_direct,
            'height_correction': height_correction.float(),
            'height_signed_correction': signed_correction.float(),
            'height_high_boost': high_boost.float(),
            'height_high_boost_raw': high_boost_raw.float(),
            'height_high_gate': high_gate.float(),
            'height_aux_raise': aux_raise.float(),
            'height_gated_residual': (residual_gate * soft_bin_residual).float(),
            'height_residual_gate': residual_gate.float(),
            'logits': logits.float(),
            'residuals': residuals_for_height.float()
        }


class InstanceCorrectionHead(nn.Module):
    def __init__(self, args=None, input_dim=11, prefix='height_instance'):
        super().__init__()
        def pvalue(name, default):
            return _cfg_value(args, f'{prefix}_{name}', _cfg_value(args, f'height_instance_{name}', default))

        hidden = int(pvalue('hidden_dim', 64))
        hidden = max(hidden, 16)
        self.clip = float(pvalue('correction_clip', 180.0))
        self.scale = float(pvalue('correction_scale', 1.0))
        last_init_std = float(pvalue('last_init_std', 0.0))
        last_init_bias = float(pvalue('last_init_bias', 0.0))
        zero_init = bool(pvalue('zero_init', True))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(self.net[-1].bias, last_init_bias)
        else:
            nn.init.normal_(self.net[-1].weight, mean=0.0, std=max(last_init_std, 1.0e-6))
            nn.init.constant_(self.net[-1].bias, last_init_bias)

    def forward(self, features):
        correction = self.net(features.float())
        if self.clip > 0:
            correction = self.clip * torch.tanh(correction / self.clip)
        return self.scale * correction


class SD2H_Height_Finetuner(nn.Module):
    def __init__(self, args):
        super(SD2H_Height_Finetuner, self).__init__()
        self.args = args
        self.current_epoch = 0
        
        self.delta_source = str(_cfg_value(args, 'height_delta_source', 'learned_base_plane')).lower()
        self.use_precomputed_delta = self.delta_source in _PRECOMPUTED_DELTA_SOURCES
        self.use_cached_disp = self.use_precomputed_delta or self.delta_source in _CACHED_DISP_LEARNED_SOURCES
        self.use_delta_residual = self.delta_source in _DELTA_RESIDUAL_SOURCES
        self.disp_net = None if self.use_cached_disp else SD2H(args)
        self.base_plane_estimator = BasePlaneEstimator()
        self.height_head = HybridHeightHead(args)
        instance_feature_map_dim = 11
        if bool(_cfg_value(args, 'height_instance_use_signed_ground_features', False)):
            instance_feature_map_dim += 3
        if bool(_cfg_value(args, 'height_instance_use_gradient_feature', False)):
            instance_feature_map_dim += 1
        instance_input_dim = instance_feature_map_dim
        stat_indices = _cfg_list(args, 'height_instance_stat_indices', [0, 1, 2, 7, 8, 9, 10])
        stat_indices = [int(i) for i in stat_indices if 0 <= int(i) < instance_feature_map_dim]
        if bool(_cfg_value(args, 'height_instance_use_shape_features', False)):
            instance_input_dim += 6
        if bool(_cfg_value(args, 'height_instance_use_min_features', False)):
            instance_input_dim += len(stat_indices)
        if bool(_cfg_value(args, 'height_instance_use_max_features', False)):
            instance_input_dim += len(stat_indices)
        if bool(_cfg_value(args, 'height_instance_use_std_features', False)):
            instance_input_dim += len(stat_indices)
        high_residual_input_dim = instance_input_dim
        high_stat_indices = _cfg_list(args, 'height_high_residual_extra_stat_indices', [])
        high_stat_indices = [int(i) for i in high_stat_indices if 0 <= int(i) < instance_feature_map_dim]
        if bool(_cfg_value(args, 'height_high_residual_use_min_features', False)):
            high_residual_input_dim += len(high_stat_indices)
        if bool(_cfg_value(args, 'height_high_residual_use_max_features', False)):
            high_residual_input_dim += len(high_stat_indices)
        if bool(_cfg_value(args, 'height_high_residual_use_std_features', False)):
            high_residual_input_dim += len(high_stat_indices)
        self.instance_correction_head = (
            InstanceCorrectionHead(args, input_dim=instance_input_dim, prefix='height_instance')
            if bool(_cfg_value(args, 'height_instance_refine', False))
            else None
        )
        self.high_residual_head = (
            InstanceCorrectionHead(args, input_dim=high_residual_input_dim, prefix='height_high_residual')
            if bool(_cfg_value(args, 'height_high_residual_refine', False))
            else None
        )
        self.high_residual_gate_head = (
            InstanceCorrectionHead(args, input_dim=high_residual_input_dim, prefix='height_high_residual_gate')
            if bool(_cfg_value(args, 'height_high_residual_learned_gate', False))
            else None
        )
        self.learned_physical_floor_head = (
            InstanceCorrectionHead(args, input_dim=instance_input_dim, prefix='height_learned_physical_floor')
            if bool(_cfg_value(args, 'height_learned_physical_floor_enabled', False))
            else None
        )
        self.learnable_physical_affine = bool(_cfg_value(args, 'height_learnable_physical_affine', False))
        if self.learnable_physical_affine:
            self.height_physical_scale_param = nn.Parameter(
                torch.tensor(float(_cfg_value(args, 'height_physical_scale', 1.0)), dtype=torch.float32)
            )
            self.height_physical_bias_param = nn.Parameter(
                torch.tensor(float(_cfg_value(args, 'height_physical_bias', 0.0)), dtype=torch.float32)
            )
        else:
            self.height_physical_scale_param = None
            self.height_physical_bias_param = None

        self.learnable_instance_feature_calibration = bool(
            _cfg_value(args, 'height_learnable_instance_feature_calibration', False)
        )
        if self.learnable_instance_feature_calibration:
            linear_coeffs = _cfg_list(args, 'height_instance_feature_linear_coeffs', [0.0] * 19)
            high_coeffs = _cfg_list(args, 'height_instance_feature_high_coeffs', linear_coeffs)
            if len(linear_coeffs) != 19 or len(high_coeffs) != 19:
                raise ValueError("Learnable instance feature calibration expects 19 low and 19 high coefficients.")
            self.height_instance_feature_linear_coeffs_param = nn.Parameter(
                torch.tensor(linear_coeffs, dtype=torch.float32)
            )
            self.height_instance_feature_linear_bias_param = nn.Parameter(
                torch.tensor(float(_cfg_value(args, 'height_instance_feature_linear_bias', 0.0)), dtype=torch.float32)
            )
            self.height_instance_feature_high_coeffs_param = nn.Parameter(
                torch.tensor(high_coeffs, dtype=torch.float32)
            )
            self.height_instance_feature_high_bias_param = nn.Parameter(
                torch.tensor(
                    float(_cfg_value(args, 'height_instance_feature_high_bias', _cfg_value(args, 'height_instance_feature_linear_bias', 0.0))),
                    dtype=torch.float32,
                )
            )
        else:
            self.height_instance_feature_linear_coeffs_param = None
            self.height_instance_feature_linear_bias_param = None
            self.height_instance_feature_high_coeffs_param = None
            self.height_instance_feature_high_bias_param = None
        if self.use_delta_residual and bool(_cfg_value(args, 'height_zero_init_delta_residual', True)):
            final_conv = self.base_plane_estimator.decoder[-1]
            if isinstance(final_conv, nn.Conv2d):
                nn.init.zeros_(final_conv.weight)
                nn.init.zeros_(final_conv.bias)
        if self.use_precomputed_delta:
            self.base_plane_estimator.requires_grad_(False)
        if bool(_cfg_value(args, 'height_freeze_pixel_head', False)):
            self.height_head.requires_grad_(False)
        if self.instance_correction_head is not None and bool(_cfg_value(args, 'height_freeze_instance_head', False)):
            self.instance_correction_head.requires_grad_(False)
        if self.high_residual_head is not None and bool(_cfg_value(args, 'height_freeze_high_residual_head', False)):
            self.high_residual_head.requires_grad_(False)
        if self.high_residual_gate_head is not None and bool(_cfg_value(args, 'height_freeze_high_residual_gate_head', False)):
            self.high_residual_gate_head.requires_grad_(False)
        if self.learned_physical_floor_head is not None and bool(_cfg_value(args, 'height_freeze_learned_physical_floor_head', False)):
            self.learned_physical_floor_head.requires_grad_(False)
        
        if self.disp_net is not None and hasattr(args, 'pretrained_disp_path') and args.pretrained_disp_path:
            self.load_pretrained_disp(args.pretrained_disp_path)
            
        self.freeze_frontend()

    def set_finetune_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _zero_anchor_for_base_plane(self, like_tensor):
        anchor = None
        for param in self.base_plane_estimator.parameters():
            if param.requires_grad:
                term = param.float().sum() * 0.0
                anchor = term if anchor is None else anchor + term
        if anchor is None:
            return torch.zeros_like(like_tensor)
        return torch.zeros_like(like_tensor) + anchor.to(device=like_tensor.device, dtype=like_tensor.dtype)

    def load_pretrained_disp(self, path):
        print(f"[*] Loading pretrained foundational disparity weights from: {path}")
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
            
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.disp_net.load_state_dict(clean_state_dict, strict=False)
        print("[*] Pretrained weights successfully injected.")

    def freeze_frontend(self):
        """Freeze the entire disparity network, including recurrent update blocks."""
        if self.disp_net is not None:
            for param in self.disp_net.parameters():
                param.requires_grad = False
            
        if self.use_precomputed_delta:
            print("[*] Using precomputed disparity/delta maps; SD2H disparity network is not instantiated.")
        elif self.use_cached_disp:
            print("[*] Using precomputed disparity with a learnable DTM/delta module; SD2H disparity network is not instantiated.")
        else:
            print(f"[*] Strict Physics Strategy: The entire SD2H disparity network is 100% FROZEN.")
        base_params = sum(p.numel() for p in self.base_plane_estimator.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.height_head.parameters() if p.requires_grad)
        inst_params = sum(
            p.numel() for p in (
                self.instance_correction_head.parameters() if self.instance_correction_head is not None else []
            ) if p.requires_grad
        )
        high_res_params = sum(
            p.numel() for p in (
                self.high_residual_head.parameters() if self.high_residual_head is not None else []
            ) if p.requires_grad
        )
        high_gate_params = sum(
            p.numel() for p in (
                self.high_residual_gate_head.parameters() if self.high_residual_gate_head is not None else []
            ) if p.requires_grad
        )
        learned_phys_params = sum(
            p.numel() for p in (
                self.learned_physical_floor_head.parameters() if self.learned_physical_floor_head is not None else []
            ) if p.requires_grad
        )
        calib_params = sum(
            p.numel() for p in (
                self.height_physical_scale_param,
                self.height_physical_bias_param,
                self.height_instance_feature_linear_coeffs_param,
                self.height_instance_feature_linear_bias_param,
                self.height_instance_feature_high_coeffs_param,
                self.height_instance_feature_high_bias_param,
            )
            if p is not None and p.requires_grad
        )
        trainable_params = base_params + head_params + inst_params + high_res_params + high_gate_params + learned_phys_params + calib_params
        print(
            f"[*] Learnable Base-Plane & Head parameters ready for finetuning: "
            f"{trainable_params / 1e6:.2f} M "
            f"(base={base_params / 1e6:.2f} M, head={head_params / 1e6:.2f} M, "
            f"instance={inst_params / 1e6:.2f} M, high_res={high_res_params / 1e6:.2f} M, "
            f"high_gate={high_gate_params / 1e6:.2f} M, learned_phys={learned_phys_params / 1e6:.2f} M, "
            f"calib={calib_params / 1e6:.4f} M)"
        )

    def _instance_mean_delta(self, delta_d_raw, mask):
        B, _, H, W = delta_d_raw.shape
        delta_d_robust = torch.zeros_like(delta_d_raw)
        flat_delta = delta_d_raw.view(B, -1)
        flat_mask = mask.view(B, -1).long()
        valid_pixel_mask = flat_mask > 0

        for b in range(B):
            b_mask = flat_mask[b][valid_pixel_mask[b]]
            b_delta = flat_delta[b][valid_pixel_mask[b]]
            if b_mask.numel() == 0:
                delta_d_robust[b] = delta_d_raw[b]
                continue

            _, inverse_indices = torch.unique(b_mask, return_inverse=True)
            inst_sum = torch.zeros(
                inverse_indices.max().item() + 1,
                device=delta_d_raw.device,
                dtype=delta_d_raw.dtype,
            ).scatter_add_(0, inverse_indices, b_delta)
            inst_cnt = torch.zeros_like(inst_sum).scatter_add_(0, inverse_indices, torch.ones_like(b_delta))
            inst_means = inst_sum / inst_cnt.clamp_min(1.0)

            flat_robust_b = delta_d_raw[b, 0].view(-1).clone()
            flat_robust_b[valid_pixel_mask[b]] = inst_means[inverse_indices]
            delta_d_robust[b, 0] = flat_robust_b.view(H, W)

        return delta_d_robust

    def _build_extra_features(self, image1, mask, disp_est=None, delta_d=None, ground_disp=None, k_tensor=None):
        features = []
        if bool(_cfg_value(self.args, 'height_use_image_features', False)):
            channels = int(_cfg_value(self.args, 'height_image_feature_channels', 1))
            img = torch.nan_to_num(image1.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if channels <= 1:
                gray = img.mean(dim=1, keepdim=True)
                mean = gray.flatten(2).mean(dim=-1).view(-1, 1, 1, 1)
                std = gray.flatten(2).std(dim=-1).view(-1, 1, 1, 1).clamp_min(1.0)
                features.append(torch.clamp((gray - mean) / std, -5.0, 5.0) / 5.0)
            else:
                mean = img.flatten(2).mean(dim=-1).view(img.shape[0], img.shape[1], 1, 1)
                std = img.flatten(2).std(dim=-1).view(img.shape[0], img.shape[1], 1, 1).clamp_min(1.0)
                norm_img = torch.clamp((img - mean) / std, -5.0, 5.0) / 5.0
                features.append(norm_img[:, :channels])

        if bool(_cfg_value(self.args, 'height_use_mask_feature', False)):
            features.append((mask > 0).float())

        if bool(_cfg_value(self.args, 'height_use_area_feature', False)):
            area_feature = torch.zeros_like(mask, dtype=torch.float32)
            B, _, H, W = mask.shape
            flat_mask = mask.view(B, -1).long()
            valid = flat_mask > 0
            norm = math.log(float(H * W) + 1.0)
            for b in range(B):
                ids = flat_mask[b][valid[b]]
                if ids.numel() == 0:
                    continue
                u_ids, inv = torch.unique(ids, return_inverse=True)
                cnt = torch.zeros(u_ids.numel(), device=mask.device, dtype=torch.float32).scatter_add_(
                    0, inv, torch.ones_like(ids, dtype=torch.float32)
                )
                vals = torch.log1p(cnt) / max(norm, 1.0)
                flat_area = area_feature[b, 0].view(-1)
                flat_area[valid[b]] = vals[inv]
            features.append(area_feature)

        if bool(_cfg_value(self.args, 'height_use_ground_feature', False)):
            if ground_disp is None and disp_est is not None and delta_d is not None:
                ground = disp_est.float() + delta_d.float()
            elif ground_disp is not None:
                ground = ground_disp.float()
            else:
                ground = torch.zeros_like(mask, dtype=torch.float32)
            features.append(torch.clamp(torch.nan_to_num(ground, nan=0.0, posinf=255.0, neginf=-255.0) / 128.0, -4.0, 4.0))

        if bool(_cfg_value(self.args, 'height_use_signed_delta_feature', False)):
            if ground_disp is not None and disp_est is not None:
                signed_delta = ground_disp.float() - disp_est.float()
            elif delta_d is not None:
                signed_delta = delta_d.float()
            else:
                signed_delta = torch.zeros_like(mask, dtype=torch.float32)
            features.append(torch.clamp(torch.nan_to_num(signed_delta, nan=0.0, posinf=80.0, neginf=-80.0) / 40.0, -4.0, 4.0))

        if bool(_cfg_value(self.args, 'height_use_physical_feature', False)):
            if delta_d is not None and k_tensor is not None:
                k = k_tensor.float().view(-1, 1, 1, 1)
                physical = k * delta_d.float()
            else:
                physical = torch.zeros_like(mask, dtype=torch.float32)
            features.append(torch.clamp(torch.nan_to_num(physical, nan=0.0, posinf=320.0, neginf=0.0) / 80.0, 0.0, 4.0))

        if not features:
            return None
        return torch.cat(features, dim=1)

    def _build_area_maps(self, mask):
        B = mask.shape[0]
        area_maps = torch.zeros((B, 2, mask.shape[-2], mask.shape[-1]), device=mask.device, dtype=torch.float32)
        flat_mask = mask.view(B, -1).long()
        valid = flat_mask > 0
        for b in range(B):
            ids = flat_mask[b][valid[b]]
            if ids.numel() == 0:
                continue
            _, inv = torch.unique(ids, return_inverse=True)
            cnt = torch.zeros(inv.max().item() + 1, device=mask.device, dtype=torch.float32).scatter_add_(
                0, inv, torch.ones_like(ids, dtype=torch.float32)
            )
            flat_log = area_maps[b, 0].view(-1)
            flat_sqrt = area_maps[b, 1].view(-1)
            flat_log[valid[b]] = torch.log1p(cnt)[inv]
            flat_sqrt[valid[b]] = torch.sqrt(cnt)[inv]
        return area_maps

    def _apply_instance_refinement(self, h_pred_dict, disp_est, delta_d, image1, mask, area_maps, ground_disp=None):
        if (
            self.instance_correction_head is None
            and self.high_residual_head is None
            and self.high_residual_gate_head is None
            and self.learned_physical_floor_head is None
        ):
            return h_pred_dict

        base_key = str(_cfg_value(self.args, 'height_instance_base_key', 'height_base'))
        if base_key not in h_pred_dict:
            base_key = 'height_base'
        base = torch.nan_to_num(h_pred_dict[base_key], nan=0.0, posinf=320.0, neginf=0.0)
        physical = torch.nan_to_num(h_pred_dict['height_physical'], nan=0.0, posinf=320.0, neginf=0.0)
        delta = torch.nan_to_num(delta_d, nan=0.0, posinf=80.0, neginf=0.0)
        disp = torch.nan_to_num(disp_est, nan=0.0, posinf=255.0, neginf=-255.0)
        if ground_disp is not None:
            ground = torch.nan_to_num(ground_disp, nan=0.0, posinf=255.0, neginf=-255.0)
            signed_delta = torch.nan_to_num(ground.float() - disp.float(), nan=0.0, posinf=80.0, neginf=-80.0).to(dtype=delta.dtype)
            order_bad = (signed_delta < 0).to(dtype=delta.dtype)
        else:
            ground = disp + delta
            signed_delta = delta
            order_bad = torch.zeros_like(delta)

        detach_instance_features = bool(_cfg_value(self.args, 'height_instance_detach_features', True)) or not self.training
        if detach_instance_features:
            base = base.detach()
            physical = physical.detach()
            delta = delta.detach()
            disp = disp.detach()
            ground = ground.detach()
            signed_delta = signed_delta.detach()
            order_bad = order_bad.detach()

        gray = torch.nan_to_num(image1.float(), nan=0.0, posinf=0.0, neginf=0.0).mean(dim=1, keepdim=True)
        gray_mean = gray.flatten(2).mean(dim=-1).view(-1, 1, 1, 1)
        gray_std = gray.flatten(2).std(dim=-1).view(-1, 1, 1, 1).clamp_min(1.0)
        gray_norm = torch.clamp((gray - gray_mean) / gray_std, -5.0, 5.0) / 5.0
        if bool(_cfg_value(self.args, 'height_instance_use_gradient_feature', False)):
            grad_x = F.pad(gray_norm[:, :, :, 1:] - gray_norm[:, :, :, :-1], (0, 1, 0, 0))
            grad_y = F.pad(gray_norm[:, :, 1:, :] - gray_norm[:, :, :-1, :], (0, 0, 0, 1))
            grad_norm = torch.sqrt(torch.clamp(grad_x * grad_x + grad_y * grad_y, min=0.0))
            grad_norm = torch.clamp(grad_norm, 0.0, 2.0)
        else:
            grad_norm = None

        if area_maps is None:
            area_maps = self._build_area_maps(mask)
        log_area = area_maps[:, 0:1].to(device=base.device, dtype=base.dtype)
        sqrt_area = area_maps[:, 1:2].to(device=base.device, dtype=base.dtype)

        feature_parts = [
            torch.clamp(base / 80.0, 0.0, 4.0),
            torch.clamp(physical / 80.0, 0.0, 4.0),
            torch.clamp(delta / 40.0, 0.0, 4.0),
            torch.clamp(disp / 128.0, -4.0, 4.0),
            gray_norm.to(dtype=base.dtype),
            torch.clamp(log_area / 14.0, 0.0, 4.0),
            torch.clamp(sqrt_area / 1024.0, 0.0, 4.0),
            torch.clamp(torch.log1p(torch.clamp(base, min=0.0)) / 5.0, 0.0, 4.0),
            torch.clamp((base / 80.0) * (log_area / 14.0), 0.0, 8.0),
            torch.clamp((physical / 80.0) * (log_area / 14.0), 0.0, 8.0),
            torch.clamp((delta / 40.0) * (log_area / 14.0), 0.0, 8.0),
        ]
        if bool(_cfg_value(self.args, 'height_instance_use_signed_ground_features', False)):
            feature_parts.extend([
                torch.clamp(ground / 128.0, -4.0, 4.0),
                torch.clamp(signed_delta / 40.0, -4.0, 4.0),
                order_bad,
            ])
        if grad_norm is not None:
            feature_parts.append(grad_norm.to(dtype=base.dtype))
        if detach_instance_features:
            bsz, _, fmap_h, fmap_w = feature_parts[0].shape
            feature_dtype_name = str(_cfg_value(self.args, 'height_instance_feature_map_dtype', 'float32')).lower()
            if feature_dtype_name in ('bf16', 'bfloat16') and feature_parts[0].is_cuda:
                feature_map_dtype = torch.bfloat16
            elif feature_dtype_name in ('fp16', 'float16', 'half') and feature_parts[0].is_cuda:
                feature_map_dtype = torch.float16
            else:
                feature_map_dtype = torch.float32
            feature_maps = torch.empty(
                (bsz, len(feature_parts), fmap_h, fmap_w),
                device=feature_parts[0].device,
                dtype=feature_map_dtype,
            )
            for feat_idx, feat in enumerate(feature_parts):
                feature_maps[:, feat_idx : feat_idx + 1].copy_(feat.to(dtype=feature_map_dtype))
            feature_maps = feature_maps.detach()
            del feature_parts
        else:
            feature_maps = torch.cat(feature_parts, dim=1).float()

        B, Fdim, H, W = feature_maps.shape
        corr_map = torch.zeros(B, 1, H, W, device=feature_maps.device, dtype=feature_maps.dtype)
        high_residual_map = torch.zeros_like(corr_map)
        high_residual_gate_logit_map = torch.zeros_like(corr_map)
        learned_floor_raw_map = torch.zeros_like(corr_map)
        flat_mask = mask.view(B, -1).long()
        flat_feat = feature_maps.permute(0, 2, 3, 1).reshape(B, -1, Fdim)
        stat_indices = _cfg_list(self.args, 'height_instance_stat_indices', [0, 1, 2, 7, 8, 9, 10])
        stat_indices = [int(i) for i in stat_indices if 0 <= int(i) < Fdim]
        use_min_features = bool(_cfg_value(self.args, 'height_instance_use_min_features', False)) and len(stat_indices) > 0
        use_max_features = bool(_cfg_value(self.args, 'height_instance_use_max_features', False)) and len(stat_indices) > 0
        use_std_features = bool(_cfg_value(self.args, 'height_instance_use_std_features', False)) and len(stat_indices) > 0
        high_stat_indices = _cfg_list(self.args, 'height_high_residual_extra_stat_indices', [])
        high_stat_indices = [int(i) for i in high_stat_indices if 0 <= int(i) < Fdim]
        use_high_min_features = bool(_cfg_value(self.args, 'height_high_residual_use_min_features', False)) and len(high_stat_indices) > 0
        use_high_max_features = bool(_cfg_value(self.args, 'height_high_residual_use_max_features', False)) and len(high_stat_indices) > 0
        use_high_std_features = bool(_cfg_value(self.args, 'height_high_residual_use_std_features', False)) and len(high_stat_indices) > 0
        use_shape_features = bool(_cfg_value(self.args, 'height_instance_use_shape_features', False))
        flat_pos = torch.arange(H * W, device=feature_maps.device)
        flat_rows = (flat_pos // W).float()
        flat_cols = (flat_pos % W).float()
        image_area_log = math.log(float(H * W) + 1.0)
        used_instance_head = False
        used_high_residual_head = False
        used_high_residual_gate_head = False
        used_learned_physical_floor_head = False
        for b in range(B):
            valid = flat_mask[b] > 0
            if not valid.any():
                continue
            ids = flat_mask[b][valid]
            _, inv = torch.unique(ids, return_inverse=True)
            num_instances = int(inv.max().item()) + 1
            inst_sum = torch.zeros(num_instances, Fdim, device=feature_maps.device, dtype=feature_maps.dtype)
            inst_sum.scatter_add_(0, inv[:, None].expand(-1, Fdim), flat_feat[b][valid])
            counts = torch.zeros(num_instances, 1, device=feature_maps.device, dtype=feature_maps.dtype)
            counts.scatter_add_(0, inv[:, None], torch.ones_like(inv[:, None], dtype=feature_maps.dtype))
            inst_features = inst_sum / counts.clamp_min(1.0)
            inst_feature_parts = [inst_features]
            if use_shape_features:
                valid_rows = flat_rows[valid].to(dtype=feature_maps.dtype)
                valid_cols = flat_cols[valid].to(dtype=feature_maps.dtype)
                inst_min_r = torch.full((num_instances, 1), torch.inf, device=feature_maps.device, dtype=feature_maps.dtype)
                inst_max_r = torch.full((num_instances, 1), -torch.inf, device=feature_maps.device, dtype=feature_maps.dtype)
                inst_min_c = torch.full((num_instances, 1), torch.inf, device=feature_maps.device, dtype=feature_maps.dtype)
                inst_max_c = torch.full((num_instances, 1), -torch.inf, device=feature_maps.device, dtype=feature_maps.dtype)
                inst_min_r.scatter_reduce_(0, inv[:, None], valid_rows[:, None], reduce='amin', include_self=False)
                inst_max_r.scatter_reduce_(0, inv[:, None], valid_rows[:, None], reduce='amax', include_self=False)
                inst_min_c.scatter_reduce_(0, inv[:, None], valid_cols[:, None], reduce='amin', include_self=False)
                inst_max_c.scatter_reduce_(0, inv[:, None], valid_cols[:, None], reduce='amax', include_self=False)
                bbox_h = (inst_max_r - inst_min_r + 1.0).clamp_min(1.0)
                bbox_w = (inst_max_c - inst_min_c + 1.0).clamp_min(1.0)
                bbox_area = (bbox_h * bbox_w).clamp_min(1.0)
                shape_features = torch.cat(
                    [
                        torch.clamp(torch.log1p(counts) / max(image_area_log, 1.0), 0.0, 4.0),
                        torch.clamp(torch.log1p(bbox_area) / max(image_area_log, 1.0), 0.0, 4.0),
                        torch.clamp(bbox_h / max(float(H), 1.0), 0.0, 1.0),
                        torch.clamp(bbox_w / max(float(W), 1.0), 0.0, 1.0),
                        torch.clamp(torch.log((bbox_w + 1.0) / (bbox_h + 1.0)), -4.0, 4.0),
                        torch.clamp(counts / bbox_area, 0.0, 1.0),
                    ],
                    dim=1,
                )
                inst_feature_parts.append(torch.nan_to_num(shape_features, nan=0.0, posinf=4.0, neginf=-4.0))
            if use_min_features:
                selected = flat_feat[b][valid][:, stat_indices]
                inst_min = torch.full(
                    (num_instances, len(stat_indices)),
                    torch.inf,
                    device=feature_maps.device,
                    dtype=feature_maps.dtype,
                )
                inst_min.scatter_reduce_(
                    0,
                    inv[:, None].expand(-1, len(stat_indices)),
                    selected,
                    reduce='amin',
                    include_self=False,
                )
                inst_feature_parts.append(torch.nan_to_num(inst_min, nan=0.0, posinf=0.0, neginf=-8.0))
            if use_max_features:
                selected = flat_feat[b][valid][:, stat_indices]
                inst_max = torch.full(
                    (num_instances, len(stat_indices)),
                    -torch.inf,
                    device=feature_maps.device,
                    dtype=feature_maps.dtype,
                )
                inst_max.scatter_reduce_(
                    0,
                    inv[:, None].expand(-1, len(stat_indices)),
                    selected,
                    reduce='amax',
                    include_self=False,
                )
                inst_feature_parts.append(torch.nan_to_num(inst_max, nan=0.0, posinf=8.0, neginf=0.0))
            if use_std_features:
                selected = flat_feat[b][valid][:, stat_indices]
                inst_sq_sum = torch.zeros(
                    num_instances,
                    len(stat_indices),
                    device=feature_maps.device,
                    dtype=feature_maps.dtype,
                )
                inst_sq_sum.scatter_add_(0, inv[:, None].expand(-1, len(stat_indices)), selected * selected)
                inst_sq = inst_sq_sum / counts.clamp_min(1.0)
                inst_var = torch.clamp(inst_sq - inst_features[:, stat_indices] ** 2, min=0.0)
                inst_std = torch.sqrt(inst_var + 1.0e-6)
                inst_feature_parts.append(torch.nan_to_num(inst_std, nan=0.0, posinf=8.0, neginf=0.0))
            inst_features = torch.cat(inst_feature_parts, dim=1)
            high_inst_features = inst_features
            if (
                self.high_residual_head is not None
                or self.high_residual_gate_head is not None
            ) and (use_high_min_features or use_high_max_features or use_high_std_features):
                high_feature_parts = [inst_features]
                selected_high = flat_feat[b][valid][:, high_stat_indices]
                if use_high_min_features:
                    high_min = torch.full(
                        (num_instances, len(high_stat_indices)),
                        torch.inf,
                        device=feature_maps.device,
                        dtype=feature_maps.dtype,
                    )
                    high_min.scatter_reduce_(
                        0,
                        inv[:, None].expand(-1, len(high_stat_indices)),
                        selected_high,
                        reduce='amin',
                        include_self=False,
                    )
                    high_feature_parts.append(torch.nan_to_num(high_min, nan=0.0, posinf=0.0, neginf=-8.0))
                if use_high_max_features:
                    high_max = torch.full(
                        (num_instances, len(high_stat_indices)),
                        -torch.inf,
                        device=feature_maps.device,
                        dtype=feature_maps.dtype,
                    )
                    high_max.scatter_reduce_(
                        0,
                        inv[:, None].expand(-1, len(high_stat_indices)),
                        selected_high,
                        reduce='amax',
                        include_self=False,
                    )
                    high_feature_parts.append(torch.nan_to_num(high_max, nan=0.0, posinf=8.0, neginf=0.0))
                if use_high_std_features:
                    high_sq_sum = torch.zeros(
                        num_instances,
                        len(high_stat_indices),
                        device=feature_maps.device,
                        dtype=feature_maps.dtype,
                    )
                    high_sq_sum.scatter_add_(
                        0,
                        inv[:, None].expand(-1, len(high_stat_indices)),
                        selected_high * selected_high,
                    )
                    high_sq = high_sq_sum / counts.clamp_min(1.0)
                    high_var = torch.clamp(high_sq - inst_features[:, high_stat_indices] ** 2, min=0.0)
                    high_std = torch.sqrt(high_var + 1.0e-6)
                    high_feature_parts.append(torch.nan_to_num(high_std, nan=0.0, posinf=8.0, neginf=0.0))
                high_inst_features = torch.cat(high_feature_parts, dim=1)
            if self.instance_correction_head is not None:
                inst_correction = self.instance_correction_head(inst_features).view(-1)
                used_instance_head = True
                corr_flat = corr_map[b, 0].view(-1)
                corr_flat[valid] = inst_correction[inv].to(dtype=corr_flat.dtype)
            if self.high_residual_head is not None:
                high_residual = self.high_residual_head(high_inst_features).view(-1)
                if bool(_cfg_value(self.args, 'height_high_residual_instance_gate_enabled', False)):
                    gate_mode = str(_cfg_value(self.args, 'height_high_residual_instance_gate_mode', 'base_area')).lower()
                    inst_base_signal = inst_features[:, 0].float() * 80.0
                    inst_log_area_signal = inst_features[:, 5].float() * 14.0 if inst_features.shape[1] > 5 else torch.zeros_like(inst_base_signal)
                    base_thr = float(_cfg_value(self.args, 'height_high_residual_instance_base_threshold', 12.0))
                    base_width = max(
                        float(_cfg_value(self.args, 'height_high_residual_instance_base_width', 2.0)),
                        1.0e-3,
                    )
                    area_thr = float(_cfg_value(self.args, 'height_high_residual_instance_log_area_threshold', 6.5))
                    area_width = max(
                        float(_cfg_value(self.args, 'height_high_residual_instance_log_area_width', 0.6)),
                        1.0e-3,
                    )
                    base_gate = torch.sigmoid((inst_base_signal - base_thr) / base_width)
                    area_gate = torch.sigmoid((inst_log_area_signal - area_thr) / area_width)
                    if gate_mode in ('base', 'base_only'):
                        inst_gate = base_gate
                    elif gate_mode in ('area', 'area_only'):
                        inst_gate = area_gate
                    elif gate_mode in ('min', 'and'):
                        inst_gate = torch.minimum(base_gate, area_gate)
                    elif gate_mode in ('max', 'or'):
                        inst_gate = torch.maximum(base_gate, area_gate)
                    else:
                        area_floor = max(
                            0.0,
                            min(1.0, float(_cfg_value(self.args, 'height_high_residual_instance_area_floor', 0.0))),
                        )
                        inst_gate = base_gate * (area_floor + (1.0 - area_floor) * area_gate)
                    gate_power = max(
                        float(_cfg_value(self.args, 'height_high_residual_instance_gate_power', 1.0)),
                        1.0e-3,
                    )
                    if abs(gate_power - 1.0) > 1.0e-6:
                        inst_gate = _safe_gate_power(inst_gate, gate_power)
                    gate_scale = float(_cfg_value(self.args, 'height_high_residual_instance_gate_scale', 1.0))
                    min_gate = max(0.0, float(_cfg_value(self.args, 'height_high_residual_instance_min_gate', 0.0)))
                    inst_gate = torch.clamp(
                        torch.nan_to_num(inst_gate * gate_scale, nan=0.0, posinf=1.0, neginf=0.0),
                        0.0,
                        1.0,
                    )
                    if min_gate > 0.0:
                        inst_gate = torch.where(inst_gate >= min_gate, inst_gate, torch.zeros_like(inst_gate))
                    high_residual = high_residual * inst_gate.to(dtype=high_residual.dtype)
                used_high_residual_head = True
                high_flat = high_residual_map[b, 0].view(-1)
                high_flat[valid] = high_residual[inv].to(dtype=high_flat.dtype)
            if self.high_residual_gate_head is not None:
                high_gate_logit = self.high_residual_gate_head(high_inst_features).view(-1)
                used_high_residual_gate_head = True
                high_gate_flat = high_residual_gate_logit_map[b, 0].view(-1)
                high_gate_flat[valid] = high_gate_logit[inv].to(dtype=high_gate_flat.dtype)
            if self.learned_physical_floor_head is not None:
                learned_floor_raw = self.learned_physical_floor_head(inst_features).view(-1)
                used_learned_physical_floor_head = True
                learned_flat = learned_floor_raw_map[b, 0].view(-1)
                learned_flat[valid] = learned_floor_raw[inv].to(dtype=learned_flat.dtype)

        if self.instance_correction_head is not None and not used_instance_head:
            dummy_input_dim = self.instance_correction_head.net[0].in_features
            dummy_features = torch.zeros(
                1,
                dummy_input_dim,
                device=feature_maps.device,
                dtype=feature_maps.dtype,
            )
            corr_map = corr_map + self.instance_correction_head(dummy_features).sum() * 0.0
        if self.high_residual_head is not None and not used_high_residual_head:
            dummy_input_dim = self.high_residual_head.net[0].in_features
            dummy_features = torch.zeros(
                1,
                dummy_input_dim,
                device=feature_maps.device,
                dtype=feature_maps.dtype,
            )
            high_residual_map = high_residual_map + self.high_residual_head(dummy_features).sum() * 0.0
        if self.high_residual_gate_head is not None and not used_high_residual_gate_head:
            dummy_input_dim = self.high_residual_gate_head.net[0].in_features
            dummy_features = torch.zeros(
                1,
                dummy_input_dim,
                device=feature_maps.device,
                dtype=feature_maps.dtype,
            )
            high_residual_gate_logit_map = (
                high_residual_gate_logit_map + self.high_residual_gate_head(dummy_features).sum() * 0.0
            )
        if self.learned_physical_floor_head is not None and not used_learned_physical_floor_head:
            dummy_input_dim = self.learned_physical_floor_head.net[0].in_features
            dummy_features = torch.zeros(
                1,
                dummy_input_dim,
                device=feature_maps.device,
                dtype=feature_maps.dtype,
            )
            learned_floor_raw_map = learned_floor_raw_map + self.learned_physical_floor_head(dummy_features).sum() * 0.0
        del feature_maps, flat_feat

        low_pos_clip = float(_cfg_value(self.args, 'height_instance_low_positive_clip', -1.0))
        if low_pos_clip >= 0:
            low_threshold = float(_cfg_value(self.args, 'height_instance_low_gate_threshold', 12.0))
            low_width = max(float(_cfg_value(self.args, 'height_instance_low_gate_width', 12.0)), 1e-3)
            phys_threshold = float(_cfg_value(self.args, 'height_instance_physical_gate_threshold', 8.0))
            phys_width = max(float(_cfg_value(self.args, 'height_instance_physical_gate_width', 18.0)), 1e-3)
            delta_threshold = float(_cfg_value(self.args, 'height_instance_delta_gate_threshold', 4.0))
            delta_width = max(float(_cfg_value(self.args, 'height_instance_delta_gate_width', 12.0)), 1e-3)
            area_threshold = float(_cfg_value(self.args, 'height_instance_log_area_gate_threshold', 6.0))
            area_width = max(float(_cfg_value(self.args, 'height_instance_log_area_gate_width', 4.0)), 1e-3)
            base_factor = torch.clamp((base.float() - low_threshold) / low_width, 0.0, 1.0)
            phys_factor = torch.clamp((physical.float() - phys_threshold) / phys_width, 0.0, 1.0)
            delta_factor = torch.clamp((delta.float() - delta_threshold) / delta_width, 0.0, 1.0)
            area_factor = torch.clamp((log_area.float() - area_threshold) / area_width, 0.0, 1.0)
            low_factor = torch.maximum(torch.maximum(base_factor, phys_factor), torch.maximum(delta_factor, area_factor))
            high_clip = self.instance_correction_head.clip if self.instance_correction_head is not None else 0.0
            max_positive = low_pos_clip + (float(high_clip) - low_pos_clip) * low_factor
            corr_map = torch.minimum(corr_map, max_positive.to(dtype=corr_map.dtype))

        pre_instance_weight = float(_cfg_value(self.args, 'height_instance_pre_weight', 0.0))
        pre_instance_key = 'height_pixel' if 'height_pixel' in h_pred_dict else 'height'
        if pre_instance_weight > 0 and pre_instance_key in h_pred_dict:
            pre_instance = torch.nan_to_num(h_pred_dict[pre_instance_key], nan=0.0, posinf=320.0, neginf=0.0)
            pre_instance_weight = max(0.0, min(1.0, pre_instance_weight))
            refined_base = (
                (1.0 - pre_instance_weight) * base.float()
                + pre_instance_weight * pre_instance.float()
            )
        else:
            refined_base = base.float()

        refine_mode = str(_cfg_value(self.args, 'height_instance_refine_mode', 'additive')).lower()
        if refine_mode in ('log_ratio', 'ratio', 'multiplicative', 'log_multiplicative'):
            ratio_clip = float(_cfg_value(self.args, 'height_instance_log_ratio_clip', self.instance_correction_head.clip))
            log_ratio = corr_map.to(dtype=base.dtype)
            if ratio_clip > 0:
                log_ratio = torch.clamp(log_ratio, min=-ratio_clip, max=ratio_clip)
            low_ratio_clip = float(_cfg_value(self.args, 'height_instance_low_log_ratio_clip', -1.0))
            if low_ratio_clip >= 0:
                low_threshold = float(_cfg_value(self.args, 'height_instance_low_gate_threshold', 12.0))
                low_width = max(float(_cfg_value(self.args, 'height_instance_low_gate_width', 12.0)), 1e-3)
                phys_threshold = float(_cfg_value(self.args, 'height_instance_physical_gate_threshold', 8.0))
                phys_width = max(float(_cfg_value(self.args, 'height_instance_physical_gate_width', 18.0)), 1e-3)
                delta_threshold = float(_cfg_value(self.args, 'height_instance_delta_gate_threshold', 4.0))
                delta_width = max(float(_cfg_value(self.args, 'height_instance_delta_gate_width', 12.0)), 1e-3)
                area_threshold = float(_cfg_value(self.args, 'height_instance_log_area_gate_threshold', 6.0))
                area_width = max(float(_cfg_value(self.args, 'height_instance_log_area_gate_width', 4.0)), 1e-3)
                base_factor = torch.clamp((base.float() - low_threshold) / low_width, 0.0, 1.0)
                phys_factor = torch.clamp((physical.float() - phys_threshold) / phys_width, 0.0, 1.0)
                delta_factor = torch.clamp((delta.float() - delta_threshold) / delta_width, 0.0, 1.0)
                area_factor = torch.clamp((log_area.float() - area_threshold) / area_width, 0.0, 1.0)
                evidence_mode = str(_cfg_value(self.args, 'height_instance_log_ratio_gate_mode', 'max')).lower()
                geom_factor = torch.maximum(base_factor, phys_factor)
                if evidence_mode in ('area_and_geom_delta', 'area_and_evidence', 'product'):
                    high_evidence = area_factor * torch.maximum(geom_factor, delta_factor)
                elif evidence_mode in ('delta_or_height_area', 'area_times_max'):
                    high_evidence = area_factor * torch.maximum(torch.maximum(base_factor, phys_factor), delta_factor)
                elif evidence_mode in ('mean', 'average'):
                    high_evidence = (base_factor + phys_factor + delta_factor + area_factor) * 0.25
                elif evidence_mode in ('min', 'all'):
                    high_evidence = torch.minimum(torch.minimum(base_factor, phys_factor), torch.minimum(delta_factor, area_factor))
                else:
                    high_evidence = torch.maximum(
                        torch.maximum(base_factor, phys_factor),
                        torch.maximum(delta_factor, area_factor),
                    )
                gate_power = float(_cfg_value(self.args, 'height_instance_log_ratio_gate_power', 1.0))
                if gate_power > 0 and abs(gate_power - 1.0) > 1.0e-6:
                    high_evidence = _safe_gate_power(high_evidence, gate_power)
                max_positive_ratio = low_ratio_clip + (max(ratio_clip, low_ratio_clip) - low_ratio_clip) * high_evidence
                log_ratio = torch.minimum(log_ratio, max_positive_ratio.to(dtype=log_ratio.dtype))
            base_floor = float(_cfg_value(self.args, 'height_instance_ratio_base_floor', 1.0))
            ratio_base = torch.clamp(refined_base, min=max(base_floor, 1.0e-4))
            refined_input = ratio_base * torch.exp(log_ratio.float()).to(dtype=ratio_base.dtype)
            h_pred_dict['height_instance_log_ratio'] = log_ratio
        elif refine_mode in ('relative', 'fractional', 'residual_ratio'):
            ratio_clip = float(_cfg_value(self.args, 'height_instance_relative_clip', self.instance_correction_head.clip))
            rel = corr_map.to(dtype=base.dtype)
            if ratio_clip > 0:
                rel = torch.clamp(rel, min=-ratio_clip, max=ratio_clip)
            refined_input = refined_base * (1.0 + rel)
            h_pred_dict['height_instance_relative'] = rel
        else:
            refined_input = refined_base + corr_map.to(dtype=base.dtype)
        refined = _activate_height(refined_input, args=self.args)
        h_pred_dict['height_pre_instance'] = h_pred_dict['height']
        h_pred_dict['height'] = refined
        h_pred_dict['height_correction'] = corr_map.to(dtype=base.dtype)
        h_pred_dict['height_instance_correction'] = corr_map.to(dtype=base.dtype)

        high_residual_after_floor = bool(_cfg_value(self.args, 'height_high_residual_after_learned_floor', False))

        def _apply_high_residual(h_pred_dict):
            current_height = torch.nan_to_num(
                h_pred_dict['height'].float(),
                nan=0.0,
                posinf=float(_cfg_value(self.args, 'height_output_clip', 320.0)),
                neginf=0.0,
            )
            high_residual_beta = max(float(_cfg_value(self.args, 'height_high_residual_softplus_beta', 2.0)), 1.0e-3)
            high_activation = str(_cfg_value(self.args, 'height_high_residual_activation', 'centered_softplus')).lower()
            high_input = torch.nan_to_num(high_residual_map.float(), nan=0.0, posinf=80.0, neginf=-80.0)
            if high_activation in ('softplus', 'positive_softplus', 'nonzero_softplus'):
                high_raw = F.softplus(high_input, beta=high_residual_beta)
                high_offset = float(_cfg_value(self.args, 'height_high_residual_softplus_offset', 0.0))
                if high_offset != 0.0:
                    high_raw = high_raw + high_offset
            elif high_activation in ('exp', 'exponential'):
                high_raw = torch.exp(torch.clamp(high_input, min=-20.0, max=8.0))
            else:
                high_raw = F.softplus(high_input, beta=high_residual_beta) - (math.log(2.0) / high_residual_beta)
                high_raw = torch.clamp(high_raw, min=0.0)
            high_raw = torch.nan_to_num(high_raw, nan=0.0, posinf=320.0, neginf=0.0)

            phys_thr = float(_cfg_value(self.args, 'height_high_residual_physical_gate_threshold', 9.0))
            phys_width = max(float(_cfg_value(self.args, 'height_high_residual_physical_gate_width', 8.0)), 1e-3)
            base_thr = float(_cfg_value(self.args, 'height_high_residual_base_gate_threshold', 14.0))
            base_width = max(float(_cfg_value(self.args, 'height_high_residual_base_gate_width', 8.0)), 1e-3)
            delta_thr = float(_cfg_value(self.args, 'height_high_residual_delta_gate_threshold', 4.5))
            delta_width = max(float(_cfg_value(self.args, 'height_high_residual_delta_gate_width', 6.0)), 1e-3)
            area_thr = float(_cfg_value(self.args, 'height_high_residual_log_area_gate_threshold', 6.2))
            area_width = max(float(_cfg_value(self.args, 'height_high_residual_log_area_gate_width', 1.4)), 1e-3)

            phys_gate = torch.sigmoid((physical.float() - phys_thr) / phys_width)
            base_signal = torch.maximum(current_height, base.float())
            base_gate = torch.sigmoid((base_signal - base_thr) / base_width)
            delta_gate = torch.sigmoid((delta.float() - delta_thr) / delta_width)
            area_gate = torch.sigmoid((log_area.float() - area_thr) / area_width)
            geom_gate = torch.maximum(torch.maximum(phys_gate, base_gate), delta_gate)
            gate_mode = str(_cfg_value(self.args, 'height_high_residual_gate_mode', 'area_and_geom')).lower()
            if gate_mode in ('product', 'all_product', 'phys_base_delta_area'):
                evidence_gate = phys_gate * base_gate * delta_gate * area_gate
            elif gate_mode in ('min', 'and'):
                evidence_gate = torch.minimum(torch.minimum(phys_gate, base_gate), torch.minimum(delta_gate, area_gate))
            elif gate_mode in ('base', 'base_only', 'height_base'):
                evidence_gate = base_gate
            elif gate_mode in ('base_and_area', 'base_area'):
                area_floor = max(0.0, min(1.0, float(_cfg_value(self.args, 'height_high_residual_area_floor', 0.15))))
                evidence_gate = base_gate * (area_floor + (1.0 - area_floor) * area_gate)
            elif gate_mode in ('base_or_delta', 'base_delta_max'):
                evidence_gate = torch.maximum(base_gate, delta_gate)
            elif gate_mode in ('area', 'area_only', 'log_area'):
                evidence_gate = area_gate
            elif gate_mode in ('area_and_geom_delta', 'area_and_geom', 'geom_area_product'):
                evidence_gate = geom_gate * area_gate
            elif gate_mode in ('soft_area_and_geom', 'geom_area_soft_product'):
                area_floor = max(0.0, min(1.0, float(_cfg_value(self.args, 'height_high_residual_area_floor', 0.15))))
                evidence_gate = geom_gate * (area_floor + (1.0 - area_floor) * area_gate)
            elif gate_mode in ('max', 'or'):
                evidence_gate = torch.maximum(geom_gate, area_gate)
            else:
                evidence_gate = geom_gate * area_gate
            gate_source = str(_cfg_value(self.args, 'height_high_residual_gate_source', 'computed')).lower()
            learned_gate = None
            learned_gate_logit = None
            if self.high_residual_gate_head is not None:
                learned_gate_logit = torch.nan_to_num(
                    high_residual_gate_logit_map.float(),
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                )
                gate_temperature = max(
                    float(_cfg_value(self.args, 'height_high_residual_learned_gate_temperature', 1.0)),
                    1.0e-3,
                )
                gate_bias = float(_cfg_value(self.args, 'height_high_residual_learned_gate_bias', 0.0))
                learned_gate = torch.sigmoid((learned_gate_logit + gate_bias) / gate_temperature)
                learned_gate = torch.clamp(
                    torch.nan_to_num(learned_gate, nan=0.0, posinf=1.0, neginf=0.0),
                    0.0,
                    1.0,
                )
            if gate_source in ('learned', 'learned_gate', 'instance_learned_gate', 'high_residual_learned_gate') and learned_gate is not None:
                evidence_gate = learned_gate
            elif gate_source in ('learned_times_computed', 'computed_times_learned', 'learned_x_computed') and learned_gate is not None:
                evidence_gate = evidence_gate * learned_gate
            elif gate_source in ('learned_max_computed', 'max_learned_computed') and learned_gate is not None:
                evidence_gate = torch.maximum(evidence_gate, learned_gate)
            if gate_source in (
                'learned_floor_evidence',
                'learned_physical_floor_evidence',
                'learned_physical_floor_evidence_gate',
            ) and 'height_learned_physical_floor_evidence_gate' in h_pred_dict:
                evidence_gate = torch.nan_to_num(
                    h_pred_dict['height_learned_physical_floor_evidence_gate'].float(),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )
            elif gate_source in (
                'learned_floor_gate',
                'learned_physical_floor_gate',
            ) and 'height_learned_physical_floor_gate' in h_pred_dict:
                evidence_gate = torch.nan_to_num(
                    h_pred_dict['height_learned_physical_floor_gate'].float(),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )
            evidence_gate = torch.clamp(torch.nan_to_num(evidence_gate, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            gate_power = max(float(_cfg_value(self.args, 'height_high_residual_gate_power', 1.0)), 1.0e-3)
            if abs(gate_power - 1.0) > 1.0e-6:
                evidence_gate = _safe_gate_power(evidence_gate, gate_power)
            min_evidence = float(_cfg_value(self.args, 'height_high_residual_min_evidence', 0.0))
            if min_evidence > 0.0:
                evidence_gate = torch.where(
                    evidence_gate >= min_evidence,
                    evidence_gate,
                    torch.zeros_like(evidence_gate),
                )
            gate_scale = float(_cfg_value(self.args, 'height_high_residual_gate_scale', 1.0))
            high_gate = torch.clamp(evidence_gate * gate_scale, 0.0, 1.0)

            if bool(_cfg_value(self.args, 'height_high_residual_low_suppress_enabled', False)):
                suppress_threshold = float(_cfg_value(self.args, 'height_high_residual_low_suppress_threshold', 14.0))
                suppress_width = max(
                    float(_cfg_value(self.args, 'height_high_residual_low_suppress_width', 3.0)),
                    1.0e-3,
                )
                suppress_floor = max(
                    0.0,
                    min(1.0, float(_cfg_value(self.args, 'height_high_residual_low_suppress_floor', 0.0))),
                )
                suppress_power = max(
                    float(_cfg_value(self.args, 'height_high_residual_low_suppress_power', 1.0)),
                    1.0e-3,
                )
                suppress_signal = torch.maximum(current_height, base.float())
                low_suppress_gate = torch.sigmoid((suppress_signal - suppress_threshold) / suppress_width)
                if abs(suppress_power - 1.0) > 1.0e-6:
                    low_suppress_gate = _safe_gate_power(low_suppress_gate, suppress_power)
                low_suppress_gate = suppress_floor + (1.0 - suppress_floor) * low_suppress_gate
                low_suppress_gate = torch.clamp(
                    torch.nan_to_num(low_suppress_gate, nan=0.0, posinf=1.0, neginf=0.0),
                    0.0,
                    1.0,
                )
                high_gate = high_gate * low_suppress_gate
                evidence_gate = evidence_gate * low_suppress_gate
                h_pred_dict['height_high_residual_low_suppress_gate'] = low_suppress_gate.to(dtype=base.dtype)

            residual_mode = str(_cfg_value(self.args, 'height_high_residual_mode', 'additive')).lower()
            if residual_mode in (
                'physical_ratio',
                'phys_ratio',
                'physical_scaled',
                'delta_physical_ratio',
                'differential_disparity_ratio',
            ):
                basis_source = str(_cfg_value(self.args, 'height_high_residual_basis_source', 'physical')).lower()
                if basis_source in ('base', 'height_base', 'current_base'):
                    residual_basis = base.float()
                elif basis_source in ('current', 'height', 'pre_height'):
                    residual_basis = current_height
                elif basis_source in ('max_base_physical', 'max_physical_base', 'max'):
                    residual_basis = torch.maximum(base.float(), physical.float())
                else:
                    residual_basis = physical.float()
                basis_min = max(0.0, float(_cfg_value(self.args, 'height_high_residual_basis_min', 0.0)))
                basis_max = float(_cfg_value(self.args, 'height_high_residual_basis_max', 120.0))
                residual_basis = torch.nan_to_num(residual_basis, nan=0.0, posinf=basis_max, neginf=0.0)
                if basis_max > 0:
                    residual_basis = torch.clamp(residual_basis, min=basis_min, max=basis_max)
                else:
                    residual_basis = torch.clamp(residual_basis, min=basis_min)
                high_residual = high_gate * high_raw * residual_basis
                h_pred_dict['height_high_residual_basis'] = residual_basis.to(dtype=base.dtype)
            else:
                high_residual = high_gate * high_raw
            clip = float(_cfg_value(self.args, 'height_high_residual_boost_clip', 80.0))
            low_clip = max(0.0, float(_cfg_value(self.args, 'height_high_residual_low_boost_clip', 0.02)))
            if clip > 0:
                adaptive_clip = low_clip + evidence_gate * max(0.0, clip - low_clip)
                high_residual = torch.minimum(torch.clamp(high_residual, min=0.0), adaptive_clip)
            else:
                high_residual = torch.clamp(high_residual, min=0.0)

            if bool(_cfg_value(self.args, 'height_high_residual_base_clip_enabled', False)):
                base_clip_thr = float(_cfg_value(self.args, 'height_high_residual_base_clip_threshold', 11.5))
                base_clip_width = max(
                    float(_cfg_value(self.args, 'height_high_residual_base_clip_width', 0.75)),
                    1.0e-3,
                )
                base_clip_low = max(
                    0.0,
                    float(_cfg_value(self.args, 'height_high_residual_base_clip_low', low_clip)),
                )
                base_clip_high = float(_cfg_value(self.args, 'height_high_residual_base_clip_high', clip))
                if base_clip_high <= 0.0:
                    base_clip_high = clip if clip > 0 else float(_cfg_value(self.args, 'height_output_clip', 320.0))
                base_clip_power = max(
                    float(_cfg_value(self.args, 'height_high_residual_base_clip_power', 1.0)),
                    1.0e-3,
                )
                base_clip_signal = torch.maximum(current_height, base.float())
                base_clip_gate = torch.sigmoid((base_clip_signal - base_clip_thr) / base_clip_width)
                if abs(base_clip_power - 1.0) > 1.0e-6:
                    base_clip_gate = _safe_gate_power(base_clip_gate, base_clip_power)
                base_clip_gate = torch.clamp(
                    torch.nan_to_num(base_clip_gate, nan=0.0, posinf=1.0, neginf=0.0),
                    0.0,
                    1.0,
                )
                base_adaptive_clip = base_clip_low + base_clip_gate * max(0.0, base_clip_high - base_clip_low)
                high_residual = torch.minimum(high_residual, base_adaptive_clip)

            if bool(_cfg_value(self.args, 'height_high_residual_piecewise_clip_enabled', False)):
                signal_key = str(_cfg_value(self.args, 'height_high_residual_piecewise_signal', 'base_current')).lower()
                if signal_key in ('base', 'height_base'):
                    clip_signal = base.float()
                elif signal_key in ('current', 'height', 'pre'):
                    clip_signal = current_height
                elif signal_key in ('physical', 'phys'):
                    clip_signal = physical.float()
                else:
                    clip_signal = torch.maximum(current_height, base.float())

                b0 = float(_cfg_value(self.args, 'height_high_residual_piecewise_boundary0', 12.0))
                b1 = float(_cfg_value(self.args, 'height_high_residual_piecewise_boundary1', 18.0))
                b2 = float(_cfg_value(self.args, 'height_high_residual_piecewise_boundary2', 24.0))
                c0 = max(0.0, float(_cfg_value(self.args, 'height_high_residual_piecewise_clip0', 0.0)))
                c1 = max(c0, float(_cfg_value(self.args, 'height_high_residual_piecewise_clip1', 1.5)))
                c2 = max(c1, float(_cfg_value(self.args, 'height_high_residual_piecewise_clip2', 5.0)))
                c3 = max(c2, float(_cfg_value(self.args, 'height_high_residual_piecewise_clip3', clip if clip > 0 else 80.0)))
                width = max(
                    float(_cfg_value(self.args, 'height_high_residual_piecewise_width', 1.5)),
                    1.0e-3,
                )
                g0 = torch.sigmoid((clip_signal - b0) / width)
                g1 = torch.sigmoid((clip_signal - b1) / width)
                g2 = torch.sigmoid((clip_signal - b2) / width)
                piece_clip = c0 + (c1 - c0) * g0 + (c2 - c1) * g1 + (c3 - c2) * g2
                if bool(_cfg_value(self.args, 'height_high_residual_piecewise_require_evidence', False)):
                    ev_thr = float(_cfg_value(self.args, 'height_high_residual_piecewise_evidence_threshold', 0.25))
                    ev_width = max(
                        float(_cfg_value(self.args, 'height_high_residual_piecewise_evidence_width', 0.08)),
                        1.0e-3,
                    )
                    ev_gate = torch.sigmoid((evidence_gate - ev_thr) / ev_width)
                    ev_floor = max(
                        0.0,
                        min(1.0, float(_cfg_value(self.args, 'height_high_residual_piecewise_evidence_floor', 0.0))),
                    )
                    ev_gate = ev_floor + (1.0 - ev_floor) * ev_gate
                    piece_clip = c0 + ev_gate * (piece_clip - c0)
                high_residual = torch.minimum(high_residual, piece_clip)
                h_pred_dict['height_high_residual_piecewise_clip'] = piece_clip.to(dtype=base.dtype)

            height_max = float(_cfg_value(self.args, 'height_output_clip', 320.0))
            h_pred_dict['height_pre_high_residual'] = h_pred_dict['height']
            h_pred_dict['height_high_residual_raw'] = high_raw.to(dtype=base.dtype)
            if learned_gate_logit is not None and learned_gate is not None:
                h_pred_dict['height_high_residual_gate_logit'] = learned_gate_logit.to(dtype=base.dtype)
                h_pred_dict['height_high_residual_learned_gate'] = learned_gate.to(dtype=base.dtype)
            h_pred_dict['height_high_residual_gate'] = high_gate.to(dtype=base.dtype)
            h_pred_dict['height_high_residual_evidence_gate'] = evidence_gate.to(dtype=base.dtype)
            h_pred_dict['height_high_residual'] = high_residual.to(dtype=base.dtype)
            h_pred_dict['height'] = torch.clamp(
                torch.nan_to_num(current_height + high_residual, nan=0.0, posinf=height_max, neginf=0.0),
                min=0.0,
                max=height_max,
            ).to(dtype=base.dtype)
            return h_pred_dict

        if self.high_residual_head is not None and not high_residual_after_floor:
            h_pred_dict = _apply_high_residual(h_pred_dict)
        if self.learned_physical_floor_head is not None:
            current = torch.nan_to_num(h_pred_dict['height'], nan=0.0, posinf=320.0, neginf=0.0).float()
            floor_raw = torch.nan_to_num(learned_floor_raw_map.float(), nan=0.0, posinf=80.0, neginf=-80.0)

            floor_base_key = str(_cfg_value(self.args, 'height_learned_physical_floor_base_key', 'height_base'))
            if floor_base_key not in h_pred_dict:
                floor_base_key = 'height_base' if 'height_base' in h_pred_dict else 'height'
            base_signal = torch.nan_to_num(h_pred_dict[floor_base_key], nan=0.0, posinf=320.0, neginf=0.0).float()
            if bool(_cfg_value(self.args, 'height_learned_physical_floor_use_current_signal', True)):
                base_signal = torch.maximum(base_signal, current)

            phys_thr = float(_cfg_value(self.args, 'height_learned_physical_floor_phys_threshold', 10.0))
            phys_width = max(float(_cfg_value(self.args, 'height_learned_physical_floor_phys_width', 4.0)), 1.0e-3)
            base_thr = float(_cfg_value(self.args, 'height_learned_physical_floor_base_threshold', 16.0))
            base_width = max(float(_cfg_value(self.args, 'height_learned_physical_floor_base_width', 4.0)), 1.0e-3)
            area_thr = float(_cfg_value(self.args, 'height_learned_physical_floor_log_area_threshold', 5.5))
            area_width = max(float(_cfg_value(self.args, 'height_learned_physical_floor_log_area_width', 0.8)), 1.0e-3)
            delta_thr = float(_cfg_value(self.args, 'height_learned_physical_floor_delta_threshold', 4.0))
            delta_width = max(float(_cfg_value(self.args, 'height_learned_physical_floor_delta_width', 6.0)), 1.0e-3)

            phys_gate = torch.sigmoid((physical.float() - phys_thr) / phys_width)
            base_gate = torch.sigmoid((base_signal - base_thr) / base_width)
            area_gate = torch.sigmoid((log_area.float() - area_thr) / area_width)
            delta_gate = torch.sigmoid((delta.float() - delta_thr) / delta_width)
            mode = str(_cfg_value(self.args, 'height_learned_physical_floor_evidence_mode', 'hard_phys_base_area')).lower()
            ratio = float(_cfg_value(self.args, 'height_learned_physical_floor_ratio', 1.35))
            bias = float(_cfg_value(self.args, 'height_learned_physical_floor_bias', 6.0))
            target_clip = float(_cfg_value(self.args, 'height_learned_physical_floor_target_clip', 120.0))
            floor_target = physical.float() * ratio + bias
            if target_clip > 0:
                floor_target = torch.clamp(floor_target, min=0.0, max=target_clip)
            target_gap = F.relu(floor_target - current)

            if mode in ('phys_base_area_delta', 'product', 'all_product'):
                evidence_gate = phys_gate * base_gate * area_gate * delta_gate
            elif mode in ('phys_base_area', 'physical_base_area'):
                evidence_gate = phys_gate * base_gate * area_gate
            elif mode in ('phys_area_delta', 'physical_area_delta'):
                evidence_gate = phys_gate * area_gate * delta_gate
            elif mode in ('geom_area_delta', 'geometry_area_delta'):
                evidence_gate = torch.maximum(phys_gate, base_gate) * area_gate * delta_gate
            elif mode in ('min', 'and'):
                evidence_gate = torch.minimum(torch.minimum(phys_gate, base_gate), torch.minimum(area_gate, delta_gate))
            elif mode in ('hard_phys_area', 'hard_physical_area'):
                hard_phys_min = float(_cfg_value(self.args, 'height_learned_physical_floor_hard_phys_min', phys_thr))
                hard_area_min = float(_cfg_value(self.args, 'height_learned_physical_floor_hard_log_area_min', area_thr))
                hard_mask = (physical.float() >= hard_phys_min) & (log_area.float() >= hard_area_min)
                evidence_gate = phys_gate * area_gate * torch.maximum(base_gate, delta_gate) * hard_mask.float()
            elif mode in ('hard_phys_base_area', 'hard_physical_base_area', 'hard_base_area'):
                hard_phys_min = float(_cfg_value(self.args, 'height_learned_physical_floor_hard_phys_min', phys_thr))
                hard_base_min = float(_cfg_value(self.args, 'height_learned_physical_floor_hard_base_min', base_thr))
                hard_area_min = float(_cfg_value(self.args, 'height_learned_physical_floor_hard_log_area_min', area_thr))
                hard_mask = (
                    (physical.float() >= hard_phys_min)
                    & (base_signal >= hard_base_min)
                    & (log_area.float() >= hard_area_min)
                )
                evidence_gate = phys_gate * base_gate * area_gate * hard_mask.float()
            elif mode in (
                'hard_phys_area_or_target_gap',
                'hard_physical_area_or_target_gap',
                'hard_base_area_or_gap',
            ):
                hard_phys_min = float(_cfg_value(self.args, 'height_learned_physical_floor_hard_phys_min', phys_thr))
                hard_base_min = float(_cfg_value(self.args, 'height_learned_physical_floor_hard_base_min', base_thr))
                hard_area_min = float(_cfg_value(self.args, 'height_learned_physical_floor_hard_log_area_min', area_thr))
                gap_min = float(_cfg_value(self.args, 'height_learned_physical_floor_hard_target_gap_min', 18.0))
                gap_thr = float(_cfg_value(self.args, 'height_learned_physical_floor_target_gap_threshold', gap_min))
                gap_width = max(float(_cfg_value(self.args, 'height_learned_physical_floor_target_gap_width', 6.0)), 1.0e-3)
                gap_gate = torch.sigmoid((target_gap - gap_thr) / gap_width)
                hard_main = (
                    (physical.float() >= hard_phys_min)
                    & (base_signal >= hard_base_min)
                    & (log_area.float() >= hard_area_min)
                )
                hard_gap = (
                    (physical.float() >= hard_phys_min)
                    & (target_gap >= gap_min)
                    & (log_area.float() >= hard_area_min)
                )
                gap_mix = max(0.0, float(_cfg_value(self.args, 'height_learned_physical_floor_gap_path_mix', 0.85)))
                main_gate = phys_gate * base_gate * area_gate * hard_main.float()
                bypass_gate = gap_mix * phys_gate * area_gate * gap_gate * hard_gap.float()
                evidence_gate = torch.maximum(main_gate, bypass_gate)
            else:
                evidence_gate = torch.maximum(phys_gate, base_gate) * area_gate
            evidence_gate = torch.clamp(torch.nan_to_num(evidence_gate, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            evidence_power = max(float(_cfg_value(self.args, 'height_learned_physical_floor_evidence_power', 1.0)), 1.0e-3)
            if abs(evidence_power - 1.0) > 1.0e-6:
                evidence_gate = _safe_gate_power(evidence_gate, evidence_power)
            min_evidence = float(_cfg_value(self.args, 'height_learned_physical_floor_min_evidence', 0.0))
            if min_evidence > 0.0:
                evidence_gate = torch.where(
                    evidence_gate >= min_evidence,
                    evidence_gate,
                    torch.zeros_like(evidence_gate),
                )
            gate_scale = float(_cfg_value(self.args, 'height_learned_physical_floor_gate_scale', 1.0))
            floor_gate = torch.clamp(evidence_gate * gate_scale, 0.0, 1.0)

            frac_min = float(_cfg_value(self.args, 'height_learned_physical_floor_fraction_min', 0.0))
            frac_max = float(_cfg_value(self.args, 'height_learned_physical_floor_fraction_max', 1.0))
            if frac_max < frac_min:
                frac_min, frac_max = frac_max, frac_min
            frac_temp = max(float(_cfg_value(self.args, 'height_learned_physical_floor_fraction_temperature', 1.0)), 1.0e-3)
            frac_bias = float(_cfg_value(self.args, 'height_learned_physical_floor_fraction_bias', 0.0))
            floor_fraction = frac_min + (frac_max - frac_min) * torch.sigmoid((floor_raw + frac_bias) / frac_temp)
            fraction_floor = float(_cfg_value(self.args, 'height_learned_physical_floor_high_fraction_floor', 0.0))
            if fraction_floor > 0.0:
                high_phys_min = float(_cfg_value(self.args, 'height_learned_physical_floor_high_fraction_phys_min', 14.0))
                high_gap_min = float(_cfg_value(self.args, 'height_learned_physical_floor_high_fraction_gap_min', 12.0))
                high_area_min = float(_cfg_value(self.args, 'height_learned_physical_floor_high_fraction_log_area_min', area_thr))
                high_evidence_min = float(_cfg_value(self.args, 'height_learned_physical_floor_high_fraction_evidence_min', 0.45))
                high_floor_mask = (
                    (physical.float() >= high_phys_min)
                    & (target_gap >= high_gap_min)
                    & (log_area.float() >= high_area_min)
                    & (evidence_gate >= high_evidence_min)
                )
                floor_fraction = torch.where(
                    high_floor_mask,
                    torch.maximum(floor_fraction, torch.full_like(floor_fraction, fraction_floor)),
                    floor_fraction,
                )

            margin = float(_cfg_value(self.args, 'height_learned_physical_floor_margin', 0.5))
            scale = float(_cfg_value(self.args, 'height_learned_physical_floor_scale', 1.0))
            floor_raise = scale * floor_gate * floor_fraction * F.relu(floor_target - current - margin)
            raise_clip = float(_cfg_value(self.args, 'height_learned_physical_floor_raise_clip', 24.0))
            low_clip = max(0.0, float(_cfg_value(self.args, 'height_learned_physical_floor_low_raise_clip', 0.0)))
            if raise_clip > 0:
                adaptive_clip_enabled = bool(_cfg_value(self.args, 'height_learned_physical_floor_adaptive_clip', True))
                if adaptive_clip_enabled:
                    adaptive_clip = low_clip + evidence_gate * max(0.0, raise_clip - low_clip)
                    floor_raise = torch.minimum(torch.clamp(floor_raise, min=0.0), adaptive_clip)
                else:
                    floor_raise = torch.clamp(floor_raise, min=0.0, max=raise_clip)
            else:
                floor_raise = torch.clamp(floor_raise, min=0.0)

            height_max = float(_cfg_value(self.args, 'height_output_clip', 320.0))
            h_pred_dict['height_pre_learned_physical_floor'] = h_pred_dict['height']
            h_pred_dict['height_learned_physical_floor_raw'] = floor_raw.to(dtype=base.dtype)
            h_pred_dict['height_learned_physical_floor_fraction'] = floor_fraction.to(dtype=base.dtype)
            h_pred_dict['height_learned_physical_floor_target'] = floor_target.to(dtype=base.dtype)
            h_pred_dict['height_learned_physical_floor_gate'] = floor_gate.to(dtype=base.dtype)
            h_pred_dict['height_learned_physical_floor_evidence_gate'] = evidence_gate.to(dtype=base.dtype)
            h_pred_dict['height_learned_physical_floor_raise'] = floor_raise.to(dtype=base.dtype)
            if bool(_cfg_value(self.args, 'height_learned_physical_floor_apply_to_output', True)):
                h_pred_dict['height'] = torch.clamp(
                    torch.nan_to_num(current + floor_raise, nan=0.0, posinf=height_max, neginf=0.0),
                    min=0.0,
                    max=height_max,
                ).to(dtype=base.dtype)
        if self.high_residual_head is not None and high_residual_after_floor:
            h_pred_dict = _apply_high_residual(h_pred_dict)
        return h_pred_dict

    def _apply_post_base_high_boost(self, h_pred_dict, mask, area_maps):
        if not bool(_cfg_value(self.args, 'height_post_base_high_boost', False)):
            return h_pred_dict

        source_key = str(_cfg_value(self.args, 'height_post_base_source_key', 'height'))
        if source_key not in h_pred_dict:
            source_key = 'height'
        base = torch.nan_to_num(h_pred_dict[source_key], nan=0.0, posinf=320.0, neginf=0.0).float()
        physical = torch.nan_to_num(h_pred_dict['height_physical'], nan=0.0, posinf=320.0, neginf=0.0).float()
        if area_maps is None:
            area_maps = self._build_area_maps(mask)
        log_area = area_maps[:, 0:1].to(device=base.device, dtype=torch.float32)

        boost_source = str(_cfg_value(self.args, 'height_post_base_boost_source', 'boost')).lower()
        raw_boost = torch.nan_to_num(
            h_pred_dict.get('height_high_boost_raw', torch.zeros_like(base)),
            nan=0.0,
            posinf=120.0,
            neginf=0.0,
        ).float()
        learned_gate = torch.clamp(
            torch.nan_to_num(
                h_pred_dict.get('height_high_gate', torch.zeros_like(base)),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).float(),
            0.0,
            1.0,
        )
        height_aux = torch.nan_to_num(
            h_pred_dict.get('height_aux', base),
            nan=0.0,
            posinf=320.0,
            neginf=0.0,
        ).float()
        aux_gap = F.relu(height_aux - base)
        aux_gap_scale = float(_cfg_value(self.args, 'height_post_base_aux_gap_scale', 1.0))
        if aux_gap_scale != 1.0:
            aux_gap = aux_gap * aux_gap_scale
        aux_gap_clip = float(_cfg_value(self.args, 'height_post_base_aux_gap_clip', 0.0))
        if aux_gap_clip > 0:
            aux_gap = torch.clamp(aux_gap, max=aux_gap_clip)

        if boost_source in ('raw', 'raw_boost', 'ungated'):
            learned_boost = raw_boost
            learned_gate_for_boost = learned_gate
        elif boost_source in ('aux_gap', 'aux_raise', 'aux_minus_base'):
            learned_boost = aux_gap
            learned_gate_for_boost = learned_gate
        elif boost_source in ('max_raw_aux', 'raw_aux_max', 'max_aux_raw'):
            learned_boost = torch.maximum(raw_boost, aux_gap)
            learned_gate_for_boost = learned_gate
        elif boost_source in ('blend_raw_aux', 'raw_aux_blend', 'blend_aux_raw'):
            aux_weight = max(0.0, min(1.0, float(_cfg_value(self.args, 'height_post_base_aux_gap_weight', 0.5))))
            learned_boost = (1.0 - aux_weight) * raw_boost + aux_weight * aux_gap
            learned_gate_for_boost = learned_gate
        else:
            learned_boost = torch.nan_to_num(
                h_pred_dict.get('height_high_boost', torch.zeros_like(base)),
                nan=0.0,
                posinf=120.0,
                neginf=0.0,
            ).float()
            learned_gate_for_boost = torch.ones_like(learned_gate)
        learned_gate_floor = max(0.0, min(1.0, float(_cfg_value(self.args, 'height_post_base_learned_gate_floor', 0.0))))
        if not bool(_cfg_value(self.args, 'height_post_base_use_learned_gate', True)):
            learned_gate_for_boost = torch.ones_like(learned_gate_for_boost)
        elif learned_gate_floor > 0 and boost_source in (
            'raw',
            'raw_boost',
            'ungated',
            'aux_gap',
            'aux_raise',
            'aux_minus_base',
            'max_raw_aux',
            'raw_aux_max',
            'max_aux_raw',
            'blend_raw_aux',
            'raw_aux_blend',
            'blend_aux_raw',
        ):
            learned_gate_for_boost = torch.maximum(
                learned_gate_for_boost,
                torch.full_like(learned_gate_for_boost, learned_gate_floor),
            )

        phys_thr = float(_cfg_value(self.args, 'height_post_base_phys_threshold', 10.0))
        phys_width = max(float(_cfg_value(self.args, 'height_post_base_phys_width', 5.0)), 1.0e-3)
        base_thr = float(_cfg_value(self.args, 'height_post_base_base_threshold', 12.0))
        base_width = max(float(_cfg_value(self.args, 'height_post_base_base_width', 6.0)), 1.0e-3)
        area_thr = float(_cfg_value(self.args, 'height_post_base_log_area_threshold', 6.5))
        area_width = max(float(_cfg_value(self.args, 'height_post_base_log_area_width', 1.0)), 1.0e-3)
        phys_gate = torch.sigmoid((physical - phys_thr) / phys_width)
        base_gate = torch.sigmoid((base - base_thr) / base_width)
        area_gate = torch.sigmoid((log_area - area_thr) / area_width)
        aux_gap_thr = float(_cfg_value(self.args, 'height_post_base_aux_gap_gate_threshold', -1.0))
        aux_gap_width = max(float(_cfg_value(self.args, 'height_post_base_aux_gap_gate_width', 4.0)), 1.0e-3)
        if aux_gap_thr >= 0.0:
            aux_gap_gate = torch.sigmoid((aux_gap.float() - aux_gap_thr) / aux_gap_width)
        else:
            aux_gap_gate = torch.ones_like(phys_gate)

        evidence_mode = str(_cfg_value(self.args, 'height_post_base_evidence_mode', 'phys_area_product')).lower()
        if evidence_mode in ('phys_area_product', 'physical_area_product', 'product'):
            evidence_gate = phys_gate * area_gate
        elif evidence_mode in ('geom_area_product', 'geometry_area_product'):
            evidence_gate = torch.maximum(phys_gate, base_gate) * area_gate
        elif evidence_mode in ('geom_area_soft_product', 'geometry_area_soft_product'):
            area_floor = max(0.0, min(1.0, float(_cfg_value(self.args, 'height_post_base_area_floor', 0.25))))
            evidence_gate = torch.maximum(phys_gate, base_gate) * (area_floor + (1.0 - area_floor) * area_gate)
        elif evidence_mode in ('min', 'and'):
            evidence_gate = torch.minimum(torch.minimum(phys_gate, base_gate), area_gate)
        elif evidence_mode in ('max', 'or'):
            evidence_gate = torch.maximum(torch.maximum(phys_gate, base_gate), area_gate)
        else:
            evidence_gate = torch.maximum(phys_gate, base_gate) * area_gate

        if str(_cfg_value(self.args, 'height_post_base_aux_gap_gate_mode', 'multiply')).lower() in ('multiply', 'product', 'and'):
            evidence_gate = evidence_gate * aux_gap_gate
        evidence_gate = torch.clamp(torch.nan_to_num(evidence_gate, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        evidence_power = max(float(_cfg_value(self.args, 'height_post_base_evidence_power', 1.0)), 1.0e-3)
        if evidence_power != 1.0:
            evidence_gate = _safe_gate_power(evidence_gate, evidence_power)
        gate_scale = float(_cfg_value(self.args, 'height_post_base_gate_scale', 1.0))
        post_gate = torch.clamp(learned_gate_for_boost * evidence_gate * gate_scale, 0.0, 1.0)
        boost_scale = float(_cfg_value(self.args, 'height_post_base_boost_scale', 1.0))
        post_boost = boost_scale * post_gate * torch.clamp(learned_boost, min=0.0)

        boost_clip = float(_cfg_value(self.args, 'height_post_base_boost_clip', 80.0))
        low_clip = max(0.0, float(_cfg_value(self.args, 'height_post_base_low_boost_clip', 0.15)))
        if boost_clip > 0:
            adaptive_clip = low_clip + evidence_gate * max(0.0, boost_clip - low_clip)
            post_boost = torch.minimum(torch.clamp(post_boost, min=0.0), adaptive_clip)
        else:
            post_boost = torch.clamp(post_boost, min=0.0)

        height_max = float(_cfg_value(self.args, 'height_output_clip', 320.0))
        h_pred_dict['height_pre_post_base_boost'] = h_pred_dict['height']
        h_pred_dict['height_post_base_gate'] = post_gate.to(dtype=h_pred_dict['height'].dtype)
        h_pred_dict['height_post_base_evidence_gate'] = evidence_gate.to(dtype=h_pred_dict['height'].dtype)
        h_pred_dict['height_post_base_high_boost'] = post_boost.to(dtype=h_pred_dict['height'].dtype)
        h_pred_dict['height_post_base_aux_gap'] = aux_gap.to(dtype=h_pred_dict['height'].dtype)
        h_pred_dict['height_post_base_aux_gap_gate'] = aux_gap_gate.to(dtype=h_pred_dict['height'].dtype)
        h_pred_dict['height'] = torch.clamp(
            torch.nan_to_num(base + post_boost, nan=0.0, posinf=height_max, neginf=0.0),
            min=0.0,
            max=height_max,
        ).to(dtype=h_pred_dict['height'].dtype)
        return h_pred_dict

    def _apply_aux_height_floor(self, h_pred_dict, mask, area_maps):
        if not bool(_cfg_value(self.args, 'height_aux_floor_enabled', False)):
            return h_pred_dict
        if 'height_aux' not in h_pred_dict or 'height' not in h_pred_dict:
            return h_pred_dict

        current = torch.nan_to_num(h_pred_dict['height'], nan=0.0, posinf=320.0, neginf=0.0).float()
        aux = torch.nan_to_num(h_pred_dict['height_aux'], nan=0.0, posinf=320.0, neginf=0.0).float()
        physical = torch.nan_to_num(
            h_pred_dict.get('height_physical', current),
            nan=0.0,
            posinf=320.0,
            neginf=0.0,
        ).float()
        if area_maps is None:
            area_maps = self._build_area_maps(mask)
        log_area = area_maps[:, 0:1].to(device=current.device, dtype=torch.float32)

        gap_base_key = str(_cfg_value(self.args, 'height_aux_floor_gap_base_key', 'height_pre_post_base_boost'))
        if gap_base_key not in h_pred_dict:
            gap_base_key = 'height_base' if 'height_base' in h_pred_dict else 'height'
        gap_base = torch.nan_to_num(h_pred_dict[gap_base_key], nan=0.0, posinf=320.0, neginf=0.0).float()
        aux_gap = F.relu(aux - gap_base)

        base_gate_key = str(_cfg_value(self.args, 'height_aux_floor_base_gate_key', 'height_pre_post_base_boost'))
        if base_gate_key not in h_pred_dict:
            base_gate_key = 'height_base' if 'height_base' in h_pred_dict else 'height'
        base_signal = torch.nan_to_num(h_pred_dict[base_gate_key], nan=0.0, posinf=320.0, neginf=0.0).float()

        phys_thr = float(_cfg_value(self.args, 'height_aux_floor_phys_threshold', 12.0))
        phys_width = max(float(_cfg_value(self.args, 'height_aux_floor_phys_width', 4.0)), 1.0e-3)
        base_thr = float(_cfg_value(self.args, 'height_aux_floor_base_threshold', 12.0))
        base_width = max(float(_cfg_value(self.args, 'height_aux_floor_base_width', 5.0)), 1.0e-3)
        area_thr = float(_cfg_value(self.args, 'height_aux_floor_log_area_threshold', 6.6))
        area_width = max(float(_cfg_value(self.args, 'height_aux_floor_log_area_width', 0.9)), 1.0e-3)
        aux_gap_thr = float(_cfg_value(self.args, 'height_aux_floor_gap_threshold', 4.0))
        aux_gap_width = max(float(_cfg_value(self.args, 'height_aux_floor_gap_width', 3.0)), 1.0e-3)

        phys_gate = torch.sigmoid((physical - phys_thr) / phys_width)
        base_gate = torch.sigmoid((base_signal - base_thr) / base_width)
        area_gate = torch.sigmoid((log_area - area_thr) / area_width)
        if aux_gap_thr >= 0:
            gap_gate = torch.sigmoid((aux_gap - aux_gap_thr) / aux_gap_width)
        else:
            gap_gate = torch.ones_like(phys_gate)

        geom_gate = torch.maximum(phys_gate, base_gate)
        mode = str(_cfg_value(self.args, 'height_aux_floor_evidence_mode', 'geom_area_gap')).lower()
        if mode in ('phys_area_gap', 'physical_area_gap'):
            evidence_gate = phys_gate * area_gate * gap_gate
        elif mode in ('geom_gap', 'geometry_gap'):
            evidence_gate = geom_gate * gap_gate
        elif mode in ('geom_soft_area_gap', 'geometry_soft_area_gap'):
            area_floor = max(0.0, min(1.0, float(_cfg_value(self.args, 'height_aux_floor_area_floor', 0.20))))
            evidence_gate = geom_gate * (area_floor + (1.0 - area_floor) * area_gate) * gap_gate
        elif mode in ('min', 'and'):
            evidence_gate = torch.minimum(torch.minimum(geom_gate, area_gate), gap_gate)
        else:
            evidence_gate = geom_gate * area_gate * gap_gate

        evidence_gate = torch.clamp(
            torch.nan_to_num(evidence_gate, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
        evidence_power = max(float(_cfg_value(self.args, 'height_aux_floor_evidence_power', 1.0)), 1.0e-3)
        if abs(evidence_power - 1.0) > 1.0e-6:
            evidence_gate = _safe_gate_power(evidence_gate, evidence_power)
        gate_scale = float(_cfg_value(self.args, 'height_aux_floor_gate_scale', 1.0))
        floor_gate = torch.clamp(evidence_gate * gate_scale, 0.0, 1.0)

        margin = float(_cfg_value(self.args, 'height_aux_floor_margin', 2.0))
        scale = float(_cfg_value(self.args, 'height_aux_floor_scale', 0.75))
        floor_raise = scale * floor_gate * F.relu(aux - current - margin)
        raise_clip = float(_cfg_value(self.args, 'height_aux_floor_raise_clip', 25.0))
        low_clip = max(0.0, float(_cfg_value(self.args, 'height_aux_floor_low_raise_clip', 0.10)))
        if raise_clip > 0:
            adaptive_clip = low_clip + evidence_gate * max(0.0, raise_clip - low_clip)
            floor_raise = torch.minimum(torch.clamp(floor_raise, min=0.0), adaptive_clip)
        else:
            floor_raise = torch.clamp(floor_raise, min=0.0)

        height_max = float(_cfg_value(self.args, 'height_output_clip', 320.0))
        h_pred_dict['height_pre_aux_floor'] = h_pred_dict['height']
        h_pred_dict['height_aux_floor_gate'] = floor_gate.to(dtype=h_pred_dict['height'].dtype)
        h_pred_dict['height_aux_floor_evidence_gate'] = evidence_gate.to(dtype=h_pred_dict['height'].dtype)
        h_pred_dict['height_aux_floor_raise'] = floor_raise.to(dtype=h_pred_dict['height'].dtype)
        h_pred_dict['height_aux_floor_gap'] = aux_gap.to(dtype=h_pred_dict['height'].dtype)
        h_pred_dict['height'] = torch.clamp(
            torch.nan_to_num(current + floor_raise, nan=0.0, posinf=height_max, neginf=0.0),
            min=0.0,
            max=height_max,
        ).to(dtype=h_pred_dict['height'].dtype)
        return h_pred_dict

    def _apply_physical_ratio_floor(self, h_pred_dict, mask, area_maps):
        if not bool(_cfg_value(self.args, 'height_physical_ratio_floor_enabled', False)):
            return h_pred_dict
        if 'height' not in h_pred_dict or 'height_physical' not in h_pred_dict:
            return h_pred_dict

        current = torch.nan_to_num(h_pred_dict['height'], nan=0.0, posinf=320.0, neginf=0.0).float()
        physical = torch.nan_to_num(
            h_pred_dict.get('height_physical', current),
            nan=0.0,
            posinf=320.0,
            neginf=0.0,
        ).float()
        if area_maps is None:
            area_maps = self._build_area_maps(mask)
        log_area = area_maps[:, 0:1].to(device=current.device, dtype=torch.float32)

        base_key = str(_cfg_value(self.args, 'height_physical_ratio_floor_base_key', 'height_pre_post_base_boost'))
        if base_key not in h_pred_dict:
            base_key = 'height_base' if 'height_base' in h_pred_dict else 'height'
        base_signal = torch.nan_to_num(h_pred_dict[base_key], nan=0.0, posinf=320.0, neginf=0.0).float()

        aux = torch.nan_to_num(
            h_pred_dict.get('height_aux', current),
            nan=0.0,
            posinf=320.0,
            neginf=0.0,
        ).float()
        gap_base_key = str(_cfg_value(self.args, 'height_physical_ratio_floor_gap_base_key', base_key))
        if gap_base_key not in h_pred_dict:
            gap_base_key = base_key
        gap_base = torch.nan_to_num(h_pred_dict[gap_base_key], nan=0.0, posinf=320.0, neginf=0.0).float()
        aux_gap = F.relu(aux - gap_base)

        phys_thr = float(_cfg_value(self.args, 'height_physical_ratio_floor_phys_threshold', 15.0))
        phys_width = max(float(_cfg_value(self.args, 'height_physical_ratio_floor_phys_width', 3.0)), 1.0e-3)
        base_thr = float(_cfg_value(self.args, 'height_physical_ratio_floor_base_threshold', 16.0))
        base_width = max(float(_cfg_value(self.args, 'height_physical_ratio_floor_base_width', 4.0)), 1.0e-3)
        area_thr = float(_cfg_value(self.args, 'height_physical_ratio_floor_log_area_threshold', 7.0))
        area_width = max(float(_cfg_value(self.args, 'height_physical_ratio_floor_log_area_width', 0.6)), 1.0e-3)
        gap_thr = float(_cfg_value(self.args, 'height_physical_ratio_floor_aux_gap_threshold', 7.0))
        gap_width = max(float(_cfg_value(self.args, 'height_physical_ratio_floor_aux_gap_width', 2.0)), 1.0e-3)

        phys_gate = torch.sigmoid((physical - phys_thr) / phys_width)
        base_gate = torch.sigmoid((base_signal - base_thr) / base_width)
        area_gate = torch.sigmoid((log_area - area_thr) / area_width)
        gap_gate = torch.sigmoid((aux_gap - gap_thr) / gap_width) if gap_thr >= 0 else torch.ones_like(phys_gate)
        geom_gate = torch.maximum(phys_gate, base_gate)

        mode = str(_cfg_value(self.args, 'height_physical_ratio_floor_evidence_mode', 'phys_area_gap')).lower()
        if mode in ('geom_area_gap', 'geometry_area_gap'):
            evidence_gate = geom_gate * area_gate * gap_gate
        elif mode in ('geom_gap', 'geometry_gap'):
            evidence_gate = geom_gate * gap_gate
        elif mode in ('phys_gap', 'physical_gap'):
            evidence_gate = phys_gate * gap_gate
        elif mode in ('base_gap',):
            evidence_gate = base_gate * gap_gate
        elif mode in ('gap', 'gap_only', 'aux_gap'):
            evidence_gate = gap_gate
        elif mode in ('phys_area_gap', 'physical_area_gap', 'product'):
            evidence_gate = phys_gate * area_gate * gap_gate
        elif mode in ('phys_pre_area_or_gap', 'physical_pre_area_or_gap', 'dual_path', 'dualpath'):
            gap_mix = max(0.0, float(_cfg_value(self.args, 'height_physical_ratio_floor_gap_mix', 0.65)))
            evidence_gate = torch.maximum(
                phys_gate * base_gate * area_gate,
                gap_mix * phys_gate * gap_gate,
            )
        elif mode in ('hard_phys_area_dual', 'hard_physical_area_dual', 'hard_dual_path'):
            gap_mix = max(0.0, float(_cfg_value(self.args, 'height_physical_ratio_floor_gap_mix', 0.65)))
            hard_phys_min = float(_cfg_value(self.args, 'height_physical_ratio_floor_hard_phys_min', phys_thr))
            hard_area_min = float(_cfg_value(self.args, 'height_physical_ratio_floor_hard_log_area_min', area_thr))
            hard_base_min = float(_cfg_value(self.args, 'height_physical_ratio_floor_hard_base_min', -1.0))
            hard_gap_min = float(_cfg_value(self.args, 'height_physical_ratio_floor_hard_aux_gap_min', -1.0))
            hard_mask = (physical >= hard_phys_min) & (log_area >= hard_area_min)
            if hard_base_min >= 0.0 and hard_gap_min >= 0.0:
                hard_mask = hard_mask & ((base_signal >= hard_base_min) | (aux_gap >= hard_gap_min))
            elif hard_base_min >= 0.0:
                hard_mask = hard_mask & (base_signal >= hard_base_min)
            elif hard_gap_min >= 0.0:
                hard_mask = hard_mask & (aux_gap >= hard_gap_min)
            evidence_gate = torch.maximum(
                phys_gate * base_gate * area_gate,
                gap_mix * phys_gate * gap_gate * area_gate,
            ) * hard_mask.float()
        elif mode in ('hard_phys_base_area', 'hard_physical_base_area', 'hard_base_area'):
            hard_phys_min = float(_cfg_value(self.args, 'height_physical_ratio_floor_hard_phys_min', phys_thr))
            hard_area_min = float(_cfg_value(self.args, 'height_physical_ratio_floor_hard_log_area_min', area_thr))
            hard_base_min = float(_cfg_value(self.args, 'height_physical_ratio_floor_hard_base_min', base_thr))
            hard_mask = (physical >= hard_phys_min) & (log_area >= hard_area_min) & (base_signal >= hard_base_min)
            evidence_gate = phys_gate * base_gate * area_gate * hard_mask.float()
        elif mode in ('min', 'and'):
            evidence_gate = torch.minimum(torch.minimum(phys_gate, area_gate), gap_gate)
        elif mode in ('hard_aux', 'hard_gap'):
            evidence_gate = phys_gate * area_gate * gap_gate * (aux_gap >= gap_thr).float()
        else:
            evidence_gate = torch.maximum(phys_gate, base_gate) * area_gate * gap_gate
        evidence_gate = torch.clamp(torch.nan_to_num(evidence_gate, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        evidence_power = max(float(_cfg_value(self.args, 'height_physical_ratio_floor_evidence_power', 1.0)), 1.0e-3)
        if abs(evidence_power - 1.0) > 1.0e-6:
            evidence_gate = _safe_gate_power(evidence_gate, evidence_power)
        min_evidence = float(_cfg_value(self.args, 'height_physical_ratio_floor_min_evidence', 0.0))
        if min_evidence > 0.0:
            evidence_gate = torch.where(
                evidence_gate >= min_evidence,
                evidence_gate,
                torch.zeros_like(evidence_gate),
            )
        gate_scale = float(_cfg_value(self.args, 'height_physical_ratio_floor_gate_scale', 1.0))
        floor_gate = torch.clamp(evidence_gate * gate_scale, 0.0, 1.0)

        ratio = float(_cfg_value(self.args, 'height_physical_ratio_floor_ratio', 1.55))
        bias = float(_cfg_value(self.args, 'height_physical_ratio_floor_bias', 4.0))
        target_clip = float(_cfg_value(self.args, 'height_physical_ratio_floor_target_clip', 120.0))
        floor_target = physical * ratio + bias
        if target_clip > 0:
            floor_target = torch.clamp(floor_target, min=0.0, max=target_clip)

        margin = float(_cfg_value(self.args, 'height_physical_ratio_floor_margin', 1.0))
        scale = float(_cfg_value(self.args, 'height_physical_ratio_floor_scale', 0.9))
        floor_raise = scale * floor_gate * F.relu(floor_target - current - margin)
        raise_clip = float(_cfg_value(self.args, 'height_physical_ratio_floor_raise_clip', 24.0))
        low_clip = max(0.0, float(_cfg_value(self.args, 'height_physical_ratio_floor_low_raise_clip', 0.02)))
        if raise_clip > 0:
            adaptive_clip_enabled = bool(_cfg_value(self.args, 'height_physical_ratio_floor_adaptive_clip', True))
            if adaptive_clip_enabled:
                adaptive_clip = low_clip + evidence_gate * max(0.0, raise_clip - low_clip)
                floor_raise = torch.minimum(torch.clamp(floor_raise, min=0.0), adaptive_clip)
            else:
                floor_raise = torch.clamp(floor_raise, min=0.0, max=raise_clip)
        else:
            floor_raise = torch.clamp(floor_raise, min=0.0)

        height_max = float(_cfg_value(self.args, 'height_output_clip', 320.0))
        h_pred_dict['height_pre_physical_ratio_floor'] = h_pred_dict['height']
        h_pred_dict['height_physical_ratio_floor_target'] = floor_target.to(dtype=h_pred_dict['height'].dtype)
        h_pred_dict['height_physical_ratio_floor_gate'] = floor_gate.to(dtype=h_pred_dict['height'].dtype)
        h_pred_dict['height_physical_ratio_floor_evidence_gate'] = evidence_gate.to(dtype=h_pred_dict['height'].dtype)
        h_pred_dict['height_physical_ratio_floor_raise'] = floor_raise.to(dtype=h_pred_dict['height'].dtype)
        h_pred_dict['height_physical_ratio_floor_aux_gap'] = aux_gap.to(dtype=h_pred_dict['height'].dtype)
        if not bool(_cfg_value(self.args, 'height_physical_ratio_floor_apply_to_output', True)):
            return h_pred_dict
        h_pred_dict['height'] = torch.clamp(
            torch.nan_to_num(current + floor_raise, nan=0.0, posinf=height_max, neginf=0.0),
            min=0.0,
            max=height_max,
        ).to(dtype=h_pred_dict['height'].dtype)
        return h_pred_dict

    def _apply_instance_linear_base(self, h_pred_dict, delta_d, mask, area_maps, ground_disp=None):
        base_mode = str(_cfg_value(self.args, 'height_base_mode', 'affine')).lower()
        if base_mode in (
            'instance_feature_linear',
            'instance_feature_huber',
            'instance_feature_soft_blend',
            'instance_feature_blend',
            'instance_feature_dual_blend',
            'inst_feature_dual_blend',
            'inst_feature_blend',
            'inst_feature_linear',
            'inst_feature_huber',
        ):
            return self._apply_instance_feature_linear_base(h_pred_dict, delta_d, mask, area_maps, ground_disp)
        if base_mode not in ('instance_linear', 'instance_huber', 'inst_linear', 'inst_huber'):
            return h_pred_dict

        coeffs = _cfg_list(
            self.args,
            'height_instance_linear_coeffs',
            [
                -0.136159462889, -0.167274058278, -0.00278506246573,
                -1.73577635409, 0.504077125036, 0.010540086411,
                0.00280257096531, 0.0129351718225, 0.00344087639816,
                0.00580925869719, 0.00875740350147, 0.0613264575258,
                0.0,
            ],
        )
        if len(coeffs) != 13:
            raise ValueError("height_instance_linear_coeffs must contain 13 coefficients.")
        bias = float(_cfg_value(self.args, 'height_instance_linear_bias', 9.13731934083))

        physical = torch.nan_to_num(h_pred_dict['height_physical'], nan=0.0, posinf=320.0, neginf=0.0)
        delta = torch.nan_to_num(delta_d, nan=0.0, posinf=80.0, neginf=0.0)
        if area_maps is None:
            area_maps = self._build_area_maps(mask)
        log_area = area_maps[:, 0:1].to(device=physical.device, dtype=physical.dtype)
        sqrt_area = area_maps[:, 1:2].to(device=physical.device, dtype=physical.dtype)
        pixel_area = sqrt_area * sqrt_area
        sqrt_phys = torch.sqrt(torch.clamp(physical, min=0.0) + 1e-6)
        ones = torch.ones_like(physical)
        terms = torch.stack(
            [
                physical,
                delta,
                pixel_area,
                log_area,
                sqrt_area,
                physical * log_area,
                physical * sqrt_area,
                delta * log_area,
                delta * sqrt_area,
                physical * physical,
                delta * delta,
                sqrt_phys,
                ones,
            ],
            dim=1,
        )
        coeff_t = torch.tensor(coeffs, device=physical.device, dtype=physical.dtype).view(1, -1, 1, 1, 1)
        linear_raw = torch.sum(terms * coeff_t, dim=1) + bias
        if bool(_cfg_value(self.args, 'height_instance_linear_pool_terms', True)):
            B, _, H, W = linear_raw.shape
            pooled_raw = torch.zeros_like(linear_raw)
            flat_mask = mask.view(B, -1).long()
            flat_terms = terms.squeeze(2).permute(0, 2, 3, 1).reshape(B, -1, len(coeffs))
            for b in range(B):
                valid = flat_mask[b] > 0
                if not valid.any():
                    continue
                ids = flat_mask[b][valid]
                _, inv = torch.unique(ids, return_inverse=True)
                num_instances = int(inv.max().item()) + 1
                inst_sum = torch.zeros(num_instances, len(coeffs), device=physical.device, dtype=physical.dtype)
                inst_sum.scatter_add_(0, inv[:, None].expand(-1, len(coeffs)), flat_terms[b][valid])
                counts = torch.zeros(num_instances, 1, device=physical.device, dtype=physical.dtype)
                counts.scatter_add_(0, inv[:, None], torch.ones_like(inv[:, None], dtype=physical.dtype))
                inst_terms = inst_sum / counts.clamp_min(1.0)
                inst_pred = torch.sum(inst_terms * torch.tensor(coeffs, device=physical.device, dtype=physical.dtype), dim=1) + bias
                pooled_flat = pooled_raw[b, 0].view(-1)
                pooled_flat[valid] = inst_pred[inv]
            linear_raw = pooled_raw

        linear_base = _activate_height(torch.nan_to_num(linear_raw, nan=0.0), args=self.args)
        h_pred_dict['height_base'] = linear_base
        h_pred_dict['height_calibrated'] = linear_base
        h_pred_dict['height'] = linear_base
        return h_pred_dict

    def _apply_instance_feature_linear_base(self, h_pred_dict, delta_d, mask, area_maps, ground_disp=None):
        coeffs = _cfg_list(self.args, 'height_instance_feature_linear_coeffs', [0.0] * 19)
        if len(coeffs) != 19:
            raise ValueError("height_instance_feature_linear_coeffs must contain 19 coefficients.")
        bias = float(_cfg_value(self.args, 'height_instance_feature_linear_bias', 0.0))

        physical = torch.nan_to_num(h_pred_dict['height_physical'], nan=0.0, posinf=320.0, neginf=0.0).float()
        if self.learnable_physical_affine:
            physical_for_features = (
                self.height_physical_scale_param.to(device=physical.device, dtype=torch.float32) * physical
                + self.height_physical_bias_param.to(device=physical.device, dtype=torch.float32)
            )
        else:
            physical_for_features = (
                float(_cfg_value(self.args, 'height_physical_scale', 1.0)) * physical
                + float(_cfg_value(self.args, 'height_physical_bias', 0.0))
            )
        physical_for_features = torch.nan_to_num(
            physical_for_features,
            nan=0.0,
            posinf=320.0,
            neginf=0.0,
        )
        poly_base = torch.nan_to_num(h_pred_dict['height_base'], nan=0.0, posinf=320.0, neginf=0.0).float()
        delta = torch.nan_to_num(delta_d, nan=0.0, posinf=80.0, neginf=0.0).float()
        disp = torch.nan_to_num(h_pred_dict.get('disp_est', torch.zeros_like(delta_d)), nan=0.0, posinf=255.0, neginf=-255.0).float()
        if ground_disp is not None:
            ground = torch.nan_to_num(ground_disp, nan=0.0, posinf=255.0, neginf=-255.0).float()
            signed_delta = torch.nan_to_num(ground - disp, nan=0.0, posinf=80.0, neginf=-80.0)
        else:
            ground = disp + delta
            signed_delta = delta
        order_bad = (signed_delta < 0).float()
        if area_maps is None:
            area_maps = self._build_area_maps(mask)
        log_area = area_maps[:, 0:1].to(device=physical.device, dtype=torch.float32)
        sqrt_area = area_maps[:, 1:2].to(device=physical.device, dtype=torch.float32)
        pixel_area = sqrt_area * sqrt_area
        sqrt_phys = torch.sqrt(torch.clamp(physical_for_features, min=0.0) + 1e-6)

        B, _, H, W = delta.shape
        flat_mask = mask.view(B, -1).long()
        valid_mask = flat_mask > 0
        out = torch.zeros(B, 1, H, W, device=delta.device, dtype=torch.float32)

        use_signed_features = bool(_cfg_value(self.args, 'height_instance_feature_use_signed_ground', False))
        if use_signed_features:
            mean_maps = torch.cat(
                [
                    physical_for_features,
                    delta,
                    poly_base,
                    log_area,
                    sqrt_area,
                    pixel_area,
                    physical_for_features * log_area,
                    delta * log_area,
                    physical_for_features * sqrt_area,
                    delta * sqrt_area,
                    physical_for_features * physical_for_features,
                    delta * delta,
                    sqrt_phys,
                    disp,
                    ground,
                    signed_delta,
                    torch.abs(signed_delta),
                    order_bad,
                ],
                dim=1,
            )
        else:
            mean_maps = torch.cat(
                [
                    physical_for_features,
                    delta,
                    poly_base,
                    log_area,
                    sqrt_area,
                    pixel_area,
                    physical_for_features * log_area,
                    delta * log_area,
                    physical_for_features * sqrt_area,
                    delta * sqrt_area,
                    physical_for_features * physical_for_features,
                    delta * delta,
                    sqrt_phys,
                    disp,
                    disp,
                    ground,
                    ground - disp,
                ],
                dim=1,
            )
        flat_mean_maps = mean_maps.permute(0, 2, 3, 1).reshape(B, -1, mean_maps.shape[1])
        flat_delta = delta.view(B, -1)

        if self.height_instance_feature_linear_coeffs_param is not None:
            coeff_t = self.height_instance_feature_linear_coeffs_param.to(device=delta.device, dtype=torch.float32)
            bias_t = self.height_instance_feature_linear_bias_param.to(device=delta.device, dtype=torch.float32)
        else:
            coeff_t = torch.tensor(coeffs, device=delta.device, dtype=torch.float32)
            bias_t = torch.tensor(bias, device=delta.device, dtype=torch.float32)
        for b in range(B):
            valid = valid_mask[b]
            if not valid.any():
                continue
            ids = flat_mask[b][valid]
            _, inv = torch.unique(ids, return_inverse=True)
            num_instances = int(inv.max().item()) + 1
            counts = torch.zeros(num_instances, 1, device=delta.device, dtype=torch.float32)
            counts.scatter_add_(0, inv[:, None], torch.ones_like(inv[:, None], dtype=torch.float32))

            values = flat_mean_maps[b][valid]
            inst_sum = torch.zeros(num_instances, values.shape[1], device=delta.device, dtype=torch.float32)
            inst_sum.scatter_add_(0, inv[:, None].expand(-1, values.shape[1]), values)
            inst_mean = inst_sum / counts.clamp_min(1.0)

            d_vals = flat_delta[b][valid].float()
            d_sum = torch.zeros(num_instances, device=delta.device, dtype=torch.float32)
            d_sum.scatter_add_(0, inv, d_vals)
            d_sq_sum = torch.zeros(num_instances, device=delta.device, dtype=torch.float32)
            d_sq_sum.scatter_add_(0, inv, d_vals * d_vals)
            d_mean = d_sum / counts.view(-1).clamp_min(1.0)
            d_std = torch.sqrt(torch.clamp(d_sq_sum / counts.view(-1).clamp_min(1.0) - d_mean * d_mean, min=0.0) + 1e-6)
            d_max = torch.full((num_instances,), -torch.inf, device=delta.device, dtype=torch.float32)
            d_max.scatter_reduce_(0, inv, d_vals, reduce='amax', include_self=False)
            d_max = torch.nan_to_num(d_max, nan=0.0, posinf=0.0, neginf=0.0)

            if use_signed_features:
                s_vals = signed_delta.view(B, -1)[b][valid].float()
                s_sq_sum = torch.zeros(num_instances, device=delta.device, dtype=torch.float32)
                s_sq_sum.scatter_add_(0, inv, s_vals * s_vals)
                s_sum = torch.zeros(num_instances, device=delta.device, dtype=torch.float32)
                s_sum.scatter_add_(0, inv, s_vals)
                s_mean = s_sum / counts.view(-1).clamp_min(1.0)
                s_std = torch.sqrt(torch.clamp(s_sq_sum / counts.view(-1).clamp_min(1.0) - s_mean * s_mean, min=0.0) + 1e-6)
                d_stat = s_std
            else:
                d_stat = d_std
            inst_features = torch.cat([inst_mean, d_stat[:, None]], dim=1)
            if not use_signed_features:
                inst_features = torch.cat([inst_features, d_max[:, None]], dim=1)
            inst_pred = torch.sum(inst_features * coeff_t.view(1, -1), dim=1) + bias_t
            flat_out = out[b, 0].view(-1)
            flat_out[valid] = inst_pred[inv]

        linear_base = _activate_height(torch.nan_to_num(out, nan=0.0), args=self.args)
        base_mode = str(_cfg_value(self.args, 'height_base_mode', 'affine')).lower()
        if base_mode in ('instance_feature_dual_blend', 'inst_feature_dual_blend'):
            high_coeffs = _cfg_list(self.args, 'height_instance_feature_high_coeffs', coeffs)
            if len(high_coeffs) != 19:
                raise ValueError("height_instance_feature_high_coeffs must contain 19 coefficients.")
            high_bias = float(_cfg_value(self.args, 'height_instance_feature_high_bias', bias))
            if self.height_instance_feature_high_coeffs_param is not None:
                high_coeff_t = self.height_instance_feature_high_coeffs_param.to(device=delta.device, dtype=torch.float32)
                high_bias_t = self.height_instance_feature_high_bias_param.to(device=delta.device, dtype=torch.float32)
            else:
                high_coeff_t = torch.tensor(high_coeffs, device=delta.device, dtype=torch.float32)
                high_bias_t = torch.tensor(high_bias, device=delta.device, dtype=torch.float32)
            high_out = torch.zeros_like(out)

            for b in range(B):
                valid = valid_mask[b]
                if not valid.any():
                    continue
                ids = flat_mask[b][valid]
                _, inv = torch.unique(ids, return_inverse=True)
                num_instances = int(inv.max().item()) + 1
                counts = torch.zeros(num_instances, 1, device=delta.device, dtype=torch.float32)
                counts.scatter_add_(0, inv[:, None], torch.ones_like(inv[:, None], dtype=torch.float32))

                values = flat_mean_maps[b][valid]
                inst_sum = torch.zeros(num_instances, values.shape[1], device=delta.device, dtype=torch.float32)
                inst_sum.scatter_add_(0, inv[:, None].expand(-1, values.shape[1]), values)
                inst_mean = inst_sum / counts.clamp_min(1.0)

                d_vals = flat_delta[b][valid].float()
                d_sum = torch.zeros(num_instances, device=delta.device, dtype=torch.float32)
                d_sum.scatter_add_(0, inv, d_vals)
                d_sq_sum = torch.zeros(num_instances, device=delta.device, dtype=torch.float32)
                d_sq_sum.scatter_add_(0, inv, d_vals * d_vals)
                d_mean = d_sum / counts.view(-1).clamp_min(1.0)
                d_std = torch.sqrt(torch.clamp(d_sq_sum / counts.view(-1).clamp_min(1.0) - d_mean * d_mean, min=0.0) + 1e-6)
                d_max = torch.full((num_instances,), -torch.inf, device=delta.device, dtype=torch.float32)
                d_max.scatter_reduce_(0, inv, d_vals, reduce='amax', include_self=False)
                d_max = torch.nan_to_num(d_max, nan=0.0, posinf=0.0, neginf=0.0)

                if use_signed_features:
                    s_vals = signed_delta.view(B, -1)[b][valid].float()
                    s_sq_sum = torch.zeros(num_instances, device=delta.device, dtype=torch.float32)
                    s_sq_sum.scatter_add_(0, inv, s_vals * s_vals)
                    s_sum = torch.zeros(num_instances, device=delta.device, dtype=torch.float32)
                    s_sum.scatter_add_(0, inv, s_vals)
                    s_mean = s_sum / counts.view(-1).clamp_min(1.0)
                    s_std = torch.sqrt(torch.clamp(s_sq_sum / counts.view(-1).clamp_min(1.0) - s_mean * s_mean, min=0.0) + 1e-6)
                    d_stat = s_std
                else:
                    d_stat = d_std
                inst_features = torch.cat([inst_mean, d_stat[:, None]], dim=1)
                if not use_signed_features:
                    inst_features = torch.cat([inst_features, d_max[:, None]], dim=1)
                inst_pred = torch.sum(inst_features * high_coeff_t.view(1, -1), dim=1) + high_bias_t
                flat_high = high_out[b, 0].view(-1)
                flat_high[valid] = inst_pred[inv]

            high_base = _activate_height(torch.nan_to_num(high_out, nan=0.0), args=self.args)
            gate_signal = str(_cfg_value(self.args, 'height_instance_feature_dual_gate', 'area')).lower()
            if gate_signal in ('physical', 'phys'):
                signal = physical_for_features
            elif gate_signal in ('poly', 'poly_base', 'base'):
                signal = poly_base
            elif gate_signal in ('low', 'low_base', 'linear'):
                signal = linear_base
            elif gate_signal in ('phys_area', 'physical_area', 'area_phys', 'geom_area'):
                phys_center = float(_cfg_value(self.args, 'height_instance_feature_dual_phys_center', 8.0))
                phys_scale = max(float(_cfg_value(self.args, 'height_instance_feature_dual_phys_scale', 12.0)), 1.0e-3)
                area_center = float(_cfg_value(self.args, 'height_instance_feature_dual_area_center', 6.2))
                area_scale = max(float(_cfg_value(self.args, 'height_instance_feature_dual_area_scale', 1.0)), 1.0e-3)
                phys_signal = (physical_for_features.float() - phys_center) / phys_scale
                area_signal = (log_area.float() - area_center) / area_scale
                signal = torch.minimum(phys_signal, area_signal)
            else:
                signal = log_area
            threshold = float(_cfg_value(self.args, 'height_instance_feature_dual_threshold', 6.62))
            width = max(float(_cfg_value(self.args, 'height_instance_feature_dual_width', 0.75)), 1.0e-3)
            alpha = torch.sigmoid((signal.float() - threshold) / width)
            min_alpha = max(0.0, min(1.0, float(_cfg_value(self.args, 'height_instance_feature_dual_min_alpha', 0.0))))
            max_alpha = max(min_alpha, min(1.0, float(_cfg_value(self.args, 'height_instance_feature_dual_max_alpha', 1.0))))
            alpha = min_alpha + (max_alpha - min_alpha) * alpha
            linear_base = (1.0 - alpha) * linear_base.float() + alpha * high_base.float()
            h_pred_dict['height_feature_dual_alpha'] = alpha.to(dtype=linear_base.dtype)

        if bool(_cfg_value(self.args, 'height_instance_feature_physical_floor', False)):
            floor_ratio = float(_cfg_value(self.args, 'height_instance_feature_physical_floor_ratio', 0.9))
            phys_center = float(_cfg_value(self.args, 'height_instance_feature_physical_floor_phys_center', 18.0))
            phys_width = max(float(_cfg_value(self.args, 'height_instance_feature_physical_floor_phys_width', 2.0)), 1.0e-3)
            area_center = float(_cfg_value(self.args, 'height_instance_feature_physical_floor_area_center', 6.6))
            area_width = max(float(_cfg_value(self.args, 'height_instance_feature_physical_floor_area_width', 0.5)), 1.0e-3)
            gate_mode = str(_cfg_value(self.args, 'height_instance_feature_physical_floor_gate', 'product')).lower()
            phys_gate = torch.sigmoid((physical_for_features.float() - phys_center) / phys_width)
            area_gate = torch.sigmoid((log_area.float() - area_center) / area_width)
            if gate_mode in ('min', 'and'):
                floor_gate = torch.minimum(phys_gate, area_gate)
            elif gate_mode in ('max', 'or'):
                floor_gate = torch.maximum(phys_gate, area_gate)
            else:
                floor_gate = phys_gate * area_gate
            floor_gate = torch.clamp(torch.nan_to_num(floor_gate, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            floor_height = torch.clamp(physical_for_features.float() * floor_ratio, min=0.0, max=320.0)
            floor_candidate = floor_gate * floor_height + (1.0 - floor_gate) * linear_base.float()
            linear_base = torch.maximum(linear_base.float(), floor_candidate)
            h_pred_dict['height_feature_physical_floor_gate'] = floor_gate.to(dtype=linear_base.dtype)

        if base_mode in ('instance_feature_soft_blend', 'instance_feature_blend', 'inst_feature_blend'):
            threshold = float(_cfg_value(self.args, 'height_instance_feature_blend_threshold', 7.0))
            width = max(float(_cfg_value(self.args, 'height_instance_feature_blend_width', 1.5)), 1.0e-3)
            alpha = torch.sigmoid((poly_base - threshold) / width)
            min_alpha = max(0.0, min(1.0, float(_cfg_value(self.args, 'height_instance_feature_blend_min_alpha', 0.0))))
            max_alpha = max(min_alpha, min(1.0, float(_cfg_value(self.args, 'height_instance_feature_blend_max_alpha', 1.0))))
            alpha = min_alpha + (max_alpha - min_alpha) * alpha
            low_bias = float(_cfg_value(self.args, 'height_instance_feature_blend_low_bias', 0.0))
            high_bias = float(_cfg_value(self.args, 'height_instance_feature_blend_high_bias', 0.0))
            blend_bias = (1.0 - alpha) * low_bias + alpha * high_bias
            linear_base = _activate_height(
                (1.0 - alpha) * poly_base + alpha * linear_base.float() + blend_bias,
                args=self.args,
            )
            h_pred_dict['height_feature_blend_alpha'] = alpha.to(dtype=linear_base.dtype)
        h_pred_dict['height_base'] = linear_base
        h_pred_dict['height_calibrated'] = linear_base
        h_pred_dict['height'] = linear_base
        return h_pred_dict

    def forward(self, image1, image2, mask, k_tensor, iters=12, test_mode=False, precomputed_disp=None, precomputed_delta=None, precomputed_ground=None):
        building_mask_binary = (mask > 0).float()
        delta_source = str(_cfg_value(self.args, 'height_delta_source', 'learned_base_plane')).lower()
        if delta_source in _PRECOMPUTED_DELTA_SOURCES:
            if precomputed_disp is None or precomputed_delta is None:
                raise ValueError("height_delta_source=precomputed requires precomputed_disp and precomputed_delta.")
            disp_est = torch.nan_to_num(precomputed_disp, nan=0.0, posinf=0.0, neginf=0.0)
            delta_d_robust = F.relu(torch.nan_to_num(precomputed_delta, nan=0.0, posinf=0.0, neginf=0.0))
        elif delta_source in _CACHED_DISP_LEARNED_SOURCES:
            if precomputed_disp is None:
                raise ValueError(f"height_delta_source={delta_source} requires precomputed_disp.")
            disp_est = torch.nan_to_num(precomputed_disp, nan=0.0, posinf=0.0, neginf=0.0)
            if delta_source in _DELTA_RESIDUAL_SOURCES:
                if precomputed_delta is None:
                    raise ValueError(f"height_delta_source={delta_source} requires precomputed_delta.")
                delta_base = torch.nan_to_num(precomputed_delta, nan=0.0, posinf=0.0, neginf=0.0)
                dtm_freeze_epochs = int(_cfg_value(self.args, 'dtm_freeze_epochs', 0))
                if self.training and self.current_epoch > 0 and self.current_epoch <= dtm_freeze_epochs:
                    delta_d_robust = F.relu(delta_base) + self._zero_anchor_for_base_plane(delta_base)
                else:
                    use_checkpoint = bool(_cfg_value(self.args, 'height_dtm_gradient_checkpoint', False)) and self.training
                    learned = self.base_plane_estimator(image1, disp_est, use_checkpoint=use_checkpoint)
                    residual_clip = float(_cfg_value(self.args, 'height_delta_residual_clip', 32.0))
                    if residual_clip > 0:
                        learned = residual_clip * torch.tanh(learned / residual_clip)
                    delta_d_robust = F.relu(delta_base + learned)
            else:
                use_checkpoint = bool(_cfg_value(self.args, 'height_dtm_gradient_checkpoint', False)) and self.training
                learned = self.base_plane_estimator(image1, disp_est, use_checkpoint=use_checkpoint)
                d_ground = learned
                delta_d_raw = d_ground - disp_est
                delta_d_robust = self._instance_mean_delta(delta_d_raw, mask)
                delta_d_robust = F.relu(delta_d_robust)
        else:
            # 1. Disparity forward pass without gradients to save memory.
            with torch.no_grad():
                disp_results = self.disp_net(image1, image2, iters=iters, test_mode=test_mode)
                
            if test_mode:
                disp_preds = [disp_results] 
            else:
                init_disp, disp_preds, depth_mono = disp_results

            disp_est = disp_preds[-1]
            disp_est = torch.nan_to_num(disp_est, nan=0.0)

            # Estimate local base-plane disparity.
            use_checkpoint = bool(_cfg_value(self.args, 'height_dtm_gradient_checkpoint', False)) and self.training
            d_ground = self.base_plane_estimator(image1, disp_est, use_checkpoint=use_checkpoint)
            
            # Raw pixel-wise differential disparity.
            delta_d_raw = d_ground - disp_est
            delta_d_robust = self._instance_mean_delta(delta_d_raw, mask)

            # Soft constraint: differential disparity should be non-negative in buildings.
            delta_d_robust = F.relu(delta_d_robust)

        delta_max = float(_cfg_value(self.args, 'height_delta_clip', 260.0))
        if delta_max > 0:
            delta_d_robust = torch.clamp(
                torch.nan_to_num(delta_d_robust.float(), nan=0.0, posinf=delta_max, neginf=0.0),
                min=0.0,
                max=delta_max,
            ).to(dtype=disp_est.dtype)

        # Use K to obtain final height estimates.
        extra_features = self._build_extra_features(
            image1,
            mask,
            disp_est=disp_est,
            delta_d=delta_d_robust,
            ground_disp=precomputed_ground,
            k_tensor=k_tensor,
        )
        base_mode = str(_cfg_value(self.args, 'height_base_mode', 'affine')).lower()
        output_mode = str(_cfg_value(self.args, 'height_output_mode', 'physical')).lower()
        output_uses_area_gate = output_mode in (
            'protected_high_correction',
            'low_protected_high_correction',
            'high_protected_correction',
            'high_boost',
            'signed_low_protected_high_correction',
            'low_protected_signed_high_correction',
        ) and (
            str(_cfg_value(self.args, 'height_high_evidence_gate_mode', 'max')).lower() not in (
                'max',
                'geom',
                'geometry',
                'phys_base',
                'phys_base_max',
                'geometry_max',
            )
            or bool(_cfg_value(self.args, 'height_low_protect_use_area', False))
        )
        need_area_maps = (
            base_mode in (
                'poly_area',
                'polyarea',
                'polynomial_area',
                'instance_feature_linear',
                'instance_feature_huber',
                'instance_feature_soft_blend',
                'instance_feature_blend',
                'instance_feature_dual_blend',
                'inst_feature_dual_blend',
                'inst_feature_blend',
                'inst_feature_linear',
                'inst_feature_huber',
            )
            or self.instance_correction_head is not None
            or output_uses_area_gate
        )
        area_maps = self._build_area_maps(mask) if need_area_maps else None
        h_pred_dict = self.height_head(
            disp_est,
            delta_d_robust,
            k_tensor,
            args=self.args,
            extra_features=extra_features,
            area_maps=area_maps,
            physical_affine=(
                (self.height_physical_scale_param, self.height_physical_bias_param)
                if self.learnable_physical_affine
                else None
            ),
        )
        h_pred_dict['disp_est'] = disp_est
        h_pred_dict['height_pixel'] = h_pred_dict['height']
        h_pred_dict = self._apply_instance_linear_base(h_pred_dict, delta_d_robust, mask, area_maps, precomputed_ground)
        h_pred_dict = self._apply_instance_refinement(h_pred_dict, disp_est, delta_d_robust, image1, mask, area_maps, precomputed_ground)
        h_pred_dict = self._apply_post_base_high_boost(h_pred_dict, mask, area_maps)
        h_pred_dict = self._apply_aux_height_floor(h_pred_dict, mask, area_maps)
        h_pred_dict = self._apply_physical_ratio_floor(h_pred_dict, mask, area_maps)

        mask_output_keys = (
            'height',
            'height_physical',
            'height_base',
            'height_calibrated',
            'height_aux',
            'height_direct',
            'height_correction',
            'height_signed_correction',
            'height_high_boost',
            'height_high_boost_raw',
            'height_high_gate',
            'height_aux_raise',
            'height_pixel',
            'height_pre_post_base_boost',
            'height_post_base_gate',
            'height_post_base_evidence_gate',
            'height_post_base_high_boost',
            'height_post_base_aux_gap',
            'height_post_base_aux_gap_gate',
            'height_pre_aux_floor',
            'height_aux_floor_gate',
            'height_aux_floor_evidence_gate',
            'height_aux_floor_raise',
            'height_aux_floor_gap',
            'height_pre_physical_ratio_floor',
            'height_physical_ratio_floor_target',
            'height_physical_ratio_floor_gate',
            'height_physical_ratio_floor_evidence_gate',
            'height_physical_ratio_floor_raise',
            'height_physical_ratio_floor_aux_gap',
            'height_pre_learned_physical_floor',
            'height_learned_physical_floor_raw',
            'height_learned_physical_floor_fraction',
            'height_learned_physical_floor_target',
            'height_learned_physical_floor_gate',
            'height_learned_physical_floor_evidence_gate',
            'height_learned_physical_floor_raise',
            'height_pre_high_residual',
            'height_high_residual_basis',
            'height_high_residual_raw',
            'height_high_residual_gate_logit',
            'height_high_residual_learned_gate',
            'height_high_residual_gate',
            'height_high_residual_evidence_gate',
            'height_high_residual',
        )

        if test_mode:
            eval_output_keys = _cfg_list(self.args, 'height_eval_output_keys', ['height'])
            keep_keys = {str(key) for key in eval_output_keys} if eval_output_keys else {'height'}
            if 'height' not in keep_keys:
                keep_keys.add('height')
            h_pred_dict = {key: value for key, value in h_pred_dict.items() if key in keep_keys}
            for output_key in mask_output_keys:
                if output_key in h_pred_dict:
                    h_pred_dict[output_key].mul_(building_mask_binary)
        else:
            train_output_keys = _cfg_list(self.args, 'height_train_output_keys', [])
            keep_keys = {str(key) for key in train_output_keys} if train_output_keys else None
            if bool(_cfg_value(self.args, 'height_train_mask_outputs', False)):
                mask_keys = (
                    [key for key in mask_output_keys if key in keep_keys]
                    if keep_keys is not None
                    else mask_output_keys
                )
                for output_key in mask_keys:
                    if output_key in h_pred_dict:
                        h_pred_dict[output_key] = h_pred_dict[output_key] * building_mask_binary
            if keep_keys is not None:
                h_pred_dict = {key: value for key, value in h_pred_dict.items() if key in keep_keys}
        
        if test_mode:
            # Return the full dictionary for validation/evaluation.
            return h_pred_dict
            
        # Return a single-element list for loss compatibility.
        return [h_pred_dict]
