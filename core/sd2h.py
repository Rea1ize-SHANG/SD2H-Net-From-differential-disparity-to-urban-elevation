import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock, BasicMultiUpdateBlock_mix2
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *
from core.refinement import REMP
from core.warp import disp_warp
import matplotlib.pyplot as plt
import timm
import math

from torch.amp import autocast

import sys
sys.path.append('/home/shangying/workspace/myrepos/SD2H/Depth-Anything-V2-list3')
from depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2_decoder


def compute_scale_shift(monocular_depth, gt_depth, mask=None):
    flattened_depth_maps = monocular_depth.clone().view(-1).contiguous()
    sorted_depth_maps, _ = torch.sort(flattened_depth_maps)
    percentile_10_index = int(0.2 * len(sorted_depth_maps))
    threshold_10_percent = sorted_depth_maps[percentile_10_index]

    if mask is None:
        mask = (gt_depth > 0) & (monocular_depth > 1e-2) & (monocular_depth > threshold_10_percent)
    
    monocular_depth_flat = monocular_depth[mask]
    gt_depth_flat = gt_depth[mask]
    
    X = torch.stack([monocular_depth_flat, torch.ones_like(monocular_depth_flat)], dim=1)
    y = gt_depth_flat
    
    A = torch.matmul(X.t(), X) + 1e-6 * torch.eye(2, device=X.device)
    b = torch.matmul(X.t(), y)
    params = torch.linalg.solve(A, b)
    
    scale, shift = params[0].item(), params[1].item()
    return scale, shift


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                   
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 

        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))

        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)
        return conv


class Feat_transfer_cnet(nn.Module):
    def __init__(self, dim_list, output_dim):
        super(Feat_transfer_cnet, self).__init__()
        self.res_16x = nn.Conv2d(dim_list[0]+192, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0]+96, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0]+48, output_dim, kernel_size=3, padding=1, stride=1)

    def forward(self, features, stem_x_list):
        features_list = []
        feat_16x = self.res_16x(torch.cat((features[2], stem_x_list[0]), 1))
        feat_8x = self.res_8x(torch.cat((features[1], stem_x_list[1]), 1))
        feat_4x = self.res_4x(torch.cat((features[0], stem_x_list[2]), 1))
        features_list.append([feat_4x, feat_4x])
        features_list.append([feat_8x, feat_8x])
        features_list.append([feat_16x, feat_16x])
        return features_list


class Feat_transfer(nn.Module):
    def __init__(self, dim_list):
        super(Feat_transfer, self).__init__()
        self.conv4x = nn.Sequential(
            nn.Conv2d(in_channels=int(48+dim_list[0]), out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(48), nn.ReLU()
            )
        self.conv8x = nn.Sequential(
            nn.Conv2d(in_channels=int(64+dim_list[0]), out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(64), nn.ReLU()
            )
        self.conv16x = nn.Sequential(
            nn.Conv2d(in_channels=int(192+dim_list[0]), out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(192), nn.ReLU()
            )
        self.conv32x = nn.Sequential(
            nn.Conv2d(in_channels=dim_list[0], out_channels=160, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(160), nn.ReLU()
            )
        self.conv_up_32x = nn.ConvTranspose2d(160, 192, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
        self.conv_up_16x = nn.ConvTranspose2d(192, 64, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
        self.conv_up_8x = nn.ConvTranspose2d(64, 48, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
        
        self.res_16x = nn.Conv2d(dim_list[0], 192, kernel_size=1, padding=0, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0], 64, kernel_size=1, padding=0, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0], 48, kernel_size=1, padding=0, stride=1)

    def forward(self, features):
        features_mono_list = []
        feat_32x = self.conv32x(features[3])
        feat_32x_up = self.conv_up_32x(feat_32x)
        feat_16x = self.conv16x(torch.cat((features[2], feat_32x_up), 1)) + self.res_16x(features[2])
        feat_16x_up = self.conv_up_16x(feat_16x)
        feat_8x = self.conv8x(torch.cat((features[1], feat_16x_up), 1)) + self.res_8x(features[1])
        feat_8x_up = self.conv_up_8x(feat_8x)
        feat_4x = self.conv4x(torch.cat((features[0], feat_8x_up), 1)) + self.res_4x(features[0])
        features_mono_list.append(feat_4x)
        features_mono_list.append(feat_8x)
        features_mono_list.append(feat_16x)
        features_mono_list.append(feat_32x)
        return features_mono_list


class MacroCostVolumeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, min_disp, max_disp):
        super().__init__()
        self.scale = 4  
        self.min_d = min_disp // self.scale
        self.max_d = max_disp // self.scale
        self.num_disp = self.max_d - self.min_d

        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)
        )

        self.macro_feat_proj = nn.Sequential(
            nn.Conv2d(self.num_disp, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_l, feat_r):
        fl = F.avg_pool2d(feat_l, self.scale, self.scale)
        fr = F.avg_pool2d(feat_r, self.scale, self.scale)
        b, c, h, w = fl.shape

        volume = torch.zeros((b, c, self.num_disp, h, w), device=fl.device, dtype=fl.dtype)
        for i, d in enumerate(range(self.min_d, self.max_d)):
            if d > 0:
                volume[:, :, i, :, d:] = fl[:, :, :, d:] * fr[:, :, :, :-d]
            elif d < 0:
                volume[:, :, i, :, :d] = fl[:, :, :, :d] * fr[:, :, :, -d:]
            else:
                volume[:, :, i, :, :] = fl * fr

        agg_vol = self.proj(volume).squeeze(1)
        macro_feat = self.macro_feat_proj(agg_vol)
        macro_feat = F.interpolate(macro_feat, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return macro_feat


class SD2H(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        context_dims = args.hidden_dims
        
        self.use_stereo = getattr(args, 'use_stereo', True)
        self.use_mono = getattr(args, 'use_mono', True)
        
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims) 

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        mono_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        dim_list_ = mono_model_configs[self.args.encoder]['features']
        dim_list = [dim_list_] 
        
        self.feat_transfer = Feat_transfer(dim_list)
        self.feat_transfer_cnet = Feat_transfer_cnet(dim_list, output_dim=args.hidden_dims[0]) 

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            ) 
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )
        self.stem_8 = nn.Sequential(
            BasicConv_IN(48, 96, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 96, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(96), nn.ReLU()
            )
        self.stem_16 = nn.Sequential(
            BasicConv_IN(96, 192, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 192, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(192), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True) 
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.macro_cost = MacroCostVolumeBlock(
            in_channels=96, 
            out_channels=args.hidden_dims[0], 
            min_disp=args.min_disp // 4, 
            max_disp=args.max_disp // 4
        )

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)
        
        depth_anything = DepthAnythingV2(**mono_model_configs[args.encoder])
        depth_anything_decoder = DepthAnythingV2_decoder(**mono_model_configs[args.encoder])
        state_dict_dpt = torch.load(f'/home/shangying/workspace/myrepos/SD2H/pretrained/depth_anything_v2_{args.encoder}.pth', map_location='cpu', weights_only=True)
        depth_anything.load_state_dict(state_dict_dpt, strict=True)
        depth_anything_decoder.load_state_dict(state_dict_dpt, strict=False)
        self.mono_encoder = depth_anything.pretrained
        self.mono_decoder = depth_anything.depth_head
        self.feat_decoder = depth_anything_decoder.depth_head
        self.mono_encoder.requires_grad_(False)
        self.mono_decoder.requires_grad_(False)
        self.feat_decoder.requires_grad_(False) 

        del depth_anything, state_dict_dpt, depth_anything_decoder
        
        self.REMP = REMP()

        self.update_block_mix_stereo = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims)
        self.update_block_mix_mono = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def infer_mono(self, image1, image2):                                
        height_ori, width_ori = image1.shape[2:]
        resize_image1 = F.interpolate(image1, scale_factor=14 / 16, mode='bilinear', align_corners=True)
        resize_image2 = F.interpolate(image2, scale_factor=14 / 16, mode='bilinear', align_corners=True)

        patch_h, patch_w = resize_image1.shape[-2] // 14, resize_image1.shape[-1] // 14
        
        features_left_encoder = self.mono_encoder.get_intermediate_layers(resize_image1, self.intermediate_layer_idx[self.args.encoder], return_class_token=True)
        features_right_encoder = self.mono_encoder.get_intermediate_layers(resize_image2, self.intermediate_layer_idx[self.args.encoder], return_class_token=True)
        
        depth_mono = self.mono_decoder(features_left_encoder, patch_h, patch_w)
        depth_mono = F.relu(depth_mono)
        
        depth_mono = F.interpolate(depth_mono, size=(height_ori, width_ori), mode='bilinear', align_corners=False)
        
        features_left_4x, features_left_8x, features_left_16x, features_left_32x = self.feat_decoder(features_left_encoder, patch_h, patch_w)
        features_right_4x, features_right_8x, features_right_16x, features_right_32x = self.feat_decoder(features_right_encoder, patch_h, patch_w)

        return depth_mono, [features_left_4x, features_left_8x, features_left_16x, features_left_32x], [features_right_4x, features_right_8x, features_right_16x, features_right_32x]
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        
        # Ensure context_upsample receives a 4D single-channel tensor [B, 1, H, W].
        if disp.dim() == 3:
            disp = disp.unsqueeze(1)
        elif disp.dim() == 4 and disp.shape[1] > 1:
            disp = disp[:, 0:1, :, :]
            
        up_disp = context_upsample(disp * 4., spx_pred)
        
        # Ensure the returned disparity is a clean 4D single-channel tensor.
        if up_disp.dim() == 3:
            up_disp = up_disp.unsqueeze(1)
        elif up_disp.dim() == 5:
            up_disp = up_disp.squeeze(1)
            
        if up_disp.shape[1] > 1:
            up_disp = up_disp[:, 0:1, :, :]
            
        return up_disp

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        image1 = (2 * (image1 / 3100.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 3100.0) - 1.0).contiguous()
        
        with torch.autocast(device_type='cuda'), torch.no_grad():
            depth_mono, features_mono_left,  features_mono_right = self.infer_mono(image1, image2)
            
        scale_factor = 0.25
        size = (int(depth_mono.shape[-2] * scale_factor), int(depth_mono.shape[-1] * scale_factor))
        disp_mono_4x = F.interpolate(depth_mono, size=size, mode='bilinear', align_corners=False)
        
        features_left = self.feat_transfer(features_mono_left)
        features_right = self.feat_transfer(features_mono_right)
        
        stem_2x = self.stem_2(image1)
        stem_4x = self.stem_4(stem_2x)
        stem_8x = self.stem_8(stem_4x)
        stem_16x = self.stem_16(stem_8x)
        stem_2y = self.stem_2(image2)
        stem_4y = self.stem_4(stem_2y)

        stem_x_list = [stem_16x, stem_8x, stem_4x]
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)
        
        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))

        gwc_volume = build_gwc_volume(match_left, match_right, self.args.min_disp//4, self.args.max_disp//4, 8)
        gwc_volume = self.corr_stem(gwc_volume)  
        gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
        geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

        prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
        init_disp = disparity_regression(prob, self.args.min_disp//4, self.args.max_disp//4)
        
        del prob, gwc_volume

        if not test_mode:
            xspx = self.spx_4(features_left[0])
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)

        cnet_list = self.feat_transfer_cnet(features_mono_left, stem_x_list)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        
        macro_prior = self.macro_cost(match_left, match_right)
        net_list[0] = net_list[0] + macro_prior
        
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [torch.relu(x) for x in inp_list]
        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]
        net_list_mono = [x.clone() for x in net_list]

        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left, match_right, geo_encoding_volume, radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1).contiguous()
        disp = init_disp
        disp_preds = []                              
        
        if not self.use_mono:
            disp_mono_4x = torch.zeros_like(disp_mono_4x)

        warmup_iters = iters // 2

        for itr in range(iters):
            disp = disp.detach()
            if itr >= int(1):
                disp_mono_4x = disp_mono_4x.detach()
            
            geo_feat = geo_fn(disp, coords)
            
            if not self.use_stereo:
                geo_feat = torch.zeros_like(geo_feat)
            
            if itr > warmup_iters:
                if itr == warmup_iters + 1:              
                    bs = disp.shape[0]
                    for i in range(bs):
                        with torch.autocast(device_type='cuda', dtype=torch.float32): 
                            scale, shift = compute_scale_shift(disp_mono_4x[i].clone().squeeze().to(torch.float32), disp[i].clone().squeeze().to(torch.float32))
                        disp_mono_4x[i] = scale * disp_mono_4x[i] + shift
                
                warped_right_mono = disp_warp(features_right[0], disp_mono_4x.clone().to(features_right[0].dtype))[0]  
                flaw_mono = warped_right_mono - features_left[0]

                warped_right_stereo = disp_warp(features_right[0], disp.clone().to(features_right[0].dtype))[0]
                flaw_stereo = warped_right_stereo - features_left[0] 
                
                geo_feat_mono = geo_fn(disp_mono_4x, coords)
                
                if not self.use_mono:
                    flaw_mono = torch.zeros_like(flaw_mono)
                    geo_feat_mono = torch.zeros_like(geo_feat_mono)
                    disp_mono_4x = torch.zeros_like(disp_mono_4x)

                if not self.use_stereo:
                    flaw_stereo = torch.zeros_like(flaw_stereo)
                    geo_feat = torch.zeros_like(geo_feat)
                    disp = torch.zeros_like(disp)

            if itr <= warmup_iters:           
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)
        
                if delta_disp.shape[1] > 1: delta_disp = delta_disp[:, 0:1, :, :]
            else:
                net_list, mask_feat_4, delta_disp = self.update_block_mix_stereo(net_list, inp_list, flaw_stereo, disp, geo_feat, flaw_mono, disp_mono_4x, geo_feat_mono, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)
     
                if delta_disp.shape[1] > 1: delta_disp = delta_disp[:, 0:1, :, :]
                
                if self.use_mono:
                    net_list_mono, mask_feat_4_mono, delta_disp_mono = self.update_block_mix_mono(net_list_mono, inp_list, flaw_mono, disp_mono_4x, geo_feat_mono, flaw_stereo, disp, geo_feat, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)
           
                    if delta_disp_mono.shape[1] > 1: delta_disp_mono = delta_disp_mono[:, 0:1, :, :]
                    
                    disp_mono_4x = disp_mono_4x + delta_disp_mono
                    disp_mono_4x_up = self.upsample_disp(disp_mono_4x, mask_feat_4_mono, stem_2x)
                    disp_preds.append(disp_mono_4x_up)
                else:
                    dummy_mono_up = torch.zeros((disp.shape[0], 1, image1.shape[-2], image1.shape[-1]), device=disp.device, dtype=torch.float32)
                    disp_preds.append(dummy_mono_up)
                    disp_mono_4x_up = dummy_mono_up

            if self.use_stereo:
                disp = disp + delta_disp
            
            if test_mode and itr < iters-1:
                continue

            if self.use_stereo:
                disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            else:
                disp_up = torch.zeros((disp.shape[0], 1, image1.shape[-2], image1.shape[-1]), device=disp.device, dtype=torch.float32)

            if itr == iters - 1:
                if self.use_stereo and self.use_mono:
                    refine_value = self.REMP(disp_mono_4x_up, disp_up, image1, image2)
                    if refine_value.shape[1] > 1: refine_value = refine_value[:, 0:1, :, :]
                    disp_up = disp_up + refine_value
                elif self.use_stereo:
                    refine_value = self.REMP(torch.zeros_like(disp_up), disp_up, image1, image2)
                    if refine_value.shape[1] > 1: refine_value = refine_value[:, 0:1, :, :]
                    disp_up = disp_up + refine_value
                elif self.use_mono:
                    disp_up = disp_mono_4x_up
                    
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up
            
        if not self.use_stereo:
            init_disp = torch.zeros((image1.shape[0], 1, image1.shape[-2], image1.shape[-1]), device=image1.device, dtype=torch.float32)
        else:
            if init_disp.dim() == 3:
                init_disp = init_disp.unsqueeze(1)
            elif init_disp.shape[1] > 1:
                init_disp = init_disp[:, 0:1, :, :]
                
            init_disp = context_upsample(init_disp * 4., spx_pred.float())
            
            if init_disp.dim() == 3:
                init_disp = init_disp.unsqueeze(1)
            elif init_disp.dim() == 5:
                init_disp = init_disp.squeeze(1)
                
            if init_disp.shape[1] > 1:
                init_disp = init_disp[:, 0:1, :, :]
            
        return init_disp, disp_preds, depth_mono
