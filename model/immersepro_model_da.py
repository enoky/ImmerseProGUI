from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import kornia

from torch.nn.modules.utils import _pair, _quadruple

from einops import rearrange

from model.modules.base_module import BaseNetwork
from model.modules.sparse_transformer import TemporalSparseTransformerBlock, SoftSplit, SoftComp
from model.modules.sparse_transformer_cross import TemporalSparseTransformerBlock as TemporalSparseTransformerBlockCross
from model.modules.spectral_norm import spectral_norm as _spectral_norm
from model.modules.flow_loss_utils import flow_warp
from model.modules.deformconv import ModulatedDeformConv2d

from transformers import AutoModelForDepthEstimation
from .misc import constant_init

from .common_arch import (
    length_sq,
    fbConsistencyCheck,
    DeformableAlignment,
    BidirectionalPropagationChecked as BidirectionalPropagation,
    deconv,
)


class BNReLU(nn.Module):
    def __init__(self, inplane, use_bn=False):
        super().__init__()
        if use_bn:
            self.bn = nn.BatchNorm2d(inplane)
        self.use_bn = use_bn
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        if self.use_bn:
            return self.leakyrelu(self.bn(x))
        return self.leakyrelu(x)


class Encoder(nn.Module):
    def __init__(self, output_sizes=None, out_indices = [3, 15]):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.out_indices = out_indices
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            BNReLU(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            BNReLU(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            BNReLU(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            BNReLU(256),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
            BNReLU(384),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
            BNReLU(512),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            BNReLU(384),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            BNReLU(256),
        ])

        self.feat_to_prop = nn.ModuleList([
            # nn.Sequential(
            #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, groups=1),
            #     nn.LeakyReLU(0.2, inplace=True)
            # ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, groups=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        ])
        self.output_sizes = output_sizes

    def forward(self, x):
        bt, c, _, _ = x.size()
        out = x
        output = []
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
                _, _, h, w = x0.size()
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x, o], 2).view(bt, -1, h, w)
            out = layer(out)
            
            if i in self.out_indices:
                o = self.feat_to_prop[len(output)](out)
                if self.output_sizes is not None:
                    o = nn.functional.interpolate(
                        o,
                        self.output_sizes[len(output)],
                        mode="bilinear",
                        align_corners=True,
                    )
                output.append(o)
        if len(output) == 1:
            return output[0]
        return output



class DepthAnythingDepthEstimationHead(nn.Module):

    def __init__(self, output_sizes=[(128, 128), (64, 64)]):
        super().__init__()

        self.output_sizes = output_sizes
        features = 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            BNReLU(features, use_bn=True),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            BNReLU(features, use_bn=True),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            BNReLU(features, use_bn=True),
        )
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            BNReLU(features, use_bn=True),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            BNReLU(features, use_bn=True),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            BNReLU(features, use_bn=True),
        )
        self.conv22 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, hidden_states: List[torch.Tensor], patch_height, patch_width):
        # hidden_states = hidden_states[-1]
        feat_early = self.conv1(hidden_states[-1])
        feat_early = nn.functional.interpolate(
            feat_early,
            self.output_sizes[0],
            mode="bilinear",
            align_corners=True,
        )
        feat_early = self.conv2(feat_early)
        feat_late = self.conv11(hidden_states[-2])
        feat_late = nn.functional.interpolate(
            feat_late,
            self.output_sizes[1],
            mode="bilinear",
            align_corners=True,
        )
        feat_late = self.conv22(feat_late)
        return feat_early, feat_late


class DepthAnythingEncoder(nn.Module):

    def __init__(self, output_sizes):
        super(DepthAnythingEncoder, self).__init__()
        self.model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-base-hf")
        self.model.head = DepthAnythingDepthEstimationHead(output_sizes)
        
        # for params in self.model.backbone.parameters():
        #     params.requires_grad = False
        # for params in self.model.backbone.encoder.layer[-8:].parameters():
        #     params.requires_grad = True
        # for params in self.model.backbone.layernorm.parameters():
        #     params.requires_grad = True

    def forward(self, x):
        disparity_feat = self.model(x).predicted_depth
        return disparity_feat


class GuidedRefiner(nn.Module):
    def __init__(self, num_depth_layers: int = 65, number_layered_depth: int = 7):
        super(GuidedRefiner, self).__init__()
        # self.conv_img = nn.Sequential(
        #     nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )
        self.conv_img = Encoder(out_indices=[15])

        self.disp_refine = nn.Sequential(
            nn.Conv2d(num_depth_layers, 128, kernel_size=3, stride=1, padding=1),
            BNReLU(128, use_bn=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            BNReLU(128, use_bn=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            BNReLU(128, use_bn=True),
        )
        self.disp_conv = nn.Sequential(
            deconv(128, 128, kernel_size=3, padding=1),
            BNReLU(128, use_bn=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            BNReLU(128, use_bn=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            BNReLU(64, use_bn=True),
            nn.Conv2d(64, number_layered_depth, kernel_size=3, stride=1, padding=1)
        )
        self.layered_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            BNReLU(64, use_bn=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            BNReLU(128, use_bn=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            BNReLU(128, use_bn=True),
        )
        self.texture_conv = nn.Sequential(
            deconv(256, 256, kernel_size=3, padding=1),
            BNReLU(256, use_bn=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, groups=4),
            BNReLU(128, use_bn=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, groups=4),
            BNReLU(64, use_bn=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            BNReLU(32, use_bn=False),
        )
        self.pre_final_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            BNReLU(64, use_bn=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            BNReLU(128, use_bn=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            BNReLU(128, use_bn=True),
        )
        self.number_layered_depth = number_layered_depth
        self.final_conv = nn.Sequential(
            nn.Conv2d(32 + 3, 128, kernel_size=3, stride=1, padding=1),
            BNReLU(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=4),
            BNReLU(128),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
        )

        self.feat_prop = BidirectionalPropagation(256, learnable=True)
        # soft split and soft composition
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        t2t_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding
        }
        self.ss = SoftSplit(128, 512, kernel_size, stride, padding)
        self.sc = SoftComp(128, 512, kernel_size, stride, padding)
        self.ss_disp = SoftSplit(128, 512, kernel_size, stride, padding)

        self.ss_im = SoftSplit(256, 512, kernel_size, stride, padding)
        self.sc_im = SoftComp(256, 512, kernel_size, stride, padding)
        # self.ss_disp_im = SoftSplit(256, 512, kernel_size, stride, padding)

        # self.median_blur = MedianPool2d(padding=1)
        
        depths = 8
        num_heads = 4
        window_size = (5, 5)
        pool_size = (4, 4)
        self.transformers = TemporalSparseTransformerBlockCross(
            dim=512,
            n_head=num_heads,
            window_size=window_size,
            pool_size=pool_size,
            depths=depths,
            t2t_params=t2t_params,
        )
        self.transformers_im = TemporalSparseTransformerBlock(
            dim=512,
            n_head=num_heads,
            window_size=window_size,
            pool_size=pool_size,
            depths=depths,
            t2t_params=t2t_params,
        )
        for block in [self.final_conv]:
            for m in block:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def make_stereo(self, image_tensor, disparity):
        """
        Args:
            max_shift: Maximum shift at the grid level (simulate lateral movement), adjust based on your needs.
        """

        B, C, H, W = image_tensor.size()
        mask_tensor = torch.ones_like(image_tensor)[:, :1]

        # Create a mesh grid
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, H, device=image_tensor.device),
            torch.linspace(-1, 1, W, device=image_tensor.device)
        )
        grid = torch.stack([yy, xx], dim=2).unsqueeze(0)  # Shape: [B, H, W, 2]
        grid = grid.repeat(B, 1, 1, 1)  # Adjust grid size to match batch size
        grid[..., 0] += disparity  # Apply shifts in x direction
        # grid[..., 1] += disparity[..., 1] * max_shift[1]  # Apply shifts in y direction

        # Use grid_sample to apply the shifts
        warped_image = F.grid_sample(image_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        warped_mask = F.grid_sample(mask_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return warped_image, warped_mask

    def flow_prop(self, enc_feat, completed_flows, original_frames, number_local_frames):
        b, t, _, ori_h, ori_w = original_frames.shape
        l_t = number_local_frames
        _, c, h, w = enc_feat.size()
        local_feat = enc_feat.view(b, t, c, h, w)[:, :l_t, ...]
        ref_feat = enc_feat.view(b, t, c, h, w)[:, l_t:, ...]

        ds_flows_f = F.interpolate(
            completed_flows[0].view(-1, 2, ori_h, ori_w),
            # scale_factor=1/4,
            size=(local_feat.size(-2), local_feat.size(-1)),
            mode='bilinear',
            align_corners=False
        ).view(b, l_t - 1, 2, h, w) / 4.0
        ds_flows_b = F.interpolate(
            completed_flows[1].view(-1, 2, ori_h, ori_w),
            # scale_factor=1/4,
            size=(local_feat.size(-2), local_feat.size(-1)),
            mode='bilinear',
            align_corners=False
        ).view(b, l_t - 1, 2, h, w) / 4.0

        _, _, local_feat = self.feat_prop(local_feat, ds_flows_f, ds_flows_b, "bilinear")
        enc_feat = torch.cat((local_feat, ref_feat), dim=1)
        return enc_feat

    def forward(self, disp, orig_x, completed_flows, num_local_frames):
        b, t, c, orig_h, orig_w = orig_x.shape
        enc_feat = self.conv_img(orig_x.view(-1, c, orig_h, orig_w))

        fold_feat_size = (enc_feat.size(-2), enc_feat.size(-1))
        _, c, h, w = enc_feat.size()

        # Disp Refiner
        cross_feat = self.disp_refine(disp)
        cross_feat_ = self.ss_disp(cross_feat.view(-1, c, h, w), b, fold_feat_size)

        trans_feat = self.ss(enc_feat.view(-1, c, h, w), b, fold_feat_size)
        trans_feat = self.transformers(cross_feat_, trans_feat, fold_feat_size, t_dilation=2)
        trans_feat = self.sc(trans_feat, t, fold_feat_size)
        # trans_feat = trans_feat.view(b, t, -1, h, w)
        cross_feat = cross_feat + trans_feat
        layered_depth = self.disp_conv(cross_feat.view(b * t, -1, h, w))

        layered_depth_ = F.interpolate(layered_depth, (orig_h, orig_w), mode="bilinear", align_corners=True)
        layered_images = orig_x.view(b * t, -1, orig_h, orig_w)[:, None].repeat(1, self.number_layered_depth, 1, 1, 1)
        output, output_mask = self.make_stereo(
            layered_images.view(-1, *layered_images.shape[2:]), layered_depth_.view(-1, *layered_depth_.shape[2:]))
        output = output.view(b, t, self.number_layered_depth, -1, orig_h, orig_w)
        output_mask = output_mask.view(b, t, -1, 1, orig_h, orig_w)

        layered_mask = torch.zeros_like(output_mask)
        total_mask = torch.zeros_like(output_mask)
        for i in range(1, self.number_layered_depth):
            if i == 0:
                layered_mask[:, :, i] = output_mask[:, :, i]
                total_mask[:, :, i] = output_mask[:, :, i]
            else:
                total_mask[:, :, i] = torch.logical_or(output_mask[:, :, i], layered_mask[:, :, i - 1])
                layered_mask[:, :, i] = total_mask[:, :, i] - output_mask[:, :, i - 1]

        layered_output = layered_mask * output

        return layered_output.sum(2), layered_output.sum(2)


class Decoder(nn.Module):
    def __init__(self, num_depth_layers: int = 69, channels_per_layer: int = 3, factor=1):
        super(Decoder, self).__init__()
        self.num_depth_layers = num_depth_layers
        self.channels_per_layer = channels_per_layer
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            BNReLU(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(128, num_depth_layers, kernel_size=1, stride=1, padding=0)
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            BNReLU(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            BNReLU(128),
            nn.ConvTranspose2d(128, num_depth_layers, kernel_size=4, stride=2, padding=1)
        )
        self.conv_final = nn.Sequential(
            nn.Conv2d(num_depth_layers, num_depth_layers, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_depth_layers, num_depth_layers, kernel_size=3, stride=1, padding=1),
        )
        # self.conv_for_refine = nn.Sequential(
        #     nn.Conv2d(num_depth_layers * 3, num_depth_layers * 3, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(num_depth_layers * 3, num_depth_layers * 3, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(num_depth_layers * 3, num_depth_layers * 3, kernel_size=3, stride=1, padding=1),
        # )
        self.soft_to_hard = nn.Conv2d(num_depth_layers * 3, 3, kernel_size=1, stride=1, padding=0)
        self.refiner = GuidedRefiner(num_depth_layers)
        # self.final_refine = UNet()

        self.register_buffer("trans_mat", self._shift_mat_precompute(
            list(range(- (num_depth_layers // 2) * factor, num_depth_layers * factor - (num_depth_layers // 2) * factor, factor))
        ))

    def _shift_mat_precompute(self, shifts):
        N = len(shifts)
        translation = torch.stack([
            torch.tensor(shifts,),
            torch.zeros(N,)
        ], dim=-1)

        trans_mat = kornia.geometry.transform.imgwarp.get_translation_matrix2d(translation)
        return trans_mat

    def shift_image(self, input, mat):
        input_stack = input[:, None].repeat(1, mat.size(0), 1, 1, 1)
        B, N, C, H, W = input_stack.size()

        trans_mat = mat.repeat(B, 1, 1)
        out = kornia.geometry.imgwarp.warp_affine(
            input_stack.reshape(-1, C, H, W), trans_mat[:, :2], (H, W), padding_mode="zeros")
        # Reshape back to original size (B, N, C, H, W)
        shifted_images = out.view(B, N, C, H, W)
        return shifted_images

    def forward(self, x1, x2, orig_x, completed_flows, num_local_frames):
        b, _, c, h, w = orig_x.size()
        x1 = x1.view(-1, *x1.shape[2:])
        x2 = x2.view(-1, *x2.shape[2:])

        out_x1 = self.decoder_1(x1)
        out_x2 = self.decoder_2(x2)
        p_out = self.conv_final(out_x1 + out_x2)
        # assert False, p_out.shape
        shifted_view = self.shift_image(orig_x.view(-1, *orig_x.shape[2:]), self.trans_mat)
        # import matplotlib.pyplot as plt
        # for j in range(shifted_view.size(0)):
        #     for i, img in enumerate(shifted_view[j]):
        #         plt.imsave(f"{j}_{i}.png", (img.cpu().detach().permute(1, 2, 0).numpy() + 1) / 2)
        # assert False
        p_out_orig = F.interpolate(p_out, size=(h, w), mode='bilinear')
        b, c = p_out_orig.size(0), p_out_orig.size(1)

        p_out_orig = F.softmax(p_out_orig.unsqueeze(2).view(b, self.num_depth_layers, 1, h, w), dim=1)
        argmax = torch.argmax(p_out_orig, dim=1)
        # print(torch.unique(argmax[0]))

        mult_soft_shift_out = torch.mul(p_out_orig, shifted_view)
        soft_output = mult_soft_shift_out.sum(1).squeeze()

        # mult_soft = self.conv_for_refine(mult_soft_shift_out.view(b, -1, h, w))
        warped_output, final_output = self.refiner(p_out_orig.view(b, -1, h, w), orig_x, completed_flows, num_local_frames)
        # hard_output = self.soft_to_hard(mult_soft)

        # final_right = self.final_refine(soft_output + rt_texture, enc_feat)
        # orig_sm = F.interpolate(
        #     orig_x.view(-1, c, h, w), size=(p_out.size(-2), p_out.size(-1)), mode='bilinear')

        # # Repeat for more channels
        # b, c, h, w = p_out.shape
        # p_out = F.softmax(p_out.unsqueeze(2).view(b, c // self.channels_per_layer, self.channels_per_layer, 1, h, w), dim=2)
        # shifted_view = shifted_view.unsqueeze(2).repeat(1, 1, self.channels_per_layer, 1, 1, 1)
        # mult_soft_shift_out = torch.mul(p_out, shifted_view).view(b, -1, h, w)

        # rt_image = self.refiner(mult_soft_shift_out, orig_x)
        return soft_output, warped_output, final_output, argmax


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True, model_path=None):
        super(InpaintGenerator, self).__init__()
        channel = 128
        hidden = 512

        self.encoder = DepthAnythingEncoder([(192, 192), (96, 96)])
        # # self.encoder = Encoder([(128, 192), (64, 96)])
        # # self.context_encoder = ContextEncoder()
        self.decoder = Decoder(num_depth_layers=79, factor=1)

        # self.encoder = DepthAnythingEncoder([(192, 192), (96, 96)])
        # self.decoder = Decoder(num_depth_layers=69, factor=1)

        # soft split and soft composition
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        t2t_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding
        }
        self.ss = nn.ModuleList([
            SoftSplit(channel, hidden, kernel_size, stride, padding),
            SoftSplit(channel, hidden, kernel_size, stride, padding),
        ])
        self.sc = nn.ModuleList([
            SoftComp(channel, hidden, kernel_size, stride, padding),
            SoftComp(channel, hidden, kernel_size, stride, padding),
        ])
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)

        # feature propagation module
        # self.img_prop_module = BidirectionalPropagation(3, learnable=False)
        self.feat_prop_module = nn.ModuleList([
            # BidirectionalPropagation(channel, learnable=True),
            # BidirectionalPropagation(channel, learnable=True),
        ])

        depths = 2
        num_heads = 4
        window_size = [(7, 7), (7, 7)]
        pool_size = [(4, 4), (4, 4)]
        self.transformers = nn.ModuleList([
            TemporalSparseTransformerBlock(
                dim=hidden,
                n_head=num_heads,
                window_size=window_size[0],
                pool_size=pool_size[0],
                depths=depths,
                t2t_params=t2t_params,
            ),
            TemporalSparseTransformerBlock(
                dim=hidden,
                n_head=num_heads,
                window_size=window_size[1],
                pool_size=pool_size[1],
                depths=depths,
                t2t_params=t2t_params,
            ),
        ])
        if init_weights:
            self.init_weights()

        if model_path is not None:
            print('Pretrained ProPainter has loaded...')
            ckpt = torch.load(model_path, map_location='cpu')
            self.load_state_dict(ckpt, strict=True)

        # print network parameter number
        self.print_network()

    def img_propagation(self, frames, completed_flows, interpolation='nearest'):
        # _, _, prop_frames = self.img_prop_module(frames, completed_flows[0], completed_flows[1], interpolation)
        # return prop_frames
        return frames

    def forward(self, frames, completed_flows, num_local_frames, original_frames, interpolation='bilinear', t_dilation=2):
        """
        Args:
            masks_in: original mask
            masks_updated: updated mask after image propagation
        """

        l_t = num_local_frames
        b, t, _, ori_h, ori_w = frames.size()
        # extracting features
        # enc_feat = self.encoder(torch.cat([masked_frames.view(b * t, 3, ori_h, ori_w),
        #                                 masks_in.view(b * t, 1, ori_h, ori_w),
        #                                 masks_updated.view(b * t, 1, ori_h, ori_w)], dim=1))
        enc_feat_early, enc_feat_late = self.encoder(frames.view(b * t, 3, ori_h, ori_w))
        enc_feat_out = []

        for i, enc_feat in enumerate([enc_feat_early, enc_feat_late]):
            _, c, h, w = enc_feat.size()
            local_feat = enc_feat.view(b, t, c, h, w)[:, :l_t, ...]
            ref_feat = enc_feat.view(b, t, c, h, w)[:, l_t:, ...]
            fold_feat_size = (h, w)

            enc_feat = torch.cat((local_feat, ref_feat), dim=1)
            trans_feat = self.ss[i](enc_feat.view(-1, c, h, w), b, fold_feat_size)
            # mask_pool_l = rearrange(mask_pool_l, 'b t c h w -> b t h w c').contiguous()
            trans_feat = self.transformers[i](trans_feat, fold_feat_size, t_dilation=t_dilation)
            trans_feat = self.sc[i](trans_feat, t, fold_feat_size)
            trans_feat = trans_feat.view(b, t, -1, h, w)

            enc_feat = enc_feat + trans_feat
            enc_feat_out.append(enc_feat)

        if self.training:
            output = self.decoder(
                enc_feat_out[0],
                enc_feat_out[1],
                original_frames.view(b, t, 3, ori_h, ori_w),
                completed_flows, num_local_frames
            )
            if isinstance(output, (tuple, list,)):
                output = (torch.tanh(output[0]).view(b, t, 3, ori_h, ori_w), *output[1:])
            else:
                output = torch.tanh(output).view(b, t, 3, ori_h, ori_w)
        # else:
        #     output = self.decoder(
        #         enc_feat_out[0],
        #         enc_feat_out[1],
        #         original_frames.view(b, t, 3, ori_h, ori_w),
        #         completed_flows, num_local_frames
        #     )
        #     if isinstance(output, (tuple, list,)):
        #         output = (
        #             torch.tanh(output[0]).view(b, t, 3, ori_h, ori_w)[:, :l_t],
        #             torch.tanh(output[1]).view(b, t, 3, ori_h, ori_w)[:, :l_t],
        #             torch.tanh(output[2]).view(b, t, 3, ori_h, ori_w)[:, :l_t],
        #             output[3][:, :l_t]
        #         )
        #     else:
        #         output = torch.tanh(output).view(b, t, 3, ori_h, ori_w)[:, :l_t]

        return output


# ######################################################################
#  Discriminator for Temporal Patch GAN
# ######################################################################
class Discriminator(BaseNetwork):

    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=nf * 1,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=1,
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 1,
                          nf * 2,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 2,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4,
                      nf * 4,
                      kernel_size=(3, 5, 5),
                      stride=(1, 2, 2),
                      padding=(1, 2, 2)))

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape (old)
        # B, T, C, H, W (new)
        xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


class Discriminator_2D(BaseNetwork):
    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=True):
        super(Discriminator_2D, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=nf * 1,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 1,
                          nf * 2,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 2,
                          nf * 4,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4,
                      nf * 4,
                      kernel_size=(1, 5, 5),
                      stride=(1, 2, 2),
                      padding=(0, 2, 2)))

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape (old)
        # B, T, C, H, W (new)
        xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out

def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module


if __name__ == "__main__":
    
    image_shape = (384, 384)
    # image_shape = (756, 256)
    num_local_frames = 4
    
    # import timm
    # frames = torch.rand(1, 8, 3, 756, 256, requires_grad=True).cuda()
    # model = timm.create_model("hrnet_w48", features_only=True).cuda()
    # assert False, [a.shape for a in model(frames[0])]
    frames = torch.rand(1, num_local_frames - 1, 3, *image_shape, requires_grad=True).cuda()
    completed_flows = [
        torch.rand(1, num_local_frames - 1, 2, *image_shape, requires_grad=True).cuda(),
        torch.rand(1, num_local_frames - 1, 2, *image_shape, requires_grad=True).cuda()
    ]
    original_frames = torch.rand(1, num_local_frames - 1, 3, *image_shape, requires_grad=True).cuda()

    out = InpaintGenerator().cuda()(frames, completed_flows, num_local_frames, original_frames)

    print(out[0].shape, out[1], torch.cuda.max_memory_allocated(device=None)/(10**9), "Gb")
