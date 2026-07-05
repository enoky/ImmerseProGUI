from typing import List, Tuple
import cv2
import os
import glob
import logging
import importlib
from PIL import Image
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from core.dataset import TestDataset
import torchvision.transforms as transforms
from core.utils import (create_random_shape_with_random_motion, Stack,
                        ToTorchFormatTensor, GroupRandomHorizontalFlip,GroupRandomHorizontalFlowFlip)


def read_video_cv2(filename):
    """Read a video into a (T, H, W, C) uint8 RGB tensor. Replaces
    torchvision.io.read_video, removed in torchvision 0.24."""
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {filename}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise IOError(f"No frames decoded from video: {filename}")
    return torch.stack(frames), fps


def write_video_cv2(filename, frames, fps):
    """Write a (T, H, W, C) RGB tensor to an mp4 file. Replaces
    torchvision.io.write_video, removed in torchvision 0.24."""
    frames = frames.to(torch.uint8).cpu().numpy()
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        raise IOError(f"Failed to open video writer: {filename}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


class Evaluator:
    def __init__(self, config, output_dir, output_index=2):
        self.config = config
        self.output_dir = output_dir
        self.num_local_frames = (6 or config['train_data_loader']['num_local_frames'])
        self.num_ref_frames = (2 or config['train_data_loader']['num_ref_frames'])
        self.output_index = output_index

        # setup models including generator and discriminator
        net = importlib.import_module('model.' + config['model']['net'])
        self.netG = net.InpaintGenerator()
        # print(self.netG)
        self.netG = self.netG.to(self.config['device'])
        
        self.interp_mode = self.config['model']['interp_mode']
        self.load()

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])

    def load(self):
        """Load netG (and netD)."""
        # get the latest checkpoint
        model_path = self.config['save_dir']
        ckpts = [
            os.path.basename(i).split('.pth')[0]
            for i in glob.glob(os.path.join(model_path, '*.pth'))
        ]
        ckpts.sort()
        latest_epoch = ckpts[-1][4:] if len(ckpts) > 0 else None
        
        loaded = False
        if latest_epoch is not None:
            gen_path = os.path.join(model_path, f'gen_{int(latest_epoch):06d}.pth')
            dataG = torch.load(gen_path, map_location=self.config['device'])
            loaded = True
            self.netG.load_state_dict(dataG)
        else:
            gen_path = self.config['trainer'].get('gen_path', None)
            if gen_path is not None:
                dataG = torch.load(gen_path, map_location=self.config['device'])
                self.netG.load_state_dict(dataG)
                loaded = True
        
        if not loaded:
            raise RuntimeError("Model not loaded. Please manual set `gen_path` or having a `latest.pth`")
        print("Loaded", gen_path)


    @torch.no_grad()
    def run_inference(self, frames):
        device = self.config['device']
        frames = frames.to(device)
        b, t, c, h, w = frames.size()
        l_t = self.num_local_frames

        results = []

        for start in range(math.ceil(t / self.num_local_frames)):
            masked_frames = frames
            # masked_frames = frames * (1 - masks)
            masked_local_frames = masked_frames[:, start * l_t:(start + 1) * l_t, ...]
        
            # ---- image propagation ----
            prop_imgs = self.netG.img_propagation(masked_local_frames, None, interpolation=self.interp_mode)
            # prop_imgs = masked_local_frames
            updated_frames = masked_frames.clone()
            # prop_local_frames = gt_local_frames * (1-local_masks) + prop_imgs.view(b, l_t, 3, h, w) * local_masks # merge
            prop_local_frames = prop_imgs.view(b, -1, 3, h, w)  # TODO: confirm
            updated_frames[:, start * l_t:(start + 1) * l_t, ...] = prop_local_frames
            original_frames = masked_frames.clone()
            original_frames[:, start * l_t:(start + 1) * l_t, ...] = masked_local_frames.view(b, -1, 3, h, w)

            # ---- feature propagation + Transformer ----
            output = self.netG(updated_frames, None, masked_local_frames.size(1), original_frames=original_frames)
            if isinstance(output, (list, tuple)) and len(output) == 3:
                pred_conversion, pred_texture, pred_imgs = output
                pred_conversion = pred_conversion.view(b, -1, c, h, w)
                pred_texture = pred_texture.view(b, -1, c, h, w)
                pred_imgs = pred_imgs.view(b, -1, c, h, w)
                selection_map = None
            elif isinstance(output, (list, tuple)) and len(output) == 4:
                pred_conversion, pred_texture, pred_imgs, selection_map = output
                pred_conversion = pred_conversion.view(b, -1, c, h, w)
                pred_imgs = pred_imgs.view(b, -1, c, h, w)
                pred_texture = pred_texture.view(b, -1, c, h, w)
                selection_map = selection_map.view(b, -1, 1, h, w)
            else:
                raise ValueError

            pred_local_frames = pred_imgs[:,  start * l_t:(start + 1) * l_t, ...]
            pred_local_conversion_frames = pred_conversion[:,  start * l_t:(start + 1) * l_t, ...]
            pred_local_texture_frames = pred_texture[:,  start * l_t:(start + 1) * l_t, ...]

            print(pred_local_conversion_frames.shape, masked_local_frames.size(1), start, start * l_t, (start + 1) * l_t)
            for t in range(masked_local_frames.size(1)):
                if self.output_index == 1:
                    pred = ((pred_local_conversion_frames.view(b, -1, 3, h, w) + 1)/2.0).cpu()
                elif self.output_index == 2:
                    pred = ((pred_local_texture_frames.view(b, -1, 3, h, w) + 1)/2.0).cpu()
                else:
                    raise IndexError

                result = pred[0][t].permute(1, 2, 0).detach().clamp(0, 1)
                results.append(result)

        return results

    @torch.no_grad()
    def inference(self, dataloader=None):
        device = self.config['device']
        for data in tqdm(dataloader or self.val_dataloader):
            frames, frames_right, flows_f, flows_b, vname, frame_names = data
            not_exist = True
            for name in frame_names:
                if os.path.exists(os.path.join(self.output_dir, vname[0], name[0])):
                    not_exist = False
                    break
            if not not_exist:
                print("skipping", frame_names[0][0], "to", frame_names[-1][0])
                continue
            
            results = self.run_inference(frames)

            os.makedirs(os.path.join(self.output_dir, vname[0]), exist_ok=True)
            for t in range(l_t):
                if self.output_index == 1:
                    pred = ((pred_local_conversion_frames.view(b,-1,3,h,w) + 1)/2.0).cpu()
                elif self.output_index == 2:
                    pred = ((pred_local_texture_frames.view(b,-1,3,h,w) + 1)/2.0).cpu()
                else:
                    raise IndexError
                try:
                    plt.imsave(os.path.join(self.output_dir, vname[0], frame_names[t][0]),
                            pred[0][t].permute(1, 2, 0).detach().clamp(0, 1).numpy())
                except Exception as e:
                    raise ValueError(f"{e}, {pred.min()}, {pred.max()}")

    def inference_image_list(self, image_list: List[Tuple[str, int]], frame_name_prefix="frame", frame_name_suffix="left.jpg"):
        for i, (folder_path, start_frame_idx) in enumerate(image_list):
            video_name = [folder_path.split("/")[-1]]

            num_iters = len(os.listdir(os.path.join(folder_path, "left"))) // self.num_local_frames

            for j in range(num_iters):
                end_frame_idx = start_frame_idx + self.num_local_frames
                frame_names = []

                frames = []
                frames_right = []
                flows_f, flows_b = [], []

                for idx in range(start_frame_idx, end_frame_idx):
                    frame_name = f"{frame_name_prefix}_{str(idx).zfill(8)}_{frame_name_suffix}"
                    frame_path = os.path.join(folder_path, "left", frame_name)
                    print(frame_path)

                    img = cv2.imread(frame_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_LINEAR)
                    img = Image.fromarray(img)

                    # right_img_path = os.path.join(folder_path, "right", frame_name.replace("left", "right"))
                    # right_img = cv2.imread(right_img_path)
                    # right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
                    # right_img = cv2.resize(right_img, (384, 384), interpolation=cv2.INTER_LINEAR)
                    # right_img = Image.fromarray(right_img)

                    frames.append(img)
                    # frames_right.append(right_img)
                    frame_names.append([frame_name])
                    
                frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
                # frames_right_tensors = self._to_tensors(frames_right) * 2.0 - 1.0

                # self.inference([
                #     (frame_tensors[None], frames_right_tensors[None], 'None', 'None', video_name, frame_names)
                # ])
            
                results = self.run_inference(frame_tensors[None])
                
                os.makedirs(os.path.join(self.output_dir, video_name[0]), exist_ok=True)
                for res, name in zip(results, frame_names):
                    print("Saving", os.path.join(self.output_dir, video_name[0], name[0]))
                    plt.imsave(os.path.join(self.output_dir, video_name[0], name[0]), res.numpy())

                start_frame_idx = end_frame_idx

    def inference_video_list(self, video_list: str, no_anaglyph=True, no_sbs=True):

        for i, frame_root in enumerate(video_list):

            video_name = os.path.basename(frame_root)[:-4]
            print("starting", video_name)
            frames, fps = read_video_cv2(frame_root) # RGB
            frames = frames.permute(0, 3, 1, 2) / 255 * 2.0 - 1.0
            size = frames.shape[-2:]
            frames = F.interpolate(frames, (384, 384), mode="bicubic", align_corners=False)[None]
            out = self.run_inference(frames)
            out = torch.stack(out).permute(0, 3, 1, 2)
            f_right = out[..., 5:-5]  # Remove borders
            f_left = (frames[0, ..., 5:-5] + 1.) / 2

            f_right = F.interpolate(f_right, size, mode="bicubic", align_corners=False).clip(0, 1)
            f_left = F.interpolate(f_left, size, mode="bicubic", align_corners=False).clip(0, 1)

            if not no_sbs:
                print(f_right.shape, f_left.shape)
                out = torch.cat([f_left, f_right], dim=-1).permute(0, 2, 3, 1)
                print(os.path.join(self.output_dir, video_name + "_immersepro_sbs.mp4"))
                write_video_cv2(os.path.join(self.output_dir, video_name + "_immersepro_sbs.mp4"), (out * 255).long(), fps=fps)

            if not no_anaglyph:
                l_r, l_g, l_b = f_left.split(1, dim=1)
                r_r, r_g, r_b = f_right.split(1, dim=1)
                out = torch.cat([l_r, r_g, r_b], dim=1).permute(0, 2, 3, 1)
                print(os.path.join(self.output_dir, video_name + "_immersepro_anaglyph.mp4"))
                write_video_cv2(os.path.join(self.output_dir, video_name + "_immersepro_anaglyph.mp4"), (out * 255).long(), fps=fps)
