"""ImmersePro — 2D to stereo 3D converter GUI.

Turns a 2D video plus a pre-rendered depth-map video into stereo output
(side-by-side and/or red-cyan anaglyph):

  1. The depth video is converted to per-pixel disparity.
  2. Each left-eye frame is forward-splatted into the right-eye view at full
     resolution (softmax splatting: bilinear, depth-weighted soft z-buffer),
     leaving real holes where background is disoccluded.
  3. The holes are filled with ProPainter video inpainting (temporally
     coherent, at a reduced internal resolution to fit GPU memory).
  4. Inpainted content is composited back into the full-resolution warp.

Inputs are matched by filename: "clip.mp4" pairs with "clip.mp4" or
"clip_depth.mp4" in the depth folder. Depth is grayscale, white = near.

Weights: weights/{raft-things,recurrent_flow_completion,ProPainter}.pth are
downloaded automatically on first run if missing.
Note: ProPainter weights are released under the NTU S-Lab license 1.0
(non-commercial use).

Run:  python app.py
"""
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(APP_DIR, ".app_settings.json")
PRETRAIN_MODEL_URL = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

DEFAULTS = {
    "mode": "batch",
    "left_dir": os.path.join(".", "videos", "left_eye"),
    "depth_dir": os.path.join(".", "videos", "depth"),
    "left_file": "",
    "depth_file": "",
    "output_dir": os.path.join(".", "videos", "output"),
    "max_disparity": 2.5,
    "convergence": 0.5,
    "splat_alpha": 50.0,
    "disp_blur": 9,
    "invert_depth": False,
    "edge_crop": True,
    "inpaint_width": 960,
    "fp32": False,
    "neighbor_length": 10,
    "ref_stride": 10,
    "subvideo_length": 80,
    "out_sbs": True,
    "out_anaglyph": True,
    "out_debug": False,
    "skip_existing": True,
    "video_codec": "h264",
    "video_crf": 18,
    "video_preset": "medium",
}

CODEC_LABELS = {  # GUI label -> internal key
    "H.264 (recommended)": "h264",
    "H.265 / HEVC (smaller files)": "h265",
    "MP4V (no ffmpeg needed)": "mp4v",
}
CODEC_KEYS = {v: k for k, v in CODEC_LABELS.items()}
FFMPEG_CODECS = {"h264": "libx264", "h265": "libx265"}
PRESETS = ["ultrafast", "veryfast", "fast", "medium", "slow", "veryslow"]

# heavy modules, loaded lazily in the worker thread so the GUI opens instantly
cv2 = np = scipy_ndimage = torch = F = None
RAFT_bi = RecurrentFlowCompleteNet = InpaintGenerator = load_file_from_url = None
_backend_loaded = False


def load_backend(log):
    global cv2, np, scipy_ndimage, torch, F
    global RAFT_bi, RecurrentFlowCompleteNet, InpaintGenerator, load_file_from_url
    global _backend_loaded
    if _backend_loaded:
        return
    log("Loading libraries (first time takes a few seconds)...")
    import cv2 as _cv2
    import numpy as _np
    import scipy.ndimage as _ndi
    import torch as _torch
    import torch.nn.functional as _F
    from propainter.modules.flow_comp_raft import RAFT_bi as _RAFT_bi
    from propainter.recurrent_flow_completion import RecurrentFlowCompleteNet as _RFC
    from propainter.propainter_arch import InpaintGenerator as _IG
    from utils.download_util import load_file_from_url as _dl
    cv2, np, scipy_ndimage, torch, F = _cv2, _np, _ndi, _torch, _F
    RAFT_bi, RecurrentFlowCompleteNet, InpaintGenerator, load_file_from_url = _RAFT_bi, _RFC, _IG, _dl
    _backend_loaded = True
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("WARNING: no CUDA GPU detected — processing will be extremely slow.")


class Cancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled()


# ----------------------------------------------------------------------------
# processing core
# ----------------------------------------------------------------------------
def read_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise IOError(f"No frames decoded from video: {path}")
    return frames, fps


_ffmpeg_exe = None


def find_ffmpeg():
    global _ffmpeg_exe
    if _ffmpeg_exe is None:
        exe = shutil.which("ffmpeg")
        if exe is None:
            try:
                import imageio_ffmpeg
                exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                exe = ""
        _ffmpeg_exe = exe or ""
    return _ffmpeg_exe or None


class FFmpegWriter:
    """Streams BGR frames to an ffmpeg subprocess for H.264/H.265 encoding
    with CRF quality control. Same write()/release() interface as cv2."""

    def __init__(self, path, w, h, fps, codec="libx264", crf=18, preset="medium"):
        exe = find_ffmpeg()
        if not exe:
            raise IOError("ffmpeg not found")
        self.path = path
        cmd = [
            exe, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
            "-i", "-", "-an",
            "-c:v", codec, "-crf", str(int(crf)), "-preset", preset,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # yuv420p needs even dims
            "-pix_fmt", "yuv420p",
        ]
        if codec == "libx265":
            cmd += ["-tag:v", "hvc1"]  # widest player compatibility for HEVC in mp4
        cmd.append(path)
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def write(self, frame_bgr):
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        except (BrokenPipeError, OSError):
            raise IOError(f"ffmpeg encoder terminated unexpectedly while writing {self.path}")

    def release(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        self.proc.wait()


def make_writer(path, w, h, fps, opt=None):
    codec = (opt or {}).get("video_codec", "mp4v")
    if codec in FFMPEG_CODECS and find_ffmpeg():
        return FFmpegWriter(path, w, h, fps, codec=FFMPEG_CODECS[codec],
                            crf=(opt or {}).get("video_crf", 18),
                            preset=(opt or {}).get("video_preset", "medium"))
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise IOError(f"Failed to open video writer: {path}")
    return writer


def depth_to_disparity(depth, w, max_disparity_pct, convergence, invert_depth, disp_blur):
    """depth: (B, 1, H, W) in [0, 1]. Returns signed disparity in pixels."""
    if invert_depth:
        depth = 1.0 - depth
    if disp_blur > 1:
        k = int(disp_blur) | 1
        pad = k // 2
        depth = F.avg_pool2d(F.pad(depth, (pad, pad, pad, pad), mode="replicate"), k, stride=1)
    max_disp_px = max_disparity_pct / 100.0 * w
    return max_disp_px * (depth - convergence), depth


def forward_splat(left, depth, disp, alpha=50.0):
    """Softmax-splatting forward warp of the left view into the right view.

    Bilinear 2-tap splat along x (displacement is horizontal-only) where each
    contribution is weighted by exp(alpha * depth), so nearer content smoothly
    dominates collisions (soft z-buffer) while edges keep sub-pixel accuracy.

    left: (B, 3, H, W) [0, 1]; depth: (B, 1, H, W) [0, 1] (1 = near);
    disp: (B, 1, H, W) signed pixels (positive = near, shifts left).
    Returns (right (B, 3, H, W), holes (B, 1, H, W) float {0,1}).
    """
    b, _, h, w = left.shape
    device = left.device
    n = h * w

    xs = torch.arange(w, device=device).view(1, 1, 1, w)
    x_dst = xs - disp  # right-eye x position for each source pixel
    x0 = torch.floor(x_dst)
    frac = (x_dst - x0).reshape(b, 1, n)
    y_idx = torch.arange(h, device=device).view(1, 1, h, 1).expand(b, 1, h, w).reshape(b, 1, n)

    # soft z-priority; subtract the max (1.0) so exp() stays in (0, 1]
    wz = torch.exp(alpha * (depth.clamp(0, 1) - 1.0)).reshape(b, 1, n)

    num = torch.zeros(b, 3, n, device=device)
    den = torch.zeros(b, 1, n, device=device)
    cov = torch.zeros(b, 1, n, device=device)  # unweighted coverage for hole detection
    color = left.reshape(b, 3, n)

    for xi, bw in ((x0, 1.0 - frac), (x0 + 1.0, frac)):
        xi = xi.reshape(b, 1, n).long()
        valid = (xi >= 0) & (xi < w)
        bw = bw * valid
        dst = (y_idx * w + xi).clamp(0, n - 1)
        den.scatter_add_(2, dst, bw * wz)
        cov.scatter_add_(2, dst, bw)
        num.scatter_add_(2, dst.expand(b, 3, n), color * (bw * wz))

    holes = cov < 0.25  # less than a quarter pixel of support -> disocclusion
    right = num / den.clamp_min(1e-30)
    right = right.reshape(b, 3, h, w) * (~holes).reshape(b, 1, h, w)
    return right, holes.reshape(b, 1, h, w).float()


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride, ref_num):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


class ProPainterInpainter:
    def __init__(self, device, use_half=True, raft_iter=20, ref_stride=10,
                 neighbor_length=10, subvideo_length=80, log=print):
        self.device = device
        self.use_half = use_half and device.type == "cuda"
        self.raft_iter = raft_iter
        self.ref_stride = ref_stride
        self.neighbor_length = neighbor_length
        self.subvideo_length = subvideo_length

        log("Loading RAFT optical flow model...")
        raft_ckpt = load_file_from_url(url=PRETRAIN_MODEL_URL + 'raft-things.pth', model_dir='weights')
        self.fix_raft = RAFT_bi(raft_ckpt, device)

        log("Loading flow completion model...")
        rfc_ckpt = load_file_from_url(url=PRETRAIN_MODEL_URL + 'recurrent_flow_completion.pth', model_dir='weights')
        self.fix_flow_complete = RecurrentFlowCompleteNet(rfc_ckpt)
        for p in self.fix_flow_complete.parameters():
            p.requires_grad = False
        self.fix_flow_complete.to(device).eval()

        log("Loading ProPainter inpainting model...")
        pp_ckpt = load_file_from_url(url=PRETRAIN_MODEL_URL + 'ProPainter.pth', model_dir='weights')
        self.model = InpaintGenerator(model_path=pp_ckpt).to(device).eval()

        if self.use_half:
            self.fix_flow_complete = self.fix_flow_complete.half()
            self.model = self.model.half()

    def __call__(self, frames_np, masks_np, progress=None, cancel=None):
        """frames_np: (T, H, W, 3) uint8 RGB (H, W divisible by 8);
        masks_np: (T, H, W) uint8 {0,1}, 1 = fill this pixel.
        Returns inpainted frames (T, H, W, 3) uint8."""
        with torch.no_grad():
            return self._run(frames_np, masks_np, progress, cancel)

    def _run(self, frames_np, masks_np, progress, cancel):
        progress = progress or (lambda stage, frac: None)
        video_length, h, w = frames_np.shape[:3]
        device = self.device

        frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float().div_(255.).unsqueeze(0)
        frames = frames.to(device) * 2 - 1
        flow_masks, masks_dilated = [], []
        for m in masks_np:
            flow_masks.append(scipy_ndimage.binary_dilation(m, iterations=4).astype(np.uint8))
            masks_dilated.append(scipy_ndimage.binary_dilation(m, iterations=2).astype(np.uint8))
        flow_masks = torch.from_numpy(np.stack(flow_masks)).unsqueeze(1).float().unsqueeze(0).to(device)
        masks_dilated = torch.from_numpy(np.stack(masks_dilated)).unsqueeze(1).float().unsqueeze(0).to(device)

        # ---- optical flow (fp32 RAFT), chunked ----
        if w <= 640:
            short_clip_len = 12
        elif w <= 720:
            short_clip_len = 8
        elif w <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2
        if video_length > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            chunk_starts = list(range(0, video_length, short_clip_len))
            for ci, f in enumerate(chunk_starts):
                check_cancel(cancel)
                end_f = min(video_length, f + short_clip_len)
                s = f if f == 0 else f - 1
                flows_f, flows_b = self.fix_raft(frames[:, s:end_f], iters=self.raft_iter)
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
                progress("flow", (ci + 1) / len(chunk_starts))
            gt_flows_bi = (torch.cat(gt_flows_f_list, dim=1), torch.cat(gt_flows_b_list, dim=1))
        else:
            gt_flows_bi = self.fix_raft(frames, iters=self.raft_iter)
            torch.cuda.empty_cache()
            progress("flow", 1.0)

        if self.use_half:
            frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())

        # ---- flow completion ----
        check_cancel(cancel)
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > self.subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            starts = list(range(0, flow_length, self.subvideo_length))
            for ci, f in enumerate(starts):
                check_cancel(cancel)
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + self.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + self.subvideo_length)
                pred_sub, _ = self.fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    flow_masks[:, s_f:e_f + 1])
                pred_sub = self.fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    pred_sub, flow_masks[:, s_f:e_f + 1])
                pred_flows_f.append(pred_sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
                pred_flows_b.append(pred_sub[1][:, pad_len_s:e_f - s_f - pad_len_e])
                torch.cuda.empty_cache()
                progress("flowcomp", (ci + 1) / len(starts))
            pred_flows_bi = (torch.cat(pred_flows_f, dim=1), torch.cat(pred_flows_b, dim=1))
        else:
            pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
            torch.cuda.empty_cache()
            progress("flowcomp", 1.0)

        # ---- image propagation ----
        check_cancel(cancel)
        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, self.subvideo_length)
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            starts = list(range(0, video_length, subvideo_length_img_prop))
            for ci, f in enumerate(starts):
                check_cancel(cancel)
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)
                b, t = masks_dilated[:, s_f:e_f].shape[:2]
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f - 1], pred_flows_bi[1][:, s_f:e_f - 1])
                prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(
                    masked_frames[:, s_f:e_f], pred_flows_bi_sub, masks_dilated[:, s_f:e_f], 'nearest')
                updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                    prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                torch.cuda.empty_cache()
                progress("prop", (ci + 1) / len(starts))
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t = masks_dilated.shape[:2]
            prop_imgs, updated_local_masks = self.model.img_propagation(
                masked_frames, pred_flows_bi, masks_dilated, 'nearest')
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            torch.cuda.empty_cache()
            progress("prop", 1.0)

        # ---- feature propagation + transformer ----
        comp_frames = [None] * video_length
        neighbor_stride = self.neighbor_length // 2
        ref_num = self.subvideo_length // self.ref_stride if video_length > self.subvideo_length else -1

        windows = list(range(0, video_length, neighbor_stride))
        for wi, f in enumerate(windows):
            check_cancel(cancel)
            neighbor_ids = list(range(max(0, f - neighbor_stride),
                                      min(video_length, f + neighbor_stride + 1)))
            ref_ids = get_ref_index(f, neighbor_ids, video_length, self.ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids]
            selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1]],
                                      pred_flows_bi[1][:, neighbor_ids[:-1]])
            l_t = len(neighbor_ids)
            pred_img = self.model(selected_imgs, selected_pred_flows_bi, selected_masks,
                                  selected_update_masks, l_t)
            pred_img = ((pred_img.view(-1, 3, h, w) + 1) / 2).float()
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids].cpu().permute(0, 2, 3, 1).float().numpy().astype(np.uint8)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = pred_img[i].astype(np.uint8) * binary_masks[i] + \
                    frames_np[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = (comp_frames[idx].astype(np.float32) * 0.5 +
                                        img.astype(np.float32) * 0.5).astype(np.uint8)
            torch.cuda.empty_cache()
            progress("inpaint", (wi + 1) / len(windows))

        return np.stack(comp_frames)


# per-clip progress = weighted sum of stages
STAGES = [
    ("read", 0.06, "Reading videos"),
    ("warp", 0.10, "Warping (softmax splat)"),
    ("flow", 0.18, "Optical flow"),
    ("flowcomp", 0.08, "Flow completion"),
    ("prop", 0.10, "Pixel propagation"),
    ("inpaint", 0.33, "Inpainting"),
    ("composite", 0.15, "Compositing + writing"),
]
STAGE_BASE = {}
_acc = 0.0
for _k, _w, _lbl in STAGES:
    STAGE_BASE[_k] = (_acc, _w, _lbl)
    _acc += _w


def process_clip(left_path, depth_path, out_dir, name, opt, inpainter, progress, cancel, log):
    """Convert one clip. progress(stage_key, frac). Raises Cancelled on cancel."""
    check_cancel(cancel)
    device = inpainter.device

    progress("read", 0.0)
    lefts, fps = read_video(left_path)
    depths, _ = read_video(depth_path)
    n = min(len(lefts), len(depths))
    if len(lefts) != len(depths):
        log(f"  WARNING: frame count mismatch (color {len(lefts)} vs depth {len(depths)}), using {n}")
    lefts, depths = lefts[:n], depths[:n]
    h, w = lefts[0].shape[:2]
    log(f"  {n} frames @ {fps:.3f} fps, {w}x{h}")
    progress("read", 1.0)

    # ---- stage 1: full-resolution forward splat ----
    right_full = np.empty((n, h, w, 3), dtype=np.uint8)
    holes_full = np.empty((n, h, w), dtype=np.uint8)
    batch = 8
    for s in range(0, n, batch):
        check_cancel(cancel)
        e = min(n, s + batch)
        left_t = torch.from_numpy(np.stack(lefts[s:e])).to(device).permute(0, 3, 1, 2).float() / 255.0
        depth_t = torch.from_numpy(np.stack([cv2.cvtColor(d, cv2.COLOR_RGB2GRAY) for d in depths[s:e]])) \
            .to(device).unsqueeze(1).float() / 255.0
        if depth_t.shape[-2:] != left_t.shape[-2:]:
            depth_t = F.interpolate(depth_t, left_t.shape[-2:], mode="bilinear", align_corners=False)
        disp, depth_t = depth_to_disparity(depth_t, w, opt["max_disparity"], opt["convergence"],
                                           opt["invert_depth"], opt["disp_blur"])
        with torch.no_grad():
            right_t, holes_t = forward_splat(left_t, depth_t, disp, alpha=opt["splat_alpha"])
        right_full[s:e] = (right_t.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        holes_full[s:e] = holes_t.squeeze(1).byte().cpu().numpy()
        progress("warp", e / n)

    # ---- edge crop: drop the border strips that are off-frame in the source
    # (unfillable -> vertical inpainting artifacts), keeping both eyes aligned;
    # frames are scaled back to the original resolution when written.
    if opt.get("edge_crop", True):
        cx = int(np.ceil(opt["max_disparity"] / 100.0 * w))
        cy = int(round(cx * h / w))
    else:
        cx = cy = 0
    if cx > 0:
        log(f"  edge crop: {cx}px sides, {cy}px top/bottom (zoomed back to {w}x{h})")
        lefts = [f[cy:h - cy, cx:w - cx] for f in lefts]
        right_full = right_full[:, cy:h - cy, cx:w - cx]
        holes_full = holes_full[:, cy:h - cy, cx:w - cx]
    ch, cw = right_full.shape[1:3]
    log(f"  disocclusion holes: {holes_full.mean() * 100:.2f}% of pixels")

    # ---- stage 2: inpaint at reduced resolution ----
    scale = min(1.0, opt["inpaint_width"] / cw)
    pw, ph = int(cw * scale) // 8 * 8, int(ch * scale) // 8 * 8
    frames_lo = np.stack([cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA) for f in right_full])
    masks_lo = np.stack([
        (cv2.resize(m.astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        for m in holes_full])
    inpainted_lo = inpainter(frames_lo, masks_lo, progress=progress, cancel=cancel)
    del frames_lo, masks_lo
    torch.cuda.empty_cache()

    # ---- stage 3: composite inpainted content into full-res holes ----
    out_paths = []
    sbs_writer = ana_writer = dbg_writer = None
    try:
        if opt["out_sbs"]:
            p = os.path.join(out_dir, f"{name}_sbs.mp4")
            sbs_writer = make_writer(p, w * 2, h, fps, opt)
            out_paths.append(p)
        if opt["out_anaglyph"]:
            p = os.path.join(out_dir, f"{name}_anaglyph.mp4")
            ana_writer = make_writer(p, w, h, fps, opt)
            out_paths.append(p)
        if opt["out_debug"]:
            p = os.path.join(out_dir, f"{name}_holes_debug.mp4")
            dbg_writer = make_writer(p, w, h, fps, opt)
            out_paths.append(p)

        kernel = np.ones((3, 3), np.uint8)
        for i in range(n):
            check_cancel(cancel)
            inp_up = cv2.resize(inpainted_lo[i], (cw, ch), interpolation=cv2.INTER_CUBIC)
            m = cv2.dilate(holes_full[i], kernel, iterations=2).astype(np.float32)
            m = cv2.GaussianBlur(m, (7, 7), 0)[..., None]
            right = (right_full[i].astype(np.float32) * (1 - m) + inp_up.astype(np.float32) * m)
            right = right.clip(0, 255).astype(np.uint8)

            left = lefts[i]
            if cx > 0:  # zoom both eyes back to the original resolution
                right = cv2.resize(right, (w, h), interpolation=cv2.INTER_CUBIC)
                left = cv2.resize(left, (w, h), interpolation=cv2.INTER_CUBIC)

            if sbs_writer is not None:
                sbs_writer.write(cv2.cvtColor(np.concatenate([left, right], axis=1), cv2.COLOR_RGB2BGR))
            if ana_writer is not None:
                ana = np.stack([left[..., 0], right[..., 1], right[..., 2]], axis=-1)
                ana_writer.write(cv2.cvtColor(ana, cv2.COLOR_RGB2BGR))
            if dbg_writer is not None:
                dbg = right_full[i].copy()
                dbg[holes_full[i] > 0] = (0, 255, 0)
                if cx > 0:
                    dbg = cv2.resize(dbg, (w, h), interpolation=cv2.INTER_CUBIC)
                dbg_writer.write(cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))
            progress("composite", (i + 1) / n)
    except BaseException:
        for wr in (sbs_writer, ana_writer, dbg_writer):
            if wr is not None:
                wr.release()
        for p in out_paths:  # remove partial outputs
            try:
                os.remove(p)
            except OSError:
                pass
        raise
    for wr in (sbs_writer, ana_writer, dbg_writer):
        if wr is not None:
            wr.release()
    for p in out_paths:
        log(f"  wrote {p}")


# ----------------------------------------------------------------------------
# input pairing
# ----------------------------------------------------------------------------
def norm_depth_stem(stem):
    s = stem.lower()
    return s[:-6] if s.endswith("_depth") else s


def find_pairs(left_dir, depth_dir):
    """Match videos by filename stem; a '_depth' suffix on depth files is optional.
    Returns (pairs [(name, left_path, depth_path)], unmatched_lefts, unmatched_depths)."""
    lefts, depths = {}, {}
    for d, store in ((left_dir, lefts), (depth_dir, depths)):
        for fn in sorted(os.listdir(d)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() in VIDEO_EXTS:
                key = stem.lower() if store is lefts else norm_depth_stem(stem)
                store.setdefault(key, os.path.join(d, fn))
    pairs, unmatched_l = [], []
    for key, lp in lefts.items():
        if key in depths:
            pairs.append((os.path.splitext(os.path.basename(lp))[0], lp, depths[key]))
        else:
            unmatched_l.append(os.path.basename(lp))
    unmatched_d = [os.path.basename(p) for k, p in depths.items() if k not in lefts]
    pairs.sort()
    return pairs, unmatched_l, unmatched_d


def expected_outputs(out_dir, name, opt):
    paths = []
    if opt["out_sbs"]:
        paths.append(os.path.join(out_dir, f"{name}_sbs.mp4"))
    if opt["out_anaglyph"]:
        paths.append(os.path.join(out_dir, f"{name}_anaglyph.mp4"))
    return paths


# ----------------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------------
def run_job(job, ui_queue, cancel):
    def log(msg):
        ui_queue.put(("log", msg))

    def status(msg):
        ui_queue.put(("status", msg))

    opt = job["opt"]
    clips = job["clips"]
    out_dir = job["output_dir"]
    done_count = err_count = skip_count = 0
    try:
        load_backend(log)
        os.makedirs(out_dir, exist_ok=True)
        if opt["video_codec"] in FFMPEG_CODECS:
            if find_ffmpeg():
                log(f"Encoder: {FFMPEG_CODECS[opt['video_codec']]} "
                    f"(CRF {opt['video_crf']}, preset {opt['video_preset']})")
            else:
                log("WARNING: ffmpeg not found — falling back to basic MP4V encoding. "
                    "Install ffmpeg (or 'pip install imageio-ffmpeg') for quality control.")
                opt["video_codec"] = "mp4v"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        status("Loading models...")
        inpainter = ProPainterInpainter(
            device, use_half=not opt["fp32"],
            ref_stride=opt["ref_stride"], neighbor_length=opt["neighbor_length"],
            subvideo_length=opt["subvideo_length"], log=log)

        total = len(clips)
        for ci, (name, left_path, depth_path) in enumerate(clips):
            check_cancel(cancel)
            if opt["skip_existing"]:
                outs = expected_outputs(out_dir, name, opt)
                if outs and all(os.path.exists(p) for p in outs):
                    log(f"[{ci + 1}/{total}] {name}: output exists, skipped")
                    skip_count += 1
                    ui_queue.put(("overall", (ci + 1) / total))
                    continue

            log(f"[{ci + 1}/{total}] {name}")
            t0 = time.time()

            def progress(stage, frac, _ci=ci):
                base, weight, label = STAGE_BASE[stage]
                clip_frac = base + weight * min(1.0, frac)
                ui_queue.put(("stage", label, frac))
                ui_queue.put(("overall", (_ci + clip_frac) / total))
                status(f"[{_ci + 1}/{total}] {name} — {label}")

            try:
                process_clip(left_path, depth_path, out_dir, name, opt,
                             inpainter, progress, cancel, log)
                log(f"  done in {time.time() - t0:.1f}s")
                done_count += 1
            except Cancelled:
                raise
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                err_count += 1
                log(f"  ERROR: CUDA out of memory on {name}. "
                    f"Try a smaller inpaint width (current: {opt['inpaint_width']}).")
            except Exception as e:
                err_count += 1
                log(f"  ERROR on {name}: {e}")
                log("  " + traceback.format_exc(limit=3).strip().replace("\n", "\n  "))
            ui_queue.put(("overall", (ci + 1) / total))

        summary = f"Finished: {done_count} converted, {skip_count} skipped, {err_count} failed."
        log(summary)
        ui_queue.put(("done", summary, err_count))
    except Cancelled:
        log("Cancelled by user.")
        ui_queue.put(("cancelled",))
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc(limit=5))
        ui_queue.put(("fatal", str(e)))


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------
class Tooltip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        if self.tip:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0",
                 relief="solid", borderwidth=1, wraplength=360, padx=6, pady=4).pack()

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ImmersePro — Depth to Stereo 3D")
        self.minsize(760, 700)
        self.ui_queue = queue.Queue()
        self.cancel_event = None
        self.worker = None

        s = self._load_settings()
        self.vars = {
            "mode": tk.StringVar(value=s["mode"]),
            "left_dir": tk.StringVar(value=s["left_dir"]),
            "depth_dir": tk.StringVar(value=s["depth_dir"]),
            "left_file": tk.StringVar(value=s["left_file"]),
            "depth_file": tk.StringVar(value=s["depth_file"]),
            "output_dir": tk.StringVar(value=s["output_dir"]),
            "max_disparity": tk.DoubleVar(value=s["max_disparity"]),
            "convergence": tk.DoubleVar(value=s["convergence"]),
            "splat_alpha": tk.DoubleVar(value=s["splat_alpha"]),
            "disp_blur": tk.IntVar(value=s["disp_blur"]),
            "invert_depth": tk.BooleanVar(value=s["invert_depth"]),
            "edge_crop": tk.BooleanVar(value=s["edge_crop"]),
            "inpaint_width": tk.IntVar(value=s["inpaint_width"]),
            "fp32": tk.BooleanVar(value=s["fp32"]),
            "neighbor_length": tk.IntVar(value=s["neighbor_length"]),
            "ref_stride": tk.IntVar(value=s["ref_stride"]),
            "subvideo_length": tk.IntVar(value=s["subvideo_length"]),
            "out_sbs": tk.BooleanVar(value=s["out_sbs"]),
            "out_anaglyph": tk.BooleanVar(value=s["out_anaglyph"]),
            "out_debug": tk.BooleanVar(value=s["out_debug"]),
            "skip_existing": tk.BooleanVar(value=s["skip_existing"]),
            "video_codec_label": tk.StringVar(
                value=CODEC_KEYS.get(s.get("video_codec", "h264"), CODEC_KEYS["h264"])),
            "video_crf": tk.IntVar(value=s["video_crf"]),
            "video_preset": tk.StringVar(value=s["video_preset"]),
        }
        self._build_ui()
        self._on_mode_change()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_queue)

    # ---------------- settings ----------------
    def _load_settings(self):
        s = dict(DEFAULTS)
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            s.update({k: v for k, v in saved.items() if k in s})
        except (OSError, ValueError):
            pass
        return s

    def _save_settings(self):
        try:
            data = {k: v.get() for k, v in self.vars.items()}
            data["video_codec"] = CODEC_LABELS.get(data.pop("video_codec_label"), "h264")
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except (OSError, tk.TclError):
            pass

    # ---------------- layout ----------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 3}
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=10, pady=8)

        # --- input ---
        inp = ttk.LabelFrame(root, text="Input")
        inp.pack(fill="x", **pad)
        mode_row = ttk.Frame(inp)
        mode_row.grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(4, 2))
        ttk.Radiobutton(mode_row, text="Batch folders", variable=self.vars["mode"],
                        value="batch", command=self._on_mode_change).pack(side="left")
        ttk.Radiobutton(mode_row, text="Single pair", variable=self.vars["mode"],
                        value="single", command=self._on_mode_change).pack(side="left", padx=12)

        self.batch_rows = [
            self._path_row(inp, 1, "Left-eye folder:", self.vars["left_dir"], is_dir=True),
            self._path_row(inp, 2, "Depth folder:", self.vars["depth_dir"], is_dir=True),
        ]
        self.single_rows = [
            self._path_row(inp, 3, "Left-eye video:", self.vars["left_file"], is_dir=False),
            self._path_row(inp, 4, "Depth video:", self.vars["depth_file"], is_dir=False),
        ]
        self._path_row(inp, 5, "Output folder:", self.vars["output_dir"], is_dir=True)
        inp.columnconfigure(1, weight=1)

        # --- stereo settings ---
        st = ttk.LabelFrame(root, text="Stereo")
        st.pack(fill="x", **pad)
        self._spin(st, 0, 0, "Depth strength (% width):", "max_disparity", 0.1, 10.0, 0.1,
                   "Maximum parallax as a percentage of frame width.\n"
                   "2.5 is comfortable; higher = stronger 3D, more inpainting.")
        self._spin(st, 0, 2, "Convergence (0-1):", "convergence", 0.0, 1.0, 0.05,
                   "Depth value placed at screen level. Content nearer pops out,\n"
                   "farther recedes. Higher values push the scene behind the screen.")
        self._spin(st, 1, 0, "Edge sharpness (alpha):", "splat_alpha", 5.0, 200.0, 5.0,
                   "Softmax splatting depth sharpness. Higher = harder occlusion\n"
                   "edges; lower = softer blending (better for hair/smoke).")
        self._spin(st, 1, 2, "Depth smoothing (px):", "disp_blur", 0, 51, 2,
                   "Blur applied to the depth map before warping; reduces halos\n"
                   "at depth edges. 0 disables.")
        cb = ttk.Checkbutton(st, text="Invert depth (black = near)", variable=self.vars["invert_depth"])
        cb.grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=2)
        Tooltip(cb, "Enable if your depth videos use black for near objects instead of white.")
        cb = ttk.Checkbutton(st, text="Crop edges (hide border artifacts)", variable=self.vars["edge_crop"])
        cb.grid(row=2, column=2, columnspan=2, sticky="w", padx=6, pady=2)
        Tooltip(cb, "Crops both eyes by the depth-strength amount and zooms back to full\n"
                    "resolution. Removes the vertical inpainting artifact at the frame\n"
                    "edges, where revealed content is off-frame and can't be recovered.")

        # --- inpainting settings ---
        ip = ttk.LabelFrame(root, text="Inpainting (ProPainter)")
        ip.pack(fill="x", **pad)
        ttk.Label(ip, text="Internal resolution (width):").grid(row=0, column=0, sticky="e", padx=6)
        combo = ttk.Combobox(ip, textvariable=self.vars["inpaint_width"], width=8, state="readonly",
                             values=[640, 768, 960, 1280, 1536])
        combo.grid(row=0, column=1, sticky="w", padx=4)
        Tooltip(combo, "Resolution the inpainting runs at (holes are composited back at full\n"
                       "resolution). Lower this if you hit CUDA out-of-memory errors.")
        cb = ttk.Checkbutton(ip, text="FP32 (slower, more VRAM)", variable=self.vars["fp32"])
        cb.grid(row=0, column=2, sticky="w", padx=14)
        self._spin(ip, 1, 0, "Neighbor frames:", "neighbor_length", 4, 30, 2,
                   "Local temporal window ProPainter propagates from.")
        self._spin(ip, 1, 2, "Reference stride:", "ref_stride", 4, 30, 2,
                   "Spacing of long-range reference frames.")
        self._spin(ip, 2, 0, "Subvideo length:", "subvideo_length", 20, 200, 10,
                   "Chunk size for long videos; lower to reduce VRAM use.")

        # --- outputs ---
        out = ttk.LabelFrame(root, text="Outputs")
        out.pack(fill="x", **pad)
        for col, (key, label) in enumerate([("out_sbs", "Side-by-side"),
                                            ("out_anaglyph", "Anaglyph (red/cyan)"),
                                            ("out_debug", "Hole debug video"),
                                            ("skip_existing", "Skip existing outputs")]):
            ttk.Checkbutton(out, text=label, variable=self.vars[key]).grid(
                row=0, column=col, sticky="w", padx=8, pady=3)

        ttk.Label(out, text="Codec:").grid(row=1, column=0, sticky="e", padx=6, pady=(2, 5))
        codec_combo = ttk.Combobox(out, textvariable=self.vars["video_codec_label"], width=26,
                                   state="readonly", values=list(CODEC_LABELS))
        codec_combo.grid(row=1, column=1, sticky="w", padx=4, pady=(2, 5))
        Tooltip(codec_combo, "H.264/H.265 use ffmpeg for high-quality encoding.\n"
                             "MP4V is a basic fallback with no quality control.")
        ttk.Label(out, text="Quality (CRF):").grid(row=1, column=2, sticky="e", padx=6, pady=(2, 5))
        crf_spin = ttk.Spinbox(out, textvariable=self.vars["video_crf"], from_=0, to=51,
                               increment=1, width=6)
        crf_spin.grid(row=1, column=3, sticky="w", padx=4, pady=(2, 5))
        Tooltip(crf_spin, "Constant Rate Factor: lower = better quality, larger files.\n"
                          "18 is visually lossless for H.264; 23 is a good size/quality\n"
                          "balance (for H.265 subtract ~5). Ignored by MP4V.")
        ttk.Label(out, text="Encode speed:").grid(row=2, column=0, sticky="e", padx=6, pady=(0, 5))
        preset_combo = ttk.Combobox(out, textvariable=self.vars["video_preset"], width=12,
                                    state="readonly", values=PRESETS)
        preset_combo.grid(row=2, column=1, sticky="w", padx=4, pady=(0, 5))
        Tooltip(preset_combo, "ffmpeg preset: slower = better compression at the same quality.\n"
                              "'medium' is a good default. Ignored by MP4V.")

        # --- run controls ---
        run = ttk.Frame(root)
        run.pack(fill="x", **pad)
        self.start_btn = ttk.Button(run, text="Start", command=self._start)
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(run, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=8)
        self.open_btn = ttk.Button(run, text="Open output folder", command=self._open_output)
        self.open_btn.pack(side="left", padx=8)

        prog = ttk.Frame(root)
        prog.pack(fill="x", **pad)
        ttk.Label(prog, text="Overall:").grid(row=0, column=0, sticky="w")
        self.overall_bar = ttk.Progressbar(prog, maximum=1.0)
        self.overall_bar.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(prog, text="Stage:").grid(row=1, column=0, sticky="w")
        self.stage_bar = ttk.Progressbar(prog, maximum=1.0)
        self.stage_bar.grid(row=1, column=1, sticky="ew", padx=6)
        prog.columnconfigure(1, weight=1)
        self.status_lbl = ttk.Label(root, text="Ready.")
        self.status_lbl.pack(fill="x", padx=10)

        self.log_box = scrolledtext.ScrolledText(root, height=12, state="disabled", wrap="word",
                                                 font=("Consolas", 9))
        self.log_box.pack(fill="both", expand=True, padx=8, pady=(4, 6))

        self._toggle_widgets = [self.start_btn, combo]

    def _path_row(self, parent, row, label, var, is_dir):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="e", padx=6, pady=2)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", pady=2)

        def browse():
            if is_dir:
                p = filedialog.askdirectory(initialdir=var.get() or APP_DIR)
            else:
                p = filedialog.askopenfilename(
                    initialdir=os.path.dirname(var.get()) or APP_DIR,
                    filetypes=[("Videos", " ".join("*" + e for e in VIDEO_EXTS)), ("All files", "*.*")])
            if p:
                var.set(p)
        btn = ttk.Button(parent, text="Browse...", command=browse, width=10)
        btn.grid(row=row, column=2, padx=6, pady=2)
        return (lbl, entry, btn)

    def _spin(self, parent, row, col, label, key, lo, hi, inc, tip):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=col, sticky="e", padx=6, pady=2)
        sp = ttk.Spinbox(parent, textvariable=self.vars[key], from_=lo, to=hi, increment=inc, width=8)
        sp.grid(row=row, column=col + 1, sticky="w", padx=4, pady=2)
        Tooltip(lbl, tip)
        Tooltip(sp, tip)

    def _on_mode_change(self):
        batch = self.vars["mode"].get() == "batch"
        for widgets in self.batch_rows:
            for w in widgets:
                w.grid() if batch else w.grid_remove()
        for widgets in self.single_rows:
            for w in widgets:
                w.grid_remove() if batch else w.grid()

    # ---------------- actions ----------------
    def _log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", time.strftime("[%H:%M:%S] ") + msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _collect_options(self):
        try:
            opt = {
                "max_disparity": float(self.vars["max_disparity"].get()),
                "convergence": float(self.vars["convergence"].get()),
                "splat_alpha": float(self.vars["splat_alpha"].get()),
                "disp_blur": int(self.vars["disp_blur"].get()),
                "invert_depth": bool(self.vars["invert_depth"].get()),
                "edge_crop": bool(self.vars["edge_crop"].get()),
                "inpaint_width": int(self.vars["inpaint_width"].get()),
                "fp32": bool(self.vars["fp32"].get()),
                "neighbor_length": max(4, int(self.vars["neighbor_length"].get())),
                "ref_stride": max(1, int(self.vars["ref_stride"].get())),
                "subvideo_length": max(20, int(self.vars["subvideo_length"].get())),
                "out_sbs": bool(self.vars["out_sbs"].get()),
                "out_anaglyph": bool(self.vars["out_anaglyph"].get()),
                "out_debug": bool(self.vars["out_debug"].get()),
                "skip_existing": bool(self.vars["skip_existing"].get()),
                "video_codec": CODEC_LABELS.get(self.vars["video_codec_label"].get(), "h264"),
                "video_crf": min(51, max(0, int(self.vars["video_crf"].get()))),
                "video_preset": self.vars["video_preset"].get() or "medium",
            }
        except (tk.TclError, ValueError) as e:
            raise ValueError(f"Invalid setting value: {e}")
        if not 0.0 <= opt["convergence"] <= 1.0:
            raise ValueError("Convergence must be between 0 and 1.")
        if not (opt["out_sbs"] or opt["out_anaglyph"] or opt["out_debug"]):
            raise ValueError("Select at least one output type.")
        return opt

    def _collect_clips(self):
        if self.vars["mode"].get() == "batch":
            left_dir = self.vars["left_dir"].get().strip()
            depth_dir = self.vars["depth_dir"].get().strip()
            if not os.path.isdir(left_dir):
                raise ValueError(f"Left-eye folder not found: {left_dir}")
            if not os.path.isdir(depth_dir):
                raise ValueError(f"Depth folder not found: {depth_dir}")
            pairs, un_l, un_d = find_pairs(left_dir, depth_dir)
            if un_l:
                self._log(f"No depth match for {len(un_l)} video(s): " + ", ".join(un_l[:5]) +
                          (" ..." if len(un_l) > 5 else ""))
            if un_d:
                self._log(f"No color match for {len(un_d)} depth video(s): " + ", ".join(un_d[:5]) +
                          (" ..." if len(un_d) > 5 else ""))
            if not pairs:
                raise ValueError("No matching video pairs found. Videos are matched by filename; "
                                 "'clip.mp4' pairs with 'clip.mp4' or 'clip_depth.mp4'.")
            return pairs
        else:
            lf = self.vars["left_file"].get().strip()
            df = self.vars["depth_file"].get().strip()
            if not os.path.isfile(lf):
                raise ValueError(f"Left-eye video not found: {lf}")
            if not os.path.isfile(df):
                raise ValueError(f"Depth video not found: {df}")
            name = os.path.splitext(os.path.basename(lf))[0]
            return [(name, lf, df)]

    def _start(self):
        try:
            opt = self._collect_options()
            clips = self._collect_clips()
            out_dir = self.vars["output_dir"].get().strip()
            if not out_dir:
                raise ValueError("Please choose an output folder.")
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return

        self._save_settings()
        self._log(f"Starting: {len(clips)} clip(s) -> {out_dir}")
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.overall_bar["value"] = 0
        self.stage_bar["value"] = 0

        self.cancel_event = threading.Event()
        job = {"opt": opt, "clips": clips, "output_dir": out_dir}
        self.worker = threading.Thread(target=run_job, args=(job, self.ui_queue, self.cancel_event),
                                       daemon=True)
        self.worker.start()

    def _cancel(self):
        if self.cancel_event is not None:
            self.cancel_event.set()
            self.status_lbl.configure(text="Cancelling (finishing current step)...")
            self.cancel_btn.configure(state="disabled")

    def _open_output(self):
        out_dir = self.vars["output_dir"].get().strip()
        if os.path.isdir(out_dir):
            os.startfile(os.path.abspath(out_dir))
        else:
            messagebox.showinfo("Not found", f"Output folder does not exist yet:\n{out_dir}")

    def _finish(self, status_text):
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.status_lbl.configure(text=status_text)
        self.cancel_event = None
        self.worker = None

    def _poll_queue(self):
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log(msg[1])
                elif kind == "status":
                    self.status_lbl.configure(text=msg[1])
                elif kind == "stage":
                    self.stage_bar["value"] = msg[2]
                elif kind == "overall":
                    self.overall_bar["value"] = msg[1]
                elif kind == "done":
                    self.overall_bar["value"] = 1.0
                    self.stage_bar["value"] = 0
                    self._finish(msg[1])
                    if msg[2] == 0:
                        messagebox.showinfo("Finished", msg[1])
                    else:
                        messagebox.showwarning("Finished with errors",
                                               msg[1] + "\nSee the log for details.")
                elif kind == "cancelled":
                    self.stage_bar["value"] = 0
                    self._finish("Cancelled.")
                elif kind == "fatal":
                    self._finish("Failed.")
                    messagebox.showerror("Error", msg[1] + "\nSee the log for details.")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _on_close(self):
        if self.worker is not None and self.worker.is_alive():
            if not messagebox.askyesno("Quit", "Processing is running. Cancel and quit?"):
                return
            if self.cancel_event is not None:
                self.cancel_event.set()
        self._save_settings()
        self.destroy()


def main():
    try:  # crisp text on Windows high-DPI displays
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
