# ImmersePro GUI — 2D to Stereo 3D Video Converter

Convert 2D video into stereo 3D (side-by-side / anaglyph) using **pre-rendered
depth-map videos**, with learned, temporally coherent inpainting of the
disoccluded regions. Runs locally on your GPU with a simple desktop GUI.

Built on the shoulders of [ImmersePro](https://github.com/shijianjian/ImmersePro),
[ProPainter](https://github.com/sczhou/ProPainter), and
[RAFT](https://github.com/princeton-vl/RAFT) — see [Credits](#credits--licenses).

## How it works

For each clip, the app takes a color video and a matching grayscale depth video
(white = near) and:

1. **Depth → disparity** — the depth map is converted to per-pixel horizontal
   parallax (configurable strength and convergence plane).
2. **Softmax splatting** — each left-eye frame is forward-warped into the
   right-eye view at full resolution using bilinear, depth-weighted splatting
   (soft z-buffer), so near content correctly occludes far content and edges
   stay sub-pixel accurate. Revealed background leaves genuine holes.
3. **Edge crop** — border strips whose content is off-frame in the source (and
   therefore unfillable) are cropped from both eyes and the frame is zoomed
   back, eliminating vertical edge artifacts.
4. **Video inpainting** — the remaining holes are filled with
   [ProPainter](https://github.com/sczhou/ProPainter) (RAFT optical flow →
   flow completion → pixel propagation → transformer), which recovers real
   background pixels from neighboring frames where possible and hallucinates
   only what was never visible.
5. **Output** — inpainted content is composited back into the full-resolution
   warp and written as side-by-side and/or red-cyan anaglyph video (H.264 /
   H.265 via ffmpeg, with CRF quality control).

## Installation

```bash
git clone <this repo>
cd ImmerseProGUI
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt   # install torch with the CUDA build matching your system
```

- A CUDA GPU is strongly recommended (tested on an RTX 5080; ~40 s for a
  4-second 1080p clip).
- [ffmpeg](https://ffmpeg.org) on your PATH enables H.264/H.265 output
  (`pip install imageio-ffmpeg` works too); without it the app falls back to
  basic MP4V.
- Model weights (~200 MB: RAFT, flow completion, ProPainter) are downloaded
  automatically to `weights/` on first run.

## Usage

```bash
python app.py
```

**Batch mode (default):** put your videos in

```
videos/left_eye/clip001.mp4        color videos
videos/depth/clip001_depth.mp4     matching depth videos (the _depth suffix is optional)
```

Pairs are matched by filename. Outputs go to `videos/output/`. **Single pair
mode** lets you pick two files directly. Already-converted clips are skipped
(toggleable), so batches can be resumed or extended.

Depth videos can be any resolution (they are upscaled to match) and should be
grayscale with **white = near**; use *Invert depth* if yours are reversed.

### Key settings

| Setting | Meaning |
|---|---|
| Depth strength | Maximum parallax as % of frame width (2.5 ≈ comfortable) |
| Convergence | Depth value placed at screen level; nearer pops out, farther recedes |
| Edge sharpness (alpha) | Occlusion hardness of the splat; lower is kinder to hair/smoke |
| Depth smoothing | Blur on the depth map to reduce halos at depth edges |
| Crop edges | Removes unfillable frame-border strips (recommended: on) |
| Internal resolution | Width the inpainting runs at; lower it on CUDA out-of-memory |
| Quality (CRF) | 18 ≈ visually lossless H.264; higher = smaller files |

## Legacy: original ImmersePro inference

This repository started as a fork of
[ImmersePro](https://github.com/shijianjian/ImmersePro) (end-to-end stereo
synthesis via implicit disparity learning, [arXiv:2410.00262](https://arxiv.org/abs/2410.00262)).
The original inference entry point is preserved:

```bash
python inference_video.py -c configs/inference.json      # MiDaS-based model
python inference_video.py -c configs/inference_da.json   # DepthAnything-based model
```

using the [published checkpoints](https://huggingface.co/shijianjian/ImmersePro)
in `experiments_model/`. In our testing the released checkpoints did not
reproduce the paper's demo quality (the DepthAnything variant's disparity
branch produces unstructured output), which motivated the depth-map-driven
pipeline above. The training code (`core/trainer*.py`, `scripts/`) is retained
and functional now that the `RAFT/` package is vendored.

## Credits & licenses

This project stands on excellent prior work:

- **[ImmersePro](https://github.com/shijianjian/ImmersePro)** — Jian Shi,
  Zhenyu Li, Peter Wonka (KAUST). Original codebase, model architectures, and
  the ImmersePro checkpoints.
- **[ProPainter](https://github.com/sczhou/ProPainter)** — Shangchen Zhou,
  Chongyi Li, Kelvin C.K. Chan, Chen Change Loy (S-Lab, NTU; ICCV 2023).
  Video inpainting models and code (vendored in `propainter/`).
  **ProPainter code and weights are released under the NTU S-Lab License 1.0,
  which permits non-commercial use only.**
- **[RAFT](https://github.com/princeton-vl/RAFT)** — Zachary Teed, Jia Deng
  (Princeton; ECCV 2020). Optical flow estimation (vendored in `RAFT/`,
  BSD-3-Clause).
- **Softmax splatting** — Simon Niklaus, Feng Liu (CVPR 2020). The
  depth-weighted forward-warping formulation used in the warp stage.

```bibtex
@article{shi2024immersepro,
  title={ImmersePro: End-to-End Stereo Video Synthesis Via Implicit Disparity Learning},
  author={Shi, Jian and Li, Zhenyu and Wonka, Peter},
  journal={arXiv preprint arXiv:2410.00262},
  year={2024}
}
@inproceedings{zhou2023propainter,
  title={{ProPainter}: Improving Propagation and Transformer for Video Inpainting},
  author={Zhou, Shangchen and Li, Chongyi and Chan, Kelvin C.K and Loy, Chen Change},
  booktitle={Proceedings of IEEE International Conference on Computer Vision (ICCV)},
  year={2023}
}
@inproceedings{teed2020raft,
  title={{RAFT}: Recurrent All-Pairs Field Transforms for Optical Flow},
  author={Teed, Zachary and Deng, Jia},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
@inproceedings{niklaus2020softmax,
  title={Softmax Splatting for Video Frame Interpolation},
  author={Niklaus, Simon and Liu, Feng},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

Please respect the upstream licenses — in particular, the ProPainter weights
restrict this pipeline's inpainting stage to **non-commercial** use.
