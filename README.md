# Minimal Thermal Adaptation for Frozen Nano-Scale RGB Detectors

Code for the paper. A small trainable thermal branch and fusion module are hooked into a frozen,
RGB-trained YOLOv8n or NanoDet-Plus detector.

## Data
- FLIR-aligned (Zhang et al. 2020) and M3FD (Liu et al. 2022). Not included; download them separately.
- Run `data_prep/` first. The fixed splits used in the paper are in `splits/`.
- Set `TF_PROJ` (this folder) and `TF_DATA` (datasets folder), or use the defaults in `lb_fusion.py`.

## What produces what
| Folder | Produces |
|---|---|
| `yolo/` | YOLOv8n Config A, all fusion runs, ablations, unfrozen, data fractions, thermal-only, B_robust |
| `nanodet/` | NanoDet-Plus Config A and fusion runs (trained on Kaggle) and their evaluation |
| `analysis/` | robustness, brightness / object-size / gate analysis, qualitative figures |
| `jetson/` | ONNX export, TensorRT build, benchmark and on-device accuracy (Jetson Orin Nano) |
| `results/` | the result JSONs behind every number in the paper |

Checkpoints are available as release assets.
