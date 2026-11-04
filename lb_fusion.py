"""Letterboxed RGB-thermal fusion training for frozen YOLOv8n (journal runs).

Lives in a .py file (not a notebook) so DataLoader worker processes can import
the dataset on Windows. Importing this file does nothing on its own; the queue
notebook calls train_and_eval(run) for each run.
"""
import gc
import json
import os
import random
import shutil
import sys
import time
import traceback
from contextlib import redirect_stdout
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import box_iou, ap_per_class
from ultralytics.utils.ops import xywh2xyxy

PROJ = Path(os.environ.get("TF_PROJ", Path(__file__).resolve().parent))   # project folder
JOURNAL = PROJ / "journal"
DATA = Path(os.environ.get("TF_DATA", PROJ.parent / "Datasets"))           # datasets folder
IMG_SIZE = 640

DATASETS = {
    "flir": {
        "root": DATA / "flir_aligned_yolo",
        "thermal": {"train": DATA / "flir_align" / "thermal" / "train",
                    "val":   DATA / "flir_align" / "thermal" / "test"},
        "ext": ".jpeg",
        "config_a": PROJ / "results" / "config_a_rgb_baseline" / "weights" / "best.pt",
        "names": {0: "car", 1: "person", 2: "bicycle"},
    },
    "m3fd": {
        "root": DATA / "m3fd_yolo",
        "thermal": {"train": DATA / "M3FD_Detection" / "Ir",
                    "val":   DATA / "M3FD_Detection" / "Ir"},
        "ext": ".png",
        "config_a": PROJ / "results" / "config_a_rgb_baseline_m3fd-2" / "weights" / "best.pt",
        "names": {0: "Bus", 1: "Car", 2: "Lamp", 3: "Motorcycle", 4: "People", 5: "Truck"},
    },
}

EXPECTED_PARAMS = {"B": 25704, "B_nothermal": 25704, "B_rgbgray": 25704, "B_unfrozen": 25704, "B_robust": 25704,
                   "B_large": 58248, "B_concat": 31776}
BASE_LR_FACTOR = 0.1   # B_unfrozen: the detector weights train at lr * 0.1, thermal branch + gate at lr


# ----------------------------------------------------------------------------- data
def letterbox(img, size, pad_value):
    """Same geometry as ultralytics LetterBox (centred padding, value 114)."""
    h0, w0 = img.shape[:2]
    r = min(size / h0, size / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    if (new_w, new_h) != (w0, h0):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)
    return img, new_w / w0, new_h / h0, left, top


def list_ids(ds, split):
    return sorted(p.stem for p in (ds["root"] / "images" / split).glob(f"*{ds['ext']}"))


class PairedLetterboxDataset(Dataset):
    def __init__(self, ds, split, image_ids, img_size=IMG_SIZE, perturb=None, train_aug=False):
        self.rgb_dir = ds["root"] / "images" / split
        self.label_dir = ds["root"] / "labels" / split
        self.thermal_dir = ds["thermal"][split]
        self.ext, self.size = ds["ext"], img_size
        self.image_ids = sorted(image_ids)
        self.perturb = perturb   # None, or a PERTURBATIONS spec applied to the thermal image only
        self.train_aug = train_aug   # B_robust: random thermal dropout / shift / noise (training only)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        rgb = cv2.cvtColor(cv2.imread(str(self.rgb_dir / f"{img_id}{self.ext}")), cv2.COLOR_BGR2RGB)
        h0, w0 = rgb.shape[:2]
        thermal_path = self.thermal_dir / f"{img_id}{self.ext}"
        thermal = cv2.imread(str(thermal_path), cv2.IMREAD_GRAYSCALE)
        if thermal is None:
            raise FileNotFoundError(f"Missing thermal image: {thermal_path}")
        if thermal.shape[:2] != (h0, w0):   # pairs are registered; bring thermal to RGB size first
            thermal = cv2.resize(thermal, (w0, h0), interpolation=cv2.INTER_LINEAR)
        if self.train_aug:
            thermal = random_thermal_aug(thermal)
        if self.perturb is not None:
            thermal = perturb_thermal(thermal, self.perturb, idx)

        rgb, rx, ry, left, top = letterbox(rgb, self.size, (114, 114, 114))
        thermal, _, _, _, _ = letterbox(thermal, self.size, 114)

        S = self.size
        boxes = []
        label_path = self.label_dir / f"{img_id}.txt"
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    cls, xc, yc, w, h = map(float, line.split())
                    boxes.append([cls, (xc * w0 * rx + left) / S, (yc * h0 * ry + top) / S,
                                  w * w0 * rx / S, h * h0 * ry / S])
        return {"image_id": img_id, "rgb": rgb, "thermal": thermal, "boxes": boxes,
                "meta": {"rx": rx, "ry": ry, "left": left, "top": top, "w0": w0, "h0": h0}}


# test-time thermal degradations (applied to the thermal image at original resolution, before letterbox)
PERTURBATIONS = {
    "clean":          None,
    "missing":        ("zero",),               # sensor dead: all-zero thermal frame
    "noise_0.05":     ("noise", 0.05),         # additive Gaussian noise, std as a fraction of the 0-255 range
    "noise_0.10":     ("noise", 0.10),
    "noise_0.20":     ("noise", 0.20),
    "shift_1pct":     ("shift", 0.01),         # misregistration: thermal shifted right+down by a fraction of width
    "shift_2.5pct":   ("shift", 0.025),
    "shift_5pct":     ("shift", 0.05),
    "lowres_160x120": ("lowres", 160, 120),    # cheap sensor (Lepton 3.5-class), upsampled back
    "lowres_80x60":   ("lowres", 80, 60),      # Lepton 2-class
}


def perturb_thermal(th, spec, idx):
    kind = spec[0]
    if kind == "zero":
        return np.zeros_like(th)
    if kind == "noise":
        rng = np.random.default_rng(1000 + idx)          # same noise for an image across all models
        return np.clip(th.astype(np.float32) + rng.normal(0, spec[1] * 255, th.shape), 0, 255).astype(np.uint8)
    if kind == "shift":
        h, w = th.shape[:2]
        d = int(round(spec[1] * w))
        if d == 0:
            return th
        out = np.zeros_like(th)
        out[d:, d:] = th[:h - d, :w - d]
        return out
    if kind == "lowres":
        h, w = th.shape[:2]
        small = cv2.resize(th, (spec[1], spec[2]), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(spec)


# B_robust training-time thermal augmentation. Deliberately covers only part of the test perturbations:
# low resolution is never seen in training, and the strongest noise (0.20) and shift (5%) are outside
# the training ranges, so those test cases measure generalisation rather than memorised corruptions.
ROBUST_AUG = {"p_drop": 0.2, "p_shift": 0.3, "shift_max": 0.025, "p_noise": 0.3, "noise_max": 0.10}


def _shift_image(th, dx, dy):
    h, w = th.shape[:2]
    out = np.zeros_like(th)
    xs, xd = (0, dx) if dx >= 0 else (-dx, 0)
    ys, yd = (0, dy) if dy >= 0 else (-dy, 0)
    out[yd:h - ys, xd:w - xs] = th[ys:h - yd, xs:w - xd]
    return out


def random_thermal_aug(th):
    """Uses Python's `random`, which PyTorch re-seeds in every DataLoader worker (numpy's global RNG is
    not, so it would repeat the same augmentations across workers)."""
    a = ROBUST_AUG
    if random.random() < a["p_drop"]:
        return np.zeros_like(th)
    if random.random() < a["p_shift"]:
        w = th.shape[1]
        dx = int(round(random.uniform(-a["shift_max"], a["shift_max"]) * w))
        dy = int(round(random.uniform(-a["shift_max"], a["shift_max"]) * w))
        th = _shift_image(th, dx, dy)
    if random.random() < a["p_noise"]:
        sigma = random.uniform(0.0, a["noise_max"]) * 255
        rng = np.random.default_rng(random.getrandbits(32))
        th = np.clip(th.astype(np.float32) + rng.normal(0, sigma, th.shape), 0, 255).astype(np.uint8)
    return th


def collate_fn(batch):
    rgb_list, thermal_list, cls_list, bbox_list, batch_idx_list = [], [], [], [], []
    for i, item in enumerate(batch):
        rgb_list.append(torch.from_numpy(item["rgb"]).permute(2, 0, 1).float() / 255.0)
        thermal_list.append(torch.from_numpy(item["thermal"]).unsqueeze(0).float() / 255.0)
        for cls_id, xc, yc, w, h in item["boxes"]:
            cls_list.append(cls_id); bbox_list.append([xc, yc, w, h]); batch_idx_list.append(i)
    return (torch.stack(rgb_list), torch.stack(thermal_list), {
        "batch_idx": torch.tensor(batch_idx_list, dtype=torch.float32),
        "cls": torch.tensor(cls_list, dtype=torch.float32),
        "bboxes": torch.tensor(bbox_list, dtype=torch.float32) if bbox_list else torch.zeros((0, 4)),
    })


def get_splits(dataset_name, smoke):
    """Fixed 10% holdout of TRAIN (seed 42, created once, reused by every run) + eval sets."""
    ds = DATASETS[dataset_name]
    split_dir = JOURNAL / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    all_train_ids = list_ids(ds, "train")
    holdout_path = split_dir / f"{dataset_name}_train_holdout.json"
    if holdout_path.exists():
        holdout = set(json.loads(holdout_path.read_text()))
        print(f"loaded holdout split: {holdout_path}")
    else:
        rng = random.Random(42)
        holdout = set(rng.sample(all_train_ids, round(0.1 * len(all_train_ids))))
        holdout_path.write_text(json.dumps(sorted(holdout)))
        print(f"created holdout split: {holdout_path}")
    fit_ids = sorted(set(all_train_ids) - holdout)
    holdout_ids = sorted(holdout)

    val_ids = list_ids(ds, "val")
    if dataset_name == "flir":
        day_night = json.loads((ds["root"] / "val_day_night_labels.json").read_text())
        eval_sets = {"night": [i for i in val_ids if day_night.get(i) == "night"], "full": val_ids}
    else:
        eval_sets = {"full": val_ids}

    if smoke:
        fit_ids, holdout_ids = fit_ids[:64], holdout_ids[:32]
        eval_sets = {k: v[:32] for k, v in eval_sets.items()}
    return fit_ids, holdout_ids, eval_sets


# ----------------------------------------------------------------------------- eval
IOUV = torch.linspace(0.5, 0.95, 10)


def _match_predictions(pred_classes, true_classes, iou, iouv):
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
    iou = (iou * (true_classes[:, None] == pred_classes)).cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.from_numpy(correct)


def evaluate_map(model, dataloader, device, class_names, img_size, conf_thres=0.001):
    model.eval()
    all_tp, all_conf, all_pred_cls, all_target_cls = [], [], [], []
    n_batches = len(dataloader)
    with torch.no_grad():
        for batch_num, (rgb_batch, thermal_batch, batch_dict) in enumerate(dataloader):
            if batch_num % 10 == 0:
                print(f"  batch {batch_num}/{n_batches}...")
            out = model(rgb_batch.to(device), thermal_batch.to(device))
            nms_out = non_max_suppression(out[0], conf_thres=conf_thres, iou_thres=0.6,
                                          nc=len(class_names), max_time_img=1.0)
            bidx, gcls, gbox = batch_dict["batch_idx"], batch_dict["cls"], batch_dict["bboxes"]
            for img_idx, det in enumerate(nms_out):
                m = bidx == img_idx
                gt_cls_i = gcls[m]
                gt_boxes_i = xywh2xyxy(gbox[m]) * img_size
                all_target_cls.append(gt_cls_i.cpu().numpy())
                if det.shape[0] == 0:
                    continue
                pb, pc, pcls = det[:, :4].cpu(), det[:, 4].cpu(), det[:, 5].cpu()
                if gt_cls_i.shape[0] == 0:
                    tp = np.zeros((pb.shape[0], IOUV.shape[0]), dtype=bool)
                else:
                    tp = _match_predictions(pcls, gt_cls_i.cpu(), box_iou(gt_boxes_i.cpu(), pb), IOUV).numpy()
                all_tp.append(tp); all_conf.append(pc.numpy()); all_pred_cls.append(pcls.numpy())
    print(f"  batch {n_batches}/{n_batches} done, computing metrics...")
    tp, fp, p, r, f1, ap, uc = ap_per_class(np.concatenate(all_tp), np.concatenate(all_conf),
                                            np.concatenate(all_pred_cls), np.concatenate(all_target_cls),
                                            names=class_names)[:7]
    res = {
        "precision": float(p.mean()), "recall": float(r.mean()),
        "map50": float(ap[:, 0].mean()), "map50_95": float(ap.mean()),
        "per_class_ap50": {class_names[int(c)]: float(ap[i, 0]) for i, c in enumerate(uc)},
    }
    print(f"P={res['precision']:.4f}  R={res['recall']:.4f}  mAP50={res['map50']:.4f}  mAP50-95={res['map50_95']:.4f}")
    for k, v in res["per_class_ap50"].items():
        print(f"  {k}: AP50={v:.4f}")
    return res


# ----------------------------------------------------------------------------- model
class ThermalBranch(nn.Module):
    def __init__(self, out_channels, size="small"):
        super().__init__()
        mid = 16 if size == "small" else 32
        self.net = nn.Sequential(
            nn.Conv2d(1, mid, 3, 2, 1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid * 2, 3, 2, 1), nn.BatchNorm2d(mid * 2), nn.ReLU(inplace=True),
            nn.Conv2d(mid * 2, out_channels, 3, 2, 1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, thermal):
        return self.net(thermal)


class ThermalBranchP4(nn.Module):
    def __init__(self, out_channels, size="small"):
        super().__init__()
        mid = 16 if size == "small" else 32
        self.net = nn.Sequential(
            nn.Conv2d(1, mid, 3, 2, 1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid * 2, 3, 2, 1), nn.BatchNorm2d(mid * 2), nn.ReLU(inplace=True),
            nn.Conv2d(mid * 2, mid * 2, 3, 2, 1), nn.BatchNorm2d(mid * 2), nn.ReLU(inplace=True),
            nn.Conv2d(mid * 2, out_channels, 3, 2, 1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, thermal):
        return self.net(thermal)


class ChannelAttentionFusion(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction), nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels * 2), nn.Sigmoid(),
        )

    def forward(self, rgb_feat, thermal_feat):
        b, c, h, w = rgb_feat.shape
        pooled = torch.cat([self.pool(rgb_feat).view(b, c), self.pool(thermal_feat).view(b, c)], dim=1)
        wts = self.gate(pooled)
        return rgb_feat * wts[:, :c].view(b, c, 1, 1) + thermal_feat * wts[:, c:].view(b, c, 1, 1)


class SimpleConcatFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, rgb_feat, thermal_feat):
        return self.proj(torch.cat([rgb_feat, thermal_feat], dim=1))


class FusionYOLOv8n(nn.Module):
    """Frozen Config A + one (thermal branch, fusion) pair per hooked layer."""

    def __init__(self, config_a_checkpoint, stages, thermal_source="thermal"):
        super().__init__()
        base = YOLO(config_a_checkpoint).model
        for p in base.parameters():
            p.requires_grad = False
        base.eval()
        self.base = base
        self.branches = nn.ModuleList([b for _, b, _ in stages])
        self.fusions = nn.ModuleList([f for _, _, f in stages])
        # "thermal": real thermal image | "zeros": all-zero input (B_nothermal, superseded)
        # "rgbgray": grayscale of the RGB image fed to the thermal branch (B_rgbgray control)
        assert thermal_source in ("thermal", "zeros", "rgbgray"), thermal_source
        self.thermal_source = thermal_source
        self.unfrozen = False
        self._thermal = None
        for k, (layer_idx, _, _) in enumerate(stages):
            self.base.model[layer_idx].register_forward_hook(self._make_hook(k))

    def _make_hook(self, k):
        def hook(module, inputs, output):
            return self.fusions[k](output, self.branches[k](self._thermal))
        return hook

    def unfreeze_base(self):
        """B_unfrozen upper bound: every detector weight becomes trainable. BatchNorm running
        statistics stay fixed (set_train keeps base in eval mode) because batch 8 is too small
        to re-estimate them reliably."""
        for p in self.base.parameters():
            p.requires_grad = True
        self.unfrozen = True

    def set_train(self):
        self.train()
        self.base.eval()          # keep BN statistics fixed (frozen and unfrozen alike)

    def trainable_parameters(self):
        return list(self.branches.parameters()) + list(self.fusions.parameters())

    def trainable_state(self):
        state = {"branches": self.branches.state_dict(), "fusions": self.fusions.state_dict()}
        if self.unfrozen:
            state["base"] = self.base.state_dict()
        return state

    def load_trainable(self, state):
        self.branches.load_state_dict(state["branches"])
        self.fusions.load_state_dict(state["fusions"])
        if "base" in state:
            self.base.load_state_dict(state["base"])

    def forward(self, rgb, thermal):
        if self.thermal_source == "zeros":
            thermal = torch.zeros_like(thermal)
        elif self.thermal_source == "rgbgray":   # rgb is RGB order, [0,1]
            thermal = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])
        self._thermal = thermal
        out = self.base(rgb)
        self._thermal = None
        return out


def build_model(config, config_a):
    if config in ("B", "B_nothermal", "B_rgbgray", "B_unfrozen", "B_robust"):
        stages = [(4, ThermalBranch(64), ChannelAttentionFusion(64))]
    elif config == "B_large":
        stages = [(4, ThermalBranch(64, "large"), ChannelAttentionFusion(64))]
    elif config == "B_concat":
        stages = [(4, ThermalBranch(64), SimpleConcatFusion(64))]
    elif config == "C":
        stages = [(4, ThermalBranch(64), ChannelAttentionFusion(64)),
                  (6, ThermalBranchP4(128), ChannelAttentionFusion(128))]
    else:
        raise ValueError(f"unknown config: {config}")
    source = {"B_nothermal": "zeros", "B_rgbgray": "rgbgray"}.get(config, "thermal")
    model = FusionYOLOv8n(str(config_a), stages, thermal_source=source)
    if config == "B_unfrozen":
        model.unfreeze_base()
    return model


# ----------------------------------------------------------------------------- one run
def run_name(run):
    frac = f"_frac{round(run['frac'] * 100)}" if run.get("frac") else ""
    return (f"lb_yolo_{run['dataset']}_{run['config']}{frac}_s{run['seed']}"
            + ("_SMOKETEST" if run.get("smoke") else ""))


def subsample_fit_ids(fit_ids, frac):
    """Nested subsets: one fixed shuffle (seed 42), take the first frac of it, so the 10% set is
    inside the 25% set, which is inside the 50% set. Independent of the run seed."""
    order = sorted(fit_ids)
    random.Random(42).shuffle(order)
    return sorted(order[:max(1, round(frac * len(order)))])


def run_paths(run):
    name = run_name(run)
    smoke = run.get("smoke", False)
    ckpt_dir = JOURNAL / "smoke" if smoke else PROJ / "checkpoints_lb"
    log_dir = JOURNAL / "smoke" if smoke else JOURNAL / "logs"
    res_dir = JOURNAL / "smoke" if smoke else JOURNAL / "results"
    for d in (ckpt_dir, log_dir, res_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "best": ckpt_dir / f"{name}_best.pt",
        "last": ckpt_dir / f"{name}_last.pt",
        "trainlog": log_dir / f"{name}_trainlog.json",
        "console": log_dir / f"{name}_console.txt",
        "results": res_dir / f"{name}_eval.json",
    }


class _Tee:
    """Print to the notebook and to a text file at the same time."""

    def __init__(self, path, stream):
        self.f = open(path, "w", encoding="utf-8")
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def close(self):
        self.f.close()


def _train_and_eval(run, epochs, batch, lr, num_workers, device):
    name = run_name(run)
    paths = run_paths(run)
    smoke = run.get("smoke", False)
    ds = DATASETS[run["dataset"]]

    seed = run["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    loader_gen = torch.Generator(); loader_gen.manual_seed(seed)

    print(f"RUN: {name}   device: {device}   num_workers: {num_workers}")
    fit_ids, holdout_ids, eval_sets = get_splits(run["dataset"], smoke)
    if run.get("frac"):
        fit_ids = subsample_fit_ids(fit_ids, run["frac"])
        print(f"data fraction {run['frac']}: training on {len(fit_ids)} pairs")
    print(f"fit: {len(fit_ids)}  holdout: {len(holdout_ids)}  eval: " +
          ", ".join(f"{k}={len(v)}" for k, v in eval_sets.items()))

    model = build_model(run["config"], ds["config_a"]).to(device)
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    exp = EXPECTED_PARAMS.get(run["config"])
    print(f"trainable params: {n_trainable}" + (f"  (expected {exp})" if exp else ""))
    if exp is not None and n_trainable != exp:
        raise RuntimeError(f"trainable params {n_trainable} != expected {exp}")

    model.base.args = copy(DEFAULT_CFG)
    criterion = v8DetectionLoss(model.base)
    if model.unfrozen:
        n_base = sum(p.numel() for p in model.base.parameters())
        print(f"UNFROZEN: also training all {n_base:,} detector weights at lr {lr * BASE_LR_FACTOR:g}")
        optimizer = torch.optim.AdamW([
            {"params": model.trainable_parameters(), "lr": lr},
            {"params": list(model.base.parameters()), "lr": lr * BASE_LR_FACTOR},
        ])
    else:
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr)

    loader_kw = dict(batch_size=batch, collate_fn=collate_fn, num_workers=num_workers,
                     pin_memory=(device.type == "cuda"), persistent_workers=num_workers > 0)
    if run["config"] == "B_robust":
        print(f"ROBUST TRAINING: random thermal augmentation on the fit set {ROBUST_AUG} (holdout stays clean)")
    fit_loader = DataLoader(PairedLetterboxDataset(ds, "train", fit_ids, train_aug=(run["config"] == "B_robust")),
                            shuffle=True,
                            generator=loader_gen, **loader_kw)
    holdout_loader = DataLoader(PairedLetterboxDataset(ds, "train", holdout_ids), shuffle=False, **loader_kw)

    n_epochs = 2 if smoke else epochs
    log = {"run": run, "run_name": name, "trainable_params": n_trainable, "epochs": []}
    best_loss = float("inf")
    t_start = time.time()

    for epoch in range(n_epochs):
        model.set_train()
        t0, train_total = time.time(), 0.0
        for i, (rgb, th, bd) in enumerate(fit_loader):
            rgb, th = rgb.to(device, non_blocking=True), th.to(device, non_blocking=True)
            bd = {k: v.to(device) for k, v in bd.items()}
            optimizer.zero_grad()
            loss, _ = criterion(model(rgb, th), bd)
            loss.sum().backward()
            optimizer.step()
            train_total += loss.sum().item()
            if i % 50 == 0:
                print(f"  epoch {epoch+1}/{n_epochs}  batch {i}/{len(fit_loader)}  loss={loss.sum().item():.4f}")
        train_loss = train_total / len(fit_loader)

        model.eval()
        holdout_total = 0.0
        with torch.no_grad():
            for rgb, th, bd in holdout_loader:
                bd = {k: v.to(device) for k, v in bd.items()}
                loss, _ = criterion(model(rgb.to(device), th.to(device)), bd)
                holdout_total += loss.sum().item()
        holdout_loss = holdout_total / len(holdout_loader)
        dt = time.time() - t0

        print(f"epoch {epoch+1}/{n_epochs}  train_loss={train_loss:.4f}  "
              f"holdout_loss={holdout_loss:.4f}  ({dt/60:.1f} min)")
        log["epochs"].append({"epoch": epoch + 1, "train_loss": train_loss,
                              "holdout_loss": holdout_loss, "seconds": dt})

        state = {"run": run, "epoch": epoch + 1, "holdout_loss": holdout_loss,
                 "trainable": model.trainable_state()}
        torch.save(state, paths["last"])
        if holdout_loss < best_loss:
            best_loss = holdout_loss
            torch.save(state, paths["best"])
            print(f"  saved new best (holdout_loss={holdout_loss:.4f})")

        log["total_train_hours"] = (time.time() - t_start) / 3600
        paths["trainlog"].write_text(json.dumps(log, indent=2))

    print(f"\nTraining done in {log['total_train_hours']:.2f} h. Best holdout loss {best_loss:.4f}")
    del fit_loader, holdout_loader   # shuts down the persistent workers

    state = torch.load(paths["best"], map_location="cpu")
    model.load_trainable(state["trainable"])
    results = {"run": run, "run_name": name, "best_epoch": state["epoch"],
               "best_holdout_loss": state["holdout_loss"], "trainable_params": n_trainable,
               "train_hours": log["total_train_hours"], "eval": {}}
    eval_kw = dict(batch_size=batch, collate_fn=collate_fn, num_workers=num_workers,
                   pin_memory=(device.type == "cuda"))
    for split_name, ids in eval_sets.items():
        loader = DataLoader(PairedLetterboxDataset(ds, "val", ids), shuffle=False, **eval_kw)
        print(f"\n=== {name}  |  {run['dataset']} {split_name} ({len(ids)} images), best epoch {state['epoch']} ===")
        results["eval"][split_name] = evaluate_map(model, loader, device, ds["names"], IMG_SIZE)

    paths["results"].write_text(json.dumps(results, indent=2))
    print(f"\nresults saved: {paths['results']}")

    del model, optimizer, criterion
    return results


def train_and_eval(run, epochs=30, batch=8, lr=1e-3, num_workers=4, device=None):
    """Train + evaluate one run. Console output is also written to journal/logs/<run>_console.txt."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = run_paths(run)
    tee = _Tee(paths["console"], sys.stdout)
    try:
        with redirect_stdout(tee):
            return _train_and_eval(run, epochs, batch, lr, num_workers, device)
    finally:
        tee.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_queue(queue, **kwargs):
    """Run every entry in order. Finished real runs (eval json exists) are skipped, so the
    queue can be restarted after a crash or a shutdown; an unfinished run restarts from
    epoch 1 and overwrites its own partial files. A run that fails is logged and the queue
    moves on to the next one. Smoke runs are never skipped."""
    summary = []
    for k, run in enumerate(queue, 1):
        name = run_name(run)
        paths = run_paths(run)
        print("\n" + "=" * 90 + f"\n[{k}/{len(queue)}] {name}\n" + "=" * 90)
        if paths["results"].exists() and not run.get("smoke", False):
            print(f"already finished -> skipping ({paths['results']})")
            res = json.loads(paths["results"].read_text())
            summary.append((name, "skipped (done)", res))
            continue
        t0 = time.time()
        try:
            if run["config"] == "T_thermalonly":
                res = train_thermal_only(run, **{k: v for k, v in kwargs.items() if k in ("num_workers", "device")})
            else:
                res = train_and_eval(run, **kwargs)
            summary.append((name, f"done in {(time.time() - t0)/3600:.2f} h", res))
        except Exception:
            err = traceback.format_exc()
            print(f"!!! {name} FAILED -- moving on to the next run\n{err}")
            with open(paths["console"], "a", encoding="utf-8") as f:
                f.write("\n!!! FAILED\n" + err)
            summary.append((name, "FAILED", None))

    print("\n" + "=" * 90 + "\nQUEUE SUMMARY\n" + "=" * 90)
    print(f"{'run':<40}{'status':<18}{'best ep':>8}{'night mAP50':>13}{'full mAP50':>12}{'full 50:95':>12}")
    for name, status, res in summary:
        if res is None:
            print(f"{name:<40}{status:<18}")
            continue
        ev = res["eval"]
        night = f"{ev['night']['map50']:.4f}" if "night" in ev else "-"
        print(f"{name:<40}{status:<18}{str(res.get('best_epoch')):>8}{night:>13}"
              f"{ev['full']['map50']:>12.4f}{ev['full']['map50_95']:>12.4f}")
    return summary


# ----------------------------------------------------------------------------- Config A, same pipeline
class PlainYOLOWrapper(nn.Module):
    """Config A (RGB only) run through the letterbox pipeline; thermal is ignored."""

    def __init__(self, checkpoint):
        super().__init__()
        self.base = YOLO(checkpoint).model
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

    def forward(self, rgb, thermal):
        return self.base(rgb)


def eval_config_a(dataset_name, batch=8, num_workers=4, device=None):
    """Evaluate Config A on the same letterboxed val split(s) as the fusion runs.
    Writes journal/results/lb_yolo_<dataset>_A_eval.json (overwrites only that file)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DATASETS[dataset_name]
    _, _, eval_sets = get_splits(dataset_name, smoke=False)
    model = PlainYOLOWrapper(str(ds["config_a"])).to(device)
    results = {"run_name": f"lb_yolo_{dataset_name}_A", "eval": {}}
    for split_name, ids in eval_sets.items():
        loader = DataLoader(PairedLetterboxDataset(ds, "val", ids), batch_size=batch, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers,
                            pin_memory=(device.type == "cuda"))
        print(f"\n=== Config A  |  {dataset_name} {split_name} ({len(ids)} images) ===")
        results["eval"][split_name] = evaluate_map(model, loader, device, ds["names"], IMG_SIZE)
    out = JOURNAL / "results" / f"lb_yolo_{dataset_name}_A_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults saved: {out}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results



# ----------------------------------------------------------------------------- thermal-only detector
class ThermalOnlyWrapper(nn.Module):
    """A YOLOv8n trained on thermal images only; the 1-channel thermal input is repeated to 3 channels
    (ultralytics also loads grayscale jpegs as 3 identical channels). The RGB input is ignored."""

    def __init__(self, checkpoint):
        super().__init__()
        self.base = YOLO(checkpoint).model
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

    def forward(self, rgb, thermal):
        return self.base(thermal.repeat(1, 3, 1, 1))


def _link_or_copy(src, dst):
    """Hard link if possible (no extra disk space, same drive), otherwise copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_thermal_yolo_dataset(dataset_name):
    """Builds Datasets/<dataset>_thermal_yolo: the thermal images under the same names and split as the
    RGB YOLO dataset, with the same label files. Only adds files that are not there yet (FLIR's copy
    from the earlier run is reused as is). New files are hard links where possible."""
    src = DATASETS[dataset_name]
    dst = DATA / f"{dataset_name}_thermal_yolo"
    n_added = 0
    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for rgb_path in sorted((src["root"] / "images" / split).glob(f"*{src['ext']}")):
            out_img = dst / "images" / split / rgb_path.name
            if not out_img.exists():
                _link_or_copy(src["thermal"][split] / rgb_path.name, out_img)
                n_added += 1
            lbl = src["root"] / "labels" / split / f"{rgb_path.stem}.txt"
            out_lbl = dst / "labels" / split / lbl.name
            if lbl.exists() and not out_lbl.exists():
                shutil.copy2(lbl, out_lbl)
    names = "".join(f"  {k}: {v}\n" for k, v in src["names"].items())
    yaml_path = JOURNAL / f"{dataset_name}_thermal.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(f"path: {dst.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}")
    print(f"thermal YOLO dataset ready at {dst} ({n_added} images added this time)")
    return yaml_path


def _train_thermal_only(run, epochs, num_workers, device):
    """Thermal-only baseline, trained exactly like Config A (stock ultralytics, COCO yolov8n.pt start,
    100 epochs max, patience 20, batch 16, 640) but on thermal images; then evaluated on the same
    letterboxed val splits as everything else."""
    name = run_name(run)
    paths = run_paths(run)
    smoke = run.get("smoke", False)
    yaml_path = prepare_thermal_yolo_dataset(run["dataset"])
    project = JOURNAL / "smoke" if smoke else PROJ / "results"
    t0 = time.time()
    model = YOLO(str(PROJ / "yolov8n.pt"))
    model.train(data=str(yaml_path), epochs=1 if smoke else epochs, patience=20, batch=16, imgsz=640,
                seed=run["seed"], project=str(project), name=name, exist_ok=True,
                fraction=0.02 if smoke else 1.0, plots=False, workers=num_workers)
    train_hours = (time.time() - t0) / 3600
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"   # where ultralytics actually saved it
    print(f"\nthermal-only training done in {train_hours:.2f} h -> {best}")

    ds = DATASETS[run["dataset"]]
    _, _, eval_sets = get_splits(run["dataset"], smoke)
    wrapper = ThermalOnlyWrapper(str(best)).to(device)
    results = {"run": run, "run_name": name, "best_epoch": None, "checkpoint": str(best),
               "train_hours": train_hours, "eval": {}}
    for split_name, ids in eval_sets.items():
        loader = DataLoader(PairedLetterboxDataset(ds, "val", ids), batch_size=8, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers, pin_memory=(device.type == "cuda"))
        print(f"\n=== {name}  |  {run['dataset']} {split_name} ({len(ids)} images) ===")
        results["eval"][split_name] = evaluate_map(wrapper, loader, device, ds["names"], IMG_SIZE)
    paths["results"].write_text(json.dumps(results, indent=2))
    print(f"\nresults saved: {paths['results']}")
    del wrapper, model
    return results


def train_thermal_only(run, epochs=100, num_workers=4, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = run_paths(run)
    tee = _Tee(paths["console"], sys.stdout)
    try:
        with redirect_stdout(tee):
            return _train_thermal_only(run, epochs, num_workers, device)
    finally:
        tee.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



# ----------------------------------------------------------------------------- robustness study (eval only)
def _model_specs(dataset_name):
    """Every YOLO model the robustness study compares, for one dataset. Missing checkpoints are skipped."""
    ds = DATASETS[dataset_name]
    ck = PROJ / "checkpoints_lb"
    specs = {"A (RGB only = detached adapter)": ("A", ds["config_a"])}
    for seed in (1, 2):
        specs[f"B s{seed}"] = ("fusion", "B", ck / f"lb_yolo_{dataset_name}_B_s{seed}_best.pt")
    for seed in (1, 2):
        specs[f"B_robust s{seed}"] = ("fusion", "B_robust", ck / f"lb_yolo_{dataset_name}_B_robust_s{seed}_best.pt")
    if dataset_name == "flir":
        specs["B_concat s1"] = ("fusion", "B_concat", ck / "lb_yolo_flir_B_concat_s1_best.pt")
        specs["B_unfrozen s1"] = ("fusion", "B_unfrozen", ck / "lb_yolo_flir_B_unfrozen_s1_best.pt")
    specs["Thermal only"] = ("thermal_only", PROJ / "results" / f"lb_yolo_{dataset_name}_T_thermalonly_s1" / "weights" / "best.pt")
    out = {}
    for name, spec in specs.items():
        if Path(spec[-1]).exists():
            out[name] = spec
        else:
            print(f"(skipping {name}: {spec[-1]} not found)")
    return out


def _load_eval_model(spec, dataset_name, device):
    if spec[0] == "A":
        return PlainYOLOWrapper(str(spec[1])).to(device)
    if spec[0] == "thermal_only":
        return ThermalOnlyWrapper(str(spec[1])).to(device)
    _, config, ckpt = spec
    model = build_model(config, DATASETS[dataset_name]["config_a"])
    state = torch.load(ckpt, map_location="cpu")
    model.load_trainable(state["trainable"])
    return model.to(device)


def run_robustness(dataset_name, perturbations=None, splits=None, smoke=False, batch=8, num_workers=4, device=None):
    """Evaluates every model under every thermal perturbation. Results are saved after each evaluation to
    journal/results/robustness_<dataset>.json (or journal/smoke/..._SMOKETEST.json); re-running skips
    entries already in that file, so it can be stopped and resumed."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DATASETS[dataset_name]
    perturbations = perturbations or list(PERTURBATIONS)
    _, _, eval_sets = get_splits(dataset_name, smoke)
    if splits:
        eval_sets = {k: v for k, v in eval_sets.items() if k in splits}
    out_dir = JOURNAL / ("smoke" if smoke else "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"robustness_{dataset_name}{'_SMOKETEST' if smoke else ''}.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    specs = _model_specs(dataset_name)
    for model_name, spec in specs.items():
        model = None
        # Config A ignores thermal, so it is only evaluated once (clean)
        perts = ["clean"] if spec[0] == "A" else perturbations
        for pert in perts:
            for split_name, ids in eval_sets.items():
                key = f"{split_name}|{model_name}|{pert}"
                if key in results:
                    continue
                if model is None:
                    model = _load_eval_model(spec, dataset_name, device)
                print(f"\n=== {dataset_name} {split_name} | {model_name} | thermal: {pert} ===")
                loader = DataLoader(PairedLetterboxDataset(ds, "val", ids, perturb=PERTURBATIONS[pert]),
                                    batch_size=batch, shuffle=False, collate_fn=collate_fn,
                                    num_workers=num_workers, pin_memory=(device.type == "cuda"))
                results[key] = evaluate_map(model, loader, device, ds["names"], IMG_SIZE)
                out_path.write_text(json.dumps(results, indent=2))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_robustness_table(results, metric="map50")
    print(f"\nsaved: {out_path}")
    return results


def print_robustness_table(results, metric="map50"):
    splits = sorted({k.split("|")[0] for k in results}, reverse=True)
    models = list(dict.fromkeys(k.split("|")[1] for k in results))
    perts = [p for p in PERTURBATIONS if any(k.endswith("|" + p) for k in results)]
    for split in splits:
        print(f"\n{metric} -- {split}")
        print(f"{'model':<34}" + "".join(f"{p:>15}" for p in perts))
        for m in models:
            row = ""
            for p in perts:
                r = results.get(f"{split}|{m}|{p}")
                row += f"{r[metric]:>15.4f}" if r else f"{'':>15}"
            print(f"{m:<34}" + row)


# ----------------------------------------------------------------------------- analysis (eval only)
def collate_meta_fn(batch):
    rgb, th, bd = collate_fn(batch)
    return rgb, th, bd, [b["image_id"] for b in batch], [b["meta"] for b in batch]


def _to_original(xyxy, m):
    """Letterboxed 640x640 pixel boxes -> original-image pixel boxes."""
    out = xyxy.copy()
    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - m["left"]) / m["rx"], 0, m["w0"])
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - m["top"]) / m["ry"], 0, m["h0"])
    return out


def collect_predictions(model, ds, ids, device, batch=8, num_workers=4, conf_thres=0.001, record_gate=False):
    """Runs a model once over `ids`. Returns predictions and ground truth in ORIGINAL image pixels, image
    sizes, and (optionally) the attention gate's per-image output from the first fusion stage."""
    loader = DataLoader(PairedLetterboxDataset(ds, "val", ids), batch_size=batch, shuffle=False,
                        collate_fn=collate_meta_fn, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    nc = len(ds["names"])
    preds, gts, sizes, gate_out = {}, {}, {}, []
    handle = None
    if record_gate:
        handle = model.fusions[0].gate.register_forward_hook(
            lambda m, i, o: gate_out.append(o.detach().float().cpu()))
    model.eval()
    with torch.no_grad():
        for b, (rgb, th, bd, img_ids, metas) in enumerate(loader):
            if b % 20 == 0:
                print(f"  batch {b}/{len(loader)}...")
            out = model(rgb.to(device), th.to(device))
            nms_out = non_max_suppression(out[0], conf_thres=conf_thres, iou_thres=0.6, nc=nc,
                                          max_time_img=1.0, max_det=300)
            for i, det in enumerate(nms_out):
                m, img_id = metas[i], img_ids[i]
                d = det.cpu().numpy().astype(np.float64)
                if len(d):
                    d[:, :4] = _to_original(d[:, :4], m)
                preds[img_id] = d
                mask = (bd["batch_idx"] == i).numpy()
                g = xywh2xyxy(bd["bboxes"][mask]).numpy().astype(np.float64) * IMG_SIZE
                gts[img_id] = (_to_original(g, m) if len(g) else g.reshape(0, 4),
                               bd["cls"][mask].numpy().astype(int))
                sizes[img_id] = (m["w0"], m["h0"])
    if handle is not None:
        handle.remove()
    gate = torch.cat(gate_out).numpy() if gate_out else None
    return preds, gts, sizes, gate


def coco_eval(preds, gts, sizes, img_ids=None):
    """COCO-style AP (101-point, maxDets 300) overall and by object size, with pycocotools.
    The interpolation differs slightly from evaluate_map(); use these numbers for the size/brightness
    tables only, and compare models within this function, not against evaluate_map() numbers."""
    import io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    keys = sorted(gts) if img_ids is None else sorted(img_ids)
    keyset = set(keys)
    id_map = {k: n for n, k in enumerate(sorted(gts))}
    images, anns, dets = [], [], []
    n_by_size = {"small": 0, "medium": 0, "large": 0}
    for k in sorted(gts):
        w0, h0 = sizes[k]
        images.append({"id": id_map[k], "width": int(w0), "height": int(h0)})
        boxes, cls = gts[k]
        for bx, c in zip(boxes, cls):
            w, h = bx[2] - bx[0], bx[3] - bx[1]
            area = float(w * h)
            anns.append({"id": len(anns) + 1, "image_id": id_map[k], "category_id": int(c),
                         "bbox": [float(bx[0]), float(bx[1]), float(w), float(h)], "area": area, "iscrowd": 0})
            if k in keyset:
                n_by_size["small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"] += 1
        for d in preds.get(k, []):
            dets.append({"image_id": id_map[k], "category_id": int(d[5]),
                         "bbox": [float(d[0]), float(d[1]), float(d[2] - d[0]), float(d[3] - d[1])],
                         "score": float(d[4])})
    cats = sorted({a["category_id"] for a in anns} | {d["category_id"] for d in dets})
    empty = {"map50": 0.0, "map50_95": 0.0, "map50_small": 0.0, "map50_medium": 0.0, "map50_large": 0.0,
             "n_images": len(keys), "n_gt_by_size": n_by_size}
    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": c} for c in cats]}
    with redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        if not dets or not anns:
            return empty
        coco_dt = coco_gt.loadRes(dets)
        E = COCOeval(coco_gt, coco_dt, "bbox")
        E.params.maxDets = [1, 10, 300]
        E.params.imgIds = [id_map[k] for k in keys]
        E.evaluate()
        E.accumulate()
    prec = E.eval["precision"]   # [IoU thr, recall, class, area (all, small, medium, large), maxDets]

    def ap(iou50, a):
        p = prec[0, :, :, a, 2] if iou50 else prec[:, :, :, a, 2]
        p = p[p > -1]
        return float(p.mean()) if p.size else float("nan")

    return {"map50": ap(True, 0), "map50_95": ap(False, 0),
            "map50_small": ap(True, 1), "map50_medium": ap(True, 2), "map50_large": ap(True, 3),
            "map50_95_small": ap(False, 1), "map50_95_medium": ap(False, 2), "map50_95_large": ap(False, 3),
            "n_images": len(keys), "n_gt_by_size": n_by_size}


def per_image_hits(pred, gt, conf=0.25, iou=0.5):
    """Number of ground-truth boxes found (same class, IoU >= iou, score >= conf; greedy by score)."""
    boxes, cls = gt
    if len(boxes) == 0 or len(pred) == 0:
        return 0
    d = pred[pred[:, 4] >= conf]
    if len(d) == 0:
        return 0
    d = d[np.argsort(-d[:, 4])]
    ious = box_iou(torch.tensor(d[:, :4]), torch.tensor(boxes)).numpy()
    used = np.zeros(len(boxes), bool)
    for j in range(len(d)):
        cand = np.where((~used) & (cls == int(d[j, 5])) & (ious[j] >= iou))[0]
        if len(cand):
            used[cand[np.argmax(ious[j, cand])]] = True
    return int(used.sum())


def _analysis_specs(dataset_name):
    specs = _model_specs(dataset_name)
    keep = ["A (RGB only = detached adapter)", "B s1", "B s2", "B_robust s1", "Thermal only"]
    specs = {k: v for k, v in specs.items() if k in keep}
    rg = PROJ / "checkpoints_lb" / f"lb_yolo_{dataset_name}_B_rgbgray_s1_best.pt"
    if rg.exists():
        specs["B_rgbgray s1"] = ("fusion", "B_rgbgray", rg)
    return specs


def _image_brightness(ds, ids):
    rgb_dir = ds["root"] / "images" / "val"
    return {i: float(cv2.imread(str(rgb_dir / f"{i}{ds['ext']}"), cv2.IMREAD_GRAYSCALE).mean()) for i in ids}


def make_qualitative_figure(ds, preds_by_model, gts, ids, out_path, n_best=3, conf=0.25):
    """Rows: the n_best images where B s1 finds the most extra objects vs A, plus the one where it loses the
    most (so the figure is not only cherry-picked successes). Columns: RGB + ground truth | A | B s1 |
    thermal-only (drawn on the thermal image)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a_name, b_name, t_name = "A (RGB only = detached adapter)", "B s1", "Thermal only"
    if not all(k in preds_by_model for k in (a_name, b_name)):
        print("(qualitative figure skipped: A or B s1 missing)")
        return None
    diff = {i: per_image_hits(preds_by_model[b_name][i], gts[i], conf)
               - per_image_hits(preds_by_model[a_name][i], gts[i], conf) for i in ids}
    order = sorted(ids, key=lambda i: diff[i])
    rows = [(i, f"fusion helps, {diff[i]:+d}") for i in order[::-1][:n_best]] + [(order[0], f"fusion hurts, {diff[order[0]]:+d}")]
    cols = [("RGB + ground truth", None, "rgb"), ("A: RGB only", a_name, "rgb"),
            ("B: frozen RGB + thermal adapter", b_name, "rgb")]
    if t_name in preds_by_model:
        cols.append(("Thermal only (on thermal image)", t_name, "thermal"))
    names = ds["names"]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(4.2 * len(cols), 3.4 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (img_id, tag) in enumerate(rows):
        rgb = cv2.cvtColor(cv2.imread(str(ds["root"] / "images" / "val" / f"{img_id}{ds['ext']}")), cv2.COLOR_BGR2RGB)
        th = cv2.imread(str(ds["thermal"]["val"] / f"{img_id}{ds['ext']}"), cv2.IMREAD_GRAYSCALE)
        th = cv2.resize(th, (rgb.shape[1], rgb.shape[0]))
        for c, (title, model_name, base) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(rgb if base == "rgb" else th, cmap=None if base == "rgb" else "gray")
            if model_name is None:
                boxes, cls = gts[img_id]
                items = [(bx, int(k)) for bx, k in zip(boxes, cls)]
                found = ""
            else:
                p = preds_by_model[model_name][img_id]
                p = p[p[:, 4] >= conf] if len(p) else p
                items = [(d[:4], int(d[5])) for d in p]
                found = f"  ({per_image_hits(preds_by_model[model_name][img_id], gts[img_id], conf)}/{len(gts[img_id][0])} found)"
            for bx, k in items:
                ax.add_patch(plt.Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False,
                                           edgecolor=colors[k % 10], linewidth=1.2))
            ax.set_title((f"{img_id} [{tag}]\n" if c == 0 else "\n") + title + found, fontsize=8)
            ax.axis("off")
    handles = [plt.Line2D([0], [0], color=colors[k % 10], lw=2, label=v) for k, v in names.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(names), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"figure saved: {out_path}")
    return [r[0] for r in rows]


def run_analysis(dataset_name, smoke=False, n_bins=5, batch=8, num_workers=4, device=None):
    """Brightness-binned AP, AP by object size, attention-gate statistics and a qualitative figure, for every
    model in _analysis_specs. Writes journal/results/analysis_<dataset>.json and
    journal/figures/qualitative_<dataset>.png (smoke test: journal/smoke/..._SMOKETEST)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DATASETS[dataset_name]
    _, _, eval_sets = get_splits(dataset_name, smoke)
    ids = eval_sets["full"]
    night = set(eval_sets.get("night", []))
    sfx = "_SMOKETEST" if smoke else ""
    out_json = JOURNAL / ("smoke" if smoke else "results") / f"analysis_{dataset_name}{sfx}.json"
    out_fig = JOURNAL / ("smoke" if smoke else "figures") / f"qualitative_{dataset_name}{sfx}.png"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    bright = _image_brightness(ds, ids)
    edges = np.quantile(np.array([bright[i] for i in ids]), np.linspace(0, 1, n_bins + 1))
    bins = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        members = [i for i in ids if (lo <= bright[i] < hi) or (b == n_bins - 1 and bright[i] == hi)]
        bins.append({"lo": float(lo), "hi": float(hi), "ids": members})
    print("brightness bins (mean gray of the RGB image): " +
          ", ".join(f"[{b['lo']:.0f}-{b['hi']:.0f}] n={len(b['ids'])}" for b in bins))

    results = {"dataset": dataset_name,
               "brightness_bins": [{"lo": b["lo"], "hi": b["hi"], "n": len(b["ids"])} for b in bins],
               "models": {}}
    preds_by_model, gts = {}, None
    for name, spec in _analysis_specs(dataset_name).items():
        print(f"\n=== {dataset_name} | {name} ===")
        model = _load_eval_model(spec, dataset_name, device)
        has_gate = spec[0] == "fusion" and isinstance(model.fusions[0], ChannelAttentionFusion)
        preds, gts_m, sizes, gate = collect_predictions(model, ds, ids, device, batch, num_workers,
                                                        record_gate=has_gate)
        gts = gts if gts is not None else gts_m
        preds_by_model[name] = preds
        r = {"full": coco_eval(preds, gts_m, sizes)}
        if night:
            r["night"] = coco_eval(preds, gts_m, sizes, img_ids=[i for i in ids if i in night])
        r["by_brightness"] = [coco_eval(preds, gts_m, sizes, img_ids=b["ids"]) if b["ids"] else None for b in bins]
        if has_gate and gate is not None and len(gate) == len(ids):
            c = gate.shape[1] // 2
            w_th = gate[:, c:]
            mean_th = w_th.mean(1)
            b_arr = np.array([bright[i] for i in ids])
            is_night = np.array([i in night for i in ids])
            r["gate"] = {
                "mean_w_rgb": float(gate[:, :c].mean()),
                "mean_w_thermal": float(w_th.mean()),
                "per_channel_std_across_images_mean": float(w_th.std(0).mean()),
                "per_image_mean_w_thermal_range": [float(mean_th.min()), float(mean_th.max())],
                "corr_brightness_vs_mean_w_thermal": float(np.corrcoef(b_arr, mean_th)[0, 1]) if len(ids) > 2 else None,
                "mean_w_thermal_night": float(mean_th[is_night].mean()) if is_night.any() else None,
                "mean_w_thermal_day": float(mean_th[~is_night].mean()) if (~is_night).any() and night else None,
            }
        results["models"][name] = r
        out_json.write_text(json.dumps(results, indent=2))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pool = [i for i in ids if i in night] if night else ids
    results["qualitative_images"] = make_qualitative_figure(ds, preds_by_model, gts, pool, out_fig)
    out_json.write_text(json.dumps(results, indent=2))
    print_analysis(results)
    print(f"\nsaved: {out_json}")
    return results


def print_analysis(results):
    models = results["models"]
    print("\nmAP50 by brightness bin of the RGB image (COCO-style, darkest -> brightest)")
    print(f"{'model':<34}" + "".join(f"{b['lo']:>6.0f}-{b['hi']:<5.0f}" for b in results["brightness_bins"]))
    for m, r in models.items():
        print(f"{m:<34}" + "".join(f"{(x['map50'] if x else float('nan')):>12.4f}" for x in r["by_brightness"]))
    first = next(iter(models.values()))["full"]["n_gt_by_size"]
    print("\nmAP50 by object size, full val (COCO-style)  [GT boxes: " +
          ", ".join(f"{k}={v}" for k, v in first.items()) + "]")
    has_night = any("night" in r for r in models.values())
    print(f"{'model':<34}{'all':>9}{'small':>9}{'medium':>9}{'large':>9}" + (f"{'night':>9}" if has_night else ""))
    for m, r in models.items():
        f = r["full"]
        print(f"{m:<34}{f['map50']:>9.4f}{f['map50_small']:>9.4f}{f['map50_medium']:>9.4f}{f['map50_large']:>9.4f}"
              + (f"{r['night']['map50']:>9.4f}" if "night" in r else ""))
    for m, r in models.items():
        if "gate" in r:
            g = r["gate"]
            print(f"\ngate {m}: mean w_rgb={g['mean_w_rgb']:.3f}  mean w_thermal={g['mean_w_thermal']:.3f}  "
                  f"per-channel std across images={g['per_channel_std_across_images_mean']:.4f}  "
                  f"corr(brightness, mean w_thermal)={g['corr_brightness_vs_mean_w_thermal']}  "
                  f"night/day mean w_thermal={g['mean_w_thermal_night']}/{g['mean_w_thermal_day']}")
