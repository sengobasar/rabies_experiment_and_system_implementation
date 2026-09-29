# Auto-extracted from WILLIE notebook (classes and functions only).
# Source: Qian-Group-HRI/Willie @ 221f2e7
# Top-level execution code was stripped by the AST extractor.

import os, sys, time, math, json, ast, warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange, repeat
import random, cv2, ast
from collections import Counter
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import time
import matplotlib
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np, cv2

# --- Module-level constants ---
SEED = 42
ROOT = Path(os.environ.get('WILLIE_ROOT', Path.cwd()))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLS_MANIFEST_DIR = ROOT / 'artifacts' / 'willie_v2' / 'manifests'
LOCKED_DIR = ROOT / 'artifacts' / 'willie_LOCKED_INPUTS' / 'tables'
CLS_TRAIN_CSV = CLS_MANIFEST_DIR / 'cls_train.csv'
CLS_VAL_CSV = CLS_MANIFEST_DIR / 'cls_val.csv'
CLS_TEST_CSV = CLS_MANIFEST_DIR / 'cls_test.csv'
SEG_TRAIN_CSV = LOCKED_DIR / 'ws_seg_manifest_fuseg_train.csv'
SEG_VAL_CSV = LOCKED_DIR / 'ws_seg_manifest_fuseg_val.csv'
DET_TRAIN_CSV = LOCKED_DIR / 'ws_det_manifest_yolo_train.csv'
DET_VAL_CSV = LOCKED_DIR / 'ws_det_manifest_yolo_val.csv'
CKPT_DIR = ROOT / 'artifacts' / '09_fuseg_csd'
CLASS_NAMES = ['diabetic', 'no_wound', 'pressure', 'surgical', 'venous']
NUM_CLASSES = 5
all_csvs = {'cls_train': CLS_TRAIN_CSV, 'cls_val': CLS_VAL_CSV, 'cls_test': CLS_TEST_CSV, 'seg_train': SEG_TRAIN_CSV, 'seg_val': SEG_VAL_CSV, 'det_train': DET_TRAIN_CSV, 'det_val': DET_VAL_CSV}
MANIFESTS = {}
cls_df = seg_df = det_df = None
CLS_LABEL_COL = 'unified_class'
CLS_INT_COL = 'unified_label'
SEG_IMG_COL = 'img'
SEG_MASK_COL = 'mask'
DET_IMG_COL = 'img'
DET_LABEL_COL = 'label'
SPLITS_FILE = CKPT_DIR / '5fold_splits_v2.pt'
MANIFESTS_CACHE = CKPT_DIR / 'data_manifests.pt'
IMG_SIZE = 518
N_FOLDS = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 4
NUM_WORKERS = 4
TRAIN_CFG = {'variant': 'MINI', 'n_folds': 5, 'epochs': 50, 'freeze_epochs': 5, 'lr_head': 0.0001, 'lr_backbone': 1e-05, 'weight_decay': 0.0001, 'patience': 12, 'grad_clip': 1.0, 'batch_size': BATCH_SIZE, 'accumulation_steps': 2, 'seg_size': 512}
PROGRESS_FILE = CKPT_DIR / 'mini_training_progress.pt'
FIG_DIR = CKPT_DIR / 'figures'
VARIANT = 'MINI'
seg_size = 512

@dataclass
class ModelConfig:
    name: str = 'MINI'
    backbone_name: str = 'dinov2_vits14'
    backbone_dim: int = 384
    backbone_layers: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    img_size: int = 518
    patch_size: int = 14
    fpn_dim: int = 256
    fpn_levels: int = 4
    wa_csa_layers: int = 1
    wa_csa_heads: int = 8
    wa_csa_dropout: float = 0.1
    num_classes: int = 5
    num_experts: int = 2
    top_k_experts: int = 2
    cls_embed_dim: int = 128
    seg_out_channels: int = 1
    pscse_reduction: int = 16
    det_max_objects: int = 20
    freeze_backbone_epochs: int = 3

    @property
    def grid_size(self) -> int:
        return self.img_size // self.patch_size

def get_model_config(variant: str='MINI') -> ModelConfig:
    configs = {'MINI': ModelConfig(name='MINI', backbone_name='dinov2_vits14', backbone_dim=384, fpn_dim=256, wa_csa_layers=1, num_experts=2, top_k_experts=2), 'BASE': ModelConfig(name='BASE', backbone_name='dinov2_vitl14', backbone_dim=1024, fpn_dim=256, wa_csa_layers=2, num_experts=4, top_k_experts=2), 'XL': ModelConfig(name='XL', backbone_name='dinov2_vitl14', backbone_dim=1024, fpn_dim=384, wa_csa_layers=4, num_experts=8, top_k_experts=2)}
    return configs[variant]

class DINOv2MultiScale(nn.Module):
    """Shared DINOv2 backbone with hooks at layers [2,5,8,11]."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.features = {}
        self.backbone = torch.hub.load('facebookresearch/dinov2', cfg.backbone_name, pretrained=True)
        self._hook_handles = []
        for idx in cfg.backbone_layers:
            h = self.backbone.blocks[idx].register_forward_hook(self._make_hook(idx))
            self._hook_handles.append(h)
        self.set_frozen(True)

    def _make_hook(self, idx):

        def hook(mod, inp, out):
            tokens = out[:, 1:, :]
            G = self.cfg.grid_size
            self.features[idx] = rearrange(tokens, 'b (h w) d -> b d h w', h=G, w=G)
        return hook

    def set_frozen(self, frozen):
        for p in self.backbone.parameters():
            p.requires_grad = not frozen

    def forward(self, x):
        self.features.clear()
        if x.shape[-1] != self.cfg.img_size:
            x = F.interpolate(x, size=self.cfg.img_size, mode='bilinear', align_corners=False)
        self.backbone(x)
        return [self.features[idx] for idx in self.cfg.backbone_layers]

class FeaturePyramidNeck(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, F = (cfg.backbone_dim, cfg.fpn_dim)
        self.laterals = nn.ModuleList([nn.Sequential(nn.Conv2d(D, F, 1, bias=False), nn.GroupNorm(32, F), nn.GELU()) for _ in range(cfg.fpn_levels)])
        self.smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(F, F, 3, padding=1, bias=False), nn.GroupNorm(32, F), nn.GELU()) for _ in range(cfg.fpn_levels)])

    def forward(self, features):
        lats = [self.laterals[i](features[i]) for i in range(len(features))]
        pyr = [None] * len(features)
        pyr[-1] = lats[-1]
        for i in range(len(features) - 2, -1, -1):
            up = F.interpolate(pyr[i + 1], size=lats[i].shape[-2:], mode='bilinear', align_corners=False)
            pyr[i] = lats[i] + up
        return [self.smooth[i](pyr[i]) for i in range(len(features))]

class WoundAwareCrossScaleAttention(nn.Module):
    """Bidirectional cross-attention + wound gate + tanh alpha residual."""

    def __init__(self, cfg: ModelConfig, level_idx: int=0):
        super().__init__()
        dim = cfg.fpn_dim
        self.heads = cfg.wa_csa_heads
        self.head_dim = dim // self.heads
        self.scale = self.head_dim ** (-0.5)
        self.use_sdpa = level_idx > 0
        self.dropout_p = cfg.wa_csa_dropout
        self.q_f = nn.Linear(dim, dim, bias=False)
        self.k_c = nn.Linear(dim, dim, bias=False)
        self.v_c = nn.Linear(dim, dim, bias=False)
        self.q_c = nn.Linear(dim, dim, bias=False)
        self.k_f = nn.Linear(dim, dim, bias=False)
        self.v_f = nn.Linear(dim, dim, bias=False)
        self.out_f = nn.Linear(dim, dim, bias=False)
        self.out_c = nn.Linear(dim, dim, bias=False)
        self.gate_f = nn.Sequential(nn.Conv2d(dim, dim // 4, 1), nn.GELU(), nn.Conv2d(dim // 4, 1, 1), nn.Sigmoid())
        self.gate_c = nn.Sequential(nn.Conv2d(dim, dim // 4, 1), nn.GELU(), nn.Conv2d(dim // 4, 1, 1), nn.Sigmoid())
        self.alpha_f = nn.Parameter(torch.zeros(1))
        self.alpha_c = nn.Parameter(torch.zeros(1))
        self.norm_f = nn.LayerNorm(dim)
        self.norm_c = nn.LayerNorm(dim)
        self.drop = nn.Dropout(cfg.wa_csa_dropout)

    def _attn(self, q, k, v):
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        if self.use_sdpa and hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        else:
            a = q @ k.transpose(-2, -1) * self.scale
            out = self.drop(a.softmax(-1)) @ v
        return rearrange(out, 'b h n d -> b n (h d)')

    def forward(self, fine, coarse):
        B, C, H, W = fine.shape
        f_seq = self.norm_f(rearrange(fine, 'b c h w -> b (h w) c'))
        c_seq = self.norm_c(rearrange(coarse, 'b c h w -> b (h w) c'))
        f_up = rearrange(self.out_f(self._attn(self.q_f(f_seq), self.k_c(c_seq), self.v_c(c_seq))), 'b (h w) c -> b c h w', h=H)
        c_up = rearrange(self.out_c(self._attn(self.q_c(c_seq), self.k_f(f_seq), self.v_f(f_seq))), 'b (h w) c -> b c h w', h=H)
        return (fine + torch.tanh(self.alpha_f) * f_up * self.gate_f(fine), coarse + torch.tanh(self.alpha_c) * c_up * self.gate_c(coarse))

class WA_CSA_Stack(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        pairs = cfg.fpn_levels - 1
        self.layers = nn.ModuleList([nn.ModuleList([WoundAwareCrossScaleAttention(cfg, p) for p in range(pairs)]) for _ in range(cfg.wa_csa_layers)])

    def forward(self, pyr):
        for layer in self.layers:
            p = list(pyr)
            for i, wa in enumerate(layer):
                p[i], p[i + 1] = wa(p[i], p[i + 1])
            pyr = p
        return pyr

class Expert(nn.Module):

    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_hid, d_out))

    def forward(self, x):
        return self.net(x)

class TopKRouter(nn.Module):

    def __init__(self, d_in, n_exp, top_k=2):
        super().__init__()
        self.n_exp, self.top_k = (n_exp, top_k)
        self.gate = nn.Linear(d_in, n_exp, bias=False)
        self.aux_loss = 0.0

    def forward(self, x):
        probs = F.softmax(self.gate(x), -1)
        w, idx = torch.topk(probs, self.top_k, -1)
        w = w / w.sum(-1, keepdim=True)
        self.aux_loss = F.mse_loss(probs.mean(0), torch.ones(self.n_exp, device=x.device) / self.n_exp)
        return (w, idx)

class ClassificationDecoder(nn.Module):
    """Multi-scale pool → MoE → logits + wound_embed (WTCS bridge)."""

    def __init__(self, cfg):
        super().__init__()
        F = cfg.fpn_dim
        self.cfg = cfg
        self.pools = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1)) for _ in range(cfg.fpn_levels)])
        self.scale_attn = nn.Sequential(nn.Linear(cfg.fpn_levels, cfg.fpn_levels), nn.Softmax(-1))
        self.pre = nn.Sequential(nn.Linear(F, F), nn.LayerNorm(F), nn.GELU())
        self.experts = nn.ModuleList([Expert(F, F * 2, cfg.num_classes) for _ in range(cfg.num_experts)])
        self.router = TopKRouter(F, cfg.num_experts, cfg.top_k_experts)
        self.embed_head = nn.Sequential(nn.Linear(F, cfg.cls_embed_dim), nn.LayerNorm(cfg.cls_embed_dim), nn.GELU())

    def forward(self, pyr):
        B = pyr[0].shape[0]
        pooled = torch.stack([self.pools[i](pyr[i]) for i in range(self.cfg.fpn_levels)], 1)
        sw = self.scale_attn(torch.ones(B, self.cfg.fpn_levels, device=pooled.device))
        fused = self.pre((pooled * sw.unsqueeze(-1)).sum(1))
        w, idx = self.router(fused)
        logits = torch.zeros(B, self.cfg.num_classes, device=fused.device)
        for k in range(self.cfg.top_k_experts):
            for e in range(self.cfg.num_experts):
                mask = idx[:, k] == e
                if mask.any():
                    logits[mask] += w[mask, k:k + 1] * self.experts[e](fused[mask])
        return (logits, self.embed_head(fused))

class ChannelSE(nn.Module):

    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 4)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(ch, mid, bias=False), nn.ReLU(True), nn.Linear(mid, ch, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x).view(x.shape[0], -1, 1, 1)

class SpatialSE(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, 1, 1, bias=False)

    def forward(self, x):
        return x * torch.sigmoid(self.conv(x))

class ParallelScSE(nn.Module):
    """Full: additive(cSE+sSE) + maxout(cSE,sSE). Shortened: additive only."""

    def __init__(self, ch, r=16, shortened=False):
        super().__init__()
        self.shortened = shortened
        self.cse_a, self.sse_a = (ChannelSE(ch, r), SpatialSE(ch))
        if not shortened:
            self.cse_m, self.sse_m = (ChannelSE(ch, r), SpatialSE(ch))

    def forward(self, x):
        add = self.cse_a(x) + self.sse_a(x)
        if self.shortened:
            return add
        return add + torch.max(self.cse_m(x), self.sse_m(x))

class FiLMConditioner(nn.Module):

    def __init__(self, embed_dim, feat_dim):
        super().__init__()
        self.gamma = nn.Sequential(nn.Linear(embed_dim, feat_dim), nn.Sigmoid())
        self.beta = nn.Linear(embed_dim, feat_dim)

    def forward(self, feat, embed):
        g = self.gamma(embed).unsqueeze(-1).unsqueeze(-1) + 1.0
        b = self.beta(embed).unsqueeze(-1).unsqueeze(-1)
        return g * feat + b

class PscSEDecoderStage(nn.Module):
    """Conv → P-scSE (middle) → Conv. Top stage uses shortened P-scSE."""

    def __init__(self, in_ch, out_ch, r=16, shortened=False):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.pscse = ParallelScSE(out_ch, r, shortened)
        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))

    def forward(self, x):
        return self.conv2(self.pscse(self.conv1(x)))

class SegmentationDecoder(nn.Module):
    """Progressive decode P4→P1 with FiLM + P-scSE at each stage."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        F = cfg.fpn_dim
        self.films = nn.ModuleList([FiLMConditioner(cfg.cls_embed_dim, F) for _ in range(cfg.fpn_levels)])
        self.stages = nn.ModuleList([PscSEDecoderStage(F * 2, F, cfg.pscse_reduction, shortened=i == cfg.fpn_levels - 2) for i in range(cfg.fpn_levels - 1)])
        self.head = nn.Sequential(nn.Conv2d(F, F // 2, 3, padding=1, bias=False), nn.BatchNorm2d(F // 2), nn.ReLU(True), nn.Conv2d(F // 2, F // 4, 3, padding=1, bias=False), nn.BatchNorm2d(F // 4), nn.ReLU(True), nn.Conv2d(F // 4, cfg.seg_out_channels, 1))

    def forward(self, pyr, wound_embed, target_size=None):
        if target_size is None:
            target_size = self.cfg.img_size
        cond = [self.films[i](pyr[i], wound_embed) for i in range(self.cfg.fpn_levels)]
        x = cond[-1]
        for s in range(self.cfg.fpn_levels - 1):
            skip = cond[self.cfg.fpn_levels - 2 - s]
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = self.stages[s](torch.cat([x, skip], 1))
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.head(x)

class DetectionDecoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        F = cfg.fpn_dim
        self.level_w = nn.Parameter(torch.ones(cfg.fpn_levels) / cfg.fpn_levels)
        self.shared = nn.Sequential(nn.Conv2d(F, F, 3, padding=1, bias=False), nn.GroupNorm(32, F), nn.GELU(), nn.Conv2d(F, F, 3, padding=1, bias=False), nn.GroupNorm(32, F), nn.GELU())
        self.obj = nn.Conv2d(F, 1, 1)
        self.bbox = nn.Sequential(nn.Conv2d(F, F // 2, 3, padding=1), nn.GELU(), nn.Conv2d(F // 2, 4, 1), nn.Sigmoid())
        self.cls = nn.Conv2d(F, cfg.num_classes, 1)

    def forward(self, pyr):
        w = F.softmax(self.level_w, 0)
        tgt = pyr[0].shape[-2:]
        fused = sum((w[i] * (F.interpolate(p, tgt, mode='bilinear', align_corners=False) if p.shape[-2:] != tgt else p) for i, p in enumerate(pyr)))
        feat = self.shared(fused)
        return {'objectness': self.obj(feat), 'bbox': self.bbox(feat), 'det_cls': self.cls(feat)}

class WILLIEModel(nn.Module):
    """
    DINOv2 → FPN → WA-CSA → [CLS(MoE) + SEG(P-scSE+FiLM) + DET]
    WTCS bridge: cls wound_embed → FiLM conditioning on seg decoder.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = DINOv2MultiScale(cfg)
        self.fpn = FeaturePyramidNeck(cfg)
        self.wa_csa = WA_CSA_Stack(cfg)
        self.cls_decoder = ClassificationDecoder(cfg)
        self.seg_decoder = SegmentationDecoder(cfg)
        self.det_decoder = DetectionDecoder(cfg)

    def unfreeze_backbone(self):
        self.encoder.set_frozen(False)
        print('  🔓 Backbone unfrozen')

    def forward(self, x, target_seg_size=None):
        pyr = self.wa_csa(self.fpn(self.encoder(x)))
        logits, embed = self.cls_decoder(pyr)
        seg = self.seg_decoder(pyr, embed, target_seg_size)
        det = self.det_decoder(pyr)
        return {'cls_logits': logits, 'wound_embed': embed, 'seg_mask': seg, 'det_objectness': det['objectness'], 'det_bbox': det['bbox'], 'det_cls': det['det_cls']}

    def get_router_aux_loss(self):
        return self.cls_decoder.router.aux_loss

class CheckpointManager:

    def __init__(self, save_dir, model_name='willie'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

    def save(self, model, optimizer=None, scheduler=None, epoch=0, fold=0, metrics=None, tag='latest'):
        ckpt = {'model_state_dict': model.state_dict(), 'epoch': epoch, 'fold': fold, 'metrics': metrics or {}, 'config': model.cfg.__dict__ if hasattr(model, 'cfg') else {}}
        if optimizer:
            ckpt['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
            ckpt['scheduler_state_dict'] = scheduler.state_dict()
        path = self.save_dir / f'{self.model_name}_fold{fold}_{tag}.pt'
        tmp = path.with_suffix('.tmp')
        torch.save(ckpt, tmp)
        tmp.rename(path)
        return path

    def load(self, model, fold=0, tag='best', optimizer=None, scheduler=None, device=None):
        path = self.save_dir / f'{self.model_name}_fold{fold}_{tag}.pt'
        if not path.exists():
            print(f'  ⚠️  No checkpoint at {path}')
            return None
        ckpt = torch.load(path, map_location=device or DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print(f"  ✅ Loaded: {path.name} (epoch {ckpt.get('epoch', '?')})")
        return ckpt.get('metrics', {})

def build_and_verify(variant='MINI'):
    cfg = get_model_config(variant)
    print(f"\n{'─' * 80}")
    print(f'  🏗️  Building WILLIE-{variant} (P-scSE Seg)')
    print(f"{'─' * 80}")
    model = WILLIEModel(cfg).to(DEVICE)
    total = sum((p.numel() for p in model.parameters()))
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'\n  📊 Params: {total / 1000000.0:.2f}M total, {trainable / 1000000.0:.2f}M trainable, {(total - trainable) / 1000000.0:.2f}M frozen')
    for name, mod in [('encoder', model.encoder), ('fpn', model.fpn), ('wa_csa', model.wa_csa), ('cls_decoder', model.cls_decoder), ('seg_decoder(P-scSE)', model.seg_decoder), ('det_decoder', model.det_decoder)]:
        n = sum((p.numel() for p in mod.parameters())) / 1000000.0
        print(f'    {name:30s}: {n:8.2f}M ({n * 1000000.0 / total * 100:5.1f}%)')
    B = 2
    dummy = torch.randn(B, 3, cfg.img_size, cfg.img_size, device=DEVICE)
    with torch.no_grad():
        out = model(dummy, target_seg_size=cfg.img_size)
    print(f'\n  🧪 Forward:')
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f'    {k}: {v.shape}')
    model.train()
    out = model(dummy, target_seg_size=cfg.img_size)
    loss = F.cross_entropy(out['cls_logits'], torch.randint(0, cfg.num_classes, (B,), device=DEVICE)) + F.binary_cross_entropy_with_logits(out['seg_mask'], torch.rand(B, 1, cfg.img_size, cfg.img_size, device=DEVICE)) + out['det_objectness'].mean() + model.get_router_aux_loss() * 0.01
    loss.backward()
    grads = sum((1 for p in model.parameters() if p.grad is not None))
    total_p = sum((1 for p in model.parameters()))
    print(f'  🔙 Backward: {grads}/{total_p} params have grads ✅')
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1000000000.0
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1000000000.0
        print(f'  💾 Memory: {peak:.1f}GB / {total_memory:.1f}GB ({total_memory - peak:.1f}GB free)')
        torch.cuda.reset_peak_memory_stats()
    print(f"\n{'=' * 80}")
    print(f'  ✅ Cell 1 COMPLETE — willie-{variant} ({total / 1000000.0:.1f}M) + All Manifests Loaded')
    print(f"{'=' * 80}")
    return (model, cfg)

def create_5fold_splits():
    if SPLITS_FILE.exists():
        print(f'\n  ✅ Loading saved splits from {SPLITS_FILE.name}')
        return torch.load(SPLITS_FILE, weights_only=False)
    print(f'\n  🔀 Creating {N_FOLDS}-fold stratified splits...')
    cls_labels = cls_all['unified_label'].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cls_folds = list(skf.split(np.arange(len(cls_all)), cls_labels))
    seg_idx = np.arange(len(seg_all))
    np.random.RandomState(SEED).shuffle(seg_idx)
    seg_size = len(seg_idx) // N_FOLDS
    det_idx = np.arange(len(det_all))
    np.random.RandomState(SEED + 1).shuffle(det_idx)
    det_size = len(det_idx) // N_FOLDS
    folds = {}
    for fold in range(N_FOLDS):
        cls_tr, cls_va = cls_folds[fold]
        s0 = fold * seg_size
        s1 = s0 + seg_size if fold < N_FOLDS - 1 else len(seg_idx)
        seg_va = seg_idx[s0:s1]
        seg_tr = np.concatenate([seg_idx[:s0], seg_idx[s1:]])
        d0 = fold * det_size
        d1 = d0 + det_size if fold < N_FOLDS - 1 else len(det_idx)
        det_va = det_idx[d0:d1]
        det_tr = np.concatenate([det_idx[:d0], det_idx[d1:]])
        folds[fold] = {'cls_tr': cls_tr.tolist(), 'cls_va': cls_va.tolist(), 'seg_tr': seg_tr.tolist(), 'seg_va': seg_va.tolist(), 'det_tr': det_tr.tolist(), 'det_va': det_va.tolist()}
        print(f'  Fold {fold}: cls={len(cls_tr)}/{len(cls_va)}, seg={len(seg_tr)}/{len(seg_va)}, det={len(det_tr)}/{len(det_va)}')
    data = {'n_folds': N_FOLDS, 'folds': folds}
    torch.save(data, SPLITS_FILE)
    print(f'  💾 Saved to {SPLITS_FILE.name}')
    return data

def get_train_transforms(sz=IMG_SIZE):
    return A.Compose([A.Resize(sz, sz), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.RandomRotate90(p=0.3), A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0), A.OneOf([A.ElasticTransform(alpha=30, sigma=5, p=0.3), A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3)], p=0.25), A.OneOf([A.GaussNoise(var_limit=(5.0, 30.0), p=0.3), A.GaussianBlur(blur_limit=(3, 5), p=0.3)], p=0.25), A.OneOf([A.RandomBrightnessContrast(0.2, 0.2, p=0.5), A.HueSaturationValue(10, 20, 15, p=0.4), A.CLAHE(clip_limit=2.0, p=0.3)], p=0.4), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])

def get_val_transforms(sz=IMG_SIZE):
    return A.Compose([A.Resize(sz, sz), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])

def get_tta_transforms(sz=IMG_SIZE):
    n = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return [A.Compose([A.Resize(sz, sz)] + n), A.Compose([A.Resize(sz, sz), A.HorizontalFlip(p=1.0)] + n), A.Compose([A.Resize(sz, sz), A.VerticalFlip(p=1.0)] + n), A.Compose([A.Resize(sz, sz), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + n), A.Compose([A.Resize(sz, sz), A.RandomRotate90(p=1.0)] + n)]

def parse_yolo_bbox(bbox_str):
    """'[[cls,cx,cy,w,h]]' → (N,4) as [x1,y1,x2,y2] normalized."""
    if pd.isna(bbox_str) or str(bbox_str).strip() in ('', '[]', 'nan'):
        return np.zeros((0, 4), dtype=np.float32)
    try:
        raw = ast.literal_eval(str(bbox_str))
    except:
        return np.zeros((0, 4), dtype=np.float32)
    if not raw:
        return np.zeros((0, 4), dtype=np.float32)
    out = []
    for bb in raw:
        if len(bb) >= 5:
            _, cx, cy, w, h = bb[:5]
        elif len(bb) == 4:
            cx, cy, w, h = bb
        else:
            continue
        out.append([max(0, cx - w / 2), max(0, cy - h / 2), min(1, cx + w / 2), min(1, cy + h / 2)])
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 4), dtype=np.float32)

def mask_to_bboxes(mask_np):
    """Connected components → [x1,y1,x2,y2] normalized."""
    if mask_np.max() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    binary = (mask_np > 0.5).astype(np.uint8)
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    H, W = mask_np.shape
    out = []
    for i in range(1, n_lab):
        x, y, w, h, a = stats[i]
        if a < 50:
            continue
        out.append([x / W, y / H, (x + w) / W, (y + h) / H])
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 4), dtype=np.float32)

class WoundMultiTaskDataset(Dataset):

    def __init__(self, cls_df, seg_df, det_df, transform=None, img_size=518):
        super().__init__()
        self.transform = transform
        self.img_size = img_size
        self.samples = []
        seen = set()
        if len(cls_df) > 0:
            for _, row in cls_df.iterrows():
                p = str(row['image_path'])
                self.samples.append({'path': p, 'cls_label': int(row['unified_label']), 'has_mask': False, 'mask_path': None, 'bbox_str': None})
                seen.add(os.path.normpath(p))
        seg_lookup = {}
        if len(seg_df) > 0:
            for _, row in seg_df.iterrows():
                seg_lookup[os.path.normpath(str(row['img']))] = str(row['mask'])
        if len(det_df) > 0:
            det_img_col = 'img' if 'img' in det_df.columns else 'image_path'
            det_bbox_col = 'label' if 'label' in det_df.columns else 'bbox_yolo'
            for _, row in det_df.iterrows():
                p = str(row[det_img_col])
                np_ = os.path.normpath(p)
                if np_ in seen:
                    continue
                mp = seg_lookup.get(np_)
                has_mp = mp is not None
                bbox_str = str(row[det_bbox_col]) if det_bbox_col in det_df.columns else '[]'
                self.samples.append({'path': p, 'cls_label': -1, 'has_mask': has_mp, 'mask_path': mp, 'bbox_str': bbox_str})
                seen.add(np_)
        for np_, mp in seg_lookup.items():
            if np_ not in seen:
                self.samples.append({'path': np_, 'cls_label': -1, 'has_mask': True, 'mask_path': mp, 'bbox_str': None})
                seen.add(np_)
        n_c = sum((1 for s in self.samples if s['cls_label'] >= 0))
        n_s = sum((1 for s in self.samples if s['has_mask']))
        print(f'    Dataset: {len(self.samples)} total ({n_c} cls, {n_s} seg)')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s['path'])
        if img is None:
            img = np.array(Image.open(s['path']).convert('RGB'))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if s['has_mask'] and s['mask_path']:
            mask = cv2.imread(s['mask_path'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.array(Image.open(s['mask_path']).convert('L'))
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = (aug['image'], aug['mask'])
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.float().unsqueeze(0)
        mask_np = mask.squeeze(0).numpy()
        if s['has_mask'] and mask_np.max() > 0:
            bboxes = mask_to_bboxes(mask_np)
        elif s['bbox_str']:
            bboxes = parse_yolo_bbox(s['bbox_str'])
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
        return {'image': img, 'cls_label': s['cls_label'], 'seg_mask': mask, 'det_bboxes': torch.from_numpy(bboxes), 'has_mask': s['has_mask']}

def collate_multitask(batch):
    images = torch.stack([b['image'] for b in batch])
    cls_labels = torch.tensor([b['cls_label'] for b in batch], dtype=torch.long)
    seg_masks = torch.stack([b['seg_mask'] for b in batch])
    has_mask = torch.tensor([b['has_mask'] for b in batch], dtype=torch.bool)
    max_b = max((b['det_bboxes'].shape[0] for b in batch), default=0)
    max_b = max(max_b, 1)
    det_bboxes = torch.zeros(len(batch), max_b, 4)
    det_valid = torch.zeros(len(batch), max_b, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b['det_bboxes'].shape[0]
        if n > 0:
            det_bboxes[i, :n] = b['det_bboxes']
            det_valid[i, :n] = True
    return {'image': images, 'cls_label': cls_labels, 'seg_mask': seg_masks, 'det_bboxes': det_bboxes, 'det_valid': det_valid, 'has_mask': has_mask}

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        p = torch.sigmoid(pred).flatten(1)
        t = target.flatten(1)
        inter = (p * t).sum(1)
        return 1.0 - ((2 * inter + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)).mean()

class MultiTaskLoss(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()
        self.cls_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.seg_bce = nn.BCEWithLogitsLoss()
        self.seg_dice = DiceLoss()
        self.det_obj = nn.BCEWithLogitsLoss()
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_det = nn.Parameter(torch.zeros(1))

    def forward(self, pred, batch, aux_loss=None):
        losses = {}
        dev = pred['cls_logits'].device
        labels = batch['cls_label'].to(dev)
        valid = labels >= 0
        losses['cls'] = self.cls_fn(pred['cls_logits'][valid], labels[valid]) if valid.any() else torch.tensor(0.0, device=dev)
        hm = batch['has_mask'].to(dev)
        if hm.any():
            sp, st = (pred['seg_mask'][hm], batch['seg_mask'][hm].to(dev))
            losses['seg'] = self.seg_bce(sp, st) + self.seg_dice(sp, st)
            do = pred['det_objectness'][hm]
            ot = F.interpolate(st, do.shape[-2:], mode='bilinear', align_corners=False)
            losses['det'] = self.det_obj(do, (ot > 0.3).float())
        else:
            losses['seg'] = torch.tensor(0.0, device=dev)
            losses['det'] = torch.tensor(0.0, device=dev)
        losses['aux'] = aux_loss if aux_loss is not None else torch.tensor(0.0, device=dev)
        wc, ws, wd = (torch.exp(-self.log_var_cls), torch.exp(-self.log_var_seg), torch.exp(-self.log_var_det))
        losses['total'] = wc * losses['cls'] + self.log_var_cls + ws * losses['seg'] + self.log_var_seg + wd * losses['det'] + self.log_var_det + 0.01 * losses['aux']
        losses['w_cls'], losses['w_seg'], losses['w_det'] = (wc.item(), ws.item(), wd.item())
        return losses

def compute_combined_metric(cls_acc, seg_dice, det_ap50):
    return {'cls_acc': cls_acc, 'seg_dice': seg_dice, 'det_ap50': det_ap50, 'combined_equal': (cls_acc + seg_dice + det_ap50) / 3, 'combined_weighted': 0.4 * cls_acc + 0.4 * seg_dice + 0.2 * det_ap50, 'combined_cls_seg': (cls_acc + seg_dice) / 2, 'min_task': min(cls_acc, seg_dice, det_ap50)}

def get_fold_dataloaders(fold, batch_size=BATCH_SIZE):
    f = splits['folds'][fold]
    cls_tr = cls_all.iloc[f['cls_tr']].reset_index(drop=True)
    cls_va = cls_all.iloc[f['cls_va']].reset_index(drop=True)
    seg_tr = seg_all.iloc[f['seg_tr']].reset_index(drop=True)
    seg_va = seg_all.iloc[f['seg_va']].reset_index(drop=True)
    det_tr = det_all.iloc[f['det_tr']].reset_index(drop=True)
    det_va = det_all.iloc[f['det_va']].reset_index(drop=True)
    print(f'\n  📦 Fold {fold}:')
    train_ds = WoundMultiTaskDataset(cls_tr, seg_tr, det_tr, get_train_transforms(), IMG_SIZE)
    val_ds = WoundMultiTaskDataset(cls_va, seg_va, det_va, get_val_transforms(), IMG_SIZE)
    labels = [s['cls_label'] for s in train_ds.samples if s['cls_label'] >= 0]
    if labels:
        counts = Counter(labels)
        tot = len(labels)
        cw = {c: tot / n for c, n in counts.items()}
        wts = [cw.get(s['cls_label'], 1.0) for s in train_ds.samples]
        sampler = WeightedRandomSampler(wts, len(wts), replacement=True)
    else:
        sampler = None
    tl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_multitask)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_multitask)
    print(f'    Train: {len(train_ds)} → {len(tl)} batches | Val: {len(val_ds)} → {len(vl)} batches')
    return (tl, vl)

def get_test_dataloader(batch_size=BATCH_SIZE):
    empty = pd.DataFrame()
    print(f'\n  🧪 Test set:')
    ds = WoundMultiTaskDataset(cls_test, empty, empty, get_val_transforms(), IMG_SIZE)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_multitask)
    print(f'    Test: {len(ds)} → {len(dl)} batches')
    return dl

def _atomic_save(data, path):
    """Write to .tmp then rename — survives crashes/disconnects."""
    tmp = Path(str(path) + '.tmp')
    torch.save(data, tmp)
    tmp.rename(path)

def _load_progress():
    if PROGRESS_FILE.exists():
        p = torch.load(PROGRESS_FILE, map_location='cpu', weights_only=False)
        print(f"  ↩️  Resuming: {len(p.get('completed_folds', {}))} folds already done")
        return p
    return {'completed_folds': {}, 'completed_histories': {}, 'start_time': time.time()}

class MultiTaskLossSafe(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()
        self.cls_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.seg_bce = nn.BCEWithLogitsLoss()
        self.seg_dice = DiceLoss()
        self.det_obj = nn.BCEWithLogitsLoss()
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_det = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _match_size(target, pred):
        if target.shape[-2:] != pred.shape[-2:]:
            return F.interpolate(target, pred.shape[-2:], mode='bilinear', align_corners=False)
        return target

    def forward(self, pred, batch, aux_loss=None):
        losses = {}
        dev = pred['cls_logits'].device
        labels = batch['cls_label'].to(dev)
        valid = labels >= 0
        losses['cls'] = self.cls_fn(pred['cls_logits'][valid], labels[valid]) if valid.any() else torch.tensor(0.0, device=dev)
        hm = batch['has_mask'].to(dev)
        if hm.any():
            sp = pred['seg_mask'][hm]
            st = self._match_size(batch['seg_mask'][hm].to(dev), sp)
            losses['seg'] = self.seg_bce(sp, st) + self.seg_dice(sp, st)
            do = pred['det_objectness'][hm]
            ot = F.interpolate(st, do.shape[-2:], mode='bilinear', align_corners=False)
            losses['det'] = self.det_obj(do, (ot > 0.3).float())
        else:
            losses['seg'] = torch.tensor(0.0, device=dev)
            losses['det'] = torch.tensor(0.0, device=dev)
        losses['aux'] = aux_loss if aux_loss is not None else torch.tensor(0.0, device=dev)
        wc, ws, wd = (torch.exp(-self.log_var_cls), torch.exp(-self.log_var_seg), torch.exp(-self.log_var_det))
        losses['total'] = wc * losses['cls'] + self.log_var_cls + ws * losses['seg'] + self.log_var_seg + wd * losses['det'] + self.log_var_det + 0.01 * losses['aux']
        losses['w_cls'], losses['w_seg'], losses['w_det'] = (wc.item(), ws.item(), wd.item())
        return losses

def compute_iou(box_a, box_b):
    """IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-08)

def mask_to_bboxes_eval(mask_np, min_area=50):
    """Connected components → list of [x1,y1,x2,y2] in pixel coords."""
    if mask_np.max() == 0:
        return []
    binary = (mask_np > 0.5).astype(np.uint8)
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    H, W = mask_np.shape
    boxes = []
    for i in range(1, n_lab):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        boxes.append([x / W, y / H, (x + w) / W, (y + h) / H])
    return boxes

def compute_ap50_single(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    AP@0.5 for a single image.
    pred_boxes: list of [x1,y1,x2,y2] (sorted by confidence, but we treat all equal)
    gt_boxes:   list of [x1,y1,x2,y2]
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0 and len(pred_boxes) > 0:
        return 0.0
    if len(gt_boxes) > 0 and len(pred_boxes) == 0:
        return 0.0
    matched_gt = set()
    tp, fp = (0, 0)
    for pb in pred_boxes:
        best_iou, best_j = (0, -1)
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = (iou, j)
        if best_iou >= iou_thresh and best_j >= 0:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp + 1e-08)
    recall = tp / (tp + fn + 1e-08)
    return precision * recall if precision + recall > 0 else 0.0

@torch.no_grad()
def compute_det_ap50_batch(pred_seg_logits, gt_seg_masks):
    """
    Real AP@0.5: pred seg mask → connected components → bboxes vs GT bboxes.
    Returns list of per-image AP@0.5 scores.
    """
    pred_masks = (torch.sigmoid(pred_seg_logits) > 0.5).float().cpu().numpy()
    gt_masks = gt_seg_masks.cpu().numpy()
    ap_scores = []
    for i in range(pred_masks.shape[0]):
        pm = pred_masks[i, 0]
        gm = gt_masks[i, 0]
        pred_boxes = mask_to_bboxes_eval(pm)
        gt_boxes = mask_to_bboxes_eval(gm)
        ap = compute_ap50_single(pred_boxes, gt_boxes)
        ap_scores.append(ap)
    return ap_scores

@torch.no_grad()
def compute_seg_dice_batch(pred_mask, gt_mask, threshold=0.5):
    pred = (torch.sigmoid(pred_mask) > threshold).float()
    dices = []
    for i in range(pred.shape[0]):
        p, g = (pred[i].flatten(), gt_mask[i].flatten())
        inter = (p * g).sum()
        union = p.sum() + g.sum()
        dices.append((2 * inter / (union + 1e-08)).item() if union > 0 else 1.0 if g.sum() == 0 else 0.0)
    return dices

@torch.no_grad()
def evaluate_fold(model, val_loader, criterion, seg_size, verbose=False):
    model.eval()
    all_cls_preds, all_cls_labels = ([], [])
    all_dices = []
    all_det_ap = []
    total_loss, n_bat = (0.0, 0)
    loader = tqdm(val_loader, desc='    Eval', leave=False) if verbose else val_loader
    for batch in loader:
        bg = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pred = model(bg['image'], target_seg_size=seg_size)
        losses = criterion(pred, bg, model.get_router_aux_loss())
        total_loss += losses['total'].item()
        n_bat += 1
        labels = bg['cls_label']
        valid = labels >= 0
        if valid.any():
            all_cls_preds.extend(pred['cls_logits'][valid].argmax(1).cpu().numpy())
            all_cls_labels.extend(labels[valid].cpu().numpy())
        hm = bg['has_mask']
        if hm.any():
            sp = pred['seg_mask'][hm]
            st = bg['seg_mask'][hm]
            if st.shape[-2:] != sp.shape[-2:]:
                st = F.interpolate(st, sp.shape[-2:], mode='bilinear', align_corners=False)
            all_dices.extend(compute_seg_dice_batch(sp, st))
            all_det_ap.extend(compute_det_ap50_batch(sp, st))
    cls_acc = accuracy_score(all_cls_labels, all_cls_preds) * 100 if all_cls_labels else 0.0
    cls_f1 = f1_score(all_cls_labels, all_cls_preds, average='macro') * 100 if all_cls_labels else 0.0
    seg_dice = np.mean(all_dices) * 100 if all_dices else 0.0
    det_ap50 = np.mean(all_det_ap) * 100 if all_det_ap else 0.0
    cm = compute_combined_metric(cls_acc, seg_dice, det_ap50)
    return {'cls_acc': cls_acc, 'cls_f1': cls_f1, 'seg_dice': seg_dice, 'seg_n': len(all_dices), 'det_ap50': det_ap50, 'det_n': len(all_det_ap), 'loss': total_loss / max(n_bat, 1), 'cls_preds': np.array(all_cls_preds), 'cls_labels': np.array(all_cls_labels), **cm}

def train_one_fold(fold, variant='MINI'):
    print(f"\n{'━' * 80}")
    print(f'  🏋️  FOLD {fold} — WILLIE-{variant}')
    print(f"{'━' * 80}")
    fold_cfg = get_model_config(variant)
    fold_model = WILLIEModel(fold_cfg).to(DEVICE)
    fold_criterion = MultiTaskLossSafe(NUM_CLASSES).to(DEVICE)
    fold_ckpt = CheckpointManager(CKPT_DIR / variant.lower(), f'willie_{variant.lower()}')
    train_loader, val_loader = get_fold_dataloaders(fold, TRAIN_CFG['batch_size'])
    backbone_params = list(fold_model.encoder.parameters())
    decoder_params = [p for n, p in fold_model.named_parameters() if 'encoder' not in n]
    loss_params = list(fold_criterion.parameters())
    optimizer = torch.optim.AdamW([{'params': decoder_params, 'lr': TRAIN_CFG['lr_head']}, {'params': loss_params, 'lr': TRAIN_CFG['lr_head']}, {'params': backbone_params, 'lr': TRAIN_CFG['lr_backbone']}], weight_decay=TRAIN_CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-07)
    scaler = GradScaler()
    start_epoch, best_combined, patience_ctr = (0, 0.0, 0)
    rm = fold_ckpt.load(fold_model, fold=fold, tag='latest', optimizer=optimizer, scheduler=scheduler)
    if rm:
        start_epoch = rm.get('epoch', 0) + 1
        best_combined = rm.get('best_combined', 0.0)
        patience_ctr = rm.get('patience_ctr', 0)
        if 'scaler_state' in rm:
            scaler.load_state_dict(rm['scaler_state'])
        print(f'  ↩️  Resumed fold {fold} at epoch {start_epoch}, best={best_combined:.2f}')
    epochs = TRAIN_CFG['epochs']
    accum = TRAIN_CFG['accumulation_steps']
    seg_size = TRAIN_CFG['seg_size']
    default_hist = {'train_loss': [], 'val_loss': [], 'cls_acc': [], 'cls_f1': [], 'seg_dice': [], 'det_ap50': [], 'combined': [], 'lr': [], 'cls_loss': [], 'seg_loss': [], 'det_loss': [], 'w_cls': [], 'w_seg': [], 'w_det': []}
    hist = rm.get('history', default_hist) if rm else default_hist
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        if epoch == TRAIN_CFG['freeze_epochs']:
            fold_model.unfreeze_backbone()
        fold_model.train()
        ep_loss, ep_cls, ep_seg, ep_det = (0.0, 0.0, 0.0, 0.0)
        n_steps = 0
        optimizer.zero_grad()
        last_w = {'cls': 1.0, 'seg': 1.0, 'det': 1.0}
        pbar = tqdm(train_loader, desc=f'  E{epoch:02d} Train', leave=False, bar_format='{l_bar}{bar:30}{r_bar}')
        for step, batch in enumerate(pbar):
            bg = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with autocast():
                pred = fold_model(bg['image'], target_seg_size=seg_size)
                losses = fold_criterion(pred, bg, fold_model.get_router_aux_loss())
                loss = losses['total'] / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fold_model.parameters(), TRAIN_CFG['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            ep_loss += losses['total'].item()
            ep_cls += losses['cls'].item()
            ep_seg += losses['seg'].item()
            ep_det += losses['det'].item()
            n_steps += 1
            last_w = {'cls': losses['w_cls'], 'seg': losses['w_seg'], 'det': losses['w_det']}
            pbar.set_postfix({'loss': f'{ep_loss / n_steps:.3f}', 'cls': f'{ep_cls / n_steps:.3f}', 'seg': f'{ep_seg / n_steps:.3f}', 'det': f'{ep_det / n_steps:.3f}'})
        pbar.close()
        scheduler.step()
        avg_tl = ep_loss / max(n_steps, 1)
        cur_lr = optimizer.param_groups[0]['lr']
        metrics = evaluate_fold(fold_model, val_loader, fold_criterion, seg_size, verbose=True)
        hist['train_loss'].append(avg_tl)
        hist['val_loss'].append(metrics['loss'])
        hist['cls_acc'].append(metrics['cls_acc'])
        hist['cls_f1'].append(metrics['cls_f1'])
        hist['seg_dice'].append(metrics['seg_dice'])
        hist['det_ap50'].append(metrics['det_ap50'])
        hist['combined'].append(metrics['combined_weighted'])
        hist['lr'].append(cur_lr)
        hist['cls_loss'].append(ep_cls / max(n_steps, 1))
        hist['seg_loss'].append(ep_seg / max(n_steps, 1))
        hist['det_loss'].append(ep_det / max(n_steps, 1))
        hist['w_cls'].append(last_w['cls'])
        hist['w_seg'].append(last_w['seg'])
        hist['w_det'].append(last_w['det'])
        elapsed = time.time() - t0
        star = '🔥' if metrics['combined_weighted'] > best_combined else '  '
        phase = 'FROZEN' if epoch < TRAIN_CFG['freeze_epochs'] else 'FULL'
        print(f"  {star} E{epoch:02d} [{phase}] [{elapsed:.0f}s] lr={cur_lr:.1e} | t_loss={avg_tl:.3f} v_loss={metrics['loss']:.3f} | cls={metrics['cls_acc']:.1f}% f1={metrics['cls_f1']:.1f}% seg={metrics['seg_dice']:.1f}% det_AP50={metrics['det_ap50']:.1f}% | comb={metrics['combined_weighted']:.1f}% | w[{last_w['cls']:.2f}/{last_w['seg']:.2f}/{last_w['det']:.2f}]")
        save_metrics = {k: v for k, v in metrics.items() if k not in ('cls_preds', 'cls_labels')}
        fold_ckpt.save(fold_model, optimizer, scheduler, epoch, fold, {'epoch': epoch, 'best_combined': max(best_combined, metrics['combined_weighted']), 'patience_ctr': patience_ctr, 'scaler_state': scaler.state_dict(), 'history': hist, **save_metrics}, tag='latest')
        if metrics['combined_weighted'] > best_combined:
            best_combined = metrics['combined_weighted']
            patience_ctr = 0
            fold_ckpt.save(fold_model, optimizer, scheduler, epoch, fold, {'epoch': epoch, 'best_combined': best_combined, **save_metrics}, tag='best')
            print(f'       💾 New best: {best_combined:.2f}%')
        else:
            patience_ctr += 1
        if patience_ctr >= TRAIN_CFG['patience']:
            print(f'  ⏹️  Early stopping at epoch {epoch}')
            break
    fold_ckpt.load(fold_model, fold=fold, tag='best')
    final = evaluate_fold(fold_model, val_loader, fold_criterion, seg_size)
    print(f"\n  ✅ Fold {fold} Best: cls={final['cls_acc']:.1f}% seg={final['seg_dice']:.1f}% det_AP50={final['det_ap50']:.1f}% comb={final['combined_weighted']:.1f}%")
    del fold_model, optimizer, scheduler, scaler, train_loader, val_loader
    torch.cuda.empty_cache()
    return (final, hist)

def _atomic_save_test(data, path):
    tmp = Path(str(path) + '.tmp')
    torch.save(data, tmp)
    tmp.rename(path)

class TTADataset(Dataset):

    def __init__(self, cls_df, seg_df=None, img_size=518):
        self.samples = []
        self.img_size = img_size
        if cls_df is not None and len(cls_df) > 0:
            for _, row in cls_df.iterrows():
                self.samples.append({'path': str(row['image_path']), 'cls_label': int(row['unified_label']), 'has_mask': False, 'mask_path': None})
        if seg_df is not None and len(seg_df) > 0:
            for _, row in seg_df.iterrows():
                self.samples.append({'path': str(row['img']), 'cls_label': -1, 'has_mask': True, 'mask_path': str(row['mask'])})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s['path'])
        if img is None:
            img = np.array(Image.open(s['path']).convert('RGB'))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = None
        if s['has_mask'] and s['mask_path']:
            mask = cv2.imread(s['mask_path'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.array(Image.open(s['mask_path']).convert('L'))
            mask = (mask > 127).astype(np.float32)
        return {'image_raw': img, 'mask_raw': mask, 'cls_label': s['cls_label'], 'has_mask': s['has_mask']}

def reverse_tta_mask(mask_tensor, view_idx):
    """
    Reverse TTA augmentation on predicted seg mask.
    mask_tensor: (1, 1, H, W)
    view_idx: 0=original, 1=hflip, 2=vflip, 3=hflip+vflip, 4=rot90
    """
    if view_idx == 0:
        return mask_tensor
    elif view_idx == 1:
        return torch.flip(mask_tensor, dims=[-1])
    elif view_idx == 2:
        return torch.flip(mask_tensor, dims=[-2])
    elif view_idx == 3:
        return torch.flip(mask_tensor, dims=[-2, -1])
    elif view_idx == 4:
        return torch.rot90(mask_tensor, k=-1, dims=[-2, -1])
    return mask_tensor

@torch.no_grad()
def tta_predict_cls(model, img_raw, transforms):
    """Apply N TTA views, average logits."""
    logits_list = []
    for tfm in transforms:
        aug = tfm(image=img_raw)
        img_t = aug['image'].unsqueeze(0).to(DEVICE)
        pred = model(img_t, target_seg_size=seg_size)
        logits_list.append(pred['cls_logits'].cpu())
    return torch.stack(logits_list, 0).mean(0).squeeze(0)

@torch.no_grad()
def tta_predict_seg(model, img_raw, mask_raw, transforms):
    """
    Apply TTA views, REVERSE transform on predicted mask, THEN average.
    This is the critical fix — masks must be in original orientation before averaging.
    """
    seg_preds = []
    for view_idx, tfm in enumerate(transforms):
        aug = tfm(image=img_raw, mask=mask_raw)
        img_t = aug['image'].unsqueeze(0).to(DEVICE)
        pred = model(img_t, target_seg_size=seg_size)
        seg_logit = pred['seg_mask'].cpu()
        seg_reversed = reverse_tta_mask(seg_logit, view_idx)
        seg_preds.append(seg_reversed)
    avg_seg = torch.stack(seg_preds, 0).mean(0)
    gt = torch.from_numpy(mask_raw).float().unsqueeze(0).unsqueeze(0)
    if gt.shape[-2:] != avg_seg.shape[-2:]:
        gt = F.interpolate(gt, avg_seg.shape[-2:], mode='bilinear', align_corners=False)
    dice_list = compute_seg_dice_batch(avg_seg, gt)
    ap_list = compute_det_ap50_batch(avg_seg, gt)
    return (dice_list[0], ap_list[0])

def _atomic_save_tta(data, path):
    tmp = Path(str(path) + '.tmp')
    torch.save(data, tmp)
    tmp.rename(path)