# ============================================================================
#  WILLIE BASE - FINAL ABLATION (single file, single run)
#  Proves components work COLLECTIVELY, not individually:
#    full          - all components
#    no_F2DCA      - remove fusion only     -> expect within noise
#    no_WACSA      - remove WA-CSA only      -> expect within noise
#    no_MoE        - remove MoE only         -> expect within noise
#    no_WTCS       - remove WTCS only        -> expect within noise
#    backbone_only - remove ALL components   -> expect big drop = COLLECTIVE WIN
#  Output: ablation_final/  (self-contained). Recipe identical to paper BASE.
# ============================================================================
#!/usr/bin/env python
"""
================================================================================
  base_ablation.py  -  WILLIE-BASE Component Ablation (standalone)
  Configs: full / no_F2DCA / no_WACSA / no_MoE / no_WTCS   x   5 folds
  Saves to: ablation/base/ckpts , ablation/results/base_results.json
  Atomic per-config + per-fold resume. Run via sbatch.

  BASE = dual backbone (DINOv2-ViT-L + ConvNeXt-Large) + F2DCA fusion
         + FPN + WA-CSA x2 + MoE-4 + P-scSE/FiLM seg + det.
  Multi-task dataset (seg active) -> WTCS ablatable.
================================================================================
"""
import os, sys, time, json, ast, warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from collections import Counter

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image
from einops import rearrange
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

# ==============================================================================
#  PATHS + DEVICE + SEED
# ==============================================================================
ROOT = Path(".")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

CLS_MANIFEST_DIR = ROOT / "artifacts" / "woundshot_v2" / "manifests"
LOCKED_DIR       = ROOT / "artifacts" / "WoundShot_LOCKED_INPUTS" / "tables"
# reuse the SAME 5-fold splits as MINI/BASE training (consistent folds)
SPLITS_FILE      = ROOT / "artifacts" / "10_fuseg_csd_base" / "5fold_splits_v2.pt"
if not SPLITS_FILE.exists():
    SPLITS_FILE  = ROOT / "artifacts" / "09_fuseg_csd" / "5fold_splits_v2.pt"

ABL_DIR     = ROOT / "ablation_final" / "base" / "ckpts"
ABL_RESULTS = ROOT / "ablation_final" / "results" / "base_final_results.json"
ABL_DIR.mkdir(parents=True, exist_ok=True)
ABL_RESULTS.parent.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["diabetic", "no_wound", "pressure", "surgical", "venous"]
NUM_CLASSES = 5
IMG_SIZE    = 518

# BASE real training config (from notebook 10 Cell 3)
ABL_CFG = {"epochs":50, "freeze_epochs":3, "lr_head":1e-4, "backbone_lr_scale":0.05,
           "weight_decay":1e-4, "patience":12, "grad_clip":1.0,
           "accumulation_steps":4, "seg_size":512, "batch_size":2, "num_workers":2}


import numpy as _np
def _to_native(o):
    if isinstance(o, dict):  return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, list):  return [_to_native(v) for v in o]
    if isinstance(o, (_np.floating,)): return float(o)
    if isinstance(o, (_np.integer,)):  return int(o)
    if isinstance(o, _np.ndarray):     return o.tolist()
    return o


def log(msg): print(msg, flush=True)

log(f"{'='*80}\n  base_ablation.py  |  {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}")
if torch.cuda.is_available():
    log(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
log(f"  ckpts   -> {ABL_DIR}")
log(f"  results -> {ABL_RESULTS}")

# ==============================================================================
#  CONFIG
# ==============================================================================
@dataclass
class BaseModelConfig:
    name: str = "BASE"
    dino_name: str = "dinov2_vitl14"
    dino_dim: int = 1024
    dino_layers: List[int] = field(default_factory=lambda: [5, 11, 17, 23])
    convnext_name: str = "convnext_large.fb_in22k_ft_in1k"
    convnext_dims: List[int] = field(default_factory=lambda: [192, 384, 768, 1536])
    img_size: int = 518
    patch_size: int = 14
    fpn_dim: int = 256
    fpn_levels: int = 4
    f2dca_heads: int = 8
    f2dca_layers: int = 2
    f2dca_dropout: float = 0.1
    wa_csa_layers: int = 2
    wa_csa_heads: int = 8
    wa_csa_dropout: float = 0.1
    num_classes: int = 5
    num_experts: int = 4
    top_k_experts: int = 2
    cls_embed_dim: int = 128
    seg_out_channels: int = 1
    pscse_reduction: int = 16
    det_max_objects: int = 20
    freeze_backbone_epochs: int = 3
    @property
    def grid_size(self) -> int:
        return self.img_size // self.patch_size

# ==============================================================================
#  BACKBONES
# ==============================================================================
class DINOv2MultiScale(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg; self.features = {}
        self.backbone = torch.hub.load("facebookresearch/dinov2", cfg.dino_name, pretrained=True)
        self._hooks = []
        for idx in cfg.dino_layers:
            self._hooks.append(self.backbone.blocks[idx].register_forward_hook(self._mk(idx)))
        self.set_frozen(True)
    def _mk(self, idx):
        def hook(m, i, o):
            t = o[:, 1:, :]; G = self.cfg.grid_size
            self.features[idx] = rearrange(t, "b (h w) d -> b d h w", h=G, w=G)
        return hook
    def set_frozen(self, fr):
        for p in self.backbone.parameters(): p.requires_grad = not fr
    def forward(self, x):
        self.features.clear()
        if x.shape[-1] != self.cfg.img_size:
            x = F.interpolate(x, self.cfg.img_size, mode="bilinear", align_corners=False)
        self.backbone(x)
        return [self.features[idx] for idx in self.cfg.dino_layers]

class ConvNeXtMultiScale(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.backbone = timm.create_model(cfg.convnext_name, pretrained=True, features_only=True)
        self.set_frozen(True)
    def set_frozen(self, fr):
        for p in self.backbone.parameters(): p.requires_grad = not fr
    def forward(self, x):
        if x.shape[-1] != self.cfg.img_size:
            x = F.interpolate(x, self.cfg.img_size, mode="bilinear", align_corners=False)
        return self.backbone(x)[:self.cfg.fpn_levels]

# ==============================================================================
#  F2DCA
# ==============================================================================
class F2DCA_Layer(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads; self.head_dim = dim // heads; self.scale = self.head_dim ** -0.5
        self.q_a=nn.Linear(dim,dim,bias=False); self.k_b=nn.Linear(dim,dim,bias=False); self.v_b=nn.Linear(dim,dim,bias=False); self.out_a=nn.Linear(dim,dim,bias=False)
        self.q_b=nn.Linear(dim,dim,bias=False); self.k_a=nn.Linear(dim,dim,bias=False); self.v_a=nn.Linear(dim,dim,bias=False); self.out_b=nn.Linear(dim,dim,bias=False)
        self.norm_a=nn.LayerNorm(dim); self.norm_b=nn.LayerNorm(dim)
        self.alpha_a=nn.Parameter(torch.zeros(1)); self.alpha_b=nn.Parameter(torch.zeros(1)); self.drop=nn.Dropout(dropout)
    def _cross(self, qp, kp, vp, op, xq, xkv):
        q=rearrange(qp(xq),"b n (h d) -> b h n d",h=self.heads); k=rearrange(kp(xkv),"b n (h d) -> b h n d",h=self.heads); v=rearrange(vp(xkv),"b n (h d) -> b h n d",h=self.heads)
        if hasattr(F,'scaled_dot_product_attention'):
            out=F.scaled_dot_product_attention(q,k,v,dropout_p=self.drop.p if self.training else 0.0)
        else:
            a=(q@k.transpose(-2,-1))*self.scale; out=self.drop(a.softmax(-1))@v
        return op(rearrange(out,"b h n d -> b n (h d)"))
    def forward(self, fa, fb):
        B,C,H,W = fa.shape
        a=self.norm_a(rearrange(fa,"b c h w -> b (h w) c")); b=self.norm_b(rearrange(fb,"b c h w -> b (h w) c"))
        au=rearrange(self._cross(self.q_a,self.k_b,self.v_b,self.out_a,a,b),"b (h w) c -> b c h w",h=H)
        bu=rearrange(self._cross(self.q_b,self.k_a,self.v_a,self.out_b,b,a),"b (h w) c -> b c h w",h=H)
        return fa+torch.tanh(self.alpha_a)*au, fb+torch.tanh(self.alpha_b)*bu

class F2DCA_Stack(nn.Module):
    def __init__(self, cfg):
        super().__init__(); dim=cfg.fpn_dim
        self.layers=nn.ModuleList([nn.ModuleList([F2DCA_Layer(dim,cfg.f2dca_heads,cfg.f2dca_dropout) for _ in range(cfg.fpn_levels)]) for _ in range(cfg.f2dca_layers)])
    def forward(self, dino_pyr, conv_pyr):
        d,c=list(dino_pyr),list(conv_pyr)
        for layer in self.layers:
            for lvl,f2dca in enumerate(layer):
                d[lvl],c[lvl]=f2dca(d[lvl],c[lvl])
        return d,c

# ==============================================================================
#  DUAL FPN
# ==============================================================================
class DualBackboneFPN(nn.Module):
    def __init__(self, cfg):
        super().__init__(); Fd=cfg.fpn_dim
        self.dino_lats=nn.ModuleList([nn.Sequential(nn.Conv2d(cfg.dino_dim,Fd,1,bias=False),nn.GroupNorm(32,Fd),nn.GELU()) for _ in range(cfg.fpn_levels)])
        self.conv_lats=nn.ModuleList([nn.Sequential(nn.Conv2d(cfg.convnext_dims[i],Fd,1,bias=False),nn.GroupNorm(32,Fd),nn.GELU()) for i in range(cfg.fpn_levels)])
        self.f2dca=F2DCA_Stack(cfg)
        self.merge_w=nn.ParameterList([nn.Parameter(torch.tensor([0.5,0.5])) for _ in range(cfg.fpn_levels)])
        self.smooth=nn.ModuleList([nn.Sequential(nn.Conv2d(Fd,Fd,3,padding=1,bias=False),nn.GroupNorm(32,Fd),nn.GELU()) for _ in range(cfg.fpn_levels)])
        self._ablate_f2dca = False   # ablation flag
    def forward(self, dino_feats, conv_feats):
        d_proj=[self.dino_lats[i](dino_feats[i]) for i in range(len(dino_feats))]
        c_proj=[]
        for i in range(len(conv_feats)):
            c=self.conv_lats[i](conv_feats[i])
            if c.shape[-2:]!=d_proj[i].shape[-2:]:
                c=F.interpolate(c,d_proj[i].shape[-2:],mode="bilinear",align_corners=False)
            c_proj.append(c)
        if self._ablate_f2dca:
            d_fused, c_fused = d_proj, c_proj   # skip cross-attention fusion
        else:
            d_fused,c_fused=self.f2dca(d_proj,c_proj)
        merged=[]
        for i in range(len(d_fused)):
            w=F.softmax(self.merge_w[i],0)
            merged.append(w[0]*d_fused[i]+w[1]*c_fused[i])
        pyr=[None]*len(merged); pyr[-1]=merged[-1]
        for i in range(len(merged)-2,-1,-1):
            up=F.interpolate(pyr[i+1],merged[i].shape[-2:],mode="bilinear",align_corners=False)
            pyr[i]=merged[i]+up
        return [self.smooth[i](pyr[i]) for i in range(len(pyr))]

# ==============================================================================
#  WA-CSA
# ==============================================================================
class WoundAwareCrossScaleAttention(nn.Module):
    def __init__(self, cfg, level_idx=0):
        super().__init__(); dim=cfg.fpn_dim
        self.heads=cfg.wa_csa_heads; self.head_dim=dim//self.heads; self.scale=self.head_dim**-0.5
        self.use_sdpa=(level_idx>0); self.dropout_p=cfg.wa_csa_dropout
        self.q_f=nn.Linear(dim,dim,bias=False); self.k_c=nn.Linear(dim,dim,bias=False); self.v_c=nn.Linear(dim,dim,bias=False)
        self.q_c=nn.Linear(dim,dim,bias=False); self.k_f=nn.Linear(dim,dim,bias=False); self.v_f=nn.Linear(dim,dim,bias=False)
        self.out_f=nn.Linear(dim,dim,bias=False); self.out_c=nn.Linear(dim,dim,bias=False)
        self.gate_f=nn.Sequential(nn.Conv2d(dim,dim//4,1),nn.GELU(),nn.Conv2d(dim//4,1,1),nn.Sigmoid())
        self.gate_c=nn.Sequential(nn.Conv2d(dim,dim//4,1),nn.GELU(),nn.Conv2d(dim//4,1,1),nn.Sigmoid())
        self.alpha_f=nn.Parameter(torch.zeros(1)); self.alpha_c=nn.Parameter(torch.zeros(1))
        self.norm_f=nn.LayerNorm(dim); self.norm_c=nn.LayerNorm(dim); self.drop=nn.Dropout(cfg.wa_csa_dropout)
    def _attn(self,q,k,v):
        q=rearrange(q,"b n (h d) -> b h n d",h=self.heads); k=rearrange(k,"b n (h d) -> b h n d",h=self.heads); v=rearrange(v,"b n (h d) -> b h n d",h=self.heads)
        if self.use_sdpa and hasattr(F,'scaled_dot_product_attention'):
            out=F.scaled_dot_product_attention(q,k,v,dropout_p=self.dropout_p if self.training else 0.0)
        else:
            a=(q@k.transpose(-2,-1))*self.scale; out=self.drop(a.softmax(-1))@v
        return rearrange(out,"b h n d -> b n (h d)")
    def forward(self, fine, coarse):
        B,C,H,W=fine.shape
        f=self.norm_f(rearrange(fine,"b c h w -> b (h w) c")); c=self.norm_c(rearrange(coarse,"b c h w -> b (h w) c"))
        f_up=rearrange(self.out_f(self._attn(self.q_f(f),self.k_c(c),self.v_c(c))),"b (h w) c -> b c h w",h=H)
        c_up=rearrange(self.out_c(self._attn(self.q_c(c),self.k_f(f),self.v_f(f))),"b (h w) c -> b c h w",h=H)
        return (fine+torch.tanh(self.alpha_f)*f_up*self.gate_f(fine), coarse+torch.tanh(self.alpha_c)*c_up*self.gate_c(coarse))

class WA_CSA_Stack(nn.Module):
    def __init__(self, cfg):
        super().__init__(); pairs=cfg.fpn_levels-1
        self.layers=nn.ModuleList([nn.ModuleList([WoundAwareCrossScaleAttention(cfg,p) for p in range(pairs)]) for _ in range(cfg.wa_csa_layers)])
    def forward(self, pyr):
        for layer in self.layers:
            p=list(pyr)
            for i,wa in enumerate(layer): p[i],p[i+1]=wa(p[i],p[i+1])
            pyr=p
        return pyr

# ==============================================================================
#  DECODERS
# ==============================================================================
class Expert(nn.Module):
    def __init__(self,d_in,d_hid,d_out):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d_in,d_hid),nn.GELU(),nn.Dropout(0.1),nn.Linear(d_hid,d_out))
    def forward(self,x): return self.net(x)
class TopKRouter(nn.Module):
    def __init__(self,d_in,n_exp,top_k=2):
        super().__init__(); self.n_exp=n_exp; self.top_k=top_k; self.gate=nn.Linear(d_in,n_exp,bias=False); self.aux_loss=0.0
    def forward(self,x):
        probs=F.softmax(self.gate(x),-1); w,idx=torch.topk(probs,self.top_k,-1); w=w/w.sum(-1,keepdim=True)
        self.aux_loss=F.mse_loss(probs.mean(0),torch.ones(self.n_exp,device=x.device)/self.n_exp)
        return w,idx
class ClassificationDecoder(nn.Module):
    def __init__(self,cfg):
        super().__init__(); Fd=cfg.fpn_dim; self.cfg=cfg
        self.pools=nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(1)) for _ in range(cfg.fpn_levels)])
        self.scale_attn=nn.Sequential(nn.Linear(cfg.fpn_levels,cfg.fpn_levels),nn.Softmax(-1))
        self.pre=nn.Sequential(nn.Linear(Fd,Fd),nn.LayerNorm(Fd),nn.GELU())
        self.experts=nn.ModuleList([Expert(Fd,Fd*2,cfg.num_classes) for _ in range(cfg.num_experts)])
        self.router=TopKRouter(Fd,cfg.num_experts,cfg.top_k_experts)
        self.embed_head=nn.Sequential(nn.Linear(Fd,cfg.cls_embed_dim),nn.LayerNorm(cfg.cls_embed_dim),nn.GELU())
        self._ablate_moe=False
    def forward(self,pyr):
        B=pyr[0].shape[0]
        pooled=torch.stack([self.pools[i](pyr[i]) for i in range(self.cfg.fpn_levels)],1)
        sw=self.scale_attn(torch.ones(B,self.cfg.fpn_levels,device=pooled.device))
        fused=self.pre((pooled*sw.unsqueeze(-1)).sum(1))
        if self._ablate_moe:
            self.router.aux_loss=torch.tensor(0.0,device=fused.device)
            return self.experts[0](fused), self.embed_head(fused)
        w,idx=self.router(fused)
        logits=torch.zeros(B,self.cfg.num_classes,device=fused.device)
        for k in range(self.cfg.top_k_experts):
            for e in range(self.cfg.num_experts):
                mask=(idx[:,k]==e)
                if mask.any(): logits[mask]+=w[mask,k:k+1]*self.experts[e](fused[mask])
        return logits, self.embed_head(fused)

class ChannelSE(nn.Module):
    def __init__(self,ch,r=16):
        super().__init__(); mid=max(ch//r,4)
        self.fc=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(1),nn.Linear(ch,mid,bias=False),nn.ReLU(True),nn.Linear(mid,ch,bias=False),nn.Sigmoid())
    def forward(self,x): return x*self.fc(x).view(x.shape[0],-1,1,1)
class SpatialSE(nn.Module):
    def __init__(self,ch):
        super().__init__(); self.conv=nn.Conv2d(ch,1,1,bias=False)
    def forward(self,x): return x*torch.sigmoid(self.conv(x))
class ParallelScSE(nn.Module):
    def __init__(self,ch,r=16,shortened=False):
        super().__init__(); self.shortened=shortened
        self.cse_a,self.sse_a=ChannelSE(ch,r),SpatialSE(ch)
        if not shortened: self.cse_m,self.sse_m=ChannelSE(ch,r),SpatialSE(ch)
    def forward(self,x):
        add=self.cse_a(x)+self.sse_a(x)
        return add if self.shortened else add+torch.max(self.cse_m(x),self.sse_m(x))
class FiLMConditioner(nn.Module):
    def __init__(self,embed_dim,feat_dim):
        super().__init__(); self.gamma=nn.Sequential(nn.Linear(embed_dim,feat_dim),nn.Sigmoid()); self.beta=nn.Linear(embed_dim,feat_dim)
    def forward(self,feat,embed):
        g=self.gamma(embed).unsqueeze(-1).unsqueeze(-1)+1.0; b=self.beta(embed).unsqueeze(-1).unsqueeze(-1)
        return g*feat+b
class PscSEDecoderStage(nn.Module):
    def __init__(self,in_ch,out_ch,r=16,shortened=False):
        super().__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_ch,out_ch,3,padding=1,bias=False),nn.BatchNorm2d(out_ch),nn.ReLU(True))
        self.pscse=ParallelScSE(out_ch,r,shortened)
        self.conv2=nn.Sequential(nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False),nn.BatchNorm2d(out_ch),nn.ReLU(True))
    def forward(self,x): return self.conv2(self.pscse(self.conv1(x)))
class SegmentationDecoder(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; Fd=cfg.fpn_dim
        self.films=nn.ModuleList([FiLMConditioner(cfg.cls_embed_dim,Fd) for _ in range(cfg.fpn_levels)])
        self.stages=nn.ModuleList([PscSEDecoderStage(Fd*2,Fd,cfg.pscse_reduction,shortened=(i==cfg.fpn_levels-2)) for i in range(cfg.fpn_levels-1)])
        self.head=nn.Sequential(nn.Conv2d(Fd,Fd//2,3,padding=1,bias=False),nn.BatchNorm2d(Fd//2),nn.ReLU(True),
                                nn.Conv2d(Fd//2,Fd//4,3,padding=1,bias=False),nn.BatchNorm2d(Fd//4),nn.ReLU(True),
                                nn.Conv2d(Fd//4,cfg.seg_out_channels,1))
        self._ablate_wtcs=False
    def forward(self,pyr,wound_embed,target_size=None):
        if target_size is None: target_size=self.cfg.img_size
        if self._ablate_wtcs:
            cond=[pyr[i] for i in range(self.cfg.fpn_levels)]   # skip FiLM conditioning
        else:
            cond=[self.films[i](pyr[i],wound_embed) for i in range(self.cfg.fpn_levels)]
        x=cond[-1]
        for s in range(self.cfg.fpn_levels-1):
            skip=cond[self.cfg.fpn_levels-2-s]
            x=F.interpolate(x,skip.shape[-2:],mode="bilinear",align_corners=False)
            x=self.stages[s](torch.cat([x,skip],1))
        x=F.interpolate(x,size=target_size,mode="bilinear",align_corners=False)
        return self.head(x)
class DetectionDecoder(nn.Module):
    def __init__(self,cfg):
        super().__init__(); Fd=cfg.fpn_dim
        self.level_w=nn.Parameter(torch.ones(cfg.fpn_levels)/cfg.fpn_levels)
        self.shared=nn.Sequential(nn.Conv2d(Fd,Fd,3,padding=1,bias=False),nn.GroupNorm(32,Fd),nn.GELU(),
                                  nn.Conv2d(Fd,Fd,3,padding=1,bias=False),nn.GroupNorm(32,Fd),nn.GELU())
        self.obj=nn.Conv2d(Fd,1,1)
        self.bbox=nn.Sequential(nn.Conv2d(Fd,Fd//2,3,padding=1),nn.GELU(),nn.Conv2d(Fd//2,4,1),nn.Sigmoid())
        self.cls=nn.Conv2d(Fd,cfg.num_classes,1)
    def forward(self,pyr):
        w=F.softmax(self.level_w,0); tgt=pyr[0].shape[-2:]
        fused=sum(w[i]*(F.interpolate(p,tgt,mode="bilinear",align_corners=False) if p.shape[-2:]!=tgt else p) for i,p in enumerate(pyr))
        feat=self.shared(fused)
        return {"objectness":self.obj(feat),"bbox":self.bbox(feat),"det_cls":self.cls(feat)}

class WILLIEBASE(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.encoder_dino=DINOv2MultiScale(cfg); self.encoder_conv=ConvNeXtMultiScale(cfg)
        self.fpn=DualBackboneFPN(cfg); self.wa_csa=WA_CSA_Stack(cfg)
        self.cls_decoder=ClassificationDecoder(cfg); self.seg_decoder=SegmentationDecoder(cfg); self.det_decoder=DetectionDecoder(cfg)
    def unfreeze_backbones(self):
        self.encoder_dino.set_frozen(False); self.encoder_conv.set_frozen(False); log("  backbones unfrozen")
    def forward(self,x,target_seg_size=None):
        dino_feats=self.encoder_dino(x); conv_feats=self.encoder_conv(x)
        pyr=self.wa_csa(self.fpn(dino_feats,conv_feats))
        logits,embed=self.cls_decoder(pyr)
        seg=self.seg_decoder(pyr,embed,target_seg_size); det=self.det_decoder(pyr)
        return {"cls_logits":logits,"wound_embed":embed,"seg_mask":seg,
                "det_objectness":det["objectness"],"det_bbox":det["bbox"],"det_cls":det["det_cls"]}
    def get_router_aux_loss(self): return self.cls_decoder.router.aux_loss

# ==============================================================================
#  CHECKPOINT MANAGER (atomic)
# ==============================================================================
class CheckpointManager:
    def __init__(self, save_dir, model_name="woundshot"):
        self.save_dir=Path(save_dir); self.save_dir.mkdir(parents=True,exist_ok=True); self.model_name=model_name
    def save(self, model, optimizer=None, scheduler=None, epoch=0, fold=0, metrics=None, tag="latest"):
        ck={"model_state_dict":model.state_dict(),"epoch":epoch,"fold":fold,"metrics":metrics or {}}
        if optimizer: ck["optimizer_state_dict"]=optimizer.state_dict()
        if scheduler: ck["scheduler_state_dict"]=scheduler.state_dict()
        path=self.save_dir/f"{self.model_name}_fold{fold}_{tag}.pt"; tmp=path.with_suffix(".tmp")
        torch.save(ck,tmp); tmp.rename(path); return path
    def load(self, model, fold=0, tag="best", optimizer=None, scheduler=None):
        path=self.save_dir/f"{self.model_name}_fold{fold}_{tag}.pt"
        if not path.exists(): return None
        ck=torch.load(path,map_location=DEVICE,weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ck: optimizer.load_state_dict(ck["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ck: scheduler.load_state_dict(ck["scheduler_state_dict"])
        log(f"  loaded {path.name} (epoch {ck.get('epoch','?')})")
        return ck.get("metrics", {})

# ==============================================================================
#  DATA  (multi-task, MINI-style -> seg active)
# ==============================================================================
log("\n  Loading manifests...")
M = {n: pd.read_csv(p) for n,p in {
    "cls_train":CLS_MANIFEST_DIR/"cls_train.csv","cls_val":CLS_MANIFEST_DIR/"cls_val.csv","cls_test":CLS_MANIFEST_DIR/"cls_test.csv",
    "seg_train":LOCKED_DIR/"ws_seg_manifest_fuseg_train.csv","seg_val":LOCKED_DIR/"ws_seg_manifest_fuseg_val.csv",
    "det_train":LOCKED_DIR/"ws_det_manifest_yolo_train.csv","det_val":LOCKED_DIR/"ws_det_manifest_yolo_val.csv"}.items()}
cls_all=pd.concat([M["cls_train"],M["cls_val"]],ignore_index=True)
seg_all=pd.concat([M["seg_train"],M["seg_val"]],ignore_index=True)
det_all=pd.concat([M["det_train"],M["det_val"]],ignore_index=True)
log(f"  cls={len(cls_all)} seg={len(seg_all)} det={len(det_all)}")
splits=torch.load(SPLITS_FILE,weights_only=False); log(f"  loaded splits: {SPLITS_FILE}")

IM_MEAN=[0.485,0.456,0.406]; IM_STD=[0.229,0.224,0.225]
def tf_train(sz=IMG_SIZE):
    return A.Compose([A.Resize(sz,sz),A.HorizontalFlip(p=0.5),A.VerticalFlip(p=0.3),A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1,scale_limit=0.15,rotate_limit=30,p=0.5,border_mode=cv2.BORDER_CONSTANT,value=0),
        A.OneOf([A.ElasticTransform(alpha=30,sigma=5,p=0.3),A.GridDistortion(num_steps=5,distort_limit=0.3,p=0.3)],p=0.25),
        A.OneOf([A.GaussNoise(var_limit=(5.0,30.0),p=0.3),A.GaussianBlur(blur_limit=(3,5),p=0.3)],p=0.25),
        A.OneOf([A.RandomBrightnessContrast(0.2,0.2,p=0.5),A.HueSaturationValue(10,20,15,p=0.4),A.CLAHE(clip_limit=2.0,p=0.3)],p=0.4),
        A.Normalize(mean=IM_MEAN,std=IM_STD),ToTensorV2()])
def tf_val(sz=IMG_SIZE):
    return A.Compose([A.Resize(sz,sz),A.Normalize(mean=IM_MEAN,std=IM_STD),ToTensorV2()])
def parse_yolo(s):
    if pd.isna(s) or str(s).strip() in ("","[]","nan"): return np.zeros((0,4),dtype=np.float32)
    try: raw=ast.literal_eval(str(s))
    except: return np.zeros((0,4),dtype=np.float32)
    if not raw: return np.zeros((0,4),dtype=np.float32)
    out=[]
    for bb in raw:
        if len(bb)>=5: _,cx,cy,w,h=bb[:5]
        elif len(bb)==4: cx,cy,w,h=bb
        else: continue
        out.append([max(0,cx-w/2),max(0,cy-h/2),min(1,cx+w/2),min(1,cy+h/2)])
    return np.array(out,dtype=np.float32) if out else np.zeros((0,4),dtype=np.float32)
def mask2bbox(m):
    if m.max()==0: return np.zeros((0,4),dtype=np.float32)
    b=(m>0.5).astype(np.uint8); n,_,st,_=cv2.connectedComponentsWithStats(b,8); H,W=m.shape; out=[]
    for i in range(1,n):
        x,y,w,h,a=st[i]
        if a<50: continue
        out.append([x/W,y/H,(x+w)/W,(y+h)/H])
    return np.array(out,dtype=np.float32) if out else np.zeros((0,4),dtype=np.float32)

class WoundDS(Dataset):
    def __init__(self, cls_df, seg_df, det_df, transform=None, img_size=518):
        super().__init__(); self.transform=transform; self.img_size=img_size; self.samples=[]; seen=set()
        if len(cls_df)>0:
            for _,r in cls_df.iterrows():
                p=str(r["image_path"]); self.samples.append({"path":p,"cls_label":int(r["unified_label"]),"has_mask":False,"mask_path":None,"bbox_str":None}); seen.add(os.path.normpath(p))
        seg_lookup={}
        if len(seg_df)>0:
            for _,r in seg_df.iterrows(): seg_lookup[os.path.normpath(str(r["img"]))]=str(r["mask"])
        if len(det_df)>0:
            dic="img" if "img" in det_df.columns else "image_path"; dbc="label" if "label" in det_df.columns else "bbox_yolo"
            for _,r in det_df.iterrows():
                p=str(r[dic]); npn=os.path.normpath(p)
                if npn in seen: continue
                mp=seg_lookup.get(npn); bs=str(r[dbc]) if dbc in det_df.columns else "[]"
                self.samples.append({"path":p,"cls_label":-1,"has_mask":mp is not None,"mask_path":mp,"bbox_str":bs}); seen.add(npn)
        for npn,mp in seg_lookup.items():
            if npn not in seen:
                self.samples.append({"path":npn,"cls_label":-1,"has_mask":True,"mask_path":mp,"bbox_str":None}); seen.add(npn)
        n_c=sum(1 for s in self.samples if s["cls_label"]>=0); n_s=sum(1 for s in self.samples if s["has_mask"])
        log(f"    Dataset: {len(self.samples)} ({n_c} cls, {n_s} seg)")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s=self.samples[idx]; img=cv2.imread(s["path"])
        if img is None: img=np.array(Image.open(s["path"]).convert("RGB"))
        else: img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if s["has_mask"] and s["mask_path"]:
            mask=cv2.imread(s["mask_path"],cv2.IMREAD_GRAYSCALE)
            if mask is None: mask=np.array(Image.open(s["mask_path"]).convert("L"))
            mask=(mask>127).astype(np.float32)
        else: mask=np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
        if self.transform:
            aug=self.transform(image=img,mask=mask); img,mask=aug["image"],aug["mask"]
        if isinstance(mask,np.ndarray): mask=torch.from_numpy(mask)
        mask=mask.float().unsqueeze(0); mn=mask.squeeze(0).numpy()
        if s["has_mask"] and mn.max()>0: bb=mask2bbox(mn)
        elif s["bbox_str"]: bb=parse_yolo(s["bbox_str"])
        else: bb=np.zeros((0,4),dtype=np.float32)
        return {"image":img,"cls_label":s["cls_label"],"seg_mask":mask,"det_bboxes":torch.from_numpy(bb),"has_mask":s["has_mask"]}

def collate(batch):
    images=torch.stack([b["image"] for b in batch]); cls=torch.tensor([b["cls_label"] for b in batch],dtype=torch.long)
    seg=torch.stack([b["seg_mask"] for b in batch]); hm=torch.tensor([b["has_mask"] for b in batch],dtype=torch.bool)
    mb=max((b["det_bboxes"].shape[0] for b in batch),default=0); mb=max(mb,1)
    db=torch.zeros(len(batch),mb,4); dv=torch.zeros(len(batch),mb,dtype=torch.bool)
    for i,b in enumerate(batch):
        n=b["det_bboxes"].shape[0]
        if n>0: db[i,:n]=b["det_bboxes"]; dv[i,:n]=True
    return {"image":images,"cls_label":cls,"seg_mask":seg,"det_bboxes":db,"det_valid":dv,"has_mask":hm}

def get_fold_loaders(fold, bs):
    f=splits["folds"][fold]
    ctr=cls_all.iloc[f["cls_tr"]].reset_index(drop=True); cva=cls_all.iloc[f["cls_va"]].reset_index(drop=True)
    str_=seg_all.iloc[f["seg_tr"]].reset_index(drop=True); sva=seg_all.iloc[f["seg_va"]].reset_index(drop=True)
    dtr=det_all.iloc[f["det_tr"]].reset_index(drop=True); dva=det_all.iloc[f["det_va"]].reset_index(drop=True)
    log(f"  Fold {fold}:")
    tds=WoundDS(ctr,str_,dtr,tf_train(),IMG_SIZE); vds=WoundDS(cva,sva,dva,tf_val(),IMG_SIZE)
    labels=[s["cls_label"] for s in tds.samples if s["cls_label"]>=0]
    if labels:
        cnt=Counter(labels); tot=len(labels); cw={c:tot/n for c,n in cnt.items()}
        wts=[cw.get(s["cls_label"],1.0) for s in tds.samples]; sampler=WeightedRandomSampler(wts,len(wts),replacement=True)
    else: sampler=None
    tl=DataLoader(tds,batch_size=bs,sampler=sampler,num_workers=ABL_CFG["num_workers"],pin_memory=True,drop_last=True,collate_fn=collate)
    vl=DataLoader(vds,batch_size=bs,shuffle=False,num_workers=ABL_CFG["num_workers"],pin_memory=True,collate_fn=collate)
    return tl, vl

# ==============================================================================
#  LOSS + METRICS
# ==============================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): super().__init__(); self.smooth=smooth
    def forward(self, pred, target):
        p=torch.sigmoid(pred).flatten(1); t=target.flatten(1); inter=(p*t).sum(1)
        return 1.0-((2*inter+self.smooth)/(p.sum(1)+t.sum(1)+self.smooth)).mean()
class MultiTaskLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.cls_fn=nn.CrossEntropyLoss(label_smoothing=0.1); self.seg_bce=nn.BCEWithLogitsLoss()
        self.seg_dice=DiceLoss(); self.det_obj=nn.BCEWithLogitsLoss()
        self.log_var_cls=nn.Parameter(torch.zeros(1)); self.log_var_seg=nn.Parameter(torch.zeros(1)); self.log_var_det=nn.Parameter(torch.zeros(1))
    def forward(self, pred, batch, aux_loss=None):
        losses={}; dev=pred["cls_logits"].device
        labels=batch["cls_label"].to(dev); valid=labels>=0
        losses["cls"]=self.cls_fn(pred["cls_logits"][valid],labels[valid]) if valid.any() else torch.tensor(0.0,device=dev)
        hm=batch["has_mask"].to(dev)
        if hm.any():
            sp=pred["seg_mask"][hm]; st=batch["seg_mask"][hm].to(dev)
            if st.shape[-2:]!=sp.shape[-2:]: st=F.interpolate(st,sp.shape[-2:],mode="bilinear",align_corners=False)
            losses["seg"]=self.seg_bce(sp,st)+self.seg_dice(sp,st)
            do=pred["det_objectness"][hm]; ot=F.interpolate(st,do.shape[-2:],mode="bilinear",align_corners=False)
            losses["det"]=self.det_obj(do,(ot>0.3).float())
        else:
            losses["seg"]=torch.tensor(0.0,device=dev); losses["det"]=torch.tensor(0.0,device=dev)
        losses["aux"]=aux_loss if aux_loss is not None else torch.tensor(0.0,device=dev)
        wc,ws,wd=torch.exp(-self.log_var_cls),torch.exp(-self.log_var_seg),torch.exp(-self.log_var_det)
        losses["total"]=(wc*losses["cls"]+self.log_var_cls+ws*losses["seg"]+self.log_var_seg+wd*losses["det"]+self.log_var_det+0.01*losses["aux"])
        return losses
def combined_metric(ca,sd,ap): return {"cls_acc":ca,"seg_dice":sd,"det_ap50":ap,"combined_weighted":0.4*ca+0.4*sd+0.2*ap}
def _iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); ua=(a[2]-a[0])*(a[3]-a[1]); ub=(b[2]-b[0])*(b[3]-b[1])
    return inter/(ua+ub-inter+1e-8)
def _m2b(m,min_area=50):
    if m.max()==0: return []
    b=(m>0.5).astype(np.uint8); n,_,st,_=cv2.connectedComponentsWithStats(b,8); H,W=m.shape; out=[]
    for i in range(1,n):
        x,y,w,h,a=st[i]
        if a<min_area: continue
        out.append([x/W,y/H,(x+w)/W,(y+h)/H])
    return out
def _ap50(pb,gb,th=0.5):
    if len(gb)==0 and len(pb)==0: return 1.0
    if len(gb)==0 or len(pb)==0: return 0.0
    mg=set(); tp=fp=0
    for p in pb:
        bi=0;bj=-1
        for j,g in enumerate(gb):
            if j in mg: continue
            i=_iou(p,g)
            if i>bi: bi=i;bj=j
        if bi>=th and bj>=0: tp+=1; mg.add(bj)
        else: fp+=1
    fn=len(gb)-len(mg); pr=tp/(tp+fp+1e-8); rc=tp/(tp+fn+1e-8)
    return pr*rc if (pr+rc)>0 else 0.0
@torch.no_grad()
def evaluate(model, vl, crit, seg_size):
    model.eval(); cp,cl,dices,aps=[],[],[],[]; tot=0.0; nb=0
    for batch in vl:
        bg={k:(v.to(DEVICE) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
        pred=model(bg["image"],target_seg_size=seg_size)
        losses=crit(pred,bg,model.get_router_aux_loss()); tot+=losses["total"].item(); nb+=1
        lab=bg["cls_label"]; val=lab>=0
        if val.any():
            cp.extend(pred["cls_logits"][val].argmax(1).cpu().numpy()); cl.extend(lab[val].cpu().numpy())
        hm=bg["has_mask"]
        if hm.any():
            sp=pred["seg_mask"][hm]; st=bg["seg_mask"][hm]
            if st.shape[-2:]!=sp.shape[-2:]: st=F.interpolate(st,sp.shape[-2:],mode="bilinear",align_corners=False)
            pm=(torch.sigmoid(sp)>0.5).float().cpu().numpy(); gm=st.cpu().numpy()
            for i in range(pm.shape[0]):
                p,g=pm[i,0],gm[i,0]; ip=(p*g).sum(); un=p.sum()+g.sum()
                dices.append((2*ip/(un+1e-8)) if un>0 else (1.0 if g.sum()==0 else 0.0)); aps.append(_ap50(_m2b(p),_m2b(g)))
    ca=accuracy_score(cl,cp)*100 if cl else 0.0
    cf=f1_score(cl,cp,average="macro")*100 if cl else 0.0
    sd=np.mean(dices)*100 if dices else 0.0; ap=np.mean(aps)*100 if aps else 0.0
    return {"cls_acc":ca,"cls_f1":cf,"seg_dice":sd,"det_ap50":ap,"loss":tot/max(nb,1),**combined_metric(ca,sd,ap)}

# ==============================================================================
#  ABLATION PATCHES  (set flags / replace modules)
# ==============================================================================
class _IdWACSA(nn.Module):
    def forward(self, pyr): return pyr
def patch_model(model, config):
    if config=="no_F2DCA":
        model.fpn._ablate_f2dca = True       # FPN skips F2DCA fusion
    elif config=="no_WACSA":
        model.wa_csa = _IdWACSA().to(DEVICE) # WA-CSA passthrough
    elif config=="no_MoE":
        model.cls_decoder._ablate_moe = True # single expert, no routing
    elif config=="no_WTCS":
        model.seg_decoder._ablate_wtcs = True# seg decoder skips FiLM conditioning
    elif config=="backbone_only":
        # ALL components OFF -> proves COLLECTIVE importance
        model.fpn._ablate_f2dca = True
        model.wa_csa = _IdWACSA().to(DEVICE)
        model.cls_decoder._ablate_moe = True
        model.seg_decoder._ablate_wtcs = True
    # full = no patch

# ==============================================================================
#  TRAIN ONE FOLD
# ==============================================================================
def train_fold(config, fold):
    cfg=BaseModelConfig(); m=WILLIEBASE(cfg).to(DEVICE); patch_model(m,config)
    crit=MultiTaskLoss(NUM_CLASSES).to(DEVICE)
    ck=CheckpointManager(ABL_DIR/config, f"base_{config}")
    tl,vl=get_fold_loaders(fold, ABL_CFG["batch_size"])
    bb=[p for n,p in m.named_parameters() if ("encoder_dino" in n or "encoder_conv" in n)]
    dec=[p for n,p in m.named_parameters() if not ("encoder_dino" in n or "encoder_conv" in n)]
    opt=torch.optim.AdamW([{"params":dec,"lr":ABL_CFG["lr_head"]},
                           {"params":list(crit.parameters()),"lr":ABL_CFG["lr_head"]},
                           {"params":bb,"lr":ABL_CFG["lr_head"]*ABL_CFG["backbone_lr_scale"]}], weight_decay=ABL_CFG["weight_decay"])
    sched=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=2,eta_min=1e-7)
    scaler=GradScaler(); start,best,pat=0,0.0,0
    rm=ck.load(m,fold=fold,tag="latest",optimizer=opt,scheduler=sched)
    if rm:
        start=rm.get("epoch",0)+1; best=rm.get("best_combined",0.0); pat=rm.get("patience_ctr",0)
        patch_model(m,config); log(f"  resume {config} f{fold} @e{start} best={best:.2f}")
    accum,seg_size=ABL_CFG["accumulation_steps"],ABL_CFG["seg_size"]
    for epoch in range(start, ABL_CFG["epochs"]):
        if epoch==ABL_CFG["freeze_epochs"]: m.unfreeze_backbones()
        m.train(); opt.zero_grad()
        for step,batch in enumerate(tl):
            bg={k:(v.to(DEVICE) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
            with autocast():
                pred=m(bg["image"],target_seg_size=seg_size); losses=crit(pred,bg,m.get_router_aux_loss()); loss=losses["total"]/accum
            scaler.scale(loss).backward()
            if (step+1)%accum==0 or (step+1)==len(tl):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(),ABL_CFG["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad()
        sched.step()
        me=evaluate(m,vl,crit,seg_size); comb=me["combined_weighted"]
        ph="FROZEN" if epoch<ABL_CFG["freeze_epochs"] else "FULL"; fl="*" if comb>best else " "
        log(f"  [{config}|f{fold}] {fl} E{epoch:02d} [{ph}] cls={me['cls_acc']:.1f} seg={me['seg_dice']:.1f} det={me['det_ap50']:.1f} comb={comb:.1f}")
        sv=dict(me)
        ck.save(m,opt,sched,epoch,fold,{"epoch":epoch,"best_combined":max(best,comb),"patience_ctr":pat,**sv},tag="latest")
        if comb>best: best=comb; pat=0; ck.save(m,opt,sched,epoch,fold,{"epoch":epoch,"best_combined":best,**sv},tag="best")
        else: pat+=1
        if pat>=ABL_CFG["patience"]: log(f"  [{config}|f{fold}] early stop @e{epoch}"); break
    ck.load(m,fold=fold,tag="best"); final=evaluate(m,vl,crit,seg_size)
    del m,opt,sched,scaler,tl,vl; torch.cuda.empty_cache()
    return final

# ==============================================================================
#  MAIN
# ==============================================================================
CONFIGS=["full","no_F2DCA","no_WACSA","no_MoE","no_WTCS","backbone_only"]
results=json.load(open(ABL_RESULTS)) if (ABL_RESULTS.exists() and ABL_RESULTS.stat().st_size>0) else {}
for config in CONFIGS:
    results.setdefault(config,{})
    for fold in range(5):
        if str(fold) in results[config]:
            log(f"  done: {config} f{fold} (comb={results[config][str(fold)]['combined_weighted']:.1f}) skip"); continue
        log(f"\n{'='*70}\n  {config}  fold {fold}\n{'='*70}")
        t0=time.time(); fm=train_fold(config,fold); fm["minutes"]=round((time.time()-t0)/60,1)
        results[config][str(fold)]=fm
        tmp=ABL_RESULTS.with_suffix(".tmp"); json.dump(_to_native(results),open(tmp,"w"),indent=2); tmp.rename(ABL_RESULTS)
        log(f"  -> {config} f{fold}: comb={fm['combined_weighted']:.1f} ({fm['minutes']} min)")

log(f"\n{'='*70}\n  BASE ABLATION SUMMARY (5-fold mean+/-std)\n{'='*70}")
def st(cfg,key):
    vs=[results[cfg][str(f)][key] for f in range(5) if str(f) in results.get(cfg,{})]
    return (np.mean(vs),np.std(vs)) if vs else (0,0)
fc=st("full","combined_weighted")[0]
log(f"  {'Config':10s} {'Cls':>8} {'Seg':>8} {'Det':>8} {'Comb':>12} {'Delta':>7}")
for cfg in CONFIGS:
    ca=st(cfg,"cls_acc")[0]; sa=st(cfg,"seg_dice")[0]; da=st(cfg,"det_ap50")[0]
    ma,ms=st(cfg,"combined_weighted"); d="" if cfg=="full" else f"{ma-fc:+.2f}"
    log(f"  {cfg:10s} {ca:7.1f} {sa:7.1f} {da:7.1f} {ma:6.1f}+/-{ms:.1f} {d:>7}")
log(f"\n  done -> {ABL_RESULTS}")


# ==============================================================================
#  FINAL SUMMARY: INDIVIDUAL (redundant) vs COLLECTIVE (essential)
# ==============================================================================
def final_summary():
    import numpy as np, json
    R = json.load(open(ABL_RESULTS))
    def stat(c,k="combined_weighted"):
        vs=[R[c][f][k] for f in R[c] if k in R[c][f]] if c in R else []
        return (np.mean(vs),np.std(vs),len(vs)) if vs else (None,None,0)
    full_m, full_s, _ = stat("full")
    if full_m is None:
        log("  (no full results yet)"); return
    log("\n"+"="*70)
    log("  BASE FINAL ABLATION  -  Individual vs Collective")
    log("="*70)
    log(f"  {'Config':14s} {'Combined':>12} {'Delta':>9}  Verdict")
    log(f"  {'full':14s} {full_m:7.1f}+/-{full_s:.1f}   {'--':>7}  baseline")
    for c in ["no_F2DCA","no_WACSA","no_MoE","no_WTCS","backbone_only"]:
        m,s,n = stat(c)
        if m is None: continue
        d=m-full_m
        if c=="backbone_only":
            v = "COLLECTIVE WIN" if abs(d)>full_s else "no collective effect (!)"
        else:
            v = "individually redundant" if abs(d)<=full_s else "individual effect"
        log(f"  {c:14s} {m:7.1f}+/-{s:.1f}   {d:+7.2f}  {v}")
    log("="*70)
    bo,_,_ = stat("backbone_only")
    if bo is not None:
        log(f"  COLLECTIVE contribution: {full_m-bo:+.1f} pts (full {full_m:.1f} vs backbone-only {bo:.1f})")
    log("="*70)

final_summary()
