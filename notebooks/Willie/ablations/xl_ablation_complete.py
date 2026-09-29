#!/usr/bin/env python
"""
================================================================================
  xl_ablation_complete.py  —  SINGLE-FILE WILLIE-XL COMPONENT ABLATION
  No local imports (no xl_core / xl_data). Only pip packages.
  Configs: full / no_F2DCA / no_WACSA / no_MoE / no_WTCS / backbone_only  x 5 folds
  Tasks evaluated: classification (acc/f1) + segmentation (Dice) + localization (AP@0.5, mask-derived)
  Resumes from checkpoints in ablation_final/xl/ckpts/. Saves ablation_final/results/xl_complete_results.json
  Bug fixes baked in: numpy->json (_to_native), 3-arg lambda patches.
  Requires on disk: manifests (CFG.MANIFEST_DIR), data/FUSeg, pretrained DINOv2/SAM2.
================================================================================
"""
import os, sys, json, time, random, warnings, math, gc, re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
warnings.filterwarnings("ignore")

# ==============================================================================
#  CONFIG  (from xl_core.CFG)
# ==============================================================================
class CFG:
    PROJECT_ROOT = "."
    ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts/11_fuseg_csd_xl")
    MANIFEST_DIR = os.path.join(PROJECT_ROOT, "artifacts/woundshot_v2/manifests")
    SEG_MASK_DIR = os.path.join(PROJECT_ROOT, "artifacts/woundshot_v2/sam2_masks")
    PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "pretrained_weights")
    VARIANT = "XL"; IMG_SIZE = 378; PATCH_SIZE = 14
    NUM_PATCHES_H = IMG_SIZE // PATCH_SIZE; NUM_PATCHES = NUM_PATCHES_H ** 2
    DINO_BACKBONE = "dinov2_vitl14"; DINO_DIM = 1024
    CONVNEXT_BACKBONE = "convnext_large"; CONVNEXT_DIM = 1536
    SAM2_WEIGHTS = "sam2.1_hiera_large.pt"; SAM2_DIM = 256
    FUSION_DIM = 512; F2DCA_LAYERS = 4; F2DCA_HEADS = 8; NUM_FREQ_BANDS = 4
    WACSA_LAYERS = 4; WACSA_HEADS = 8; COARSE_TOKENS = 64
    NUM_CLASSES = 5; CLASS_NAMES = ["diabetic","pressure","surgical","venous","no_wound"]
    MOE_EXPERTS = 8; MOE_TOP_K = 2; MOE_HIDDEN = 1024
    SEG_DECODER_CHANNELS = [512,256,128,64]; SEG_TARGET_SIZE = 512; FILM_DIM = 256
    FCOS_CHANNELS = 256; FCOS_NUM_CONVS = 4; FPN_DIM = 384
    N_FOLDS = 5; BATCH_SIZE = 2; GRAD_ACCUM = 4; LR = 5e-5
    BACKBONE_LR_SCALE = 0.05; WEIGHT_DECAY = 0.05; EPOCHS = 100
    WARMUP_EPOCHS = 5; PATIENCE = 20
    SEED = 42; DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True; USE_GRAD_CKPT = False; DROPOUT = 0.15
    DINO_UNFREEZE_LAST_N = 8; CONVNEXT_UNFREEZE_LAST_N = 2; SAM2_FROZEN = True

os.makedirs(CFG.ARTIFACT_DIR, exist_ok=True)
DEVICE = CFG.DEVICE

def seed_everything(seed=CFG.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
seed_everything()

def log(m): print(m, flush=True)
def _to_native(o):
    if isinstance(o, dict):  return {k:_to_native(v) for k,v in o.items()}
    if isinstance(o, list):  return [_to_native(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    return o

# ==============================================================================
#  MODEL  (verbatim from xl_core.py — DO NOT rename: checkpoint compatibility)
# ==============================================================================
class DINOv2Backbone(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.model = torch.hub.load('facebookresearch/dinov2', cfg.DINO_BACKBONE, pretrained=True)
        self.embed_dim = self.model.embed_dim
        self.features = {}
        n_blocks = len(self.model.blocks)
        self.hook_layers = [n_blocks//4-1, n_blocks//2-1, 3*n_blocks//4-1, n_blocks-1]
        for idx in self.hook_layers:
            self.model.blocks[idx].register_forward_hook(self._make_hook(f"layer_{idx}"))
        for param in self.model.parameters(): param.requires_grad = False
        if cfg.DINO_UNFREEZE_LAST_N > 0:
            for block in self.model.blocks[-cfg.DINO_UNFREEZE_LAST_N:]:
                for param in block.parameters(): param.requires_grad = True
        self.proj = nn.Sequential(nn.Linear(self.embed_dim, cfg.FUSION_DIM),
            nn.LayerNorm(cfg.FUSION_DIM), nn.GELU(), nn.Dropout(cfg.DROPOUT*0.5))
    def _make_hook(self, name):
        def hook(module, inp, output): self.features[name] = output
        return hook
    def forward(self, x):
        self.features = {}
        tokens = self.model.forward_features(x)
        if isinstance(tokens, dict): tokens = tokens["x_norm_patchtokens"]
        elif tokens.dim()==3 and tokens.shape[1] > CFG.NUM_PATCHES: tokens = tokens[:,1:,:]
        projected = self.proj(tokens)
        ms_features = []
        for idx in self.hook_layers:
            feat = self.features.get(f"layer_{idx}", None)
            if feat is not None:
                if isinstance(feat, tuple): feat = feat[0]
                if feat.dim()==3 and feat.shape[1] > CFG.NUM_PATCHES: feat = feat[:,1:,:]
                ms_features.append(feat)
        return projected, ms_features

class ConvNeXtBackbone(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model("convnext_large.fb_in22k_ft_in1k", pretrained=True, features_only=True)
        with torch.no_grad():
            dummy = torch.randn(1,3,cfg.IMG_SIZE,cfg.IMG_SIZE); outs = self.model(dummy)
            self.stage_dims = [o.shape[1] for o in outs]
            self.stage_sizes = [(o.shape[2],o.shape[3]) for o in outs]
        self.final_dim = self.stage_dims[-1]
        self.proj = nn.Sequential(nn.Linear(self.final_dim, cfg.FUSION_DIM),
            nn.LayerNorm(cfg.FUSION_DIM), nn.GELU(), nn.Dropout(cfg.DROPOUT*0.5))
        for param in self.model.parameters(): param.requires_grad = False
        if cfg.CONVNEXT_UNFREEZE_LAST_N > 0:
            unfreeze_idx = set(range(4-cfg.CONVNEXT_UNFREEZE_LAST_N, 4))
            for name, param in self.model.named_parameters():
                for idx in unfreeze_idx:
                    if f"stages.{idx}" in name or f"stages_{idx}" in name:
                        param.requires_grad = True; break
    def forward(self, x):
        features = self.model(x); last = features[-1]
        B,C,H,W = last.shape; tokens = last.flatten(2).transpose(1,2)
        projected = self.proj(tokens)
        return projected, features

class SAM2Backbone(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        sam2_path = os.path.join(cfg.PRETRAINED_DIR, cfg.SAM2_WEIGHTS)
        self.has_sam2 = os.path.exists(sam2_path)
        if self.has_sam2:
            ckpt = torch.load(sam2_path, map_location="cpu", weights_only=False)
            self._build_from_checkpoint(ckpt); del ckpt; gc.collect()
            log(f"     SAM2: Loaded from {cfg.SAM2_WEIGHTS}")
            with torch.no_grad():
                _sz = 224 if self.use_timm_hiera else cfg.IMG_SIZE
                dummy = torch.randn(1,3,_sz,_sz)
                if self.use_timm_hiera:
                    feats = self.encoder(dummy); self.out_dim = feats[-1].shape[1]
                else:
                    feat = self.encoder(dummy); self.out_dim = feat.shape[1] if feat.dim()==4 else cfg.SAM2_DIM
                del dummy
            log(f"     SAM2 encoder output dim: {self.out_dim}")
        else:
            log(f"     SAM2: weights not found, using Swin-Base fallback")
            self.fallback = timm.create_model("swin_base_patch4_window7_224.ms_in22k_ft_in1k",
                pretrained=True, features_only=True, img_size=cfg.IMG_SIZE)
            with torch.no_grad():
                dummy = torch.randn(1,3,cfg.IMG_SIZE,cfg.IMG_SIZE); outs = self.fallback(dummy)
                self.out_dim = outs[-1].shape[1]
        self.proj = nn.Sequential(nn.Linear(self.out_dim, cfg.FUSION_DIM),
            nn.LayerNorm(cfg.FUSION_DIM), nn.GELU(), nn.Dropout(cfg.DROPOUT*0.5))
        if cfg.SAM2_FROZEN:
            if self.has_sam2:
                for name, param in self.named_parameters():
                    if 'proj' not in name: param.requires_grad = False
            else:
                for param in self.fallback.parameters(): param.requires_grad = False
    def _build_from_checkpoint(self, ckpt):
        state = ckpt.get("model", ckpt)
        encoder_keys = {k.replace("image_encoder.",""):v for k,v in state.items() if k.startswith("image_encoder.")}
        if len(encoder_keys) > 0:
            trunk_keys = {k.replace("trunk.",""):v for k,v in encoder_keys.items() if k.startswith("trunk.")}
            try:
                self.encoder = timm.create_model("hiera_large_224.mae_in1k_ft_in1k", pretrained=False, features_only=True)
                missing, unexpected = self.encoder.load_state_dict(trunk_keys, strict=False)
                if len(missing) > len(trunk_keys)*0.5:
                    raise RuntimeError(f"Too many missing keys ({len(missing)} vs {len(trunk_keys)})")
                log(f"     SAM2 trunk: loaded ({len(missing)} missing, {len(unexpected)} unexpected)")
                self.use_timm_hiera = True
            except Exception as e:
                log(f"     SAM2 trunk load failed ({e})")
                log(f"     Using pretrained Hiera-Large from timm (native 224)")
                self.encoder = timm.create_model("hiera_large_224.mae_in1k_ft_in1k", pretrained=True, features_only=True)
                self.use_timm_hiera = True
        else:
            log(f"     SAM2: no image_encoder keys, using pretrained Hiera-Large")
            self.encoder = timm.create_model("hiera_large_224.mae_in1k_ft_in1k", pretrained=True, features_only=True)
            self.use_timm_hiera = True
    def forward(self, x):
        if self.has_sam2:
            if self.use_timm_hiera:
                if x.shape[-1]!=224 or x.shape[-2]!=224:
                    x_resized = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
                else: x_resized = x
                features = self.encoder(x_resized); feat = features[-1]
            else: feat = self.encoder(x)
        else:
            features = self.fallback(x); feat = features[-1]
        if feat.dim()==4:
            B,C,H,W = feat.shape; tokens = feat.flatten(2).transpose(1,2)
        else: tokens = feat
        projected = self.proj(tokens)
        return projected

class TripleBackboneEncoder(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.bb1 = DINOv2Backbone(cfg); self.bb2 = ConvNeXtBackbone(cfg); self.bb3 = SAM2Backbone(cfg)
        log(f"  Triple Backbone: DINOv2-ViT-L + ConvNeXt-Large + SAM2-Hiera-Large")
    def forward(self, x):
        tok1, ms1 = self.bb1(x); tok2, ms2 = self.bb2(x); tok3 = self.bb3(x)
        return {"bb1":tok1, "bb2":tok2, "bb3":tok3, "ms_vit":ms1, "ms_cnn":ms2}

class FrequencyDecomposition(nn.Module):
    def __init__(self, num_bands=4):
        super().__init__(); self.num_bands = num_bands
    def forward(self, x):
        B,N,D = x.shape; bands = []
        for i in range(self.num_bands):
            if i==0: bands.append(x)
            else:
                k = min(2**i, N)
                pool = F.adaptive_avg_pool1d(x.transpose(1,2), max(N//k,1))
                upsampled = F.interpolate(pool, size=N, mode='linear', align_corners=False)
                bands.append(upsampled.transpose(1,2))
        return bands

class F2DCA_Layer(nn.Module):
    def __init__(self, dim, num_heads=12, num_freq_bands=4, dropout=0.1):
        super().__init__()
        self.dim=dim; self.num_heads=num_heads; self.head_dim=dim//num_heads; self.num_freq_bands=num_freq_bands
        self.freq_decomp = FrequencyDecomposition(num_freq_bands)
        self.q_proj=nn.Linear(dim,dim); self.k_proj=nn.Linear(dim,dim); self.v_proj=nn.Linear(dim,dim); self.out_proj=nn.Linear(dim,dim)
        self.freq_weights = nn.Parameter(torch.ones(num_freq_bands)/num_freq_bands)
        self.wound_gate = nn.Sequential(nn.Linear(dim,dim//4), nn.GELU(), nn.Linear(dim//4,num_heads), nn.Sigmoid())
        self.norm1=nn.LayerNorm(dim); self.norm2=nn.LayerNorm(dim); self.norm3=nn.LayerNorm(dim); self.dropout=nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(dim,dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4,dim), nn.Dropout(dropout))
        self.ffn_norm = nn.LayerNorm(dim)
    def forward(self, bb1_tokens, bb2_tokens, bb3_tokens):
        B = bb1_tokens.shape[0]
        context = torch.cat([bb2_tokens, bb3_tokens], dim=1)
        x = self.norm1(bb1_tokens); ctx = self.norm2(context)
        freq_bands = self.freq_decomp(x); weights = F.softmax(self.freq_weights, dim=0)
        attended_bands = []
        for band_idx, band_q in enumerate(freq_bands):
            q = self.q_proj(band_q).reshape(B,-1,self.num_heads,self.head_dim).transpose(1,2)
            k = self.k_proj(ctx).reshape(B,-1,self.num_heads,self.head_dim).transpose(1,2)
            v = self.v_proj(ctx).reshape(B,-1,self.num_heads,self.head_dim).transpose(1,2)
            attn = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(self.head_dim)
            attn = F.softmax(attn, dim=-1); attn = self.dropout(attn)
            out = torch.matmul(attn, v).transpose(1,2).reshape(B,-1,self.dim)
            attended_bands.append(out * weights[band_idx])
        fused = sum(attended_bands)
        gate = self.wound_gate(bb1_tokens.mean(dim=1))
        N = fused.shape[1]
        gate = gate.unsqueeze(1).unsqueeze(-1).expand(B,N,self.num_heads,self.head_dim).reshape(B,N,self.dim)
        fused = fused * gate
        fused = self.out_proj(fused); fused = self.dropout(fused)
        out = bb1_tokens + fused
        out = out + self.ffn(self.ffn_norm(out))
        return out

class WA_CSA_Layer(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim=dim; self.num_heads=num_heads; self.head_dim=dim//num_heads
        self.f2c_q=nn.Linear(dim,dim); self.f2c_k=nn.Linear(dim,dim); self.f2c_v=nn.Linear(dim,dim); self.f2c_out=nn.Linear(dim,dim)
        self.c2f_q=nn.Linear(dim,dim); self.c2f_k=nn.Linear(dim,dim); self.c2f_v=nn.Linear(dim,dim); self.c2f_out=nn.Linear(dim,dim)
        self.wound_gate_fine = nn.Sequential(nn.Linear(dim,dim//4), nn.GELU(), nn.Linear(dim//4,1), nn.Sigmoid())
        self.wound_gate_coarse = nn.Sequential(nn.Linear(dim,dim//4), nn.GELU(), nn.Linear(dim//4,1), nn.Sigmoid())
        self.alpha_f2c = nn.Parameter(torch.zeros(1)); self.alpha_c2f = nn.Parameter(torch.zeros(1))
        self.norm_fine=nn.LayerNorm(dim); self.norm_coarse=nn.LayerNorm(dim); self.dropout=nn.Dropout(dropout)
    def _cross_attend(self, q_proj, k_proj, v_proj, out_proj, query, key_value):
        B,N,_ = query.shape; M = key_value.shape[1]
        q = q_proj(query).reshape(B,N,self.num_heads,self.head_dim).transpose(1,2)
        k = k_proj(key_value).reshape(B,M,self.num_heads,self.head_dim).transpose(1,2)
        v = v_proj(key_value).reshape(B,M,self.num_heads,self.head_dim).transpose(1,2)
        attn = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1); attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1,2).reshape(B,N,self.dim)
        return out_proj(out)
    def forward(self, fine, coarse):
        fine_n = self.norm_fine(fine); coarse_n = self.norm_coarse(coarse)
        f2c = self._cross_attend(self.f2c_q,self.f2c_k,self.f2c_v,self.f2c_out, coarse_n, fine_n)
        gate_c = self.wound_gate_coarse(coarse_n)
        coarse = coarse + torch.tanh(self.alpha_f2c)*f2c*gate_c
        c2f = self._cross_attend(self.c2f_q,self.c2f_k,self.c2f_v,self.c2f_out, fine_n, coarse_n)
        gate_f = self.wound_gate_fine(fine_n)
        fine = fine + torch.tanh(self.alpha_c2f)*c2f*gate_f
        return fine, coarse

class DualBackboneFPN(nn.Module):
    def __init__(self, vit_dim=1024, cnn_dims=[192,384,768,1536], fpn_dim=384, num_patches_h=27):
        super().__init__()
        self.fpn_dim=fpn_dim; self.num_patches_h=num_patches_h
        self.vit_projs = nn.ModuleList([nn.Sequential(nn.Linear(vit_dim,fpn_dim), nn.LayerNorm(fpn_dim)) for _ in range(4)])
        self.cnn_projs = nn.ModuleList([nn.Sequential(nn.Conv2d(dim,fpn_dim,1), nn.GroupNorm(16,fpn_dim), nn.GELU()) for dim in cnn_dims])
        self.merge_weights = nn.ParameterList([nn.Parameter(torch.ones(2)*0.5) for _ in range(4)])
        self.lateral = nn.ModuleList([nn.Conv2d(fpn_dim,fpn_dim,1) for _ in range(4)])
        self.smooth = nn.ModuleList([nn.Conv2d(fpn_dim,fpn_dim,3,padding=1) for _ in range(4)])
    def forward(self, ms_vit, ms_cnn):
        B = ms_cnn[0].shape[0]; fpn_features = []
        n_vit = min(len(ms_vit),4); n_cnn = min(len(ms_cnn),4)
        for i in range(4):
            merged = None
            if i < n_vit:
                vit_feat = ms_vit[i]
                if vit_feat.dim()==3:
                    if vit_feat.shape[1] > self.num_patches_h**2: vit_feat = vit_feat[:,1:,:]
                    vit_proj = self.vit_projs[i](vit_feat)
                    h = w = int(math.sqrt(vit_proj.shape[1]))
                    vit_spatial = vit_proj.transpose(1,2).reshape(B,self.fpn_dim,h,w)
                else: vit_spatial = self.vit_projs[i](vit_feat)
                merged = vit_spatial
            if i < n_cnn:
                cnn_proj = self.cnn_projs[i](ms_cnn[i])
                if merged is not None:
                    target_h, target_w = merged.shape[2], merged.shape[3]
                    cnn_resized = F.interpolate(cnn_proj, size=(target_h,target_w), mode='bilinear', align_corners=False)
                    w = F.softmax(self.merge_weights[i], dim=0)
                    merged = w[0]*merged + w[1]*cnn_resized
                else: merged = cnn_proj
            if merged is None:
                h = self.num_patches_h//(2**i)
                merged = torch.zeros(B,self.fpn_dim,max(h,1),max(h,1), device=ms_cnn[0].device)
            fpn_features.append(merged)
        for i in range(3,0,-1):
            lat = self.lateral[i](fpn_features[i])
            upsampled = F.interpolate(lat, size=fpn_features[i-1].shape[2:], mode='bilinear', align_corners=False)
            fpn_features[i-1] = fpn_features[i-1] + upsampled
        fpn_out = [self.smooth[i](fpn_features[i]) for i in range(4)]
        return fpn_out

class Expert(nn.Module):
    def __init__(self, dim, hidden, num_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim,hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden,hidden//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden//2,num_classes))
    def forward(self, x): return self.net(x)

class TopKRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k=2):
        super().__init__(); self.gate = nn.Linear(dim,num_experts); self.top_k = top_k
    def forward(self, x):
        logits = self.gate(x)
        top_k_vals, top_k_idx = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_vals, dim=-1)
        return top_k_weights, top_k_idx, logits

class MoEClassifier(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.num_experts = cfg.MOE_EXPERTS; self.top_k = cfg.MOE_TOP_K
        self.experts = nn.ModuleList([Expert(cfg.FUSION_DIM,cfg.MOE_HIDDEN,cfg.NUM_CLASSES,cfg.DROPOUT) for _ in range(cfg.MOE_EXPERTS)])
        self.router = TopKRouter(cfg.FUSION_DIM, cfg.MOE_EXPERTS, cfg.MOE_TOP_K)
        self.wound_embed = nn.Linear(cfg.FUSION_DIM, cfg.FILM_DIM)
    def forward(self, pooled):
        weights, idx, gate_logits = self.router(pooled)
        B = pooled.shape[0]
        all_expert_out = torch.stack([e(pooled) for e in self.experts], dim=1)
        idx_expanded = idx.unsqueeze(-1).expand(-1,-1,all_expert_out.shape[-1])
        selected = torch.gather(all_expert_out, 1, idx_expanded)
        logits = (selected * weights.unsqueeze(-1)).sum(dim=1)
        router_probs = F.softmax(gate_logits, dim=-1)
        avg_probs = router_probs.mean(dim=0)
        balance_loss = (self.num_experts * (avg_probs**2).sum())
        wound_emb = self.wound_embed(pooled)
        return logits, wound_emb, balance_loss, idx

class ChannelSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels,channels//reduction), nn.ReLU(), nn.Linear(channels//reduction,channels), nn.Sigmoid())
    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1); return x*w

class SpatialSE(nn.Module):
    def __init__(self, channels):
        super().__init__(); self.conv = nn.Conv2d(channels,1,1)
    def forward(self, x):
        w = torch.sigmoid(self.conv(x)); return x*w

class PscSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__(); self.cse = ChannelSE(channels,reduction); self.sse = SpatialSE(channels)
    def forward(self, x): return self.cse(x) + self.sse(x)

class FiLMLayer(nn.Module):
    def __init__(self, film_dim, channels):
        super().__init__()
        self.gamma = nn.Linear(film_dim, channels); self.beta = nn.Linear(film_dim, channels)
        nn.init.ones_(self.gamma.weight.data[:, :channels]); nn.init.zeros_(self.beta.weight.data)
    def forward(self, x, condition):
        gamma = self.gamma(condition).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(condition).unsqueeze(-1).unsqueeze(-1)
        return gamma*x + beta

class FUSegNetDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, film_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = nn.Sequential(nn.Conv2d(in_ch+skip_ch,out_ch,3,padding=1), nn.GroupNorm(16,out_ch), nn.GELU(),
            nn.Conv2d(out_ch,out_ch,3,padding=1), nn.GroupNorm(16,out_ch), nn.GELU())
        self.pscse = PscSE(out_ch); self.film = FiLMLayer(film_dim, out_ch)
    def forward(self, x, skip, wound_embed):
        x = self.up(x)
        if skip.shape[2:] != x.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1); x = self.conv(x); x = self.pscse(x); x = self.film(x, wound_embed)
        return x

class FUSegNetDecoder(nn.Module):
    def __init__(self, fpn_dim=384, decoder_channels=[512,256,128,64], film_dim=256, target_size=512):
        super().__init__()
        self.target_size = target_size
        self.input_proj = nn.Sequential(nn.Conv2d(fpn_dim,decoder_channels[0],1), nn.GroupNorm(16,decoder_channels[0]), nn.GELU())
        self.blocks = nn.ModuleList()
        for i in range(len(decoder_channels)-1):
            self.blocks.append(FUSegNetDecoderBlock(decoder_channels[i], fpn_dim, decoder_channels[i+1], film_dim))
        self.final_up = nn.Sequential(nn.ConvTranspose2d(decoder_channels[-1],decoder_channels[-1],2,stride=2),
            nn.GroupNorm(16,decoder_channels[-1]), nn.GELU())
        self.head = nn.Sequential(nn.Conv2d(decoder_channels[-1],32,3,padding=1), nn.GELU(), nn.Conv2d(32,1,1))
    def forward(self, fpn_features, wound_embed):
        x = self.input_proj(fpn_features[-1])
        for i, block in enumerate(self.blocks):
            skip_idx = len(fpn_features)-2-i; skip = fpn_features[max(skip_idx,0)]
            if CFG.USE_GRAD_CKPT and self.training:
                x = grad_ckpt(block, x, skip, wound_embed, use_reentrant=False)
            else: x = block(x, skip, wound_embed)
        x = self.final_up(x); mask = self.head(x)
        mask = F.interpolate(mask, size=(self.target_size,self.target_size), mode='bilinear', align_corners=False)
        return mask

class FCOSHead(nn.Module):
    def __init__(self, in_channels=384, hidden=256, num_convs=4, num_classes=5):
        super().__init__()
        layers = []
        for i in range(num_convs):
            ch_in = in_channels if i==0 else hidden
            layers.extend([nn.Conv2d(ch_in,hidden,3,padding=1), nn.GroupNorm(16,hidden), nn.GELU()])
        self.tower = nn.Sequential(*layers)
        self.cls_head = nn.Conv2d(hidden,num_classes,3,padding=1)
        self.reg_head = nn.Conv2d(hidden,4,3,padding=1)
        self.center_head = nn.Conv2d(hidden,1,3,padding=1)
    def forward(self, fpn_features):
        all_cls, all_reg, all_center = [], [], []
        for feat in fpn_features:
            shared = self.tower(feat)
            all_cls.append(self.cls_head(shared).flatten(2).transpose(1,2))
            all_reg.append(self.reg_head(shared).flatten(2).transpose(1,2))
            all_center.append(self.center_head(shared).flatten(2).transpose(1,2))
        return torch.cat(all_cls,dim=1), torch.cat(all_reg,dim=1), torch.cat(all_center,dim=1)

class WILLIECSD_XL(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.encoder = TripleBackboneEncoder(cfg)
        self.f2dca_layers = nn.ModuleList([F2DCA_Layer(cfg.FUSION_DIM,cfg.F2DCA_HEADS,cfg.NUM_FREQ_BANDS,cfg.DROPOUT) for _ in range(cfg.F2DCA_LAYERS)])
        self.wacsa_layers = nn.ModuleList([WA_CSA_Layer(cfg.FUSION_DIM,cfg.WACSA_HEADS,cfg.DROPOUT) for _ in range(cfg.WACSA_LAYERS)])
        self.coarse_pool = nn.AdaptiveAvgPool1d(cfg.COARSE_TOKENS)
        self.fpn = DualBackboneFPN(cfg.DINO_DIM, [192,384,768,1536], cfg.FPN_DIM, cfg.NUM_PATCHES_H)
        self.cls_norm = nn.LayerNorm(cfg.FUSION_DIM); self.cls_pool = nn.AdaptiveAvgPool1d(1)
        self.moe = MoEClassifier(cfg)
        self.seg_decoder = FUSegNetDecoder(cfg.FPN_DIM, cfg.SEG_DECODER_CHANNELS, cfg.FILM_DIM, cfg.SEG_TARGET_SIZE)
        self.det_head = FCOSHead(cfg.FPN_DIM, cfg.FCOS_CHANNELS, cfg.FCOS_NUM_CONVS, cfg.NUM_CLASSES)
        log(f"  WILLIE-XL CSD model built")
    def forward(self, images, tasks=("cls","seg","det")):
        out = {}
        enc = self.encoder(images)
        tok1, tok2, tok3 = enc["bb1"], enc["bb2"], enc["bb3"]
        ms_vit, ms_cnn = enc["ms_vit"], enc["ms_cnn"]
        fused = tok1
        for layer in self.f2dca_layers:
            if self.cfg.USE_GRAD_CKPT and self.training:
                fused = grad_ckpt(layer, fused, tok2, tok3, use_reentrant=False)
            else: fused = layer(fused, tok2, tok3)
        fine = fused
        coarse = self.coarse_pool(fused.transpose(1,2)).transpose(1,2)
        for layer in self.wacsa_layers:
            fine, coarse = layer(fine, coarse)
        if "cls" in tasks:
            pooled = self.cls_pool(self.cls_norm(fine).transpose(1,2)).squeeze(-1)
            logits, wound_embed, balance_loss, expert_idx = self.moe(pooled)
            out["cls_logits"]=logits; out["wound_embed"]=wound_embed
            out["moe_balance_loss"]=balance_loss; out["expert_idx"]=expert_idx
        if "seg" in tasks or "det" in tasks:
            fpn_features = self.fpn(ms_vit, ms_cnn)
        if "seg" in tasks:
            wound_emb = out.get("wound_embed", torch.zeros(images.shape[0], self.cfg.FILM_DIM, device=images.device))
            out["seg_mask"] = self.seg_decoder(fpn_features, wound_emb)
        if "det" in tasks:
            det_cls, det_reg, det_center = self.det_head(fpn_features)
            out["det_cls"]=det_cls; out["det_reg"]=det_reg; out["det_center"]=det_center
        return out

# ==============================================================================
#  DATA  (condensed from xl_data.py — manifests, splits, dataset)
# ==============================================================================
IMAGENET_MEAN = [0.485,0.456,0.406]; IMAGENET_STD = [0.229,0.224,0.225]
img_col = "image_path"; label_col = "unified_label"; class_col = "unified_class"

log("\nLoading manifests...")
cls_train_df = pd.read_csv(os.path.join(CFG.MANIFEST_DIR,"cls_train.csv"))
cls_val_df   = pd.read_csv(os.path.join(CFG.MANIFEST_DIR,"cls_val.csv"))
cls_test_df  = pd.read_csv(os.path.join(CFG.MANIFEST_DIR,"cls_test.csv"))
unique_classes = sorted(cls_train_df[class_col].unique())
data_class_to_idx = {c:i for i,c in enumerate(unique_classes)}
cfg_class_to_idx = {c:i for i,c in enumerate(CFG.CLASS_NAMES)}
label_remap = {}
for cls_name in unique_classes:
    di = data_class_to_idx[cls_name]; ci = cfg_class_to_idx.get(cls_name, di); label_remap[di]=ci
for df in [cls_train_df, cls_val_df, cls_test_df]:
    df["label"] = df[label_col].map(label_remap)

def find_manifest(name, dirs):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p): return p
    return None
search_dirs = [CFG.MANIFEST_DIR, os.path.join(CFG.PROJECT_ROOT,"artifacts/woundshot_v2"),
    os.path.join(CFG.PROJECT_ROOT,"artifacts/woundshot_v2/manifests"),
    os.path.join(CFG.PROJECT_ROOT,"artifacts/WoundShot_FIXED"), os.path.join(CFG.PROJECT_ROOT,"artifacts"),
    os.path.join(CFG.PROJECT_ROOT,"artifacts/09_fuseg_csd_mini"), os.path.join(CFG.PROJECT_ROOT,"artifacts/10_fuseg_csd_base")]

# seg manifests — build from FUSeg dir if not found (mirrors xl_data fallback)
seg_train_path = None
for name in ["ws_seg_manifest_fuseg_train.csv","seg_train.csv","seg_manifest_train.csv","fuseg_train.csv","ws_seg_train.csv"]:
    seg_train_path = find_manifest(name, search_dirs)
    if seg_train_path:
        seg_val_path = find_manifest(name.replace("train","val"), search_dirs)
        if seg_val_path: break
        else: seg_train_path = None
if seg_train_path is None:
    fuseg_base = os.path.join(CFG.PROJECT_ROOT,"data/FUSeg")
    str_recs, sva_recs = [], []
    for split, recs in [("train",str_recs),("val",sva_recs)]:
        img_dir = os.path.join(fuseg_base,split,"images"); mask_dir = os.path.join(fuseg_base,split,"labels")
        if not os.path.isdir(img_dir): continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif')): continue
            ip = os.path.join(img_dir,fname); base = os.path.splitext(fname)[0]; mp=None
            for ext in ['.png','.jpg','.bmp','.tif']:
                c = os.path.join(mask_dir, base+ext)
                if os.path.exists(c): mp=c; break
            recs.append({"img":ip,"mask":mp if mp else ""})
    seg_train_df = pd.DataFrame(str_recs); seg_val_df = pd.DataFrame(sva_recs)
    seg_img_col = "img"; seg_mask_col = "mask"
else:
    seg_train_df = pd.read_csv(seg_train_path); seg_val_df = pd.read_csv(seg_val_path)
    seg_img_col = [c for c in seg_train_df.columns if "img" in c.lower() or "image" in c.lower()][0]
    seg_mask_col = [c for c in seg_train_df.columns if "mask" in c.lower()][0]

det_train_df = pd.DataFrame(); det_val_df = pd.DataFrame(); det_img_col="image_path"; det_label_col=None
cls_trainval_df = pd.concat([cls_train_df, cls_val_df], ignore_index=True)
seg_trainval_df = pd.concat([seg_train_df, seg_val_df], ignore_index=True)

seg_lookup = {}
for _, row in seg_trainval_df.iterrows():
    ip = str(row[seg_img_col]); mp = str(row[seg_mask_col])
    if pd.notna(row[seg_mask_col]) and mp != 'nan':
        seg_lookup[os.path.basename(ip)] = (ip, mp)
det_lookup = {}
log(f"  cls_trainval={len(cls_trainval_df)} seg_trainval={len(seg_trainval_df)} seg_lookup={len(seg_lookup)}")

# fold splits — reuse saved if present else stratified
splits_path = os.path.join(CFG.ARTIFACT_DIR,"5fold_splits_v2.pt")
if os.path.exists(splits_path):
    fold_splits = torch.load(splits_path, map_location="cpu", weights_only=False)
elif os.path.exists(os.path.join(CFG.PROJECT_ROOT,"artifacts/10_fuseg_csd_base/5fold_splits_v2.pt")):
    fold_splits = torch.load(os.path.join(CFG.PROJECT_ROOT,"artifacts/10_fuseg_csd_base/5fold_splits_v2.pt"), map_location="cpu", weights_only=False)
else:
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    labels = cls_trainval_df["label"].values; fold_splits = {}
    for fold,(tr,va) in enumerate(skf.split(np.arange(len(cls_trainval_df)), labels)):
        fold_splits[fold] = {"cls_train_idx":tr.tolist(), "cls_val_idx":va.tolist()}
    torch.save(fold_splits, splits_path)
# normalize keys
_new = {}
for k,v in fold_splits.items():
    if not isinstance(v, dict): continue
    if "cls_train_idx" not in v and "train_idx" not in v: continue
    try: fn = int(k)
    except: 
        m = re.search(r'(\d+)', str(k)); fn = int(m.group(1)) if m else None
    if fn is not None: _new[fn] = v
if _new: fold_splits = _new

def get_val_transform(img_size=CFG.IMG_SIZE):
    return A.Compose([A.Resize(img_size,img_size), A.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD), ToTensorV2()])
def get_train_transform(img_size=CFG.IMG_SIZE):
    return A.Compose([A.Resize(img_size,img_size), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5), A.ShiftScaleRotate(shift_limit=0.1,scale_limit=0.15,rotate_limit=30,border_mode=0,p=0.6),
        A.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD), ToTensorV2()])
train_transform = get_train_transform(); val_transform = get_val_transform()

class MultiTaskDataset(Dataset):
    def __init__(self, cls_df, seg_df, det_df, seg_lookup_dict, det_lookup_dict, transform,
                 img_size=CFG.IMG_SIZE, seg_size=CFG.SEG_TARGET_SIZE):
        self.transform=transform; self.img_size=img_size; self.seg_size=seg_size
        self.seg_lookup=seg_lookup_dict; self.det_lookup=det_lookup_dict
        self.samples=[]; seen=set()
        if cls_df is not None and len(cls_df)>0:
            for _,row in cls_df.iterrows():
                ip=str(row[img_col]); bn=os.path.basename(ip); lab=int(row["label"])
                hs = bn in self.seg_lookup; hd = bn in self.det_lookup
                si = self.seg_lookup.get(bn,(None,None)); di = self.det_lookup.get(bn,(None,None))
                self.samples.append({"img_path":ip,"cls_label":lab,"has_cls":True,"has_seg":hs,"has_det":hd,
                    "seg_img":si[0],"seg_mask":si[1],"det_img":di[0],"det_label":di[1]}); seen.add(bn)
        if seg_df is not None and len(seg_df)>0:
            for _,row in seg_df.iterrows():
                ip=str(row[seg_img_col]); bn=os.path.basename(ip)
                if bn not in seen:
                    hd = bn in self.det_lookup; di = self.det_lookup.get(bn,(None,None))
                    self.samples.append({"img_path":ip,"cls_label":-1,"has_cls":False,"has_seg":True,"has_det":hd,
                        "seg_img":ip,"seg_mask":str(row[seg_mask_col]),"det_img":di[0],"det_label":di[1]}); seen.add(bn)
    def __len__(self): return len(self.samples)
    def _load_image(self, path):
        if path is None or not os.path.exists(path): return np.zeros((self.img_size,self.img_size,3),dtype=np.uint8)
        img = cv2.imread(path)
        if img is None: return np.zeros((self.img_size,self.img_size,3),dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    def _load_mask(self, path):
        if path is None or str(path)=='nan' or not os.path.exists(str(path)):
            return np.zeros((self.seg_size,self.seg_size),dtype=np.float32)
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None: return np.zeros((self.seg_size,self.seg_size),dtype=np.float32)
        mask = cv2.resize(mask,(self.seg_size,self.seg_size)); return (mask>127).astype(np.float32)
    def __getitem__(self, idx):
        s = self.samples[idx]; img = self._load_image(s["img_path"])
        mask = self._load_mask(s["seg_mask"]) if s["has_seg"] else None
        if mask is not None:
            t = self.transform(image=img, mask=mask); img_t = t["image"]
            _m = t["mask"].unsqueeze(0).unsqueeze(0).float()
            _m = F.interpolate(_m, size=(self.seg_size,self.seg_size), mode="nearest"); mask_t = _m.squeeze(0)
        else:
            t = self.transform(image=img); img_t = t["image"]
            mask_t = torch.zeros(1,self.seg_size,self.seg_size,dtype=torch.float32)
        cl = s["cls_label"] if s["has_cls"] else -1
        targets = {"cls_label":torch.tensor(cl,dtype=torch.long), "seg_mask":mask_t,
            "has_cls":torch.tensor(s["has_cls"],dtype=torch.bool), "has_seg":torch.tensor(s["has_seg"],dtype=torch.bool),
            "has_det":torch.tensor(s["has_det"],dtype=torch.bool)}
        return img_t, targets

def mt_collate(batch):
    images = torch.stack([b[0] for b in batch]); targets = {}
    for k in batch[0][1].keys(): targets[k] = torch.stack([b[1][k] for b in batch])
    return images, targets

# class weights + loss
cls_counts = np.zeros(CFG.NUM_CLASSES)
for _,row in cls_trainval_df.iterrows(): cls_counts[int(row["label"])] += 1
class_weights = 1.0/(cls_counts+1e-6); class_weights = class_weights/class_weights.sum()*CFG.NUM_CLASSES
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

class MultiTaskLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__(); self.num_classes=num_classes
        self.log_var_cls=nn.Parameter(torch.zeros(1)); self.log_var_seg=nn.Parameter(torch.zeros(1)); self.log_var_det=nn.Parameter(torch.zeros(1))
    def focal_loss(self, pred, target, gamma=2.0, alpha=0.25):
        ce = F.cross_entropy(pred, target, reduction='none'); pt = torch.exp(-ce)
        return (alpha*(1-pt)**gamma*ce).mean()
    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred).flatten(1); target = target.flatten(1)
        inter = (pred*target).sum(dim=1); return 1-(2*inter+1)/(pred.sum(1)+target.sum(1)+1)
    def forward(self, outputs, targets):
        losses={}; total=0.0
        if "cls_logits" in outputs and "cls_label" in targets:
            hc = targets.get("has_cls", torch.ones(outputs["cls_logits"].shape[0],dtype=torch.bool))
            if hc.any():
                cl = self.focal_loss(outputs["cls_logits"][hc], targets["cls_label"][hc])
                total += torch.exp(-self.log_var_cls)*cl + self.log_var_cls; losses["cls"]=cl.item()
        if "seg_mask" in outputs and "seg_mask" in targets:
            hs = targets.get("has_seg", torch.ones(outputs["seg_mask"].shape[0],dtype=torch.bool))
            if hs.any():
                ps = outputs["seg_mask"][hs]; gs = targets["seg_mask"][hs]
                if gs.shape[-2:]!=ps.shape[-2:]: gs = F.interpolate(gs.float(), size=ps.shape[-2:], mode='nearest')
                bce = F.binary_cross_entropy_with_logits(ps, gs); dice = self.dice_loss(ps, gs).mean()
                sl = bce+dice; total += torch.exp(-self.log_var_seg)*sl + self.log_var_seg; losses["seg"]=sl.item()
        if "moe_balance_loss" in outputs:
            ml = outputs["moe_balance_loss"]*0.01; total += ml; losses["moe"]=ml.item()
        losses["total"] = total.item() if isinstance(total,torch.Tensor) else total
        return total, losses

# ==============================================================================
#  ABLATION HARNESS
# ==============================================================================
ABL_DIR = Path(CFG.PROJECT_ROOT)/"ablation_final"/"xl"/"ckpts"
OUT     = Path(CFG.PROJECT_ROOT)/"ablation_final"/"results"/"xl_complete_results.json"
ABL_DIR.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
CONFIGS = ["full","no_F2DCA","no_WACSA","no_MoE","no_WTCS","backbone_only"]
N_FOLDS = CFG.N_FOLDS; BATCH_SIZE = CFG.BATCH_SIZE; NUM_WORKERS = 2
EPOCHS = CFG.EPOCHS; PATIENCE = CFG.PATIENCE; GRAD_ACCUM = CFG.GRAD_ACCUM

_kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CFG.SEED)
_seg_splits = list(_kf.split(range(len(seg_trainval_df))))

def build_loaders(fold):
    f = fold_splits[fold]
    cls_tr = cls_trainval_df.iloc[f["cls_train_idx"]]; cls_va = cls_trainval_df.iloc[f["cls_val_idx"]]
    str_idx, sva_idx = _seg_splits[fold]
    seg_tr = seg_trainval_df.iloc[str_idx]; seg_va = seg_trainval_df.iloc[sva_idx]
    tds = MultiTaskDataset(cls_tr, seg_tr, None, seg_lookup, det_lookup, train_transform)
    vds = MultiTaskDataset(cls_va, seg_va, None, seg_lookup, det_lookup, val_transform)
    nc = sum(1 for s in tds.samples if s["has_cls"]); ns = sum(1 for s in tds.samples if s["has_seg"])
    log(f"  Fold {fold}: train {len(tds)} ({nc} cls, {ns} seg) | val {len(vds)}")
    labels = [s["cls_label"] for s in tds.samples if s["has_cls"]]
    if labels:
        cnt = Counter(labels); tot = len(labels); cw = {c: tot/n for c,n in cnt.items()}
        wts = [cw.get(s["cls_label"],1.0) if s["has_cls"] else 1.0 for s in tds.samples]
        sampler = WeightedRandomSampler(wts, len(wts), replacement=True)
    else: sampler = None
    tl = DataLoader(tds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=True, collate_fn=mt_collate)
    vl = DataLoader(vds, batch_size=max(1,BATCH_SIZE), shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, collate_fn=mt_collate)
    return tl, vl

def patch_model(model, config):
    if config == "no_F2DCA":
        for layer in model.f2dca_layers: layer.forward = (lambda fused,tok2,tok3: fused)
    elif config == "no_WACSA":
        for layer in model.wacsa_layers: layer.forward = (lambda fine,coarse: (fine,coarse))
    elif config == "no_MoE":
        moe = model.moe
        def _se(pooled, _moe=moe):
            lg=_moe.experts[0](pooled); we=_moe.wound_embed(pooled)
            bal=torch.tensor(0.0,device=pooled.device); idx=torch.zeros(pooled.shape[0],_moe.top_k,dtype=torch.long,device=pooled.device)
            return lg, we, bal, idx
        moe.forward = _se
    elif config == "no_WTCS":
        seg = model.seg_decoder; _orig = seg.forward
        def _nf(fpn, we, _orig=_orig): return _orig(fpn, torch.zeros_like(we))
        seg.forward = _nf
    elif config == "backbone_only":
        for layer in model.f2dca_layers: layer.forward = (lambda fused,tok2,tok3: fused)
        for layer in model.wacsa_layers: layer.forward = (lambda fine,coarse: (fine,coarse))
        moe = model.moe
        def _se2(pooled, _moe=moe):
            lg=_moe.experts[0](pooled); we=_moe.wound_embed(pooled)
            bal=torch.tensor(0.0,device=pooled.device); idx=torch.zeros(pooled.shape[0],_moe.top_k,dtype=torch.long,device=pooled.device)
            return lg, we, bal, idx
        moe.forward = _se2
        seg = model.seg_decoder; _ob = seg.forward
        def _nf2(fpn, we, _orig=_ob): return _orig(fpn, torch.zeros_like(we))
        seg.forward = _nf2
    # full = no patch

# mask-derived localization (paper Table 5 s->l)
def _iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); ua=(a[2]-a[0])*(a[3]-a[1]); ub=(b[2]-b[0])*(b[3]-b[1])
    return inter/(ua+ub-inter+1e-8)
def _m2b(m, min_area=50):
    if m.max()==0: return []
    b=(m>0.5).astype(np.uint8); n,_,stt,_=cv2.connectedComponentsWithStats(b,8); H,W=m.shape; out=[]
    for i in range(1,n):
        x,y,w,h,a=stt[i]
        if a<min_area: continue
        out.append([x/W,y/H,(x+w)/W,(y+h)/H])
    return out
def _ap50(pb,gb,th=0.5):
    if len(gb)==0 and len(pb)==0: return 1.0
    if len(gb)==0 or len(pb)==0: return 0.0
    mg=set(); tp=fp=0
    for p in pb:
        bi=0; bj=-1
        for j,g in enumerate(gb):
            if j in mg: continue
            i=_iou(p,g)
            if i>bi: bi=i; bj=j
        if bi>=th and bj>=0: tp+=1; mg.add(bj)
        else: fp+=1
    fn=len(gb)-len(mg); pr=tp/(tp+fp+1e-8); rc=tp/(tp+fn+1e-8)
    return pr*rc if (pr+rc)>0 else 0.0

@torch.no_grad()
def evaluate(model, vl):
    model.eval(); preds, labels, dices, aps = [], [], [], []
    for images, targets in vl:
        images = images.to(DEVICE, non_blocking=True)
        hc = targets["has_cls"]; hs = targets["has_seg"]
        if hc.any():
            with autocast(enabled=CFG.USE_AMP):
                out = model(images, tasks=("cls",))
            lg = out["cls_logits"][hc].float().cpu()
            preds.extend(lg.argmax(-1).numpy()); labels.extend(targets["cls_label"][hc].numpy())
        if hs.any():
            with autocast(enabled=CFG.USE_AMP):
                out = model(images, tasks=("seg",))
            pm = (torch.sigmoid(out["seg_mask"][hs])>0.5).float()
            gt = targets["seg_mask"][hs].to(DEVICE)
            if gt.shape[-2:]!=pm.shape[-2:]: gt = F.interpolate(gt, pm.shape[-2:], mode="nearest")
            pm_np=pm.cpu().numpy(); gt_np=gt.cpu().numpy()
            for i in range(pm_np.shape[0]):
                p=pm_np[i,0]; g=gt_np[i,0]; inter=(p*g).sum(); uni=p.sum()+g.sum()
                dices.append((2*inter/(uni+1e-8)) if uni>0 else (1.0 if g.sum()==0 else 0.0))
                aps.append(_ap50(_m2b(p), _m2b(g)))
    acc = accuracy_score(labels,preds)*100 if labels else 0.0
    f1  = f1_score(labels,preds,average="macro")*100 if labels else 0.0
    dice= np.mean(dices)*100 if dices else 0.0
    ap  = np.mean(aps)*100 if aps else 0.0
    combined = 0.4*acc + 0.4*dice + 0.2*ap
    return {"cls_acc":acc,"cls_f1":f1,"seg_dice":dice,"det_ap50":ap,"combined":combined}

def atomic_save(obj, path):
    tmp = str(path)+".tmp"; torch.save(obj, tmp); os.replace(tmp, path)

def train_fold(config, fold):
    cdir = ABL_DIR/config; cdir.mkdir(parents=True, exist_ok=True)
    resume_p = cdir/f"xl_{config}_fold{fold}_latest.pt"
    best_p   = cdir/f"xl_{config}_fold{fold}_best.pt"
    model = WILLIECSD_XL(CFG).to(DEVICE); patch_model(model, config)
    tl, vl = build_loaders(fold)
    if best_p.exists():
        sd = torch.load(best_p, map_location=DEVICE, weights_only=False)
        if isinstance(sd, dict) and "model_state" in sd: sd = sd["model_state"]
        model.load_state_dict(sd); patch_model(model, config)
        me = evaluate(model, vl)
        log(f"  [EVAL-ONLY] {config} f{fold}: cls={me['cls_acc']:.1f} seg={me['seg_dice']:.1f} det={me['det_ap50']:.1f} comb={me['combined']:.1f}")
        del model; torch.cuda.empty_cache(); return me
    if best_p.exists():
        sd = torch.load(best_p, map_location=DEVICE, weights_only=False)
        if isinstance(sd, dict) and "model_state" in sd: sd = sd["model_state"]
        model.load_state_dict(sd); patch_model(model, config)
        me = evaluate(model, vl)
        log(f"  [EVAL-ONLY] {config} f{fold}: cls={me['cls_acc']:.1f} seg={me['seg_dice']:.1f} det={me['det_ap50']:.1f} comb={me['combined']:.1f}")
        del model; torch.cuda.empty_cache(); return me
    # EVAL-ONLY short-circuit: if a trained best checkpoint exists, just score it.
    if best_p.exists():
        sd = torch.load(best_p, map_location=DEVICE, weights_only=False)
        if isinstance(sd, dict) and "model_state" in sd: sd = sd["model_state"]
        model.load_state_dict(sd); patch_model(model, config)
        me = evaluate(model, vl)
        log(f"  [EVAL-ONLY] {config} f{fold}: cls={me['cls_acc']:.1f} seg={me['seg_dice']:.1f} det={me['det_ap50']:.1f} comb={me['combined']:.1f}")
        del model; torch.cuda.empty_cache(); return me
    crit = MultiTaskLoss(CFG.NUM_CLASSES).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad] + list(crit.parameters())
    opt = torch.optim.AdamW(params, lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = GradScaler(enabled=CFG.USE_AMP)
    start=0; best=-1; pat=0
    if resume_p.exists():
        ck = torch.load(resume_p, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"]); patch_model(model, config)
        try:
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        except Exception as _e:
            log(f"  (opt resume skipped: {_e})")
        crit.load_state_dict(ck["crit"]); start=ck["epoch"]+1; best=ck["best"]; pat=ck["pat"]
        log(f"  resume {config} f{fold} @e{start} best={best:.2f}")
    for epoch in range(start, EPOCHS):
        model.train(); opt.zero_grad()
        for bi,(images,targets) in enumerate(tl):
            images = images.to(DEVICE, non_blocking=True)
            targets = {k:(v.to(DEVICE) if isinstance(v,torch.Tensor) else v) for k,v in targets.items()}
            with autocast(enabled=CFG.USE_AMP):
                out = model(images, tasks=("cls","seg"))
                loss, _ = crit(out, targets); loss = loss/GRAD_ACCUM
            scaler.scale(loss).backward()
            if (bi+1)%GRAD_ACCUM==0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
        sched.step()
        me = evaluate(model, vl); comb = me["combined"]; fl = "*" if comb>best else " "
        log(f"  [{config}|f{fold}] {fl} E{epoch:02d} cls={me['cls_acc']:.1f} seg={me['seg_dice']:.1f} det={me['det_ap50']:.1f} comb={comb:.1f}")
        atomic_save({"model_state":model.state_dict(),"opt":opt.state_dict(),"sched":sched.state_dict(),
            "scaler":scaler.state_dict(),"crit":crit.state_dict(),"epoch":epoch,"best":best,"pat":pat}, resume_p)
        if comb>best: best=comb; pat=0; atomic_save(model.state_dict(), best_p)
        else:
            pat+=1
            if pat>=PATIENCE: log(f"  early stop @e{epoch}"); break
    if best_p.exists():
        model.load_state_dict(torch.load(best_p, map_location=DEVICE, weights_only=False)); patch_model(model, config)
    me = evaluate(model, vl)
    del model; torch.cuda.empty_cache()
    return me

# ==============================================================================
#  MAIN
# ==============================================================================
if __name__ == "__main__":
    log("="*80); log(f"  XL ABLATION COMPLETE (single file) | {time.strftime('%Y-%m-%d %H:%M:%S')}"); log("="*80)
    results = {}
    if OUT.exists():
        try: results = json.load(open(OUT))
        except: results = {}
    for config in CONFIGS:
        results.setdefault(config, {})
        for fold in range(N_FOLDS):
            if str(fold) in results[config]:
                log(f"  done: {config} f{fold} (comb={results[config][str(fold)]['combined']:.1f}) skip"); continue
            t0 = time.time()
            me = train_fold(config, fold); me["minutes"] = round((time.time()-t0)/60,1)
            results[config][str(fold)] = me
            tmp=str(OUT)+".tmp"; json.dump(_to_native(results), open(tmp,"w"), indent=2); os.replace(tmp, OUT)
            log(f"  -> {config} f{fold}: comb={me['combined']:.1f} ({me['minutes']}m)")
    # summary
    log("\n"+"="*70); log("  XL ABLATION SUMMARY (5-fold mean)"); log("="*70)
    def mean(c,k):
        vs=[results[c][f][k] for f in results.get(c,{}) if k in results[c][f]]
        return np.mean(vs) if vs else float('nan')
    log(f"  {'Config':14s} {'Cls':>6} {'Dice':>6} {'AP':>6} {'Comb':>7}")
    for c in CONFIGS:
        log(f"  {c:14s} {mean(c,'cls_acc'):6.1f} {mean(c,'seg_dice'):6.1f} {mean(c,'det_ap50'):6.1f} {mean(c,'combined'):7.1f}")
    log(f"\n  written -> {OUT}")
