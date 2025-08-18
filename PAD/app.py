# app.py
import io, math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI, UploadFile, File, Request, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------
# Config (edit thresholds/paths)
# -------------------------------
CFG = {
    "ckpt_path": "C:\\Users\\gonth\\Downloads\\minivit_best.pth",  # <- change if needed
    "img_size": 256,
    "patch_size": 16,
    "embed_dim": 256,
    "depth": 6,
    "num_heads": 4,
    "mlp_ratio": 4.0,
    # thresholds measured in your runs:
    "thr_min_acer": 0.523,
    "thr_bpc5":    0.477,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# -------------------------------
# MiniViT definition (same as training)
# -------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0): super().__init__(); self.drop_prob=drop_prob
    def forward(self, x):
        if self.drop_prob==0.0 or not self.training: return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,)*(x.ndim-1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)                   # (B,C,H/ps,W/ps)
        x = x.flatten(2).transpose(1,2)    # (B,N,C)
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden); self.act = nn.GELU(); self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.num_heads = heads; self.head_dim = dim // heads; self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim); self.proj_drop = nn.Dropout(proj_drop)
        self.register_buffer("head_mask", torch.ones(heads), persistent=False)
        self._last_attn = None
    def forward(self, x):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        self._last_attn = attn.detach()
        attn = self.attn_drop(attn)
        hm = self.head_mask.view(1,-1,1,1)
        x = (attn*hm) @ v
        x = x.transpose(1,2).reshape(B,N,C)
        x = self.proj(x); x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path>0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, int(dim*mlp_ratio), drop=drop)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MiniViT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3,
                 num_classes=1, embed_dim=256, depth=6, heads=4,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.1):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+n_patches, embed_dim))
        self.pos_drop  = nn.Dropout(drop)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([Block(embed_dim, heads, mlp_ratio, drop, attn_drop, dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02); nn.init.constant_(self.head.bias, 0)
    def forward(self, x):
        B = x.size(0)
        x = self.patch(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x); cls_out = x[:,0]
        return self.head(cls_out).squeeze(-1)

# -------------------------------
# Load weights (pos_embed safe)
# -------------------------------
def load_minivit_flex(ckpt_path, img_size, cfg):
    device = cfg["device"]
    model = MiniViT(img_size=img_size, patch_size=cfg["patch_size"], embed_dim=cfg["embed_dim"],
                    depth=cfg["depth"], heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    if "pos_embed" in state and state["pos_embed"].shape != model.pos_embed.shape:
        pe = state["pos_embed"]; cls, patch = pe[:, :1, :], pe[:, 1:, :]
        n_old = patch.shape[1]; gs_old = int(math.sqrt(n_old))
        n_new = model.pos_embed.shape[1] - 1; gs_new = int(math.sqrt(n_new))
        patch = patch.transpose(1,2).reshape(1, pe.size(-1), gs_old, gs_old)
        patch = F.interpolate(patch, size=(gs_new, gs_new), mode="bicubic", align_corners=False)
        patch = patch.flatten(2).transpose(1,2)
        state["pos_embed"] = torch.cat([cls, patch], dim=1)
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=True)
    model.eval()
    return model

# -------------------------------
# Preprocess + predict
# -------------------------------
def preprocess_pil(img_pil, img_size=256):
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((img_size,img_size)), T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])
    return tf(img_pil).unsqueeze(0)

@torch.no_grad()
def predict_pil(model, img_pil, threshold=0.477):
    x = preprocess_pil(img_pil, CFG["img_size"]).to(CFG["device"])
    prob_attack = torch.sigmoid(model(x)).item()
    return {"prob_attack": float(prob_attack),
            "decision": "attack" if prob_attack >= threshold else "bona_fide",
            "threshold": float(threshold)}

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="PatchGuard Live PAD")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL = load_minivit_flex(CFG["ckpt_path"], CFG["img_size"], {
    "patch_size": CFG["patch_size"], "embed_dim": CFG["embed_dim"],
    "depth": CFG["depth"], "num_heads": CFG["num_heads"],
    "mlp_ratio": CFG["mlp_ratio"], "device": CFG["device"]
})

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/healthz")
def healthz():
    return {"status":"ok", "device": CFG["device"], "img_size": CFG["img_size"]}

@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  op: str = Query("bpc5", enum=["bpc5","minacer"])):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    thr = CFG["thr_bpc5"] if op=="bpc5" else CFG["thr_min_acer"]
    res = predict_pil(MODEL, img, threshold=thr)
    return JSONResponse(res)