# main.py  (repo root)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from io import BytesIO
from PIL import Image
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms, models

app = FastAPI(title="Neuro Scan Assist API", version="1.0")

# Allow your Vercel site + local dev (set in Render env: CORS_ORIGINS)
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Model (same architecture, no external downloads) ----
class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.cnn = models.resnet18(weights=None)  # avoid downloads on Render
        self.cnn.fc = nn.Identity()
        enc = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)          # [B, 512]
        x = x.unsqueeze(1)       # [B, 1, 512]
        x = self.transformer_encoder(x)
        x = x.squeeze(1)         # [B, 512]
        return self.fc(x)

device = torch.device("cpu")
_model = None

# NumPy-free, safe transforms
test_transforms = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Adjust to your training class order
CLASSES: List[str] = ["glioma", "meningioma", "pituitary", "no_tumor"]

def get_model():
    global _model
    if _model is None:
        m = HybridCNNTransformer(num_classes=len(CLASSES))
        # model.pth must be next to this file
        state = torch.load(os.path.join(os.path.dirname(__file__), "model.pth"),
                           map_location="cpu")
        if isinstance(state, nn.Module):
            m = state
        else:
            m.load_state_dict(state)
        m.to(device).eval()
        _model = m
    return _model

class PredictResponse(BaseModel):
    pred_class: str
    probs: Dict[str, float]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict", response_model=PredictResponse)
async def predict(image: UploadFile = File(...)):
    if image.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Upload a PNG or JPEG image.")
    img = Image.open(BytesIO(await image.read())).convert("RGB")
    x = test_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = get_model()(x)
        probs_t = F.softmax(logits, dim=1)[0].cpu()
        pred_idx = int(torch.argmax(probs_t).item())
        probs = {c: float(probs_t[i].item()) for i, c in enumerate(CLASSES)}

    return PredictResponse(pred_class=CLASSES[pred_idx], probs=probs)

@app.get("/")
def root():
    return {"message": "FastAPI up. POST /predict with form field 'image'."}
