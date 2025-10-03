import io, base64, torch
from fastapi import FastAPI
from torchvision.utils import save_image
from src.lit_diffusion import LitDDPM
from src.api.schemas import GenRequest, GenResponse

app = FastAPI(title="MNIST Diffusion API", version="1.0.0")
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
async def load_model():
    global _model
    # Adjust path if your checkpoint lives elsewhere
    _model = LitDDPM.load_from_checkpoint("ddpm.ckpt").to(_device).eval()

@app.get("/health")
async def health():
    return {"status": "ok", "device": _device}

@app.post("/generate", response_model=GenResponse)
async def generate(req: GenRequest):
    global _model
    n = max(1, min(512, req.count))  # simple guardrail
    with torch.no_grad():
        imgs = _model.sample(n)      # [-1,1]
        imgs = (imgs + 1) / 2        # [0,1]
    out = []
    for i in range(imgs.size(0)):
        buf = io.BytesIO()
        save_image(imgs[i], buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return GenResponse(images=out)
