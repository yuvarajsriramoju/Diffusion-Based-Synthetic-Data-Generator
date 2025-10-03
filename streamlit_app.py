import io, base64, requests, numpy as np
from PIL import Image, ImageOps
import streamlit as st

st.cache_resource(show_spinner=False)
def load_local_model(ckpt_path: str):
    from src.lit_diffusion import LitDDPM
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LitDDPM.load_from_checkpoint(ckpt_path).to(device).eval()
    return model, device

def tensor_to_pil(img_t):
    """img_t: (1, H, W) in [0,1] torch tensor -> PIL Image"""
    import torch
    img = (img_t.detach().cpu().clamp(0,1) * 255).to(torch.uint8).squeeze(0).numpy()
    return Image.fromarray(img, mode="L")

def call_remote_api(api_url: str, count: int):
    r = requests.post(f"{api_url.rstrip('/')}/generate", json={"count": count}, timeout=120)
    r.raise_for_status()
    data = r.json()
    imgs = []
    for b64 in data["images"]:
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
        imgs.append(img)
    return imgs

st.set_page_config(page_title="MNIST Diffusion Demo", layout="centered")
st.title("MNIST Diffusion Synthetic Generator - By Yuvaraj Sriramoju")

st.markdown(
    "Upload a 28×28 digit (optional), choose how many synthetic images to generate, "
    "and click **Generate**. In this demo, the upload is shown for context and not used to condition the generator."
)

mode = st.sidebar.selectbox("Generation backend", ["Local (ckpt)", "Remote API"], index=0)
count = st.sidebar.slider("Number of images", min_value=1, max_value=64, value=16, step=1)
st.sidebar.caption("Tip: keep ≤64 for snappy responses.")

api_url = None
if mode == "Remote API":
    api_url = st.sidebar.text_input("Remote API base URL", value="http://127.0.0.1:8000")
    st.sidebar.caption("E.g., your Cloud Run URL or local FastAPI http://127.0.0.1:8000")


uploaded = st.file_uploader("Optional: upload a digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    img = Image.open(uploaded).convert("L")
    preview = ImageOps.fit(img, (28,28), method=Image.LANCZOS)
    st.image([img, preview.resize((140,140), Image.NEAREST)], caption=["Original", "28×28 normalized"], width=140)

generate = st.button("Generate")

if generate:
    with st.spinner("Sampling synthetic images..."):
        try:
            if mode == "Local (ckpt)":
                model, device = load_local_model("ddpm.ckpt")
                import torch
                with torch.no_grad():
                    imgs_t = model.sample(count)      # [-1,1]
                    imgs_t = (imgs_t + 1) / 2         # [0,1]
                imgs = [tensor_to_pil(imgs_t[i]) for i in range(imgs_t.size(0))]
            else:
                if not api_url:
                    st.error("Please provide an API URL in the sidebar.")
                    st.stop()
                imgs = call_remote_api(api_url, count)

            cols = st.columns(8)
            for i, im in enumerate(imgs):
                cols[i % 8].image(im.resize((112,112), Image.NEAREST), use_container_width=False, caption=f"{i+1}")
        except Exception as e:
            st.exception(e)

st.markdown("---")
st.caption("Note: This demo uses an unconditional diffusion model. The uploaded file is not used to condition generation.")
