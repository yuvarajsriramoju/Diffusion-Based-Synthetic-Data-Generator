import requests, base64, os

# API endpoint
url = "http://127.0.0.1:8000/generate"

# Request 4 images
resp = requests.post(url, json={"count": 4})
resp.raise_for_status()
data = resp.json()

# Create output folder
outdir = "api_samples"
os.makedirs(outdir, exist_ok=True)

# Save each image as PNG
for i, img_b64 in enumerate(data["images"]):
    img_bytes = base64.b64decode(img_b64)
    path = os.path.join(outdir, f"sample_{i}.png")
    with open(path, "wb") as f:
        f.write(img_bytes)
    print(f"Saved {path}")
