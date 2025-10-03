import argparse, torch
from pathlib import Path
from torchvision.utils import save_image, make_grid
from src.lit_diffusion import LitDDPM

@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LitDDPM.load_from_checkpoint(args.ckpt).to(device).eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    preview = model.sample(64)
    preview = (preview + 1) / 2  # [-1,1] -> [0,1]
    save_image(make_grid(preview, nrow=8), out_dir / "preview_grid.png")

    total = 0
    while total < args.count:
        bs = min(512, args.count - total)
        imgs = model.sample(bs)
        imgs = (imgs + 1) / 2
        for i in range(imgs.size(0)):
            save_image(imgs[i], out_dir / f"{total + i:07d}.png")
        total += bs
    print(f"Saved {total} images to {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="ddpm.ckpt")
    p.add_argument("--out", type=str, default="synthetic/mnist_ddpm")
    p.add_argument("--count", type=int, default=60000)
    main(p.parse_args())
