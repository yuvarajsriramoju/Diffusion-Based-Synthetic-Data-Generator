from torchvision import datasets, transforms
from torchvision.utils import save_image
from pathlib import Path

def dump_mnist(out_dir="data/mnist/test_samples"):
    ds = datasets.MNIST("data/mnist", train=False, download=True,
                        transform=transforms.ToTensor())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for i, (x, _) in enumerate(ds):
        save_image(x, f"{out_dir}/{i:07d}.png")
    print(f"Saved {len(ds)} real MNIST test images to {out_dir}")

if __name__ == "__main__":
    dump_mnist()
