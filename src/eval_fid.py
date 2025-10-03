import argparse
from src.fid_utils import compute_fid

def main(args):
    fid = compute_fid(args.real, args.fake)
    print(f"FID Score: {fid:.2f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real", type=str, default="data/mnist/test_samples") 
    p.add_argument("--fake", type=str, default="synthetic/mnist_ddpm")
    args = p.parse_args()
    main(args)
