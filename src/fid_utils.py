import torch
from torch_fidelity import calculate_metrics

def compute_fid(real_dir, fake_dir, cuda=True):
    metrics = calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        cuda=torch.cuda.is_available() and cuda,
        isc=False,
        fid=True,
        kid=False
    )
    return metrics["frechet_inception_distance"]
