from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def mnist_dataloaders(batch_size=128, num_workers=2):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST("data/mnist", train=True, download=True, transform=tf)
    test = datasets.MNIST("data/mnist", train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )