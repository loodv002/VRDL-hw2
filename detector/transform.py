import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.RandomAdjustSharpness(1.5, 1.0),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.RandomAdjustSharpness(1.5, 1.0),
    transforms.ToTensor()
])