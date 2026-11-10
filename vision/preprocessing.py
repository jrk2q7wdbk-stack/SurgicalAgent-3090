from torchvision import transforms

def get_test_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((256, 448)),
        transforms.ToTensor(),
        normalize,
    ])