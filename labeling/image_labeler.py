from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

def label_images(image_paths):
    model = models.resnet18(pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Load ImageNet class names
    import json
    from urllib import request
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    class_names = request.urlopen(url).read().decode("utf-8").splitlines()

    results = []

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = outputs.max(1)
            label = class_names[predicted.item()]
            results.append((path.split("/")[-1], label))

    return results
