import torch
from PIL import Image
from torchvision import transforms


def predict_painter(image_path, model, painters, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path)
    img = transform(img)
    img = img.unsqueeze(0)  
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    
    painter = painters[predicted.item()].replace('_', ' ')
    
    return painter