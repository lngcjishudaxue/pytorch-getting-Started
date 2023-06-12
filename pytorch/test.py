import torch
import torchvision
from PIL import Image
from model import  *

image_path = "imgs/dog.jpg"
image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print (image.shape)


model = torch.load('best.pth',map_location=torch.device('cuda:0'))
print(model)
image = torch.reshape(image,[1,3,32,32])
print(image.shape)
image = image.to('cuda:0')
model.eval()
with torch.no_grad():
    output = model(image)
print(output)