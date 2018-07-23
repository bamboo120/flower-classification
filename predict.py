import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import PIL 
from PIL import Image
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, help='Image to predict')
parser.add_argument('--checkpoint', type=str, help='checkpoint to use when predicting')
parser.add_argument('--topk', type=int, help='Return top K predictions')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
args, _ = parser.parse_known_args()
data_dir = 'flowers'
image_path = (data_dir + '/test' + '/18/' + '/image_04272.jpg')
hidden_units = 4096
checkpoint = 'checkpoint.pth'

arch = 'vgg16'
if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif arch == 'vgg13':
    model = models.vgg13(pretrained=True)
else:
    model = models.vgg16(pretrained=True)
    print('undefined architecture')


checkpoint_dict = torch.load(checkpoint)
model = torch.load(checkpoint)['model']


model.cuda()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image_process = Image.open(image)
    w, h = image_process.size
    if w < h:
        image_process.thumbnail((256,h))
    else:
        image_process.thumbnail((w,256))
    w_new, h_new = image_process.size
    image_process = image_process.crop((w_new//2-112,h_new//2-112,w_new//2+112,h_new//2+112))
    tensor = transforms.ToTensor()
    image_process = tensor(image_process)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
    image_process = normalize(image_process)
    np_image = np.array(image_process)
    np_image = np.ndarray.transpose(np_image)
    return np_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if args.image_path:
        image_path = args.image_path             
    if args.checkpoint:
        checkpoint = args.checkpoint
    if args.topk:
        topk = args.topk           
    if args.gpu:
        gpu = args.gpu

    # TODO: Implement the code to predict the class from an image file
    to_tensor=transforms.ToTensor()
    image_path = to_tensor(image_path)
    image_path = image_path.unsqueeze(0)
    image_path = image_path.to('cuda')
    output = model.forward(image_path)
    ps = torch.exp(output)
    prob, classa = ps.topk(topk)
    #print(classa)
    prob = prob.cpu()
    probs1 = prob.detach().numpy()
    classa = classa.cpu()
    classes1 = classa.detach().numpy()
    probs1 = probs1.tolist()
    classes1 = classes1.tolist()
    probs2 = probs1[0]
    classes2 = classes1[0]
    flower_dict = {}
    for i in range(len(classes2)):
        flower_dict[classes2[i]] = probs2[i]
    return probs2, classes2

image_final = process_image(image_path)
probs, classes = predict(image_final, model)
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
flowerclass =[]
for i in classes:    
    j = list(cat_to_name.items())[i][1]
    flowerclass.append(j)
    
print(flowerclass)
print(probs)
    
