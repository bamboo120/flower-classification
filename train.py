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
parser.add_argument('--data_dir', type=str, help='Path to dataset ')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--arch', type=str, help='Model architecture')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')
args, _ = parser.parse_known_args()
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)


# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def validation(model, validloader, criterion):
    valid_loss = 0
    accuracy = 0
    for images, labels in validloader:
        images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()    
    return valid_loss, accuracy

learning_rate = 0.001
hidden_units = 4096
epochs = 9
print_every = 40
steps = 0
running_loss = 0 
arch = 'vgg16'
if args.learning_rate:
    learning_rate = args.learning_rate
if args.hidden_units:
    hidden_units = args.hidden_units
if args.arch:
    arch = args.arch     
if args.epochs:
    epochs = args.epochs
if args.gpu:
    gpu = args.gpu
if args.checkpoint:
    checkpoint = args.checkpoint            
if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif arch == 'vgg13':
    model = models.vgg13(pretrained=True)
else:
    model = models.vgg16(pretrained=True)
    print('undefined architecture')

for param in model.parameters():
    param.requires_grad = False
        
model.classifier = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(25088, hidden_units)),
                      ('relu', nn.ReLU()),
                      ('dropout', nn.Dropout(p=0.5)),
                      ('fc2', nn.Linear(hidden_units, hidden_units)),
                      ('relu', nn.ReLU()),
                      ('dropout', nn.Dropout(p=0.5)),
                      ('fc2', nn.Linear(hidden_units, 102)),
                      ('output', nn.LogSoftmax(dim=1))
                      ]))
model = model.cuda()
cost = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()
    

for e in range(epochs):
    model.train()
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1      
        inputs, labels = inputs.to('cuda'), labels.to('cuda')        
        optimizer.zero_grad()
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()       
        running_loss += loss.item()      
        if steps % print_every == 0:
            model.eval()
            valid_loss = 0
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validloader, criterion)

            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "runing_Loss: {:.3f}".format(running_loss/print_every),
                  "Vadid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))
            running_loss = 0            

model.class_to_idx = train_data.class_to_idx
checkpoint = {
              'model': model,
              'state_dict': model.state_dict(),
              'datasets' : model.class_to_idx,
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),              
             }
torch.save(checkpoint, 'checkpoint.pth')



