# Imports
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models    
from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import json
import time

#Function to load data
def load_data(data_dir='flowers'):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),  #Rotate 30 degrees each way
                                           transforms.RandomResizedCrop(224), #Crop to 224 x 224 pixel
                                           transforms.RandomHorizontalFlip(), #Flip horizonally
                                           transforms.ToTensor(),  #Convert Tensor
                                           transforms.Normalize([0.485, 0.456, 0.406],  #Mean
                                                                [0.229, 0.224, 0.225])]) #Standard Dev

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)

    return train_data, trainloader, testloader, validloader


#Classifer function to build layers attach to existing model eg VGG19
def classifier(model, inputs, hidden,outputs):
    for param in model.parameters():
        param.requires_grad = False


    #List of different operations and pass a tensor through sequentially
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(inputs, hidden)), #layer 1 
                            ('relu', nn.ReLU()),
                            ('dropout',nn.Dropout(0.5)),
                            ('fc3',nn.Linear(hidden,outputs)),#output size = 102
                            ('output', nn.LogSoftmax(dim=1))]))# For using NLLLoss()
    return classifier
    

        
#Save the checkpoint 
def save_checkpoint(train_data, model, optimiser, filepath='checkpoint.pth', arch='vgg16', inputs=25088, hidden=4096, outputs=102, epochs=2, learn_rate=0.003):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'arch': arch,
                  'hidden': hidden,                 
                  'inputs': inputs,
                  'outputs': outputs,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimiser_state': optimiser.state_dict(),
                  'epochs': epochs,
                  'learning_rate': learn_rate}
    torch.save(checkpoint, filepath)

#Load saved checkpoint
def load_checkpoint(filepath='checkpoint.pth'):
    criterion = nn.NLLLoss()
    checkpoint = torch.load(filepath)
    
    #Load Model with checkpoints
    if checkpoint['arch'] == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print("Not a vgg model - vgg11, vgg13, vgg16 or vgg19")
    
    model.classifier = classifier(model, checkpoint['inputs'], checkpoint['hidden'], checkpoint['outputs'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    #Load Optimiser
    optimiser = optim.Adam(model.classifier.parameters(), checkpoint['learning_rate'])      
    optimiser.load_state_dict(checkpoint['optimiser_state'])
    epochs = checkpoint['epochs']
    
    return model, optimiser, criterion, epochs    
 
#Function to process image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])
    # Transforms image   
    image_process =  img_transform(img)
    return image_process

# Function to show image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def load_labels(filepath='cat_to_name.json'):
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)    
    return cat_to_name