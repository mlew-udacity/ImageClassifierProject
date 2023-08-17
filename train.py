# Imports
import utility as utils
import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from collections import OrderedDict
from workspace_utils import active_session
import os

# Function to parse arguments
def arg_parser():
    parser = argparse.ArgumentParser(description="Training Neural Network")
    parser.add_argument('--data_dir', help='Set Data Directory', dest="data_dir", action="store", default="flowers")
    parser.add_argument('--arch', help='Set Network Architecture - VGG11, VGG13, VGG16 OR VGG19', dest="arch", action="store", default="vgg16")
    parser.add_argument('--checkpoint', help='Save Checkpoint', dest="checkpoint", action="store", default="checkpoint.pth")
    parser.add_argument('--learning_rate', help='Set Learning Rate', type=float, dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', help='Hidden Units for Hidden Layer 1', type=int, dest="hidden_units", action="store", default=1096)
    parser.add_argument('--epochs',help='Epochs', dest="epochs", action="store", type=int, default=3)
    parser.add_argument('--gpu', help='Use GPU', dest="gpu", action="store_true")
    args = parser.parse_args()
    return args

# Function to build 1 layer neural network - Arguments include model, inputs, 2 hidden inputs, outputs
def build_model(model, inputs=25088, hidden=1096, outputs=102, device="gpu", learn_rate=0.001):
    
    # Run Classifer function and attach to model
    model.classifier = utils.classifier(model, inputs, hidden, outputs)
    criterion = nn.NLLLoss() # Define Loss - negative likelihood loss
    optimiser = optim.Adam(model.classifier.parameters(), lr=learn_rate) # Optimiser - get the parameters from model, set learning rate

    #move our model to device e.g. cuda or cpu
    model.to(device)
    return model, criterion, optimiser

# Function to validate network
def validate_network(model, validloader, criterion, device):
    valid_loss = 0
    accuracy = 0
    model.to(device)
    
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        output_logps = model.forward(images)
        valid_loss += criterion(output_logps, labels).item()

        # Calculate accuracy
        probability = torch.exp(output_logps)
        top_p, top_class = probability.topk(1, dim=1) #top 1 of each row (along columns)
        equals = top_class == labels.view(*top_class.shape) #compares class to labels (reshapes labels tensor)
        accuracy += equals.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

# Function to Train and Validate Network - Arguments include Epochs and validation_step for when to run validation function.
def train_network(model, criterion, optimiser, epochs, validation_step, trainloader, validloader, device):
    steps=0
    running_loss = 0
    
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1 #increment steps

            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad() #zero out gradients
            output_logps = model.forward(images)             #Log probability from model with forward pass
            
            # Calculate loss and Increment
            loss = criterion(output_logps, labels) 
            loss.backward() #backwards pass - calculate gradient
            optimiser.step() #Adjust parameters
            running_loss += loss.item() 
            
           # Validation - For every Xth step - test networks accuracy and loss on validation data set.
            if steps % validation_step == 0:
                valid_loss = 0
                accuracy = 0
                model.eval() #turns off model inference mode - ie. turns off dropout so we can use network for make predictions
                with torch.no_grad(): #no grad turn off for validation 
                    # Run Validation function to calculation valiation loss and accuracy
                    valid_loss, accuracy = validate_network(model, validloader, criterion)
                           
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Training loss: {running_loss/validation_step:.3f}.. " #average of training loss
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. " # average loss - len(validloader) tells us how many batches we are getting
                      f"Validation accuracy: {(accuracy/len(validloader))*100:.2f}%") #average accuracy - summing up accuracy for each batch / total number of batches 
                running_loss = 0 #set running loss back to 0
                model.train() #set model back into training mode 

                
# Function Test Network and print accuracy - Arguments include the model
def test_network(model, data_loader, device):
    accuracy = 0
    model.eval() #Turns off model inference mode
    model.to(device)
    
    #Turn off gradients for testing 
    with torch.no_grad(): 
        # Calculate accuracy and loss
        for images, labels in data_loader: #Get images and labels from test data to do validation
            images, labels = images.to(device), labels.to(device) #set to device
            output_logps = model.forward(images)  # get log probability from test set
            probability = torch.exp(output_logps) #logps = returning softmax - log probability of classes
            top_p, top_class = probability.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape) #compares class to labels
            accuracy += equals.type(torch.FloatTensor).mean()

        print( f"Test accuracy: {(accuracy/len(data_loader)*100):.2f}%") #average accuracy - summing up accuracy for each batch / total number of batches 
        model.train() #set model back into training mode
        
        
def main():
    #Load arguments
    args = arg_parser()
    arch = args.arch
    checkpoint = args.checkpoint    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    learn_rate = args.learning_rate
    epochs = args.epochs    
    inputs = 25088
    hidden = args.hidden_units    
    outputs = 102
    data_dir = args.data_dir
    validation_step = 5

    # Load Data for Training
    if os.path.exists(data_dir):          
        print("Loading Data for Training...")
        train_data, trainloader, testloader, validloader = utils.load_data(data_dir)
    else:
        print(f"{data_dir} path does not exist. Please try again")
        return

    # Build Model - based on a selection of VGG models
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print(f"{arch} is not a VGG model. Please choose vgg11, vgg13, vgg16 or vgg19")
        return
    
    print(f"Building Model Architecture {arch}")        
    model, criterion, optimiser = build_model(model, inputs, hidden, outputs, device, learn_rate)
    
    # Train Model and Validate
    print(f"Training Model at Learning Rate {learn_rate}")     
    active_session = True
    while active_session:
        train_network(model, criterion, optimiser, epochs, validation_step, trainloader, validloader, device)  
        active_session = False
        
    # Save Checkpoint
    print("Saving Checkpoint...")        
    utils.save_checkpoint(train_data, model, optimiser, checkpoint, arch, inputs, hidden, outputs, epochs)    

    # Testing Model with Testing Set 
    print("Testing Model accuracy...")
    test_network(model)
    
if __name__ == '__main__': main()        