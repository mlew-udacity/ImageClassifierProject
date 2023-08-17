# Imports
import utility as utils
import argparse
import torch
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image

# Arg Parser
def arg_parser():
    parser = argparse.ArgumentParser(description='Predict flower name.')
    parser.add_argument('image_path', help='Image Directory', action="store")
    parser.add_argument('checkpoint', help='Load Checkpoint File', action="store", default="checkpoint.pth")
    parser.add_argument('--top_k', help='Top K', type=int, dest="top_k", action="store", default=5)
    parser.add_argument('--category_names', help='Category Names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', help='Use GPU', dest="gpu", action="store_true")
    args = parser.parse_args()
    return args

#Predict function to return top number of predictions for image
def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    with torch.no_grad():    # Turn off gradients
        device = torch.device('cpu') #set to CPU mode as GPU not required
        model.eval()
        model.to(device) #set model to device

        # Run function to process image 
        image = utils.process_image(image_path)    
        image = image.unsqueeze_(0)
        image.to(device) #set image to device

        output_logps = model.forward(image)  # Forward pass - get log probability 
        probability = torch.exp(output_logps) # Probability
        top_p, top_class = probability.topk(topk, dim=1)
        
        #Convert tensors to numpy array
        top_p=top_p.numpy() 
        top_class=top_class.numpy()

        # store class_to_idx values from modelin a dictionary
        idx_to_class = {model.class_to_idx[i]: i for i in model.class_to_idx} 
        classes = [idx_to_class[i] for i in top_class[0]] # Iterate get top classes
        probs = top_p[0] # get Probabilities 
    
        return probs, classes
    
    
# Function: Display an image along with the top classes
def display_predict(image_path, model, device, cat_to_name, top_k=5):
    
    #Based on image get probability of image
    probs, classes= predict(image_path, model, device, top_k)

    #Get labels for top 5 - classes returned from predict function and add labels and probs to dictionary
    labels = [cat_to_name[i] for i in classes]
    flower_dict = dict(zip(labels, probs))
    
    print(f"Top {top_k} Predictions...")  
    
    # Plot flower and assign title
    i = 1
    for label, prob in flower_dict.items():
       print(f"Flower {i}: {label} - {prob*100:.2f}%")
       i+=1
    
def main():
    #Load arguments
    args = arg_parser()
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")  

    #Load Saved Checkpoint
    if os.path.isfile(checkpoint):        
        model, optimiser = utils.load_checkpoint(checkpoint, device)
    else: 
        print(f"Checkpoint file {checkpoint} does not exist.")   
        return
    
    #Load category names
    if os.path.isfile(category_names):        
        cat_to_name = utils.load_labels(category_names)
    else:    
        print(f"Category Names file {category_names} does not exist.")   
        return
    
    # Predict Probabilities and display image and bar graph based on topK
    if os.path.isfile(image_path):  
        display_predict(image_path, model, device, cat_to_name, top_k)
    else:    
        print(f"Image filepath {image_path} does not exist.")   
        return
    
if __name__ == '__main__': main()      

    