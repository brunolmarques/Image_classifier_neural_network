# Imports statements
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import numpy as np
import copy
from PIL import Image
import argparse
import json
import matplotlib.pyplot as plt

def load_checkpoint(args):
    '''
        Load a previously saved trained model
    '''
    # Load the trained model
    checkpoint = torch.load(args.filepath, map_location=lambda storage, loc: storage)
    arch = checkpoint['model']
    
    if arch == 'vgg':
        model = models.vgg16(pretrained=True)        
    elif arch == 'densenet':
        model = models.densenet201(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    elif arch == 'resnet':
        model = models.resnet101(pretrained = True)
    
    # Identify if GPU is enabled and set the device
    if args.gpu == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
        
    # Set model to GPU or CPU
    model = model.to(device)    
    
     # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Update the classifier
    classifier = Network(model, checkpoint['input_size'], checkpoint['output_size'], checkpoint['hidden_layers'], checkpoint['dropout'])
    model.classifier = classifier
    
    # Update the model state and weights
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint

def process_image(args):
    ''' 
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Create the image setup
    img_process = transforms.Compose([transforms.Resize(256), 
                                    transforms.CenterCrop(224)])
    
    # Load the image and process it
    pil_image = Image.open(args.image_path)
    pil_image = img_process(pil_image)
    
    # Convert image to NumPy array
    np_image = np.array(pil_image, dtype = float) / 255  
        
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    processed_image = (np_image - mean)/std    
    processed_image = np.transpose(processed_image, (2, 0, 1))
    
    return processed_image

def imshow(image, ax=None, title=None):
    '''
        Function to re-process the image so it can be displayed.
    '''
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    
    return ax

def predict(args, model, idx_to_class):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Identify if GPU is enabled and set the device
    if args.gpu == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    
    # Load the image
    image = torch.FloatTensor([process_image(args)])
    image = image.to(device)
    model.eval()
    
    # Forward Pass the image
    output = model.forward(image)
    ps = torch.exp(output)
    
    prob = ps.topk(args.topk)[0].cpu().data.numpy()[0]
    top_idx = ps.topk(args.topk)[1].cpu().data.numpy()[0]

    top_k = [idx_to_class[x] for x in top_idx]
    
    return prob, top_k

class Network(nn.Module):
    def __init__(self, model, input_size, output_size, hidden_layers, drop_p):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' 
            Forward pass through the network, returns the output logits.
        '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

def wrapler_func(args):
    '''
        Consolidates all the arguments and call the necessary functions and print final result.
    '''
    # Get index to class mapping
    model, checkpoint = load_checkpoint(args)
    idx_to_class = { v : k for k,v in checkpoint['class_to_idx'].items()}
    
    # Load test image
    img_path = args.image_path
    image = Image.open(img_path)

    # Call probabilities and find respective indexes
    probs , idx = predict(args, model, idx_to_class)

     # Load category names
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    # Set test image name
    category_folder = img_path.split('/')[-2]
    flower = cat_to_name[category_folder]

    # Print results
    print("\nTest flower is {}".format(flower.upper()))
    
    for prob, i in zip(probs, idx):
        print("\n{} has a probability of {:.2%}".format(cat_to_name[i], prob))

def main():
    '''
        Main function parsing all the functions arguments.
    '''
    
    # Parser function, receives inputs from console
    parser = argparse.ArgumentParser(description='Model predict')
    parser.add_argument('--image_path', type = str, default = 'flowers/test/1/image_06743.jpg', help = 'Images to test prediction')
    parser.add_argument('--filepath', type = str, default = 'my_checkpoint.pth', help = 'File path to the saved trained model')
    parser.add_argument('--gpu', type = bool, default = False, help = 'GPU enabled or disabled')
    parser.add_argument('--topk', type = int, default = 5, help = 'Number of possible results to be displayed')
    args = parser.parse_args()
    
    # Call the kick-off function
    wrapler_func(args)
    
if __name__ == "__main__":
    main()   