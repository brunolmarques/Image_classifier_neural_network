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
import copy
import argparse
import json

def process_data(*args):
    '''
        Function that receives a data directory of images and access them.
    '''

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test':transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # TODO: Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(root = args[0] + '/' + x, transform = data_transforms[x]) for x in ['train', 'valid', 'test']}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True) for x in ['train', 'valid', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    
    return image_datasets, dataloaders, dataset_sizes

def train_model(args, model, criterion, optimizer, scheduler, num_epochs=10):
    '''
        Train model function based on the begginer function example proposed on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#id1
    '''
    since = time.time()
   
    # Identify if GPU is enabled and set the device
    if args.gpu == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    
    # Load datasets and loaders
    image_datasets, dataloaders, dataset_sizes =  process_data(args.data_dir)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward Pass
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

class Network(nn.Module):
    def __init__(self, model, args):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        
        input_size = model.classifier.in_features
        output_size = 102
        hidden_layers = args.hidden_layers
        drop_p = args.dropout
        
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
    
def construct_train(args):
    ''' 
        Consolidates all the inputs and functions to train the model.
    '''
    # Identify if GPU is enabled and set the device
    if args.gpu == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    
    # Load the images datasets and loaders
    image_datasets, dataloaders, dataset_sizes =  process_data(args.data_dir)
    
    # Load the choosen model, if model inputed do not exist raise an error
    if args.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif args.arch == 'densenet':
        model = models.densenet201(pretrained=True)
    elif args.arch == 'resnet':
        model = models.resnet101(pretrained=True)
    elif args.arch == 'vgg':
        model = models.vgg16(pretrained=True)
    else:
        raise ErrorValue('Invalid model name! Choose alexnet, densenet, resnet or vgg.')
        
    # Freeze parameters so it don't back propagate through the whole model
    for param in model.parameters():
        param.requires_grad = False
    
    # Set model to GPU or CPU
    model = model.to(device)
    
    # Create a new classifier and appends it to the model
    classifier = Network(model, args)
    model.classifier = classifier
    
    # Define criterion (loss function) and optimizer (learning algorithm)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    
    # Decay Learn rate by a factor of 0.5 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train the model
    model = train_model(args, model, criterion, optimizer, exp_lr_scheduler, num_epochs=args.epochs)
    
    # Save the model
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    model.epochs = args.epochs

    checkpoint = {'input_size': 1920,
                  'output_size': 102,
                  'model': args.arch,
                  'dropout': args.dropout,
                  'hidden_layers': args.hidden_layers,
                  'state_dict': model.state_dict(),
                  'optimizer_dict':optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'epochs': model.epochs}

    torch.save(checkpoint, args.save_name)
    
def main():
    '''
        Main function parsing all the functions arguments.
    '''
    
    # Parser function, receives inputs from console
    parser = argparse.ArgumentParser(description='Model trainer')
    parser.add_argument('--data_dir', type = str, default = 'flowers/', help = 'Images directory')
    parser.add_argument('--arch', type = str, default = 'densenet', help = 'Architecture model (available: alexnet, densenet, resnet, vgg)', required = True)
    parser.add_argument('--lr', type = float, default = 0.001, help = 'Learning rate')
    parser.add_argument('--hidden_layers', type = int, default = [500], help = 'Sizes of the hidden layers (list of integers)')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Number of epochs')
    parser.add_argument('--gpu', type = bool, default = False, help = 'GPU enabled or disabled')
    parser.add_argument('--save_name' , type = str, default = 'my_checkpoint.pth', help = 'Name of the saved trained model')
    parser.add_argument('--dropout', type = float, default = 0.5, help = '')
    args = parser.parse_args()

    # Load category names
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Call kick-off function
    construct_train(args)

if __name__ == "__main__":
    main()