#Import dependencies.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms

#parser for reading command-line arguments
import argparse

import time
import os
import json
import shutil
import io


def test(model, test_loader, criterion, device):
    '''
        This function takes a model and a testing data loader and get the
        test accuray/loss of the model.
    '''
    # set model to evaluate mode
    model.eval() #because finetuning (pre-trained)
    
    correction = 0
    loss_counter = 0
    
    with torch.no_grad():
        for image, target in test_loader:
            image = image.to(device)
            target = target.to(device)
            
            output = model(image)
            loss = criterion(output, target) ## nn.CrossEntropyLoss()
            #loss = nn.NLLLoss(output, target, reduction="sum")
            pred = torch.argmax(input = output, dim = 1)
            
            loss_counter += loss.item()
            correction += torch.sum(pred == target.data).item()
            
    # print statistics
    print(f"Test set: Average loss: {loss_counter/len(test_loader.dataset)}")
    print(f"Test Accuracy {100*correction/len(test_loader.dataset)}%\n")



def train(model, train_loader, criterion, optimizer, device):
    '''
        This function takes a model and data loaders for training 
        and will get train the model.
    '''
    
    loss_counter = 0
    
    # set model to training mode
    model.train()

    # training loop
    for batch_n, (image, target) in enumerate(train_loader, 1):
        image = image.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        #loss = nn.NLLLoss(output, target, reduction="sum")
        loss.backward()
        optimizer.step()

        #statistics
        loss_counter += loss.item()
        
    # return loss
    return loss_counter/len(train_loader.dataset)



def vgg():
    '''
        This function initializes vgg pretrained model
    '''

    # download pretrained model
    model = models.vgg16(pretrained = True)
    
    # print model structure before replacing output layers
    print(model)
    
    # freeze all the pre-trained layers
    for p in model.parameters():
        p.requires_grad = False
    
    # get the numbers of input features in the last FC layer
    nin_features = model.classifier[6].in_features
    print(f"Number of imput features in the last classifier: {nin_features}")

    # replace the linear layers of the classifier
    model.classifier[0].requires_grad = True  ## IMPROVE ACC
    model.classifier[3].requires_grad = True  ## IMPROVE ACC

    # in the last linear layer, match the number of classes in the dataset
    # note: there're 107 n_classes in the dataset
    n_classes = 107
    
    ### BEST FIT + Adadelta: acc. ~ 20%
    model.classifier[6] = nn.Sequential(nn.Linear(in_features = nin_features, out_features = n_classes), nn.LogSoftmax(dim = 1))
    
    ### GOOD FIT + Adadelta: acc. ~ 5%
    #model.classifier[6] = nn.Sequential(nn.Linear(nin_features, 214), nn.ReLU(), nn.Linear(214, n_classes), nn.LogSoftmax(dim = 1))
    
    # print final model structure
    print(model)
    
    return model



def net18():
    '''
        This function initializes ResNet18 pretrained model.
    '''

    # download pretrained model
    model = models.resnet18(pretrained = True)
    
    # freeze all the pre-trained layers
    for p in model.parameters():
        p.requires_grad = False
    
    # get the numbers of input features in the last FC layer
    nin_features = model.fc.in_features
    
    # replace the last FC layer to meet the number of classes (out_features) in the actual dataset
    model.fc = nn.Sequential(nn.Linear(in_features = nin_features, out_features = 133)) #oss. there're 133 n_classes in the dataset
    
    return model



def net34():
    '''
        This function initializes ResNet34 pretrained model
    '''

    # download pretrained model
    model = models.resnet34(pretrained = True)

    # print model structure before replacing output layers
    print(model)
    
    # freeze all the pre-trained layers
    for p in model.parameters():
        p.requires_grad = False
    
    # get the numbers of input features in the last FC layer
    nin_features = model.fc.in_features
    print(f"Number of imput features in the last linear layer: {nin_features}")
    
    n_classes = 107

    # replace the last FC layer to meet the number of classes (out_features) in the actual dataset
    model.fc = nn.Sequential(nn.Linear(in_features = nin_features, out_features = n_classes), nn.LogSoftmax(dim = 1)) ### only 2% acc

    # print final model structure
    print(model)
    
    return model



def create_data_loaders(data_path, batch_size):
    '''
    This is an optional function that you may or may not need
    depending on whether you need to use data loaders or not
    '''

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    return loader



def save_model(model, path_to_model):
    '''
        This function Save the trained model to 'SM_MODEL_DIR' /opt/ml/model
    '''

    print('Saving the model!!!')

    model = model.to('cpu')
    modeldir_path = os.path.join(path_to_model, "model.pth")
    torch.save(model.state_dict(), modeldir_path)
    #with open(modeldir_path, 'wb') as f:
    #torch.save(model.state_dict(), f)
    
    print(f"Model saved to {modeldir_path}")



def main(args):
    
    # define device to run the model
    use_cuda = args.num_gpus > 0
    print(f"Number of gpus available - {args.num_gpus}")
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Running on device {device}")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Running on device {device}")
    
    # print out hyperparameters
    print("Hyperparameters\n epochs: {},\n lr: {},\n batch_size: {},\n test_batch_size: {}".format(args.epochs, args.lr, args.batch_size, args.test_batch_size))
    
    ## Initialize the model
    #model = net34()
    model = vgg()
    
    # move model to GPU memory
    model = model.to(device)
    
    ## Create your loss
    loss_criterion = nn.CrossEntropyLoss()
    
    # define optimizer criteria
    #optimizer = optim.Adam(model.parameters(), lr=args.lr) ## for resnet18
    #optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9) ## for vgg16
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr) ### BEST  FIT!

    # create_data_loaders load data from S3
    train_loader = create_data_loaders(args.data_train, args.batch_size)
    
    ## call the train function to start training your model
    print("\n")
    for epoch in range(1, args.epochs + 1):
        loss = train(model, train_loader, loss_criterion, optimizer, device)
        # print out training progress
        print("Train Epoch: {}\t|\tAverage Loss: {}".format(epoch, loss))

    ## Test the model and show its accuracy
    test_loader = create_data_loaders(args.data_test, args.test_batch_size) 
    test(model, test_loader, loss_criterion, device)

    # save the trained model
    save_model(model, args.model_dir)
    


if __name__=='__main__':
    
    parser=argparse.ArgumentParser(description = "define hyperparameters for fine tuning Pytorch CIFAR10 model")
    
    ## Specify training args (hyperparameters sent by user)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=30,
        metavar="N",
        help="training batch size (default: 64)"
    )
    
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=20,
        metavar="N",
        help="test batch size (default: 100)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        metavar="N",
        help="number of epochs to train (default: 10)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.15,
        metavar="LR",
        help="learning rate (default: 3e-2)"
    )
    
    # get container environment variables
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--num_gpus", type=int, default=os.environ["SM_NUM_GPUS"]) ## number of gpus in the current container
    
    # Model directory (the default set by SageMaker, /opt/ml/model)
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"]) ## path to model artifacts
    
    # Data directories
    parser.add_argument('--channel_names', default=json.loads(os.environ['SM_CHANNELS'])) ## names of data channels
    parser.add_argument("--data_train", type=str, default=os.environ["SM_CHANNEL_TRAIN"]) ## path to training data
    parser.add_argument('--data_test', type=str, default=os.environ['SM_CHANNEL_TEST']) ## path to test data
    parser.add_argument('--data_valid', type=str, default=os.environ['SM_CHANNEL_VALID']) ## path to validation data
    parser.add_argument("--data_train_lst", type=str, default=os.environ["SM_CHANNEL_TRAIN_LST"]) ## path to training metadata
    parser.add_argument('--data_test_lst', type=str, default=os.environ['SM_CHANNEL_TEST_LST']) ## path to test metadata
    parser.add_argument('--data_valid_lst', type=str, default=os.environ['SM_CHANNEL_VALID_LST']) ## path to validation metadata
    
    # end argument definition
    args=parser.parse_args()
    
    main(args)

    
