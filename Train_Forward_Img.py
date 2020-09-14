import torch
import os
import numpy as np
import pandas as pd
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter

# To run the NN on GPU
use_cuda = 0
device = torch.device("cuda" if use_cuda else "cpu")

#writer = SummaryWriter('runs/CNN_expt_1')

# Hyperparameters
batchSize = 32
lr = 0.001
momentum = 0.9
decayRate = 0.99
trainRatio = 0.9    # Splitting dataset into training and Validation
inputDim = 24      # Input to the NN : Hole-Vectors(5x20) + Structural Parameters(6)
outputDimS = 64      # Output from the NN : Coupling Efficiency (Near field + Far field)
outputDimH = 64
outputDim = 40
epochs = 3
TImSize = 224

# Helper functions

# Function to write in files 

def writeInFiles(pred, gt, st, iter):  # gt : Ground truth spectrum, st : Structure parameters
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    st = st.detach().cpu().numpy()
    folder = "Results_Forward1/"
    for i in range(0, pred.shape[0]):
        f = folder + str(iter+1) + "_" + str(i) + "_spec_pred.txt"
        p = pred[i] 
        np.savetxt(f, p,  delimiter='\t')

    for i in range(0, gt.shape[0]):
        f = folder + str(iter+1) + "_" + str(i) + "_spec_real.txt"
        p = gt[i]
        np.savetxt(f, p,  delimiter='\t')

    for i in range(0, st.shape[0]):
        f = folder + str(iter+1) + "_" + str(i) + "_structure.txt"
        p = st[i]
        np.savetxt(f, p,  delimiter='\t')


# Function to load saved checkpoint file
        
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.eval()
    return model


# Function to read data from csv files
    
def readData(holeVectorPath, structurePath, effPath):
    structure = pd.read_csv(structurePath).values
    efficiency = pd.read_csv(effPath).values
    holeVectors = pd.read_csv(holeVectorPath).values[:structure.shape[0], :]
    input_ = np.hstack((structure,structure,structure, structure))  # Horizontally staking hole vectors and structure
    return torch.tensor(input_).float(), torch.tensor(efficiency).float()


def readDataImg(holeVectorPath, structurePath, effPath):
    structure = pd.read_csv(structurePath).values
    efficiency = pd.read_csv(effPath).values
    transform = transforms.Compose([transforms.Resize((TImSize, TImSize)), transforms.ToTensor()])
    
    dataset = datasets.ImageFolder(root = "./holes/", transform = transform)
    print(type(dataset))
    #holeVectors = pd.read_csv(holeVectorPath).values[:structure.shape[0], :]
    input_ = np.hstack((structure, structure,structure, structure))  # Horizontally staking hole vectors and structure
    return torch.tensor(input_).float(), torch.tensor(efficiency).float(), dataset



# Defining the forward network 
class forwardNet(nn.Module):
    def __init__(self, lr, inputDim, outputDim):
        super(forwardNet, self).__init__()
        modelStr = [nn.Linear(inputDim, 300), nn.BatchNorm1d(300), nn.LeakyReLU(),
                 nn.Linear(300, 300), nn.BatchNorm1d(300), nn.LeakyReLU(),nn.Dropout(p=0.5),
                 nn.Linear(300, 300), nn.BatchNorm1d(300), nn.LeakyReLU(),nn.Dropout(p=0.5),
                 nn.Linear(300, 300), nn.BatchNorm1d(300), nn.LeakyReLU(),nn.Dropout(p=0.5),
                 nn.Linear(300, 300), nn.BatchNorm1d(300), nn.LeakyReLU(),nn.Dropout(p=0.5),
                 nn.Linear(300, outputDimS), nn.Sigmoid()]

        modelHole = models.resnet18(pretrained=True)
        num_fltrs = modelHole.fc.in_features
        modelHole.fc = nn.Linear(num_fltrs,outputDimH)
        

        
        outputDecoder = [nn.Linear(outputDimH + outputDimS, 100), nn.BatchNorm1d(100), nn.ReLU(), nn.Dropout(p=0.5),
                 nn.Linear(100, 200), nn.BatchNorm1d(200), nn.ReLU(), nn.Dropout(p=0.5),
                 nn.Linear(200, 300), nn.BatchNorm1d(300), nn.LeakyReLU(),nn.Dropout(p=0.5),
                 nn.Linear(300, outputDim), nn.ReLU()]


        self.modelStr = nn.Sequential(*modelStr)
        #self.modelHole = nn.Sequential(*modelHole)
        self.modelHole = modelHole
        self.outputDecoder = nn.Sequential(*outputDecoder)

        self.lr = lr
        self.decayRate = decayRate
        self.momentum = momentum
        self.optim = torch.optim.RMSprop(list(self.modelStr.parameters()) + list(self.modelHole.parameters()) + list(self.outputDecoder.parameters()), self.lr, self.decayRate)
        self.criterion = nn.MSELoss()
        self.accuracy = nn.L1Loss()

    def train(self, x, y,z):      # For training dataset
        out1 = self.modelStr(x)
        print(type(out1))
        out2 = self.modelHole(z)
        
        print(type(out2))
        out = torch.cat((out1,out2),1)
        print("$$$$$$$$$$$$$$$$$$")
        out = self.outputDecoder(out)
        

        self.modelStr.zero_grad()
        self.modelHole.zero_grad()
        self.outputDecoder.zero_grad()


        loss = self.criterion(out, y)
        acc = self.accuracy(out, y)
        loss.backward()
        self.optim.step()

        return loss, acc, out

    def eval(self, x, y, z):         # For validation dataset

        out1 = self.modelStr(x)
        out2 = self.modelHole(z)
        out = torch.cat((out1,out2),1)
        out = self.outputDecoder(out)


        self.modelStr.zero_grad()
        self.modelHole.zero_grad()
        self.outputDecoder.zero_grad()

        loss = self.criterion(out, y)
        acc = self.accuracy(out, y)
        return loss, acc, out


forwardModel = forwardNet(lr, inputDim, outputDim).to(device)
trainlossFile = open("train_loss_forward.txt", "+w")
trainaccFile = open("train_acc_forward.txt", "+w")
vallossFile = open("val_loss_forward.txt", "+w")
valaccFile = open("val_acc_forward.txt", "+w")

#writer.add_graph(forwardModel)
#writer.close()


if __name__ == '__main__': 
    inputData, outputData, images = readDataImg("Hole_vectors.csv", "Structural_Parameters.csv", "Coupling_Efficiency.csv")
    
    '''
    trainInput = inputData[0:int(inputData.shape[0]*trainRatio), :]
    valInput = inputData[int(inputData.shape[0]*trainRatio):inputData.shape[0]:, :]
    trainOutput = outputData[0:int(outputData.shape[0]*trainRatio), :]
    valOutput = outputData[int(outputData.shape[0]*trainRatio):outputData.shape[0], :]
    print(trainOutput.shape)
    print(valOutput.shape)
    print(type(images))
    #trainInputHoles = images[0:int(inputData.shape[0]*trainRatio), :]
    #valInputHoles = images[int(inputData.shape[0]*trainRatio):inputData.shape[0]:, :]

    trainInput = torch.utils.data.DataLoader(trainInput, batchSize, True)
    trainOutput = torch.utils.data.DataLoader(trainOutput, batchSize, True)
    valInput = torch.utils.data.DataLoader(valInput, len(valInput), True)
    valOutput = torch.utils.data.DataLoader(valOutput, len(valOutput), True)
    trainInputHoles = torch.utils.data.DataLoader(images, batchSize, True)
    #valInputHoles = torch.utils.data.DataLoader(trainOutput, batchSize, True)
    print(type(trainInputHoles))
    '''

    data_size = inputData.shape[0]
    train_size = int(inputData.shape[0]*trainRatio)

    inputData = torch.utils.data.DataLoader(inputData, batchSize, True)
    outputData = torch.utils.data.DataLoader(outputData, batchSize, True)
    images = torch.utils.data.DataLoader(images, batchSize, True)

    for epoch in range(0, epochs):

        print("-----------------------EPOCH NUMBER--------------------------", epoch + 1)
        avgtrainLoss, avgtrainAcc, avgvalLoss, avgvalAcc = 0.0, 0.0, 0.0, 0.0
        # Training iteration in each epoch

        for (iterx, x), (itery, y), (iterz, z) in zip(enumerate(inputData), enumerate(outputData), enumerate(images)):
            #print(type(x))
            #print(type(y))
            #print((x.shape))
            if iterx%100 == 0:
                print(iterx)
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            z = Variable(z[0]).to(device)
            loss, acc, out = forwardModel.train(x, y, z)
            avgtrainLoss += loss
            avgtrainAcc += torch.mean(acc)
            if iterx == data_size:
                break

        ''' 
        for (iterx, x), (itery, y) in zip(enumerate(trainInput), enumerate(trainOutput)):
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            loss, acc, out = forwardModel.train(x, y)
            avgtrainLoss += loss
            avgtrainAcc += torch.mean(acc)

        '''
        avgtrainLoss = avgtrainLoss/(iterx+1)
        avgtrainAcc = avgtrainAcc/(iterx+1)
        print("AVERAGE TRAINING LOSS : ", avgtrainLoss.item())
        print("AVERAGE TRAINING ACCURACY : ", avgtrainAcc.item())
        trainlossFile.write(str(avgtrainLoss.item()))
        trainlossFile.write("\n")
        trainaccFile.write(str(avgtrainAcc.item()))
        trainaccFile.write("\n")
        
        
        if(epoch and epoch %1 ==0):
            checkpoint = {'model': forwardNet(lr, inputDim, outputDim).to(device) ,'state_dict': forwardModel.state_dict(), 'optimizer' : forwardModel.optim.state_dict()}
            torch.save(checkpoint, 'Models_Forward/' + str(epoch) + '_forward_Model.pth')
        
        # Validation iteration in each epoch
        '''
        for (iterx, x), (itery, y) in zip(enumerate(valInput), enumerate(valOutput)):
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            loss, acc, out = forwardModel.eval(x, y)
            avgvalLoss += loss
            avgvalAcc += torch.mean(acc)
            if epoch %10 ==0:
                writeInFiles(out, y, x, epoch)

        '''

        for (iterx, x), (itery, y), (iterz, z) in zip(enumerate(inputData), enumerate(outputData), enumerate(images)):
            if iterx<data_size:
                pass
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            z = Variable(z[0]).to(device)
            loss, acc, out = forwardModel.eval(x, y, z)
            avgvalLoss += loss
            avgvalAcc += torch.mean(acc)
            if epoch %10 ==0:
                writeInFiles(out, y, x, epoch)

        avgvalLoss = avgvalLoss/(iterx+1)
        avgvalAcc = avgvalAcc/(iterx+1)
        print("AVERAGE VALIDATION LOSS : ", avgvalLoss.item())
        print("AVERAGE VALIDATION ACCURACY : ", avgvalAcc.item())
        vallossFile.write(str(avgvalLoss.item()))
        vallossFile.write("\n")
        valaccFile.write(str(avgvalAcc.item()))
        valaccFile.write("\n")
        
        
        
        
        
        
        
        