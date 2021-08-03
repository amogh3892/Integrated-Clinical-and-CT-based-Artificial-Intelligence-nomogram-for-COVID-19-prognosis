import matplotlib.pyplot as plt
import time, math
import tables
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import h5py
from torch.autograd import Variable
import pandas as pd 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random 
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cf
import numpy as np 
import copy
import torchvision
import os
import matplotlib.pyplot as plt 
import pickle
from pytorchtools import EarlyStopping
from resnet import resnet10, resnet50, resnet101
from densenet import generate_model
from segUtil import Modified3DNet
from progressbar import *
import sys



def getProgressbar(message,size):
    """
    A function to implement progress bar: 
    Specifically to check the status of epoch while training (percentage of epoch completed)

    message: What message to display along with the progress bar 
    size: Total loops involved for progressbar to reach 100% 
    """
    widgets = [message, Percentage(), ' ', Bar(marker='-',left='[',right=']'),
            ' ', ETA()] #see docs for other options

    pbar = ProgressBar(widgets=widgets, maxval=size)

    return pbar

def visualizeImagesTorch(data_train,samp):
    """
    A function for visualizing a few sample images from pyorch dataset 
    data_train: A pytorch dataset 
    samp: sample index no. 
    """
    for i in range(3):     
        timg,lb,name = data_train.__getitem__(i + samp)        
        timg = np.asarray(timg)
                            
        print(lb)
        print(name)
        print(timg[0].min())
        print(timg[0].max())

                
        plt.subplot(131)
        plt.imshow(timg[1,70,:,:],cmap = 'gray')
        plt.subplot(132)
        plt.imshow(timg[0,70,:,:],cmap = 'gray')

        plt.show()
        


class ProstateDatasetHDF5(Dataset):
    """
    Pytorch dataset class 
    Outputs two channels: 
        CT volume as the first channel 
        Corresponding binary mask of COVID lesions as second channel. 
    """

    def __init__(self, fname,transforms = None):
        self.fname=fname
        self.file = tables.open_file(fname)
        self.tables = self.file.root
        self.nitems=self.tables.data.shape[0]
        self.file.close()
        self.data = None
        self.mask = None 
        self.names = None
        self.labels = None 
         
    def __getitem__(self, index):
                
        self.file = tables.open_file(self.fname)
        self.tables = self.file.root
        self.data = self.tables.data
        self.labels = self.tables.labels
        self.mask = self.tables.mask

        if "names" in self.tables:
            self.names = self.tables.names

        img = self.data[index,:,:,:]
        mask = self.mask[index,:,:,:]

        if self.names is not None:
            name = self.names[index]


        label = self.labels[index]
        self.file.close() 

        out = np.vstack((img[None],mask[None]))

        out = torch.from_numpy(out)
        return out,label,name

    def __len__(self):
        return self.nitems

def getData(foldername, batch_size, num_workers,cv):

    """
    A function to create pytorch data loaders
    dataset: foldername which contains train.h5, val.h5 and test.h5 

    """

    trainfilename = fr"{foldername}/train.h5"
    valfilename =fr"{foldername}/val.h5"
    testfilename = fr"{foldername}/test.h5"

    train = h5py.File(trainfilename,libver='latest',mode='r')
    val = h5py.File(valfilename,libver='latest',mode='r')
    test = h5py.File(testfilename,libver='latest',mode='r')

    trainlabels = np.array(train["labels"])
    vallabels = np.array(val["labels"])
    testlabels = np.array(test["labels"])

    train.close()
    test.close()
    val.close()
    
    zeros = (trainlabels == 1).sum()
    ones = (trainlabels != 1).sum()

    data_train = ProstateDatasetHDF5(trainfilename)
    data_val = ProstateDatasetHDF5(valfilename)
    data_test  = ProstateDatasetHDF5(testfilename)

    # Obtaining the train, val and test dataloader instances and loading them to a dictionary 
    trainLoader = torch.utils.data.DataLoader(dataset=data_train,batch_size = batch_size,num_workers = num_workers,shuffle = True)
    valLoader = torch.utils.data.DataLoader(dataset=data_val,batch_size = batch_size,num_workers = num_workers,shuffle = False) 
    testLoader = torch.utils.data.DataLoader(dataset=data_test,batch_size = batch_size,num_workers = num_workers,shuffle = False) 

    dataLoader = {}
    dataLoader['train'] = trainLoader
    dataLoader['val'] = valLoader
    dataLoader['test'] = testLoader

    return dataLoader, zeros, ones 



def run(mn, device, dataset, zeros, ones, num_epochs, learning_rate, weightdecay, patience, cv):

    # choosing the architecture 
    if mn == "resnet10":
        model = resnet10(num_classes=2)
    if mn == "resnet50":
        model = resnet50(num_classes=2)
    if mn == "resnet101":
        model = resnet101(num_classes=2)
    if mn == "densenet121":
        model = generate_model(121,n_input_channels=2,num_classes=2)
    if mn == "densenet169":
        model = generate_model(169,n_input_channels=2,num_classes=2)
    if mn == "densenet201":
        model = generate_model(201,n_input_channels=2,num_classes=2)
    if mn == "aip":
        model = Modified3DNet(in_channels=2, n_classes=2)


    model.to(device)

    total = zeros + ones 

    # define weights based on how the training set is balanced
    weights = [zeros/float(total),ones/float(total)]

    class_weights = torch.FloatTensor(weights).cuda(device)

    criterion=nn.CrossEntropyLoss(weight = class_weights)

    # defining the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weightdecay)

    niter_total=len(dataLoader['train'].dataset)/batch_size

    display = ["test","val"]

    results = {} 
    results["patience"] = patience

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    modelname = fr"{dataset}_{mn}_{learning_rate}_{weightdecay}"
    parentfolder = r"Data/"

    print(modelname)


    # start training 
    # the predictions and checkpoint will be saved in <parentfolder>/<modelname>

    for epoch in range(num_epochs):

        pred_df_dict = {} 
        results_dict = {} 
        
        
        for phase in ["train","val",'test',]:


            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            confusion_matrix=np.zeros((2,2))
            
            loss_vector=[]
            ytrue = [] 
            ypred = [] 
            ynames = [] 
            features = None 

            niter_total_phase=len(dataLoader[phase].dataset)/batch_size
            pbar = getProgressbar(fr'{phase} epoch {epoch}  : ',niter_total_phase)
            pbar.start()

            for ii,(data,label,name) in enumerate(dataLoader[phase]):
                if data.shape[0] != 1:

                    label=label.squeeze().long().to(device)
                    data = Variable(data.float().cuda(device))


                    with torch.set_grad_enabled(phase == 'train'):

                        output,feat = model(data)
                        output = output.squeeze()

                        feat = feat.detach().data.cpu().numpy()
                        features = feat if features is None else np.vstack((features,feat))

                        try:
                            _,pred_label=torch.max(output,1)

                        except:
                            import pdb 
                            pdb.set_trace()

                        probs = F.softmax(output,dim = 1)

                        loss = criterion(probs, label)

                        probs = probs[:,1]

                        loss_vector.append(loss.detach().data.cpu().numpy())

                        if phase=="train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()        

                        ypred.extend(probs.cpu().data.numpy().tolist())
                        ytrue.extend(label.cpu().data.numpy().tolist())
                        ynames.extend(list(name))

                        pred_label=pred_label.cpu()
                        label=label.cpu()
                        for p,l in zip(pred_label,label):
                            confusion_matrix[p,l]+=1
                        
                pbar.update(ii)
            pbar.finish()

            total=confusion_matrix.sum()        
            acc=confusion_matrix.trace()/total
            loss_avg=np.mean(loss_vector)
            auc = roc_auc_score(ytrue,ypred)



            columns = ["FileName","True", "Pred","Phase"]

            for fno in range(features.shape[1]):
                columns.append(fr"feat_{fno}")

            
            pred_df = pd.DataFrame(np.column_stack((ynames,ytrue,ypred,[phase]*len(ynames),features)), 
                                columns = columns)

            pred_df_dict[phase] = pred_df

            results_dict[phase] = {} 
            results_dict[phase]["loss"] = loss_avg
            results_dict[phase]["auc"] = auc 
            results_dict[phase]["acc"] = acc 

            if phase == 'train':
                print("Epoch : {}, Phase : {}, Loss : {}, Acc: {}, Auc : {}".format(epoch,phase,loss_avg,acc,auc))
            elif phase in display:
                print("                 Epoch : {}, Phase : {}, Loss : {}, Acc: {}, Auc : {}".format(epoch,phase,loss_avg,acc,auc))
                

            for cl in range(confusion_matrix.shape[0]):
                cl_tp=confusion_matrix[cl,cl]/confusion_matrix[:,cl].sum()

            if phase == 'test':
                df = pred_df_dict["test"].append(pred_df_dict["train"], ignore_index=True)
                early_stopping(1-auc, model, modelname, df, results_dict,parentfolder =None)

            if phase == 'test':
                if auc > 0.8:
                    sys.exit()


            if early_stopping.early_stop:
                print("Early stopping")
                break

        if early_stopping.early_stop:
            break


if __name__ == "__main__":
    
    # Cross validation splits 
    cvs = range(3)

    device = torch.device(f"cuda:{sys.argv[2]}" if torch.cuda.is_available() else "cpu")

    # model name as first command line argument
    mn = sys.argv[1]

    # The foldername will be appended by _{cv}. The cross validation split 
    foldername = "<foldername>"

    # batch size as second command line argument. 
    batch_size = int(sys.argv[3])
    
    # number of workers for parallel processing
    num_workers = 8

    # patient criteria for early stopping
    # if validation loss increases consecutively for the defined 'patience', network training is stopped
    patience = 15

    # max number of epochs 
    num_epochs = 200

    # list of learning rates and weight decays to loop through. 
    learning_rates = [1e-3, 1e-4, 1e-5]
    weightdecays = [1e-3, 1e-2, 1e-1]


    for learning_rate in learning_rates:
        for weightdecay in weightdecays:
            for cv in cvs:
                _foldername = f"{foldername}_{cv}"

                dataLoader, zeros, ones = getData(_foldername, batch_size, num_workers,cv)
                run(mn,device,dataset, zeros, ones, num_epochs, learning_rate, weightdecay, patience, cv)



