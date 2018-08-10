
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
#LOADING DATASET#LOADIN 

batch_size = 128

train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=False)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


accuracy=[]

class Logistic_Regression(nn.Module):
    def __init__(self):
        super(Logistic_Regression,self).__init__()
        self.linear = nn.Linear(28*28,10)


    def forward(self,x):
        x  = x.view(-1, 28*28)
        return self.linear(x)


model = Logistic_Regression()

criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lr=0.01,params = model.parameters())


for epoch in range(10):
    epoch_loss = []
    step_acc  = []
    for i,(images,labels) in enumerate(train_loader):
        pred = model(images)
        loss = criterian(pred,labels)
        model.zero_grad()
        loss.backward()
        _,p = pred.max(1)
        acc = sum(p.eq(labels)).numpy()/batch_size
        step_acc.append(acc)
        
        #print(p)
        
        optimizer.step()
        epoch_loss.append(loss.item()/batch_size)
    
    print("Training Loss: ",sum(epoch_loss)/len(epoch_loss))
    print("Training Acc : ",np.sum(step_acc)/len(step_acc))
    print("----------------------------------------")


with torch.no_grad():

    model.eval()

    for epoch in range(1):
        #epoch_loss = []
        step_acc  = []
        for i,(images,labels) in enumerate(test_loader):
            pred = model(images)
            #loss = criterian(pred,labels)
            #model.zero_grad()
            #loss.backward()
            _,p = pred.max(1)
            acc = sum(p.eq(labels)).numpy()/batch_size
            step_acc.append(acc)
            
            #print(p)
            
            #optimizer.step()
            #epoch_loss.append(loss.item()/batch_size)
        
        #print("Test Loss: ",sum(epoch_loss)/len(epoch_loss))
        print("Test Acc : ",np.sum(step_acc)/len(step_acc))
        print("----------------------------------------")
    #    for i,(images,labels) in enumerate(test_loader):




