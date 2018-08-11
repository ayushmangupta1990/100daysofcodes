
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
#LOADING DATASET#LOADIN 


# Loss function of Logistic
## -y* np.log(h) - (1-y) * np.log(1-h)


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

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,output_size):
        super(NeuralNet,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size1)
        self.linear2 = nn.Linear(hidden_size1,hidden_size2)
        self.linear3 = nn.Linear(hidden_size2,output_size)

    def forward(self,x):
        out = F.relu(self.linear1(x.view(-1,28*28)))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        return out


model = NeuralNet(28*28,128,256,10)

criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lr=0.01,params = model.parameters())


for epoch in range(20):
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




