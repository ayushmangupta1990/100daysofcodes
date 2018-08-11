
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




class ConvNN(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(ConvNN,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim,6,3,padding=1,stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5,padding=0,stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )

        self.fc = nn.Sequential(
            nn.Linear(400,128),
            nn.Linear(128,128),
            nn.Linear(128,output_dim)
        )



    def forward(self,x):
        out = self.conv(x)
        #print(out.size())
        #torch.Size([128, 16, 5, 5])  # batch size , channel, kernel , kernel
        out = self.fc(out.view(out.size(0),-1))
        return out



model = ConvNN(1,10)

criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lr=0.001,params = model.parameters())


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

