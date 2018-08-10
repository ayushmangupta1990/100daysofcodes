import torch
import torch.nn as nn
from torch.autograd import Variable
x = torch.rand(10)
x = x.view(-1,1)

print("x: ",x)
y = (x * 0.5) + 3
print("y: ",y)



class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(LinearRegressionModel, self).__init__() 
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function

        out = self.linear(x)
        return out

input_dim = 1
output_dim = 1



model = LinearRegressionModel(input_dim,output_dim)
optimizer = torch.optim.Adam(lr=.01,params = model.parameters())
criterian = nn.MSELoss()

for i in range(10000):
    x_data = Variable(x)
    y_data = Variable(y)
    y_ = model(x_data)
    #print("y_: ",y_)
    loss = criterian(y_data,y_)
    model.zero_grad()
    loss.backward()
    optimizer.step()
    if i%100==0:
        print("Loss: ",loss.item())
        print("W: ",model.linear.weight, "b: ", model.linear.bias)
