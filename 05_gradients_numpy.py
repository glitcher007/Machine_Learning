import torch
import torch.nn as nn

# Compute every step manually

# Linear regression
# f = w * x 

# here : f = 2 * x
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)


n_samples, n_features=X.shape
print(f'#samples: {n_samples}, #features: {n_features}')

X_test=torch.tensor([5],dtype=torch.float32)

print(n_samples,n_features)

input_size=n_features
output_size=n_features

#model=nn.Linear(input_size,output_size)
# model output


class LinearRegretion(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegretion,self).__init__()
        
        self.lin=nn.Linear(input_dim,output_dim)
    
    def forward(self,x):
        return self.lin(x)
    

model=LinearRegretion(input_size,output_size)


# loss = MSE

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01 
n_iters = 10

loss=nn.MSELoss()

optimiser=torch.optim.SGD(model.parameters(),lr=learning_rate)



for epoch in range(n_iters):
    # predict = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)
    
    
    optimiser.step()
    
    # calculate gradients=backward pass
    l.backward()

    # update weights
    
    optimiser.step()
    
        
    optimiser.zero_grad()    
         

    if epoch % 2 == 0:
        [w,b]=model.parameters()
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l)
     
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')