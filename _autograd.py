import torch


x=torch.randn(3,requires_grad=True)
# /*

#print(x)

y=x+2
#print(y)

z=y+y*2
z=z.mean()
#print(z)
# */


#z.backward() #dz/dx
# print(x.grad)
#

weights=torch.ones(4,requires_grad=True)

for epoch in range(5):
    model_output=(weights*3).sum()
    model_output.backward()
    
    #print(weights.grad)
    weights.grad.zero_()
    #waps se use zero krdo koi increament nhi hoga
    
with torch.no_grad():
    y=x+2
  #  print(y)
  

#optimiser

weights=torch.ones(4,requires_grad=True)

optimiser=torch.optim.SGD(weights,lr=0.01)

optimiser.step()
optimiser.zero_grad()
  
  
    
    
    

