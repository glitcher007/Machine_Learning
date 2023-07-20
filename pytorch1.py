import torch
import numpy as np


a=torch.ones(5)

print(a)

b=a.numpy()
print(b)
a.add_(1)
print(a)
print(b)



x=torch.ones(5,requires_grad=False)

print(x)