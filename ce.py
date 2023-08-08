import torch
import torch.nn as nn
import numpy as np

loss=nn.CrossEntropyLoss()


Y=torch.tensor([2,0])
# nsample*nclasses=1*3
Y_pred_good=torch.tensor([[2.0,1.0,0.1],[0.5,3.0,0.3] ])
Y_pred_bad=torch.tensor([[0.5,3.0,0.3],[2.0,1.0,0.1] ])

l1=loss(Y_pred_good,Y)
l2=loss(Y_pred_bad,Y)

print(l1.item())
print(l2.item())
# from here we can see l2 loss is quite hight,coz of data give

_, prediction1=torch.max(Y_pred_good,1)
_, prediction2=torch.max(Y_pred_bad,1)

print(prediction1)
print(prediction2)




