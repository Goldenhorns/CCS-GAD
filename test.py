import torch
x=torch.tensor([[2,3,4],[2,3,4],[2,3,5]])
x1=torch.tensor([[2,3,4],[2,3,4],[2,3,4]])
x3=torch.pow(x-x1,2)
print(x3)