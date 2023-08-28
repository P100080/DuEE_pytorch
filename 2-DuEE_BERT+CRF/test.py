import torch
a = torch.tensor([1,0,1])
print(a, a.dtype)
b=a.bool()
a+=1
print(a, a.dtype)
print(b, b.dtype)
