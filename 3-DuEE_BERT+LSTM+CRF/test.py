'''import torch
a = torch.tensor([1,0,1])
print(a, a.dtype)
b=a.bool()
a+=1
print(a, a.dtype)
print(b, b.dtype)'''

# -*- encoding:utf-8 -*-
import torch
import numpy as np
import transformers
print(torch.__version__)  # 1.7.1
print(transformers.__version__) # 2.1.1