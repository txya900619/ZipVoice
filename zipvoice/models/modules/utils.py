import torch
from torch import nested

NJT = lambda ls: nested.nested_tensor(ls, layout=torch.jagged)
