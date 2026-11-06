import torch.nn as nn
import torch.distributed as dist

class NaiveDDPModel(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, src=0)

    def forward(self, x):
        return self.module(x)

    def ddp_backward(self):
        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

