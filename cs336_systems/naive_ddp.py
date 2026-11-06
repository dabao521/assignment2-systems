import torch.nn as nn
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class NaiveDDPModel(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, src=0)

    def forward(self, x):
        return self.module(x)

    def ddp_backward(self):
        params_with_grad = [param for param in self.module.parameters() if param.grad is not None]
        grads = [param.grad for param in self.module.parameters() if param.grad is not None]

        flatten = _flatten_dense_tensors(grads)

        # all-redue
        dist.all_reduce(flatten, op=dist.ReduceOp.AVG)

        # unflatten
        new_grads = _unflatten_dense_tensors(flatten, grads)

        # assign to the new grads
        assert len(new_grads) == len(params_with_grad)
        for param, new_grad in zip(params_with_grad, new_grads):
            param.grad = new_grad

