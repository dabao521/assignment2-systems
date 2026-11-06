import torch.nn as nn
import torch.distributed as dist

class DdpOverlapModel(nn.Module):
    def __init__(
        self,
        module: nn.Module,
    ):
        super().__init__()
        self.module = module

        # broadcast to make sure all ranks having the same start
        # paramters value,
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # now register a param update callback hook to
        # do the all_reduce. Note that we need track
        # the handles globally so later we can block
        # waiting on them before optimizer.step
        self.handles = []
        def callback(param):
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
            self.handles.append(handle)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(callback)

    def forward(self, *inputs, **kargs):
        return self.module(*inputs, **kargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

