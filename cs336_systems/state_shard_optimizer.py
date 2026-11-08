from torch import nn
from torch import optim
import torch.distributed as dist

class StateShardOptimizer(optim.Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls: type[optim.Optimizer],
        **kwargs: any,
    ):
        self.local_param_groups: list[dict[str, any]] = []
        self.global_param_list: list[tuple[nn.Parameter, int]] = []
        self.optimizer = None
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        super().__init__(params, defaults=kwargs) # this will call add_param_group to intialize param list
        assert len(self.local_param_groups) > 0
        # now we have collect all paramters for current rank, create the
        # intenral optimizer
        self.optimizer = optimizer_cls(self.local_param_groups, **kwargs)

        print(f"rank {self.rank}: {len(self.local_param_groups)} local groups and {len(self.global_param_list)} global params")

    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure=closure, **kwargs)

        # boradcast to make sure all ranks having the same params
        for param, owner_rank in self.global_param_list:
            dist.broadcast(param.data, src=owner_rank)

    def add_param_group(self, param_group: dict[str, any]):
        super().add_param_group(param_group)

        new_group = {k:v for k, v in param_group.items() if k != "params"}
        old_params = param_group["params"]
        new_params = []
        for param in old_params:
            owner_rank = len(self.global_param_list) % self.world_size
            self.global_param_list.append((param, owner_rank))
            if self.rank == owner_rank:
                new_params.append(param)
        new_group["params"] = new_params
        self.local_param_groups.append(new_group)
        if self.optimizer is not None:
            self.optimizer.add_param_group(new_group)
