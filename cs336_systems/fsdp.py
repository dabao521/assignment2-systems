import torch
import torch.distributed as dist
from torch import nn
import weakref
from collections import deque
from cs336_basics.model import Linear, Embedding

class FSDPModel(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        compute_dype: torch.dtype | None = None,
    ):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.module = module
        self.compute_dtype = compute_dype

        # states accumuated from the first iteration to learn the actual execution order
        self.order_discovered = False
        self.fsdp_execution_order = [] # tracking Linear and Embedding only
        self.fsdp_module_to_idx = {}

        # prefetch states
        self.forward_gathers = {} # module index to tuple[handle, flat_buffer]
        self.backward_gathers = {} # module index to tuple[handle, flat_buffer]

        # backward grads states
        self.pending_grad_syncs = deque() # tuple[mod, flat_grad, local_grad, handle]
        self.pending_nonlinear_handles = deque()

        # for Linear / Embedding modules
        #  1. flatten the original weight
        #  2. extract the current rank's shard and clone it.
        #  3. replase mod.weight with a new sharde-shaped nn.Parameter
        #  4. reigster forward and backward hooks

        # For other modules
        # 1. iterate over m.paramters
        # 2. broadcast each paramter from rank 0
        # 3. register a post accuuate backward paramter hook to launch async all-reduce

        for mod in module.modules():
            if isinstance(mod, (Linear, Embedding)):
                flat = mod.weight.data.flatten()
                assert flat.numel() % self.world_size == 0
                sz = flat.numel() // self.world_size
                # clone is needed so that we don't hold reference to the large sharded memory
                local_shard = flat[self.rank*sz : (self.rank+1) * sz].clone()
                mod._fsdp_enabled = True
                mod._fsdp_model = weakref.ref(self) # weakref avoids circulr dep on modules
                mod._fsdp_original_shape = mod.weight.shape
                mod._fsdp_original_numel = mod.weight.numel()
                mod.weight = nn.Parameter(local_shard)
                mod.register_forward_pre_hook(self._forward_pre_hook)
                mod.register_forward_hook(self._forward_post_hook)
                mod.register_full_backward_pre_hook(self._backward_pre_hook)
            else:
                for param in mod.parameters(recurse=False):
                    if param.requires_grad:
                        dist.broadcast(param.data, src=0)
                        param.register_post_accumulate_grad_hook(self._backward_non_linear_param_post_hook)

    def forward(self, *inputs, **kwargs):
        outputs = self.module(*inputs, **kwargs)
        if not self.order_discovered:
            # print(f"[rank={self.rank}], first forward pass has completed..")
            self.order_discovered = True
        return outputs

    def finish_gradient_synchronization(self):
        while self.pending_grad_syncs:
            self.finalize_oldest_grad_sync()
        
        while self.pending_nonlinear_handles:
            handle = self.pending_nonlinear_handles.popleft()
            handle.wait()
    
    def async_all_gather_weights(self, mod, idx):
        flat_buffer = torch.empty(
            mod._fsdp_original_numel,
            dtype=self.compute_dtype or mod.weight.dtype,
            device=mod.weight.device,
        )

        local_shard = mod.weight.detach().to(flat_buffer.dtype)

        handle = dist.all_gather_into_tensor(
            flat_buffer,
            local_shard,
            async_op=True,
        )
        return handle, flat_buffer, local_shard
    
    def sync_all_gather_weights(self, mod, idx):
        handle, flat_buffer, local_shard = self.async_all_gather_weights(mod, idx)
        handle.wait()
        return flat_buffer.reshape(mod._fsdp_original_shape)

    def install_fsdp_full_weight(self, mod, full_weight):
        assert not hasattr(mod, "_fsdp_full_weight")
        mod._fsdp_full_weight = full_weight

    def _forward_pre_hook(self, mod: nn.Module, inputs):
        # for the first pass:
        # 1. add the module to the execution order list
        # 2. update module to idx map for later prefetch scheduling
        # 3. synchronously all gather the module's weight into mod._fsdp_full_weight
        #    to be used later for auto grad forward function

        # for the future passes
        # 1. if there is pre-fetch handle wait for it and releae the buffer
        # 2. otherwise syncrhoously all gather the modeule weight
        # 3. store the gathered weight into mod._fsdp_full_weight
        if not self.order_discovered:
            idx = len(self.fsdp_execution_order)
            self.fsdp_execution_order.append(mod)
            self.fsdp_module_to_idx[mod] = idx 
            # print(f"[rank={self.rank}], register mod={mod} with idx={idx}")

            self.install_fsdp_full_weight(mod, self.sync_all_gather_weights(mod, idx))
        else:
            assert mod in self.fsdp_module_to_idx
            idx = self.fsdp_module_to_idx[mod]

            if idx in self.forward_gathers:
                handle, flat_buffer, local_shard = self.forward_gathers[idx] 
                handle.wait()
                full_weight = flat_buffer.reshape(mod._fsdp_original_shape)
                self.install_fsdp_full_weight(mod, full_weight)
                # free the flat buffer
                del self.forward_gathers[idx]
            else:
                self.install_fsdp_full_weight(mod, self.sync_all_gather_weights(mod, idx))
        # self.check_finite("pre forward hook for full weight", mod._fsdp_full_weight)

    def _forward_post_hook(self, mod: nn.Module, inputs, output):
        # for the first pass:
        # 1. since forward is done, releset the fsdp full weight
        
        # for future passes:
        # 1. releae current module's full weight.
        # 2. prefetch idx+2 if it exists

        del mod._fsdp_full_weight

        if self.order_discovered:
            idx = self.fsdp_module_to_idx[mod]

            prefetch_idx = idx + 2
            if prefetch_idx < len(self.fsdp_execution_order):
                prefetch_mod = self.fsdp_execution_order[prefetch_idx]
                handle, flat_buffer, local_shard = self.async_all_gather_weights(prefetch_mod, prefetch_idx)
                # track this in the forward_gathers
                assert prefetch_idx not in self.forward_gathers
                self.forward_gathers[prefetch_idx] = (handle, flat_buffer, local_shard)

    def prefetch_backward(self, mod):
        assert mod in self.fsdp_module_to_idx
        idx = self.fsdp_module_to_idx[mod]
        prefetch_idx = idx - 2
        if prefetch_idx >= 0:
            prefetch_mod = self.fsdp_execution_order[prefetch_idx]
            handle, flat_buffer, local_shard = self.async_all_gather_weights(prefetch_mod, prefetch_idx)
            # track this in the backward_gathers
            assert prefetch_idx not in self.backward_gathers
            self.backward_gathers[prefetch_idx] = (handle, flat_buffer, local_shard)

    def finalize_oldest_grad_sync(self):
        mod, flat_full_grad, local_grad, handle = self.pending_grad_syncs.popleft()
        handle.wait()
        grad = local_grad.to(mod.weight.dtype)
        assert mod.weight.grad is None
        mod.weight.grad = grad

    def enqueue_reduce_scatter(self, mod, grad_full_weight):
        # grad_full_weight has shape [d_out, d_in]
        # the goal here is install the weight.grad after reduce
        if len(self.pending_grad_syncs) >= 2:
            # make sure at most two inflight reduction
            self.finalize_oldest_grad_sync()

        full_grad_flattened = grad_full_weight.contiguous().flatten()
        local_grad_buffer = torch.empty(
            full_grad_flattened.numel() // self.world_size,
            device=full_grad_flattened.device,
            dtype=full_grad_flattened.dtype,
        )

        handle = dist.reduce_scatter_tensor(
            output=local_grad_buffer,
            input=full_grad_flattened,
            op=dist.ReduceOp.AVG,
            async_op=True,
        )

        self.pending_grad_syncs.append((mod, full_grad_flattened, local_grad_buffer, handle))

    def _backward_pre_hook(self, mod: nn.Module, grad_output):
        assert mod in self.fsdp_module_to_idx
        idx = self.fsdp_module_to_idx[mod]

        if idx in self.backward_gathers:
            # this module was prefetched
            handle, flat_buffer, local_shard = self.backward_gathers[idx]
            handle.wait()
            full_weight = flat_buffer.reshape(mod._fsdp_original_shape)
            self.install_fsdp_full_weight(mod, full_weight)
            del self.backward_gathers[idx]
        else:
            # otherwise all gather synchronously
            self.install_fsdp_full_weight(mod, self.sync_all_gather_weights(mod, idx))
        # self.check_finite("pre backward hook for full weight", mod._fsdp_full_weight)


    def _backward_non_linear_param_post_hook(self, param):
        assert param.grad is not None
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
        self.pending_nonlinear_handles.append(handle)

    def check_finite(self, name, tensor):
        if not torch.isfinite(tensor).all():
            print(
                f"[Rank {dist.get_rank()}] {name}: "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                f"NaN={torch.isnan(tensor).sum().item()}, "
                f"Inf={torch.isinf(tensor).sum().item()}",
                flush=True,
            )
            raise RuntimeError(f"Non-finite values detected in {name}")
