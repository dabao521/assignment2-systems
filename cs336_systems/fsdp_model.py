import torch
from einops import einsum

class ShardedLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, local_weight, mod):
        ctx.save_for_backward(x)
        ctx.mod = mod

        full_weight = mod._fsdp_full_weight
        outputs = einsum(x, full_weight, " ... d_in, d_out d_in -> ... d_out")

        return outputs
    
    @staticmethod
    def backward(ctx, grad_out):
        (x, ) = ctx.saved_tensors
        mod = ctx.mod

        assert hasattr(mod, "_fsdp_full_weight")
        assert hasattr(mod, "_fsdp_model")

        full_weight = mod._fsdp_full_weight
        dx = grad_out @ full_weight

        del mod._fsdp_full_weight

        fsdp = mod._fsdp_model()
        fsdp.prefetch_backward(mod)

        # compute the full-weight gradient locally
        grad_full_weight = (
            grad_out.reshape(-1, grad_out.shape[-1]).T
            @ x.reshape(-1, x.shape[-1])
        )
        fsdp.enqueue_reduce_scatter(mod, grad_full_weight)


        #fsdp.check_finite("linear backward full weight", full_weight)
        #fsdp.check_finite("linear backward dx", dx)
        #fsdp.check_finite("linear backward grad out", grad_out)
        #fsdp.check_finite("linear backward grad full weight", grad_full_weight)

        return dx, None, None

class ShardedEmbeddingFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_ids, local_weight, mod):
        ctx.save_for_backward(token_ids)
        ctx.mod = mod

        full_weight = mod._fsdp_full_weight
        outputs = full_weight[token_ids, :]

        #fsdp = mod._fsdp_model()
        #fsdp.check_finite("embedding forward token ids", token_ids)
        #fsdp.check_finite("embedding forward full_weight", full_weight)
        #fsdp.check_finite("embedding forward outputs", outputs)

        return outputs


    @staticmethod
    def backward(ctx, grad_output):
        (token_ids,) = ctx.saved_tensors
        mod = ctx.mod

        assert hasattr(mod, "_fsdp_full_weight")
        assert hasattr(mod, "_fsdp_model")

        del mod._fsdp_full_weight
        fsdp = mod._fsdp_model()
        fsdp.prefetch_backward(mod)

        # only need to compute the full weight grads
        flat_full_grad = torch.zeros(
            mod._fsdp_original_shape,
            device=grad_output.device,
            dtype=grad_output.dtype,
        )

        # token_ids are the indexes from grad_out to impact falt_full_grad
        flat_full_grad.index_add_(
            0,
            token_ids.flatten(),
            grad_output.reshape(-1, grad_output.shape[-1]),
        )
        fsdp.enqueue_reduce_scatter(mod, flat_full_grad)

        #fsdp.check_finite("embedding backward flat_full_grad", flat_full_grad)

        return None, None, None

