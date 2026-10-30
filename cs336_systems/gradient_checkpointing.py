import torch
from torch import nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from cs336_basics.model import TransformerBlock

class Transformer(nn.Module):
    def __init__(self, num_layers: int = 32):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model = 2560,
                    num_heads = 32,
                    d_ff = 10240,
                    positional_encoder=None,
                )
                for _ in range(num_layers)
            ]
        )

    def checkpoint_group(self, x: Tensor, start: int, end: int):
        assert end > start
        for i in range(start, end):
            x = self.blocks[i](x)
        return x

    def forward(self, x: Tensor, group_size: int | None) -> Tensor:
        if group_size is None:
            for block in self.blocks:
                x = block(x)
            return x
        
        # do checkpointing
        assert group_size > 0
        for start in range(0, len(self.blocks), group_size):
            end = min(start + group_size, len(self.blocks))

            x = checkpoint(
                self.checkpoint_group,
                x,
                start,
                end,
                use_reentrant=False,
            )
        return x

global_saved_bytes: int = 0

def pack_hook(t):
    if isinstance(t, nn.Parameter):
        return t

    global global_saved_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    global_saved_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def main() -> None:
    model = Transformer()
    model = model.to("cuda")
    group_sizes = [1, 2, 3, 4, 6, 8, 16, 32]
    x = torch.randn((4, 2048, 2560), requires_grad=True, device="cuda")

    for group_size in group_sizes:
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            y = model(x, group_size)
            loss = y.sum()
            loss.backward()

        print(f"group_size={group_size}, peak_saved={global_saved_bytes/2**20:.2f}MiB")

if __name__ == "__main__":
    main()
