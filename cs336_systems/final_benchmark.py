from cs336_basics.model import BasicsTransformerLM
import triton
import torch.optim as optim
from cs336_basics.nn_utils import cross_entropy
import torch

#class Config:
#    ctx_len = 32768
#    vocab_size = 151936
#    d_model = 4096
#    d_ff = 11008
#    num_layers = 34
#    num_heads = 32
#    torch_dtype = torch.bfloat16
#    is_causal = True
#    batch_size = 2
class Config:
    ctx_len = 1024
    vocab_size = 256
    d_model = 128
    d_ff = 256
    num_layers = 2
    num_heads = 4
    torch_dtype = torch.bfloat16
    is_causal = True
    batch_size = 2
cfg = Config()

def test_timing_forward_backward():
    labels, targets = torch.randint(
        high=cfg.vocab_size,
        size=(2, cfg.batch_size, cfg.ctx_len),
        device="cuda",
    )
    
    model = BasicsTransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.ctx_len,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
    ).to(device="cuda", dtype=cfg.torch_dtype)
    optimizer = optim.AdamW(model.parameters())

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        res = model(labels)
        loss = cross_entropy(res, targets).sum()
        loss.backward()
        optimizer.step()
    
    # timing_results = triton.testing.do_bench(train_step, rep=30_000, warmup=10_000)
    timing_results = triton.testing.do_bench(train_step, rep=100, warmup=10)
    print(timing_results)
