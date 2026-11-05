from tabulate import tabulate
from cs336_systems.flash_forward import FlashAttentionForwardPyTorchFunc, FlashAttentionTritonFunc
import triton
import torch
import time

def benchmark_forward(fn: type, q, k, v, bq, bk) -> float:
    try:
        return triton.testing.do_bench(
            lambda: fn.apply(q,k,v,True, bq, bk),
            warmup=5,
            rep=10,
        )
    except Exception as e:
        print(f"Benchmark failed: {type(e).__name__}: {e}")
        return None

def benchmark_backward(fn: type, q, k, v, do, bq, bk) -> float:
    out = fn.apply(q,k,v,True, bq,bk)

    def backward():
        torch.autograd.grad(
            out,
            (q, k, v),
            grad_outputs=do,
            retain_graph=True,
        )
    try:
        return triton.testing.do_bench(
            backward,
            warmup=5,
            rep=10,
        )
    except Exception as e:
        print(f"Benchmark failed: {type(e).__name__}: {e}")
        return None

def benchmark_forward_backward(fn:type, q,k,v,do,bq,bk) -> float:
    def forward_backward():
        out = fn.apply(q,k,v,True,bq,bk)
        torch.autograd.grad(
            out,
            (q, k, v),
            grad_outputs=do,
        )

    try:
        return triton.testing.do_bench(
            forward_backward,
            warmup=5,
            rep=10,
        )
    except Exception as e:
        print(f"Benchmark failed: {type(e).__name__}: {e}")
        return None

def choose_tile_sizes(d_model: int, context_len: int) -> tuple[int, int]:
    if context_len <= 1024:
        bq, bk = 32, 32
    elif context_len <= 8192:
        bq, bk = 64, 64
    else:
        bq, bk = 128, 64

    if d_model >= 128:
        bq = min(bq, 64)

    return bq, bk


def main() -> None:
    rows = []

    context_lens = [2**i for i in range(7,14)]
    d_models = [2**i for i in range(4,8)]
    dtypes = [torch.bfloat16]

    deadline = time.monotonic() + 45 * 60

    headers = ["Impl", "Mode", "contextLen", "dmodel", "dtype", "Time (ms)"]
    for context_len in context_lens:
        for d_model in d_models:
            bq, bk = choose_tile_sizes(d_model, context_len)
            for dtype in dtypes:
                print(f"contextlen={context_len}, d_mode={d_model}, dtype={dtype}")
                q = torch.randn(1, context_len, d_model, device="cuda", dtype=dtype, requires_grad=True)
                k = torch.randn(1, context_len, d_model, device="cuda", dtype=dtype, requires_grad=True)
                v = torch.randn(1, context_len, d_model, device="cuda", dtype=dtype, requires_grad=True)
                do = torch.randn(1, context_len, d_model, device="cuda", dtype=dtype, requires_grad=True)
                forward_time_pytorch = benchmark_forward(FlashAttentionForwardPyTorchFunc,q,k,v,bq,bk)
                forward_time_triton = benchmark_forward(FlashAttentionTritonFunc,q,k,v,bq,bk)

                backward_time_pytorch = benchmark_backward(FlashAttentionForwardPyTorchFunc,q,k,v,do, bq,bk)
                backward_time_triton = benchmark_backward(FlashAttentionTritonFunc,q,k,v,do,bq,bk)

                e2e_pytorch = benchmark_forward_backward(FlashAttentionForwardPyTorchFunc,q,k,v,do, bq,bk)
                e2e_triton = benchmark_forward_backward(FlashAttentionTritonFunc,q,k,v,do,bq,bk)

                rows.extend([
                    ["pytorch", "forward", context_len, d_model, dtype, forward_time_pytorch],
                    ["triton", "forward", context_len, d_model, dtype, forward_time_triton],
                    ["pytorch", "backward", context_len, d_model, dtype, backward_time_pytorch],
                    ["triton", "backward", context_len, d_model, dtype, backward_time_triton],
                    ["pytorch", "forward-backward", context_len, d_model, dtype, e2e_pytorch],
                    ["triton", "forward-backward", context_len, d_model, dtype, e2e_triton],
                ])
                if time.monotonic() >= deadline:
                    print(tabulate(rows, headers=headers, tablefmt="github"))
                    return

    print(tabulate(rows, headers=headers, tablefmt="github"))

if __name__ == "__main__":
    main()
