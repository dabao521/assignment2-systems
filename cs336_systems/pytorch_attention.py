import argparse
import torch
from torch import nn
from cs336_basics.model import scaled_dot_product_attention
from jaxtyping import Float
from torch import Tensor
from collections.abc import Callable
from statistics import mean

class Attention(nn.Module):
    def __init__(
        self,
        Q: Float[Tensor, " ... context_len d_model"],
        K: Float[Tensor, " ... context_len d_model"],
        V: Float[Tensor, " ... context_len d_model"],
    ):
        self.Q = Q
        self.K = K
        self.V = V
        super().__init__()

    def forward(
        self,
        x: Float[Tensor, " ... context_len d_model"],
    ) -> Float[Tensor, " ... context_len d_model"]:
        *batch_dims, context_len, d_model = x.size()

        iota = torch.arange(context_len, device=x.device)
        mask = iota[:,None] >= iota[None,:].to("cuda")

        attn_output = scaled_dot_product_attention(K=self.K, Q=self.Q, V=self.V, mask=mask)
        return attn_output

loss: torch.Tensor | None = None
inputs: Float[Tensor, " ... context_len d_model"] | None = None
model: Attention | None = None

def setup_backward(iter: int) -> None:
    global loss
    try:
        model.zero_grad(set_to_none=True)
        loss = model(inputs).sum()
        if iter == 0:
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
            print(f" before backward, {memory_before/(2**20):.2f} MiB")
    except torch.cuda.OutOfMemoryError:
        print("OOM in setup backward")

def benchmark(
    run: Callable[[], None],
    num_warmups: int,
    num_trials: int,
    setup: Callable[[], None] | None = None,
) -> tuple[float, float]:
    for _ in range(num_warmups):
        if setup is not None:
            setup(-1)
        run()

    torch.cuda.synchronize()

    times: list[float] = []
    memory: list[float] = []
    for trial in range(num_trials):
        if setup is not None:
            setup(trial)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        memory_start = torch.cuda.memory_allocated()
        try:
            output = run() # noqa: F841
        except torch.cuda.OutOfMemoryError:
            print(f"OOM in iteration {trial}")
            torch.cuda.synchronize()
            continue
        memory_end = torch.cuda.memory_allocated()

        end_event.record()

        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
        memory.append(memory_end - memory_start)

    mean_time = mean(times)
    mean_memory = mean(memory)
    return (mean_time, mean_memory)

def main() -> None:
    parser = argparse.ArgumentParser(description="benchmark attention")

    parser.add_argument("--mode", type=str, required=True)
    args = parser.parse_args()
    mode = args.mode

    batch_size = 8
    context_lens = [256, 1024, 4096, 8192, 16384]
    d_models =  [16, 32, 64, 128]

    global model
    global inputs

    for context_len in context_lens:
        for d_model in d_models:
            # create random QKV
            Q = torch.randn((batch_size, context_len, d_model), device="cuda", requires_grad=True)
            K = torch.randn((batch_size, context_len, d_model), device="cuda", requires_grad=True)
            V = torch.randn((batch_size, context_len, d_model), device="cuda", requires_grad=True)

            model = torch.compile(Attention(Q,K,V))
            inputs = torch.randn((batch_size, context_len, d_model), device="cuda", requires_grad=True)

            print(f"benchmark {mode}, context_len={context_len}, d_model={d_model}")

            if mode == "forward":
                mean_time, mean_memory = benchmark(
                    run=lambda : model(inputs),
                    num_warmups = 5,
                    num_trials=100,
                )
            else:
                assert mode == "backward"
                mean_time, mean_memory = benchmark(
                    run=lambda: loss.backward(),
                    num_warmups=5,
                    num_trials=100,
                    setup = lambda x: setup_backward(x)
                )
            print(f" mean_time={mean_time} ms, mean_memory={mean_memory//(2**20):.2f} MiB")

if __name__ == "__main__":
    main()
