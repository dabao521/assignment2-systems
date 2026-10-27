import argparse
import timeit
import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from jaxtyping import Int, Float
from torch import Tensor
import numpy as np
import pandas as pd
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext

def synchronize_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    if device == "mps":
        torch.mps.synchronize()

def current_time(device: str) -> float:
    synchronize_device(device)
    return timeit.default_timer()

def test_iteration(
    inputs: Int[Tensor, " ... context_len"],
    targets: Int[Tensor, " ... context_len"],
    is_warmup: bool,
    mode: str,
    model: BasicsTransformerLM,
    optimizer: AdamW,
    device,
) -> float:
    # zero the gards and not need to time it
    optimizer.zero_grad(set_to_none=True)

    start: float | None = None

    if mode != "backward":
        start = current_time(device)

    # forward
    with nvtx.range("forward"):
        logits: Float[Tensor, " ... context_len vocab_size"] = model(inputs)

    if mode == "forward":
        assert start is not None
        return current_time(device) - start

    loss = cross_entropy(logits, targets)

    if mode == "backward":
        assert start is None
        start = current_time(device)

    # backward
    with nvtx.range("backward"):
        loss.backward()

    if mode == "backward":
        return current_time(device) - start

    # run optimizer
    with nvtx.range("optimizer"):
        optimizer.step()

    assert start is not None
    return current_time(device) - start

def get_random_batch(
    batch_size: int,
    context_len: int,
    vocab_size: int,
    device: str,
) -> tuple[
    Int[Tensor, " ... batch_size context_len"],
    Int[Tensor, " ... batch_size context_len"]
    ]:
    inputs: Int[Tensor, " ... batch_size context_len"] = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_len),
        dtype=torch.long,
        device=device,
    )

    targets: Int[Tensor, " ... batch_size context_len"] = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_len),
        dtype=torch.long,
        device=device,
    )

    return (inputs, targets)

def main() -> None:
    parser = argparse.ArgumentParser(description="CS336 Assignent 2 - Benchmarking")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["forward", "backward", "forward_backward"])
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--use-mixed-precision", type=bool, required=False)
    parser.add_argument("--profile-memory", type=bool, required=False)

    args = parser.parse_args()

    context_len: int = args.context_length
    batch_size: int = 4
    vocab_size:int = 10000
    device: str = args.device
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    )

    model = model.to(device)

    optimizer=AdamW(model.parameters())
    use_mixed_precision = args.use_mixed_precision
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_mixed_precision
        else nullcontext()
    )

    profile_memory = args.profile_memory

    # warmup
    for _ in range(args.warmup):
        inputs, targets = get_random_batch(batch_size, context_len, vocab_size, device)
        with context:
            test_iteration(
                inputs,
                targets,
                is_warmup=True,
                mode=args.mode,
                model=model,
                optimizer=optimizer,
                device=args.device,
            )

    # start 
    timings: list[float] = []

    if device.startswith("cuda"):
        torch.cuda.cudart().cudaProfilerStart()

    if profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    try:
        with nvtx.range("benchmark"):
            for i in range(args.iterations):
                #print(f"iteration: {i}")
                inputs, targets = get_random_batch(batch_size, context_len, vocab_size, device)

                
                with context:
                    latency: float = test_iteration(
                        inputs,
                        targets,
                        is_warmup=False,
                        mode=args.mode,
                        model=model,
                        optimizer=optimizer,
                        device=args.device,
                    )
                timings.append(latency)
    finally:
        synchronize_device(device)
        if device.startswith("cuda"):
            torch.cuda.cudart().cudaProfilerStop()
        if profile_memory:
            torch.cuda.memory._dump_snapshot("/data/memory_snapshot.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)

    # generate table report
    rows: list[dict[str, str | float]] = [{
        "mode": args.mode,
        "mean": float(np.mean(timings)),
        "std": float(np.std(timings)),
    }]
    df: np.DataFrame = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format="%.3f"))

    
if __name__ == "__main__":
    main()
