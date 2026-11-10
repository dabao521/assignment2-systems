
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

from cs336_basics.model import BasicsTransformerLM
# from cs336_basics.nn_utils import cross_entropy
import torch.nn.functional as F
from cs336_systems.fsdp import FSDPModel


class Config:
    ctx_len = 32768
    vocab_size = 151936
    d_model = 4096
    d_ff = 11008
    num_layers = 34
    num_heads = 32
    torch_dtype = torch.bfloat16
    is_causal = True
    batch_size = 2


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

def report_mem(label: str) -> None:
    torch.cuda.synchronize()
    print(
        f"{label}: "
        f"allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GiB, "
        f"peak={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
        flush=True,
    )

def train_worker(
    rank: int,
    world_size: int,
    port: int,
    cfg: Config,
    warmup_steps: int,
    benchmark_steps: int,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    try:
        assert cfg.batch_size % world_size == 0
        local_batch_size = cfg.batch_size // world_size

        # Same initialization on both ranks.
        torch.manual_seed(42)

        # Construct directly on GPU to avoid a full CPU-to-GPU
        # parameter copy during initialization.
        with torch.device(device):
            model = BasicsTransformerLM(
                vocab_size=cfg.vocab_size,
                context_length=cfg.ctx_len,
                d_model=cfg.d_model,
                d_ff=cfg.d_ff,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                rope_theta=10000.0,
            )

        model = FSDPModel(
            model,
            compute_dype=cfg.torch_dtype,
        )

        optimizer = optim.AdamW(model.parameters())

        # Generate the same global input on both ranks, then
        # select each rank's portion.
        generator = torch.Generator(device="cpu").manual_seed(123)
        labels = torch.randint(
            cfg.vocab_size,
            (cfg.batch_size, cfg.ctx_len),
            generator=generator,
        )
        targets = torch.randint(
            cfg.vocab_size,
            (cfg.batch_size, cfg.ctx_len),
            generator=generator,
        )

        start = rank * local_batch_size
        end = start + local_batch_size

        labels = labels[start:end].to(device)
        targets = targets[start:end].to(device)

        def train_step() -> None:
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.nvtx.range("model_forward"):
                logits = model(labels)

            report_mem("Before cross_entropy")
            torch.cuda.reset_peak_memory_stats()

            # loss = cross_entropy(logits, targets).sum()
            with torch.cuda.nvtx.range("cross_entropy"):
                loss = F.cross_entropy(
                        logits.reshape(-1, cfg.vocab_size),
                        targets.reshape(-1),
                )

            report_mem("After cross_entropy")

            # FSDP averages gradients across ranks.
            # Undo that averaging to reproduce the gradient
            # of the global summed loss.
            loss = loss * world_size

            with torch.cuda.nvtx.range("backward"):
                loss.backward()

            with torch.cuda.nvtx.range("gradient_sync"):
                model.finish_gradient_synchronization()

            with torch.cuda.nvtx.range("optimizer"):
                optimizer.step()

        if rank == 0:
            print(
                f"Starting benchmark: {world_size} GPUs, "
                f"global batch={cfg.batch_size}, "
                f"sequence={cfg.ctx_len}",
                flush=True,
            )

        # Warmup
        for _ in range(warmup_steps):
            train_step()

        print("warm up is done...", flush=True)

        torch.cuda.synchronize()
        dist.barrier()

        torch.cuda.reset_peak_memory_stats(device)

        # Measure complete training steps.
        step_times = []

        for i in range(benchmark_steps):
            print(f"step {i} ...", flush=True)
            dist.barrier()
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            train_step()

            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            step_times.append(elapsed_ms)

            if rank == 0:
                print(f"Step {i}: {elapsed_ms:.2f} ms", flush=True)

        local_avg_ms = sum(step_times) / len(step_times)
        local_peak_mib = (
            torch.cuda.max_memory_allocated(device) / 1024**2
        )

        # Wall-clock completion is determined by the slower rank.
        metrics = torch.tensor(
            [local_avg_ms, local_peak_mib],
            dtype=torch.float64,
            device=device,
        )

        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)

        if rank == 0:
            print(
                f"Full training step: {metrics[0].item():.2f} ms",
                flush=True,
            )
            print(
                f"Peak memory per GPU (maximum): "
                f"{metrics[1].item():.2f} MiB",
                flush=True,
            )

    finally:
        dist.destroy_process_group()


def test_timing_forward_backward(
    warmup_steps: int = 1,
    benchmark_steps: int = 3,
) -> None:
    cfg = Config()
    world_size = 2

    assert torch.cuda.device_count() >= world_size

    mp.spawn(
        train_worker,
        args=(
            world_size,
            get_free_port(),
            cfg,
            warmup_steps,
            benchmark_steps,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    test_timing_forward_backward(
        warmup_steps=1,
        benchmark_steps=1,
    )

