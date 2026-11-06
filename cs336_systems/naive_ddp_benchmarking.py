import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.common import (
    _cleanup_process_group,
    _setup_process_group,
)
import time
# from cs336_systems.naive_ddp import NaiveDDPModel
from cs336_systems.ddp_overlap_model import DdpOverlapModel
import torch.optim as optim
from cs336_basics.model import BasicsTransformerLM

def get_data(batch_size, context_len, vocab_size):
    data = torch.randint(0, vocab_size, (batch_size, context_len))
    targets = torch.randint(0, vocab_size, (batch_size, context_len))
    return data, targets

def sync_if_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def run_benchmark(rank, world_size, data, targets, vocab_size, context_length):
    device = _setup_process_group(rank, world_size=world_size, backend="nccl")
    dist.barrier()
    print(f"device: {device}")

    model = DdpOverlapModel(BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=2560,
        d_ff=10240,
        num_layers=32,
        num_heads=32,
        rope_theta=10000.0,
    ).to(device)).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    batch_size = data.shape[0]
    local_batch_size = batch_size // world_size
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size
    data = data[start_index:end_index,...].to(device)
    targets = targets[start_index:end_index,...].to(device)

    # warm up
    for _ in range(3):
        optimizer.zero_grad()
        outputs = model(data)
        loss = nn.functional.cross_entropy(
            outputs.reshape(-1, vocab_size),
            targets.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        # model.ddp_backward()
        model.finish_gradient_synchronization()

    sync_if_gpu()
    dist.barrier()

    # test runs
    backward_timings = []
    start_time = time.time()
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(data)
        loss = nn.functional.cross_entropy(
            outputs.reshape(-1, vocab_size),
            targets.reshape(-1),
        )
        loss.backward()

        sync_if_gpu()
        backward_start_time = time.time()
        # model.ddp_backward()
        model.finish_gradient_synchronization()
        sync_if_gpu()
        backward_timings.append(time.time() - backward_start_time)

        optimizer.step()
    elapsed = (time.time() - start_time) / 5
    backward_elapsed = sum(backward_timings) / 5
    dist.barrier()

    # aggregate across ranks
    elapsed_timings = [0.0] * world_size
    backward_elapsed_timings = [0.0] * world_size

    dist.all_gather_object(elapsed_timings, elapsed)
    dist.all_gather_object(backward_elapsed_timings, backward_elapsed)

    if rank == 0:
        avg_elapsed = 1000 * sum(elapsed_timings) / world_size 
        avg_backward = 1000 * sum(backward_elapsed_timings) / world_size
        print(f"iteration time {avg_elapsed} ms, and backward time {avg_backward} ms")

    _cleanup_process_group()

def main() -> None:
    world_size = 2
    batch_size=4
    vocab_size = 10_000
    context_length = 128
    data, targets = get_data(batch_size, context_length, vocab_size)

    mp.spawn(
        run_benchmark,
        args=(world_size, data, targets,vocab_size, context_length,),
        nprocs=world_size,
        join=True,
    )

if __name__ == "__main__":
    main()
