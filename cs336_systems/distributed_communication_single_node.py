from tabulate import tabulate
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time

def setup(rank, world_size, device):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    if device == "cuda":
        torch.cuda.set_device(rank)

    group = "gloo"
    if device == "cuda":
        group = "nccl"

    dist.init_process_group(group, rank=rank, world_size=world_size)

def clean():
    dist.destroy_process_group()

def cuda_sync_if_available(device):
    if device == "cuda":
        torch.cuda.synchronize()

def benchmark_run(rank, world_size, data_len, queue, device):
    setup(rank, world_size, device)

    # create the data
    data = torch.randint(0, 10, (data_len,), device=device)

    # warm up
    for _ in range(5):
        dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
        cuda_sync_if_available(device)

    dist.barrier()

    # perform the actual runs
    start_time = time.time()
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    cuda_sync_if_available(device)
    elapsed = (time.time() - start_time) * 1000
    dist.barrier()

    # aggregate all elpased time from all ranks
    timings = [0.0] * world_size
    dist.all_gather_object(timings, elapsed)

    if rank == 0:
        queue.put(sum(timings) / world_size)
    clean()

def main() -> None:
    headers = ["Num GPUs", "data size (MB)", "time (ms)"]

    world_sizes = [2, 4]

    data_lens = [i * 1024**2 // 4 for i in [1, 10, 100, 1024]]
    rows = []
    device="cuda"
    

    for world_size in world_sizes:
        for data_len in data_lens:
            queue = mp.get_context("spawn").SimpleQueue()

            mp.spawn(
                fn=benchmark_run,
                args=(world_size, data_len, queue, device,),
                nprocs=world_size,
                join=True,
            ) 

            rows.append([world_size, data_len*4 // (1024**2), queue.get()])

    print(tabulate(rows, headers=headers, tablefmt="github"))

if __name__ == "__main__":
    main()
