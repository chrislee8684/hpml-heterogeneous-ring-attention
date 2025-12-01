import torch
import torch.distributed as dist
import os

def run(rank, world_size):
    """
    A simple distributed function to test NCCL initialization.
    """
    print(f"--> Starting process on rank {rank} of {world_size}.")
    
    # This is the line that is likely hanging in your environment
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    print(f"✅ Rank {rank} has successfully initialized the process group.")
    
    # Perform a single collective operation (all_reduce) as a sanity check
    tensor = torch.ones(1).to(rank)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    if tensor[0] == world_size:
        print(f"✅ Rank {rank} has successfully completed an all_reduce operation.")
    else:
        print(f"❌ Rank {rank} FAILED the all_reduce operation. Tensor value: {tensor[0]}")

    dist.destroy_process_group()
    print(f"--> Process on rank {rank} is done.")

def setup(rank, world_size):
    """
    Sets up the environment for distributed training.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355" # A common default port
    run(rank, world_size)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs. Starting NCCL test...")
    torch.multiprocessing.spawn(setup, args=(world_size,), nprocs=world_size, join=True)
