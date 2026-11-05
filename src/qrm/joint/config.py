"""Global action-batch partitioning."""

def accumulation_steps(cfg, world_size):
    batch = cfg['sft']['global_batch_size']
    if world_size < 1 or batch % world_size:
        raise ValueError(f'{world_size} GPUs cannot divide global batch {batch}; use 1/2/4/8 GPUs')
    return batch // world_size
