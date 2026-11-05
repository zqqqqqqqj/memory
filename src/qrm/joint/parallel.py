"""FSDP communication policies with the same global accumulated objective."""
from contextlib import nullcontext

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy


STRATEGIES = {
    'fsdp_full_shard': ShardingStrategy.FULL_SHARD,
    'fsdp_shard_grad_op': ShardingStrategy.SHARD_GRAD_OP,
}


def wrap_sft(model, sft, device):
    """Keep optimizer/EMA shards and FP32 reductions in both modes.

    SHARD_GRAD_OP plus no_sync retains full parameters and gradients between
    microbatches. The last backward reduces and shards before clip/Adam/EMA.
    This trades higher memory residency for fewer collective operations.
    """
    device = torch.device(device)
    return FSDP(model, sharding_strategy=STRATEGIES[sft['parallelism']],
                ignored_states=[p for p in model.parameters() if not p.requires_grad],
                device_id=device, use_orig_params=True,
                sync_module_states=device.type == 'cuda')


def accumulation_context(model, micro, accumulation, gradient_sync='every_microbatch'):
    """Cover both forward and backward; always synchronize the final microbatch."""
    if accumulation < 1 or not 0 <= micro < accumulation:
        raise ValueError('Invalid accumulation microbatch index')
    if gradient_sync not in {'every_microbatch', 'optimizer_step'}:
        raise ValueError(f'Unknown gradient synchronization: {gradient_sync}')
    if (isinstance(model, FSDP) and gradient_sync == 'optimizer_step'
            and micro < accumulation-1):
        return model.no_sync()
    return nullcontext()


def communication_metadata(sft, world_size, accumulation):
    sync = sft.get('gradient_sync', 'every_microbatch')
    return dict(sharding_strategy=sft['parallelism'], gradient_sync=sync,
                gradient_sync_rounds_per_step=(
                    0 if world_size == 1 else 1 if sync == 'optimizer_step' else accumulation),
                retains_full_parameters_between_microbatches=(
                    world_size > 1 and accumulation > 1 and sync == 'optimizer_step'
                    and sft['parallelism'] == 'fsdp_shard_grad_op'))
