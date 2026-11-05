"""Batch diagnostic communication; never change loss or optimizer gradients."""
import torch
import torch.distributed as dist


def accumulate_metrics(totals, metrics, device):
    """Keep detached scalar sums on the training device until the update ends."""
    metrics = dict(metrics)
    # Weight diagnostics before summing: empty or shorter prefixes must not
    # dilute reconstruction measurements from actual historical queries.
    if 'history_zero_residual_mse' in metrics:
        metrics['history_zero_residual_sse'] = (metrics['history_zero_residual_mse']
                                               * metrics['history_query_elements'])
    if 'history_mean_target_age' in metrics:
        metrics['history_target_age_sum'] = (metrics['history_mean_target_age']
                                            * metrics['history_history_queries'])
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            totals[key] = totals.get(key, 0.)+value
            continue
        if not isinstance(value, torch.Tensor):
            continue
        value = torch.as_tensor(value, device=device).detach()
        if value.numel() != 1:
            raise ValueError(f'Expected scalar diagnostic {key}')
        value = value.to(torch.float64).reshape(())
        totals[key] = totals.get(key, 0.)+value


def reduce_metric_sums(context, totals, denominator, *, history_energy_floor=None):
    """One scalar-vector all-reduce per update, instead of one per metric."""
    if denominator <= 0:
        raise ValueError('Diagnostic denominator must be positive')
    keys = sorted(totals)
    if not keys:
        return {}
    packed = torch.stack([torch.as_tensor(totals[key], dtype=torch.float64,
                                          device=context.device).detach() for key in keys])
    if context.world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    sums = dict(zip(keys, packed.cpu().tolist(), strict=True))
    result = {key: value/denominator for key, value in sums.items()}
    if history_energy_floor is None:
        return result
    if history_energy_floor < 0:
        raise ValueError('History energy floor must be nonnegative')
    statistics = ('delta_sse', 'target_delta_energy', 'query_elements', 'history_queries',
                  'eligible_history_queries', 'observed_past_queries', 'low_energy_queries',
                  'history_sse', 'history_elements', 'zero_residual_sse', 'target_age_sum')
    for key, value in sums.items():
        if key.startswith('history_') and any(key.endswith('_'+suffix) for suffix in statistics):
            result[key] = value
    for prefix in ('history_', 'history_near_', 'history_mid_', 'history_far_'):
        elements = sums.get(prefix+'query_elements', 0)
        if prefix+'delta_sse' in sums:
            result[prefix+'delta_mse'] = sums[prefix+'delta_sse']/elements if elements else None
        if prefix+'target_delta_energy' in sums:
            energy = sums[prefix+'target_delta_energy']
            valid = elements > 0 and energy > history_energy_floor*elements
            result[prefix+'relative_error_valid'] = int(valid)
            result[prefix+'relative_error'] = sums[prefix+'delta_sse']/energy if valid else None
        if prefix+'low_energy_queries' in sums:
            queries = sums.get(prefix+'history_queries', 0)
            result[prefix+'low_energy_coverage'] = sums[prefix+'low_energy_queries']/queries if queries else None
    for target, numerator, denominator_key in (
            ('history_history_mse', 'history_history_sse', 'history_history_elements'),
            ('history_zero_residual_mse', 'history_zero_residual_sse', 'history_query_elements'),
            ('history_stored_mean_baseline_mse', 'history_target_delta_energy', 'history_query_elements'),
            ('history_mean_target_age', 'history_target_age_sum', 'history_history_queries'),
            ('history_eligible_query_coverage', 'history_eligible_history_queries', 'history_observed_past_queries')):
        if target in sums:
            count = sums.get(denominator_key, 0)
            result[target] = sums[numerator]/count if count else None
    return result


@torch.no_grad()
def gradient_rms(context, modules):
    """One reduction for all disjoint FSDP parameter-shard gradient statistics."""
    names = sorted(modules)
    packed = []
    for name in names:
        sums, count = [], 0
        for param in modules[name].parameters():
            if param.grad is not None:
                sums.append(param.grad.detach().float().square().sum())
                count += param.grad.numel()
        square = torch.stack(sums).sum() if sums else torch.zeros((), device=context.device)
        packed.extend((square.to(torch.float64), square.new_tensor(count, dtype=torch.float64)))
    values = torch.stack(packed)
    if context.world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values.cpu().tolist()
    return {name: (values[2*i]/max(values[2*i+1], 1.))**.5 for i, name in enumerate(names)}
