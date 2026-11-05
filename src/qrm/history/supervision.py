"""Strictly past, block-local residual supervision with auditable denominators."""
import torch


AGE_BUCKETS = ("near", "mid", "far")


def merged_past_indices(state, times):
    """Only genuine observed times absorbed into non-singleton blocks."""
    if not state.blocks:
        return []
    if any(int(time) != time for time in times):
        raise ValueError("Historical times must be integers")
    times = [int(time) for time in times]
    if any(right <= left for left, right in zip(times[:-1], times[1:])):
        raise ValueError("Historical times must be strictly increasing")
    now = state.blocks[-1].end
    if not times or times[-1] != now or times[0] != state.blocks[0].start:
        raise ValueError("Targets must describe exactly the observed prefix")
    if len(times) != sum(block.n for block in state.blocks):
        raise ValueError("Target count does not match stored observation count")
    for block in state.blocks:
        inside = [time for time in times if block.start <= time <= block.end]
        if len(inside) != block.n or inside[0] != block.start or inside[-1] != block.end:
            raise ValueError("Target times do not match stored block observation metadata")
    return [index for index, time in enumerate(times) if time < now
            and any(block.n > 1 and block.start <= time <= block.end for block in state.blocks)]


def age_bucket(time, times):
    """Relative age thirds of the observed prefix, never of the whole episode."""
    relative_age = (times[-1]-time)/max(times[-1]-times[0], 1)
    return AGE_BUCKETS[0 if relative_age <= 1/3 else 1 if relative_age <= 2/3 else 2]


def uniform_temporal_queries(eligible, times, max_queries, generator=None):
    """Uniform systematic sampling of observed candidates, without replacement.

    Sort the N eligible candidates by time, select Q=min(max_queries, N), and
    use indices floor((j*N + offset)/Q), where offset is uniform on [0, N).
    Every candidate has exactly Q/N marginal inclusion probability. Selected
    ranks are evenly spaced, with a fresh shared random offset per prefix;
    selections within a prefix are correlated, not independent random draws.
    Only observed candidates are used, including when timestamps have gaps.
    Short prefixes use every eligible time and consume no random numbers.
    """
    if not isinstance(max_queries, int) or isinstance(max_queries, bool) or max_queries < 1:
        raise ValueError("Need a positive integer query count")
    if len(set(eligible)) != len(eligible):
        raise ValueError("Eligible query indices must be unique")
    ordered = sorted(eligible, key=lambda index: times[index])
    count = len(ordered)
    if count <= max_queries:
        return ordered
    offset = int(torch.randint(count, (), generator=generator))
    return [ordered[(query*count+offset)//max_queries] for query in range(max_queries)]


def stratified_queries(eligible, times, max_queries, generator=None):
    """Compatibility name for uniform systematic sampling; ages are metrics."""
    return uniform_temporal_queries(eligible, times, max_queries, generator)


def history_loss(memory, state, times, targets, *, max_queries=32, generator=None,
                 diagnostics=True, mode=None, tensor_metrics=False):
    """Teach x_s - actual_stored_c for merged observed s < current time.

    ``coarse``/``time`` select independently trainable control heads. ``full``
    uses the action decoder's residual value function on the selected block;
    zero-z inference is separately named and is not an independent baseline.
    All ratios include additive sufficient statistics for correct aggregation.
    Invalid zero-energy ratios use a 0 placeholder plus an explicit valid flag;
    report them as N/A, never as evidence of successful reconstruction.
    """
    mode = memory.reader_mode if mode is None else mode
    if mode not in {"full", "coarse", "time"}:
        raise ValueError("mode must be full, coarse, or time")
    if targets.ndim != 2 or targets.shape != (len(times), memory.coarse_dim):
        raise ValueError("Expected target features [prefix_length, coarse_dim]")
    if max_queries < 1 or not torch.isfinite(targets).all():
        raise ValueError("Need positive query count and finite teacher targets")
    eligible = merged_past_indices(state, times)
    selected = uniform_temporal_queries(eligible, times, max_queries, generator)
    module = memory.history_train_modules(mode)[-1]
    zero = next(module.parameters()).sum() * 0.
    device = zero.device
    teacher = targets[selected].detach().to(device=device, dtype=torch.float32)
    if selected:
        stored_c = torch.stack([memory.historical_block(state, times[index])[1].c.detach().float()
                                for index in selected])
        delta_target = (teacher-stored_c).detach()
        prediction = torch.stack([memory.reconstruct(state, times[index], mode) for index in selected])
        objective_target = delta_target if memory.objective == "residual" else teacher
        predicted_delta = prediction if memory.objective == "residual" else prediction-stored_c
        errors = (prediction.float()-objective_target).square().sum(-1)
        delta_errors = (predicted_delta.float()-delta_target).square().sum(-1)
        energies = delta_target.square().sum(-1)
        loss = errors.sum()/(len(selected)*memory.coarse_dim)
    else:
        errors = delta_errors = energies = zero.detach().new_empty(0)
        loss = zero

    def value(tensor):
        return tensor.detach() if tensor_metrics else float(tensor.detach())

    def summaries(indices):
        elements = len(indices)*memory.coarse_dim
        delta_sse = delta_errors[indices].sum()
        energy = energies[indices].sum()
        valid = (energy > memory.relative_energy_floor*elements) & (elements > 0)
        return dict(history_queries=len(indices), query_elements=elements,
                    delta_sse=value(delta_sse), target_delta_energy=value(energy),
                    delta_mse=value(delta_sse/max(elements, 1)),
                    relative_error=value(torch.where(valid, delta_sse/energy.clamp_min(1e-30), 0.)),
                    relative_error_valid=value(valid.float()),
                    low_energy_queries=value((energies[indices] <= memory.relative_energy_floor*memory.coarse_dim).sum()))

    metrics = summaries(list(range(len(selected))))
    metrics.update(eligible_history_queries=len(eligible),
                   observed_past_queries=max(len(times)-1, 0),
                   history_sse=value(errors.sum()), history_elements=len(selected)*memory.coarse_dim,
                   history_mse=value(loss),
                   mean_target_age=sum(times[-1]-times[index] for index in selected)/max(len(selected), 1),
                   eligible_query_coverage=len(eligible)/max(len(times)-1, 1))
    for bucket in AGE_BUCKETS:
        indices = [i for i, index in enumerate(selected) if age_bucket(times[index], times) == bucket]
        metrics.update({f"{bucket}_{key}": item for key, item in summaries(indices).items()})
    if diagnostics:
        with torch.no_grad():
            if selected:
                if mode == "full" and memory.objective == "residual":
                    # Bias-free residual decoder/head map zero residual to zero.
                    # Keep subtraction/square/mean identical to the diagnostic.
                    zero_prediction = torch.zeros_like(objective_target)
                else:
                    zero_prediction = torch.stack([memory.reconstruct(state, times[index], mode,
                                                       zero_residual=True) for index in selected])
                zero_error = (zero_prediction-objective_target).square().mean()
            else:
                zero_error = zero.detach()
            metrics["zero_residual_mse"] = value(zero_error)
            metrics["stored_mean_baseline_mse"] = value(energies.sum()/max(len(selected)*memory.coarse_dim, 1))
    return loss, metrics
