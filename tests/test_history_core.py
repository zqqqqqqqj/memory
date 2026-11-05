"""Small CPU tests of information flow, not only output dimensions."""
import pytest
import torch

from qrm.history.core import HistoryMemory, HistoryCodec
from qrm.history.teacher import FixedSpatialTeacher, spatial_regions
from qrm.history.supervision import history_loss, merged_past_indices
from qrm.memory.stream_codec import MemoryState


def small_memory(**kwargs):
    config = dict(prefix_dim=8, state_dim=3, coarse_dim=8, residual_dim=4,
                  query_dim=8, read_dim=8, hidden_dim=16, slots=2, image_grid=2,
                  storage_dtype=torch.bfloat16, merge_rule="shortest")
    config.update(kwargs)
    return HistoryMemory(**config)


def test_spatial_regions_preserve_camera_and_quadrant_order_and_masks():
    # Each of 4 quadrants in each camera has a distinguishable constant.
    images = torch.empty(1, 3, 4, 4, 2)
    for camera in range(3):
        for row in range(2):
            for col in range(2):
                images[:, camera, row*2:(row+1)*2, col*2:(col+1)*2] = camera*10+row*2+col
    mask = torch.ones(1, 48, dtype=torch.bool)
    mask[:, 32:] = False
    images[:, 2] = float("nan")
    pooled, valid = spatial_regions(images.flatten(1, 3), mask, image_grid=4)
    assert torch.equal(pooled[0, :, 0], torch.tensor([0., 1., 2., 3., 10., 11., 12., 13.]))
    assert valid.all()
    mask[:, 16:32] = False
    pooled, valid = spatial_regions(images.flatten(1, 3), mask, image_grid=4)
    assert torch.equal(pooled[:, 4:], torch.zeros_like(pooled[:, 4:]))
    assert not valid[:, 4:].any()


def test_teacher_targets_fixed_fitted_and_detached():
    torch.manual_seed(1)
    teacher = FixedSpatialTeacher(prefix_dim=8, image_grid=2, region_dim=1, seed=17)
    twin = FixedSpatialTeacher(prefix_dim=8, image_grid=2, region_dim=1, seed=17)
    assert torch.equal(teacher.projection, twin.projection)
    visual = torch.randn(12, 12, 8, requires_grad=True)
    mask = torch.ones(12, 12, dtype=torch.bool)
    mask[:, 8:] = False
    with pytest.raises(RuntimeError, match="Fit teacher"):
        teacher(visual, mask)
    teacher.fit_normalization(teacher.raw(visual, mask))
    target = teacher(visual, mask)
    assert not target.requires_grad and not list(teacher.parameters())
    assert torch.allclose(target.mean(0), torch.zeros(8), atol=2e-7)
    assert torch.allclose(target.std(0, unbiased=False), torch.ones(8), atol=2e-6)
    with pytest.raises(RuntimeError, match="already fitted"):
        teacher.fit_normalization(teacher.raw(visual, mask))
    twin.load_state_dict(teacher.state_dict())
    assert torch.equal(target, twin(visual, mask))
    changed = visual.detach().clone()
    changed[:, 8:] = 10000.
    assert torch.equal(target, teacher(changed, mask))


def test_fixed_visual_has_no_student_and_leaves_action_query_trainable():
    memory = small_memory()
    assert not hasattr(memory.feature, "visual_project")
    assert all(parameter.requires_grad for parameter in memory.feature.query.parameters())
    visual, mask = torch.randn(6, 12, 8), torch.ones(6, 12, dtype=torch.bool)
    mask[:, 8:] = False
    memory.teacher.fit_normalization(*memory.teacher._project(visual, mask))
    assert torch.equal(memory.visual(visual, mask), memory.teacher(visual, mask))
    x = memory.visual(visual[:1], mask[:1])
    query = memory.current_query(x, torch.randn(1, 5, 8), torch.ones(1, 5, dtype=torch.bool), torch.randn(1, 3))
    state = memory.codec.append(MemoryState(), x[0], 0)
    coarse, residual = memory.read(state, query[0])
    assert coarse.shape == residual.shape == (8,)
    assert torch.count_nonzero(residual) == 0


def test_history_gradient_reaches_early_writes_through_multiple_bf16_merges():
    torch.manual_seed(4)
    memory = small_memory(slots=1)
    features = torch.randn(6, 8, requires_grad=True)
    writes = []

    def retain_write(module, args, output):
        output.retain_grad()
        writes.append(output)

    hook = memory.codec.writer.register_forward_hook(retain_write)
    states = list(memory.codec.replay(features, range(6)))
    hook.remove()
    targets = torch.randn(6, 8, requires_grad=True)
    loss, metrics = history_loss(memory, states[-1], list(range(6)), targets,
                                 max_queries=4, generator=torch.Generator().manual_seed(3))
    loss.backward()
    assert metrics["history_queries"] == 4
    assert states[-1].blocks[0].z.dtype == torch.bfloat16
    assert writes[0].grad is not None and writes[0].grad.abs().sum() > 0
    assert features.grad[0].abs().sum() > 0
    assert memory.codec.writer.contrast.weight.grad.abs().sum() > 0
    assert memory.decoder.residual_skip.weight.grad.abs().sum() > 0
    assert targets.grad is None


def test_sampling_excludes_current_and_unmerged_items_and_rejects_future():
    memory = small_memory(slots=3)
    features = torch.randn(4, 8)
    times = [0, 10, 20, 30]
    states = list(memory.codec.replay(features, times))
    assert merged_past_indices(states[-1], times) == [0, 1]
    assert merged_past_indices(states[1], times[:2]) == []
    with pytest.raises(ValueError, match="precede current"):
        memory.reconstruct(states[-1], 30)
    with pytest.raises(ValueError, match="precede current"):
        memory.reconstruct(states[-1], 40)
    with pytest.raises(ValueError, match="exactly the observed prefix"):
        history_loss(memory, states[-1], [0, 10, 20, 40], torch.randn(4, 8))
    # Appending later observations cannot mutate an already constructed prefix.
    before = states[-1].packed(3, 8, 4)
    memory.codec.append(states[-1], torch.randn(8), 40)
    after = states[-1].packed(3, 8, 4)
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_coarse_and_time_ablation_remove_residual_exactly():
    memory = small_memory(slots=1)
    state = list(memory.codec.replay(torch.randn(4, 8), range(4)))[-1]
    query = torch.randn(8)
    assert memory.read(state, query)[1].abs().sum() > 0
    assert torch.count_nonzero(memory.read(state, query, "coarse")[1]) == 0
    assert torch.count_nonzero(memory.read(state, query, "time")[1]) == 0


@pytest.mark.parametrize("diagnostics", [True, False])
def test_short_and_merged_prefixes_report_identical_metric_keys(diagnostics):
    memory = small_memory(slots=2)
    states = list(memory.codec.replay(torch.randn(5, 8), range(5)))
    _, short = history_loss(memory, states[1], [0, 1], torch.randn(2, 8), diagnostics=diagnostics)
    _, long = history_loss(memory, states[-1], list(range(5)), torch.randn(5, 8), diagnostics=diagnostics)
    assert short.keys() == long.keys()
    assert short["history_queries"] == 0
    assert short["mean_target_age"] == 0


def test_independent_coarse_training_does_not_update_residual_writer():
    memory = small_memory(slots=1)
    state = list(memory.codec.replay(torch.randn(5, 8), range(5)))[-1]
    loss, _ = history_loss(memory, state, list(range(5)), torch.randn(5, 8), mode="coarse")
    loss.backward()
    assert all(parameter.grad is None for parameter in memory.codec.writer.parameters())
    assert memory.control_heads["coarse"][0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in memory.history_head.parameters())


def test_order_pair_with_identical_coarse_summary_can_learn_old_observation():
    """A tiny constructive check that the shared residual path can learn order.

    This is a synthetic mechanism test, not evidence of robot task success.
    """
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(9)
        memory = small_memory(slots=1)
        positive = torch.ones(8)
        negative = -positive
        sequences = [torch.stack((positive, negative, torch.zeros(8))),
                     torch.stack((negative, positive, torch.zeros(8)))]
        optimizer = torch.optim.Adam(memory.parameters(), lr=.01)
        initial = None
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            states = [list(memory.codec.replay(sequence, range(3)))[-1] for sequence in sequences]
            assert torch.equal(states[0].blocks[0].c, states[1].blocks[0].c)
            predictions = torch.stack([memory.reconstruct(state, 0) for state in states])
            expected = torch.stack((positive, negative))
            loss = (predictions-expected).square().mean()
            if initial is None:
                initial = float(loss.detach())
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            states = [list(memory.codec.replay(sequence, range(3)))[-1] for sequence in sequences]
            full = torch.stack([memory.reconstruct(state, 0) for state in states])
            coarse = torch.stack([memory.reconstruct(state, 0, "coarse") for state in states])
            full_error = float((full-expected).square().mean())
            coarse_error = float((coarse-expected).square().mean())
        assert full_error < min(.03, initial*.1)
        assert coarse_error > .9
    finally:
        torch.set_num_threads(old_threads)


def test_cached_projection_normalization_matches_live_with_missing_camera():
    teacher = FixedSpatialTeacher(prefix_dim=8, image_grid=2, region_dim=1)
    visual = torch.randn(12, 12, 8)
    mask = torch.ones(12, 12, dtype=torch.bool)
    mask[:, 4:8] = False
    raw, valid = teacher._project(visual, mask)
    raw[:, 4:] = float('nan')
    teacher.fit_normalization(raw, valid)
    assert torch.equal(teacher.fit_valid_count, torch.tensor([12]*4+[0]*4))
    assert torch.equal(teacher.mean[4:], torch.zeros(4))
    assert torch.equal(teacher.raw_std[4:], torch.zeros(4))
    assert torch.equal(teacher.scale[4:], torch.full((4,), teacher.scale_floor))
    assert torch.equal(teacher(visual, mask), teacher.normalize_raw(raw, valid))
    assert not teacher(visual, mask).requires_grad


def test_zero_residual_history_prediction_is_exact_zero_without_coarse_time_bypass():
    from qrm.memory.stream_codec import Block
    memory = small_memory(slots=1)
    for time in [2, 500, 10000]:
        state = MemoryState((Block(torch.randn(8).bfloat16(), torch.zeros(4).bfloat16(),
                                   3, 0, time),))
        prediction = memory.reconstruct(state, 0)
        assert torch.count_nonzero(prediction) == 0
    assert all(layer.bias is None for layer in memory.history_head if isinstance(layer, torch.nn.Linear))


def test_residual_target_uses_actual_quantized_stored_coarse_and_is_detached():
    memory = small_memory(slots=1)
    for parameter in memory.history_head.parameters():
        parameter.data.zero_()
    features = torch.tensor([[1.003]*8, [1.009]*8, [1.017]*8], requires_grad=True)
    state = list(memory.codec.replay(features, [0, 10, 20]))[-1]
    loss, metrics = history_loss(memory, state, [0, 10, 20], features, max_queries=8)
    expected = (features[:2].detach()-state.blocks[0].c.detach().float()).square().mean()
    assert torch.allclose(loss.detach(), expected)
    assert metrics['stored_mean_baseline_mse'] == pytest.approx(float(expected))
    assert metrics['relative_error'] == pytest.approx(1.)
    assert metrics['relative_error_valid'] == 1
    loss.backward()
    # A zeroed residual head prevents any prediction gradient; detached target
    # must not provide the writer with a gradient through the stored mean.
    assert features.grad is None or torch.count_nonzero(features.grad) == 0


def test_zero_energy_is_explicitly_invalid_and_queryless_prefix_is_not_success():
    memory = small_memory(slots=1)
    state = list(memory.codec.replay(torch.zeros(3, 8), [0, 1, 2]))[-1]
    _, metrics = history_loss(memory, state, [0, 1, 2], torch.zeros(3, 8))
    assert metrics['relative_error_valid'] == 0
    assert metrics['low_energy_queries'] == metrics['history_queries'] == 2
    assert metrics['target_delta_energy'] == 0
    short = memory.codec.append(MemoryState(), torch.zeros(8), 0)
    _, empty = history_loss(memory, short, [0], torch.zeros(1, 8))
    assert empty['history_queries'] == empty['history_elements'] == 0
    assert empty['relative_error_valid'] == 0


def test_stratified_sampling_covers_age_groups_and_rejects_singletons():
    memory = small_memory(slots=1)
    state = list(memory.codec.replay(torch.randn(10, 8), range(10)))[-1]
    _, metrics = history_loss(memory, state, list(range(10)), torch.randn(10, 8),
                              max_queries=3, generator=torch.Generator().manual_seed(3))
    assert all(metrics[bucket+'_history_queries'] == 1 for bucket in ['near', 'mid', 'far'])
    singles = small_memory(slots=3)
    state = list(singles.codec.replay(torch.randn(3, 8), range(3)))[-1]
    with pytest.raises(ValueError, match='merged block'):
        singles.reconstruct(state, 0)


@pytest.mark.parametrize('mode', ['coarse', 'time'])
def test_controls_are_independently_trainable_without_residual_parameters(mode):
    memory = small_memory(slots=1)
    memory.requires_grad_(False)
    for module in memory.history_train_modules(mode):
        module.requires_grad_(True)
    state = list(memory.codec.replay(torch.randn(4, 8), range(4)))[-1]
    loss, _ = history_loss(memory, state, list(range(4)), torch.randn(4, 8), mode=mode)
    loss.backward()
    assert all(parameter.grad is None for parameter in memory.codec.writer.parameters())
    assert memory.control_heads[mode][0].weight.grad.abs().sum() > 0
    assert all(not parameter.requires_grad for parameter in memory.time_query.parameters())


def test_history_and_actions_call_the_same_residual_value_function():
    from unittest.mock import patch
    memory = small_memory(slots=1)
    state = list(memory.codec.replay(torch.randn(4, 8), range(4)))[-1]
    with patch.object(memory.decoder, 'residual_value', wraps=memory.decoder.residual_value) as shared:
        memory.reconstruct(state, 0)
        memory.read(state, torch.randn(8))
        assert shared.call_count == 2
        assert shared.call_args_list[0].args[0].shape == (4,)
        assert shared.call_args_list[1].args[0].shape == (1, 4)


def test_full_reconstruction_is_explicit_objective_control():
    memory = small_memory(slots=1, objective='reconstruction')
    features = torch.randn(4, 8)
    state = list(memory.codec.replay(features, range(4)))[-1]
    loss, metrics = history_loss(memory, state, list(range(4)), features)
    expected = torch.stack([memory.reconstruct(state, t) for t in range(3)])
    assert float(loss.detach()) == pytest.approx(float((expected-features[:3]).square().mean()))
    assert metrics['history_elements'] == 3*8
    assert memory.reconstruction_head in memory.history_train_modules()
    assert memory.history_head not in memory.history_train_modules()


def test_eight_slot_ward_order_pair_strictly_matches_every_coarse_and_time_input():
    """A genuine 8-slot collision: only the two first observations are swapped."""
    memory = small_memory(slots=8, merge_rule='ward')
    u = torch.full((8,), 1.003)
    v = torch.full((8,), .031)
    suffix = torch.stack([torch.full((8,), 10.+5*k) for k in range(7)])
    sequences = [torch.cat((torch.stack((u+v, u-v)), suffix)),
                 torch.cat((torch.stack((u-v, u+v)), suffix))]
    states = [list(memory.codec.replay(features, range(9)))[-1] for features in sequences]
    assert torch.equal(sequences[0][-1], sequences[1][-1])
    for left, right in zip(states[0].blocks, states[1].blocks, strict=True):
        assert torch.equal(left.c, right.c)
        assert (left.n, left.start, left.end) == (right.n, right.start, right.end)
    assert states[0].blocks[0].n == 2
    assert not torch.equal(states[0].blocks[0].z, states[1].blocks[0].z)
    stored_mean = states[0].blocks[0].c.float()
    delta = torch.stack((u+v-stored_mean, u-v-stored_mean))
    # Quantized coarse mean is not exactly u; the best common prediction is b.
    b = u-stored_mean
    assert torch.count_nonzero(b) > 0
    lower_bound = v.square().mean()
    for mode in ['coarse', 'time']:
        predictions = torch.stack([memory.reconstruct(state, 0, mode) for state in states])
        assert torch.equal(predictions[0], predictions[1])
        actual = (predictions-delta).square().mean()
        identity = (predictions[0]-b).square().mean()+lower_bound
        assert torch.allclose(actual, identity, atol=1e-5, rtol=1e-5)
        assert actual >= lower_bound
