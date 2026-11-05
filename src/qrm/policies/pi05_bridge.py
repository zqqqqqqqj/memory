"""PyTorch OpenPI bridge with explicit dataset normalization shared across stages."""
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import torch
from qrm.memory.reference_reader import freeze
from .memory_receiver import injected_prefix


@dataclass
class PreparedObservation:
    observation: object
    prefix: tuple
    preprocessing: tuple
    proprio: torch.Tensor
    normalized_actions: torch.Tensor | None = None


class Pi05Bridge:
    def __init__(self, checkpoint, *, repo_id="yinchenghust/robomemarena_lerobot", device="cuda:0", norm_stats=None, action_horizon=None):
        checkpoint = Path(checkpoint)
        if not (checkpoint/"model.safetensors").is_file():
            raise FileNotFoundError("Provide an OpenPI PyTorch checkpoint containing model.safetensors")
        from openpi.training import config
        from openpi.policies import policy_config
        base = config.get_config("pi05_libero")
        # Preserve pi05_base's state-token input; LIBERO's image-only preset disables it.
        train_cfg = replace(base, model=replace(base.model, pytorch_compile_mode=None, discrete_state_input=True),
                            data=replace(base.data, repo_id=repo_id))
        if action_horizon is not None:
            train_cfg = replace(train_cfg, model=replace(train_cfg.model, action_horizon=action_horizon))
        if norm_stats is not None:
            from openpi.shared.normalize import deserialize_json
            norm_stats = deserialize_json(Path(norm_stats).read_text())
        self.policy = policy_config.create_trained_policy(train_cfg, checkpoint, pytorch_device=device, norm_stats=norm_stats)
        if not self.policy._is_pytorch_model:
            raise TypeError("Memory injection requires the PyTorch backend")
        self.model = freeze(self.policy._model)
        self.device = torch.device(device)
        self.horizon = train_cfg.model.action_horizon
        self.prefix_dim = self.model.paligemma_with_expert.paligemma.config.text_config.hidden_size

    def transform(self, raw, actions=None):
        return self.transform_prepared(self.transform_cpu(raw, actions))

    def transform_cpu(self, raw, actions=None, *, pin_memory=False):
        """Same deterministic input transforms; safe to prepare ahead on CPU."""
        import jax
        inputs = dict(raw)
        if actions is not None:
            inputs["actions"] = np.asarray(actions, np.float32)
        inputs = self.policy._input_transform(inputs)
        def to_tensor(x):
            tensor = torch.from_numpy(np.array(x))
            # JSON normalization quantiles become float64 in NumPy. The action
            # projection uses float32; do not silently send doubles into it.
            if tensor.is_floating_point():
                tensor = tensor.float()
            return tensor.pin_memory() if pin_memory else tensor
        return jax.tree.map(to_tensor, inputs)

    def transform_prepared(self, inputs, *, non_blocking=False):
        import jax
        from openpi.models.model import Observation
        # Keep the original dtype, layout, batch dimension and CUDA stream.
        # Pinned source tensors are never mutated or reused by the producer.
        inputs = jax.tree.map(lambda x: x.to(self.device, non_blocking=non_blocking)[None], inputs)
        observation = Observation.from_dict(inputs)
        return observation, inputs.get("actions")

    def prepare(self, raw, actions=None):
        observation, normalized_actions = self.transform(raw, actions)
        with torch.no_grad():
            prep = self.model._preprocess_observation(observation, train=False)
            prefix = tuple(v.detach() for v in self.model.embed_prefix(*prep[:4]))
        return PreparedObservation(observation, prefix, prep,
            torch.tensor(np.array(raw["observation/state"], copy=True), device=self.device, dtype=torch.float32)[None],
            normalized_actions)

    def supervised_loss(self, prepared, receiver, memory):
        if prepared.normalized_actions is None:
            raise ValueError("Expert action targets are required")
        # Backbone parameters are frozen; gradients through its operations reach receiver/memory.
        with injected_prefix(self.model, prepared.prefix, prepared.preprocessing, receiver, memory):
            return self.model(prepared.observation, prepared.normalized_actions).mean()

    @torch.no_grad()
    def infer(self, prepared, receiver, memory):
        with injected_prefix(self.model, prepared.prefix, prepared.preprocessing, receiver, memory):
            actions = self.model.sample_actions(self.device, prepared.observation)
        outputs = {"state": prepared.observation.state[0].cpu().numpy(),
                   "actions": actions[0].float().cpu().numpy()}
        return self.policy._output_transform(outputs)["actions"]
