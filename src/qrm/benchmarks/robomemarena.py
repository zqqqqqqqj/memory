"""Use official episode loops/checkers; expose only legal observations to policies.

This module intentionally does not import torch and runs in the simulator venv.
"""
from pathlib import Path
import logging
import sys
import time
import numpy as np

LEGAL_FIELDS = ("observation/image", "observation/wrist_image", "observation/state")


def construct_environment(factory, kwargs, *, seed, placement_error, max_attempts=5):
    """Retry only constructor placement failures, before any policy action."""
    np.random.seed(seed)
    for attempt in range(1, max_attempts+1):
        try:
            return factory(**kwargs), attempt
        except placement_error:
            if attempt == max_attempts:
                raise
            logging.getLogger(__name__).warning(
                'Environment construction placement failed; retry %s/%s; reset seed remains %s',
                attempt+1, max_attempts, seed)


def dataset_axis_angle(quat):
    """Match the pinned LeRobot state convention, including negative quaternion w.

    The dataset stores the 0..2*pi angle from the simulator's signed xyzw
    quaternion. Canonicalizing q to w>=0 changes the learned state representation.
    """
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("Expected a finite xyzw proprioceptive quaternion")
    # Do not normalize or canonicalize a valid signed quaternion: either would
    # change the recorded-data convention. Reject malformed inputs explicitly.
    if not np.isclose(np.linalg.norm(q), 1.0, rtol=0.0, atol=1e-4):
        raise ValueError("Expected a unit xyzw proprioceptive quaternion")
    w = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w*w)
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (q[:3]*(2.0*np.arccos(w))/den).astype(np.float32)


def legal_observation(obs, prompt):
    clean = {key: np.asarray(obs[key]).copy() for key in LEGAL_FIELDS}
    clean["prompt"] = str(prompt)
    if clean["observation/state"].shape != (8,):
        raise ValueError("Expected LIBERO position/axis-angle/gripper state, 8 dimensions")
    state = clean["observation/state"]
    if not np.issubdtype(state.dtype, np.floating) or not np.isfinite(state).all():
        raise ValueError("Expected a finite floating-point robot state; integer states truncate orientation")
    # The common benchmark adapter canonicalizes axis-angle; demonstrations do
    # not. Rebuild only the legal robot orientation before dropping raw fields.
    raw = obs.get("_raw_obs", obs)
    quat = raw.get("robot0_eef_quat", raw.get("eef_quat"))
    if quat is None:
        raise ValueError("Raw robot quaternion is required to match demonstration state encoding")
    clean["observation/state"][3:6] = dataset_axis_angle(quat)
    for key in LEGAL_FIELDS[:2]:
        if clean[key].ndim != 3 or clean[key].shape[-1] != 3 or clean[key].dtype != np.uint8:
            raise ValueError("Expected HWC uint8 camera data")
    return clean


class AuditEnv:
    def __init__(self, env):
        self.env, self.controls, self.decisions = env, [], []
        self.native_rewards, self.dones = [], []
        self.errors = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        self.controls.clear()
        self.native_rewards.clear()
        self.dones.clear()
        self.decisions.clear()
        return self.env.reset()

    def step(self, action):
        try:
            result = self.env.step(action)
        except Exception as exc:
            self.errors.append(repr(exc))
            raise
        self.controls.append(np.asarray(action).tolist())
        self.native_rewards.append(float(result[1]))
        self.dones.append(bool(result[2]))
        return result


class ZeroPolicy:
    """Explicit installation probe, never a learned baseline."""
    def reset(self, metadata):
        pass

    def infer(self, obs, env_step, previous_action):
        actions = np.zeros((10, 7), dtype=np.float32)
        actions[:, -1] = -1
        return actions

    def finish(self, result):
        pass


class RemotePolicy:
    def __init__(self, host="127.0.0.1", port=8011):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self.client = WebsocketClientPolicy(host=host, port=port)

    def reset(self, metadata):
        self.client.infer({"op": "reset", "metadata": metadata})

    def infer(self, obs, env_step, previous_action):
        return self.client.infer({"op": "act", "observation": obs, "env_step": env_step,
                                  "previous_action": previous_action})["actions"]

    def finish(self, result):
        self.client.infer({"op": "finish", "result": result})

    def close(self):
        self.client._ws.close()


class ErrorCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.errors = []

    def emit(self, record):
        self.errors.append(record.getMessage())


def run_episode(cfg, task_id, seed, policy, out_dir, *, max_steps=None):
    b = cfg["benchmark"]
    if task_id not in b["full_tasks"]:
        raise ValueError("Unregistered task")
    root = Path(b["root"])
    sys.path.insert(0, str(root / "evaluation_benchmark/scripts"))
    import eval_common as ec
    import eval_tasks2_26 as tasks
    import eval_task1_only as task1
    from policy_adapter import BasePolicyAdapter
    from robosuite.utils.errors import RandomizationError
    raw, construction_attempts = construct_environment(ec._get_env_class(), dict(
        bddl_file_name=str(ec._resolve_bddl_path(task_id)),
        camera_heights=b["camera_height"], camera_widths=b["camera_width"],
        ignore_done=True, reward_shaping=True, control_freq=20, initialization_noise=None),
        seed=seed, placement_error=RandomizationError)
    env = AuditEnv(raw)
    latency = []

    class Adapter(BasePolicyAdapter):
        def infer_actions(self, obs, prompt, resize_size):
            clean = legal_observation(obs, prompt)
            start = time.perf_counter()
            last_action = env.controls[-1] if env.controls else [0.0]*6+[-1.0]
            actions = np.asarray(policy.infer(clean, len(env.controls), np.asarray(last_action, np.float32)), np.float32)
            latency.append(time.perf_counter()-start)
            if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0 or not np.isfinite(actions).all():
                raise ValueError("Policy returned invalid actions")
            env.decisions.append(len(env.controls))
            return actions

    prompt = ec.get_prompt(f"task{task_id}")
    limit = b["max_steps"] if max_steps is None else max_steps
    if limit < 1:
        raise ValueError("max_steps must be positive")
    metadata = dict(benchmark="robomemarena", benchmark_commit=b["commit"], task_id=task_id,
                    seed=seed, episode_id=f"task{task_id}-seed{seed}", protocol="policy_query_only")
    handler = ErrorCapture()
    logging.getLogger().addHandler(handler)
    started = time.perf_counter()
    try:
        np.random.seed(seed)
        env.seed(seed)
        policy.reset(metadata)
        common = dict(env=env, adapter=Adapter(), prompt=prompt, resize_size=b["resize_size"],
            replan_steps=b["replan_steps"], num_steps_wait=b["num_steps_wait"], max_steps=limit,
            post_goal_steps=b["post_goal_steps"], goal_monitor_dict={})
        if task_id == 1:
            score, stages, success, video, wrist = ec.run_episode_with_stages(
                **common, stage_checks=task1.STAGE_CHECKS, stage_checks_sequential=True)
            diagnostics = dict(failure_reason=None if success else "incomplete_stage")
        else:
            score, stages, success, diagnostics, video, wrist = tasks.run_episode_with_stateful_stages(
                **common, task_id=task_id, stage_specs=tasks._task_specs(task_id),
                goal_check_override=tasks._goal_override_check(task_id),
                fail_on_extra_pour=b["fail_on_extra_pour"], extra_pour_monitor_steps=b["extra_pour_monitor_steps"])
        errors = env.errors+handler.errors
        ends = env.decisions[1:]+[len(env.controls)]
        actions = [env.controls[start:end] for start, end in zip(env.decisions, ends, strict=True)]
        rewards = [[0.0]*len(chunk) for chunk in actions]
        if rewards and rewards[-1] and not errors:
            rewards[-1][-1] = float(bool(success))
        timed_out = len(env.controls) >= limit+b["num_steps_wait"]
        result = dict(**metadata, status="error" if errors else "completed", errors=errors,
            construction_attempts=construction_attempts,
            TSR=float(bool(success)) if not errors else 0.0, CSR=float(score)/100,
            native_env_steps=len(env.controls), controlled_env_steps=sum(map(len, actions)),
            observation_times=env.decisions, executed_actions=actions, step_rewards=rewards,
            terminated=not timed_out, truncated=timed_out, hard_time_limit=timed_out,
            native_rewards=env.native_rewards, stage_done=stages, diagnostics=diagnostics,
            wall_seconds=time.perf_counter()-started,
            query_latency_p50=float(np.percentile(latency, 50)) if latency else None,
            query_latency_p95=float(np.percentile(latency, 95)) if latency else None,
            installation_probe=isinstance(policy, ZeroPolicy))
        if out_dir is not None:
            import imageio.v2 as imageio
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            for suffix, frames in (("main", video), ("wrist", wrist)):
                if frames:
                    imageio.mimwrite(out / f"{metadata['episode_id']}_{suffix}.mp4", frames, fps=10)
        policy.finish(result)
        return result
    finally:
        logging.getLogger().removeHandler(handler)
        env.close()
