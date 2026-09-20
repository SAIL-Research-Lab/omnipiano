"""Joint transition transport for the native learners."""
from __future__ import annotations

import multiprocessing as mp
import traceback
import numpy as np

from omnipiano.multiagent.training.runtime import shared_team_reward


def inspect_env(env):
    agents = list(env.possible_agents)
    meta = {
        "agents": agents, "obs_dims": [], "action_dims": [],
        "own_slices": [], "state_slices": [],
    }
    for agent in agents:
        observation = env.observation_space(agent)
        action = env.action_space(agent)
        if len(observation.shape) != 1 or len(action.shape) != 1:
            raise ValueError("Native backend requires flat Box observations/actions")
        if not np.all(action.low == -1) or not np.all(action.high == 1):
            raise ValueError("Native implementation requires canonical [-1,1] actions")
        layout = env.obs_layout(agent)
        if "own" not in layout or "global_state" not in layout:
            raise ValueError("Native transport requires explicit own/global slices")
        meta["obs_dims"].append(observation.shape[0])
        meta["action_dims"].append(action.shape[0])
        meta["own_slices"].append(list(layout["own"]))
        meta["state_slices"].append(list(layout["global_state"]))
    widths = [hi - lo for lo, hi in meta["state_slices"]]
    if len(set(widths)) != 1 or widths[0] <= len(agents):
        raise ValueError("Inconsistent centralized-state layout")
    meta["state_dim"] = widths[0] - len(agents)
    return meta


class Codec:
    def __init__(self, meta):
        self.meta = meta
        self.agents = meta["agents"]
        self.action_dims = meta["action_dims"]
        self.action_dim = sum(self.action_dims)
        self.own_dim = sum(hi - lo for lo, hi in meta["own_slices"])
        self.state_dim = meta["state_dim"]

    def pack(self, observations):
        locals_batch, states = [], []
        identity = np.eye(len(self.agents), dtype=np.float32)
        for observation in observations:
            if set(observation) != set(self.agents):
                raise ValueError("Joint observation agent keys changed")
            local, reference = [], None
            for i, agent in enumerate(self.agents):
                value = np.asarray(observation[agent], dtype=np.float32)
                if value.shape != (self.meta["obs_dims"][i],):
                    raise ValueError(f"Observation shape mismatch for {agent}")
                if not np.isfinite(value).all():
                    raise FloatingPointError("Non-finite observation")
                lo, hi = self.meta["own_slices"][i]
                local.append(value[lo:hi])
                lo, hi = self.meta["state_slices"][i]
                global_state = value[lo:hi]
                if not np.array_equal(global_state[-len(self.agents):], identity[i]):
                    raise ValueError("Unexpected global-state agent identity layout")
                state = global_state[:-len(self.agents)]
                if reference is not None and not np.array_equal(state, reference):
                    raise ValueError("Agents do not expose the same environmental state")
                reference = state
            locals_batch.append(np.concatenate(local))
            states.append(reference)
        return {
            "o": np.stack(locals_batch).astype(np.float32),
            "s": np.stack(states).astype(np.float32),
        }

    def split(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError("Invalid joint action shape")
        if not np.isfinite(actions).all():
            raise FloatingPointError("Non-finite action")
        actions = np.clip(actions, -1, 1)
        result = []
        for row in actions:
            offset, per_agent = 0, {}
            for agent, width in zip(self.agents, self.action_dims):
                per_agent[agent] = row[offset:offset + width].copy()
                offset += width
            result.append(per_agent)
        return result


def make_env(task, seed, penalty):
    from omnipiano.multiagent.compile.environment import make_parallel_from_task
    return make_parallel_from_task(
        task, seed=seed, flatten_obs=True, include_global_state=True,
        inter_agent_collision_penalty_coef=penalty)


def transition(env, actions):
    obs, rewards, terminated, truncated, _ = env.step(actions)
    agents = set(env.possible_agents)
    if set(rewards) != agents or set(terminated) != agents or set(truncated) != agents:
        raise ValueError("Native backend requires synchronous, fixed-agent episodes")
    terms = [bool(terminated[a]) for a in env.possible_agents]
    truncs = [bool(truncated[a]) for a in env.possible_agents]
    if len(set(terms)) != 1 or len(set(truncs)) != 1:
        raise ValueError("Agents terminated at different times")
    done = terms[0] or truncs[0]
    if bool(env.agents) == done:
        raise ValueError("env.agents disagrees with termination/truncation")
    return {
        "obs": obs, "r": shared_team_reward(rewards),
        "term": float(terms[0]), "trunc": float(truncs[0]),
    }


def worker(connection, task, seed, penalty):
    env = None
    try:
        env = make_env(task, seed, penalty)
        connection.send(("ok", (inspect_env(env), env.reset()[0])))
        while True:
            command, argument = connection.recv()
            if command == "close":
                break
            if command == "step":
                result = transition(env, argument)
            elif command == "reset":
                result = env.reset()[0]
            else:
                raise ValueError(command)
            connection.send(("ok", result))
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class EnvPool:
    def __init__(self, task, seed, penalty, workers, timeout):
        self.timeout = timeout
        self.connections, self.processes, self.local = [], [], None
        self.n = max(1, workers)
        self.episode = [0] * self.n
        self.timestep = [0] * self.n
        self.observations = []
        try:
            if workers == 0:
                self.local = make_env(task, seed, penalty)
                self.meta = inspect_env(self.local)
                self.observations = [self.local.reset()[0]]
            else:
                context = mp.get_context("spawn")
                for i in range(workers):
                    parent, child = context.Pipe()
                    process = context.Process(
                        target=worker,
                        args=(child, task, seed + 1000 * (i + 1), penalty),
                    )
                    process.start()
                    child.close()
                    self.connections.append(parent)
                    self.processes.append(process)
                metas = []
                for i in range(workers):
                    meta, observation = self.receive(i)
                    metas.append(meta)
                    self.observations.append(observation)
                if any(meta != metas[0] for meta in metas):
                    raise ValueError("Worker environment spaces differ")
                self.meta = metas[0]
        except BaseException:
            self.close()
            raise

    def receive(self, i):
        connection = self.connections[i]
        if not connection.poll(self.timeout):
            raise TimeoutError(f"Native environment worker {i} timed out")
        status, value = connection.recv()
        if status != "ok":
            raise RuntimeError(value)
        return value

    def keys(self, count):
        return np.asarray([
            [i, self.episode[i], self.timestep[i]] for i in range(count)
        ], dtype=np.int64)

    def step(self, actions):
        count = len(actions)
        if self.local is not None:
            results = [transition(self.local, actions[0])]
        else:
            for i in range(count):
                self.connections[i].send(("step", actions[i]))
            results = [self.receive(i) for i in range(count)]
        for i, result in enumerate(results):
            if result["term"] or result["trunc"]:
                self.episode[i] += 1
                self.timestep[i] = 0
                if self.local is not None:
                    self.observations[i] = self.local.reset()[0]
                else:
                    self.connections[i].send(("reset", None))
                    self.observations[i] = self.receive(i)
            else:
                self.timestep[i] += 1
                self.observations[i] = result["obs"]
        return results

    def close(self):
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (OSError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()  # Only workers owned by this pool.
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for connection in self.connections:
            connection.close()
        self.connections, self.processes = [], []
        if self.local is not None:
            self.local.close()
            self.local = None


def add_gae(batch, gamma, gae_lambda, streams):
    size, values = batch["v"].shape
    future = np.zeros((streams, values), dtype=np.float32)
    advantages = np.zeros_like(batch["v"])
    for row in range(size - 1, -1, -1):
        stream = int(batch["keys"][row, 0])
        terminated = batch["term"][row]
        done = max(terminated, batch["trunc"][row])
        delta = (
            batch["r"][row]
            + gamma * (1 - terminated) * batch["nv"][row]
            - batch["v"][row]
        )
        advantages[row] = delta + gamma * gae_lambda * (1 - done) * future[stream]
        future[stream] = advantages[row]
    batch["adv"] = advantages
    batch["ret"] = advantages + batch["v"]


class Replay:
    def __init__(self, codec, capacity, maximum_gib):
        shapes = {
            "o": (codec.own_dim,), "s": (codec.state_dim,),
            "a": (codec.action_dim,), "no": (codec.own_dim,),
            "ns": (codec.state_dim,), "r": (), "term": (), "trunc": (),
        }
        required = capacity * sum(int(np.prod(s)) for s in shapes.values()) * 4
        if required > maximum_gib * 1024**3:
            raise MemoryError(
                f"Replay requires {required / 1024**3:.2f} GiB, "
                f"configured limit is {maximum_gib} GiB")
        self.capacity, self.position, self.size = capacity, 0, 0
        self.arrays = {
            name: np.empty((capacity, *shape), np.float32)
            for name, shape in shapes.items()
        }

    def add(self, batch):
        count = len(batch["a"])
        if count > self.capacity:
            raise ValueError("Collection batch exceeds replay capacity")
        indices = (np.arange(count) + self.position) % self.capacity
        for name, array in self.arrays.items():
            array[indices] = batch[name]
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, size):
        indices = np.random.randint(0, self.size, size=size)
        return {name: array[indices] for name, array in self.arrays.items()}

    def state(self):
        return {
            "position": self.position, "size": self.size,
            "arrays": {name: array[:self.size] for name, array in self.arrays.items()},
        }

    def restore(self, state):
        size = state["size"]
        if not 0 <= size <= self.capacity or set(state["arrays"]) != set(self.arrays):
            raise ValueError("Invalid replay checkpoint")
        for name, array in self.arrays.items():
            value = state["arrays"][name]
            if value.shape != array[:size].shape:
                raise ValueError("Replay shape mismatch")
            array[:size] = value
        self.size, self.position = size, state["position"]
        if not 0 <= self.position < self.capacity:
            raise ValueError("Invalid replay cursor")