from omnipiano import make

env = make("OmniPiano-ClairDeLune-CollisionSafe-v0")
obs, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    cost = info["step_safety/cost_total"]           # per-step safety cost

    if terminated or truncated:
        obs, info = env.reset()

env.close()