"""Native OmniSafe 0.5.0 online, model-free algorithms (no reimplementations)."""

ON_POLICY = (
    "PolicyGradient", "NaturalPG", "TRPO", "PPO", "PPOLag", "TRPOLag",
    "RCPO", "PDO", "FOCOPS", "CPO", "PCPO", "OnCRPO", "IPO", "P3O",
    "CUP", "CPPOPID", "TRPOPID", "PPOEarlyTerminated", "TRPOEarlyTerminated",
    "PPOSaute", "TRPOSaute", "PPOSimmerPID", "TRPOSimmerPID",
)
OFF_POLICY = ("DDPG", "TD3", "SAC", "DDPGLag", "TD3Lag", "SACLag",
              "DDPGPID", "TD3PID", "SACPID")
SUPPORTED_ALGORITHMS = ON_POLICY + OFF_POLICY
UNCONSTRAINED = ("PolicyGradient", "NaturalPG", "TRPO", "PPO", "DDPG", "TD3", "SAC")
LAGRANGIAN = ("PPOLag", "TRPOLag", "RCPO", "PDO", "FOCOPS", "CUP",
              "CPPOPID", "TRPOPID", "DDPGLag", "TD3Lag", "SACLag",
              "DDPGPID", "TD3PID", "SACPID")
DIRECT_BUDGET = ("CPO", "PCPO", "OnCRPO", "IPO", "P3O",
                 "PPOEarlyTerminated", "TRPOEarlyTerminated")
AUGMENTED = ("PPOSaute", "TRPOSaute", "PPOSimmerPID", "TRPOSimmerPID")


def evaluation_budget(algorithm, algo_cfgs):
    """Fixed-target evaluation, not restoration of Simmer's unsaved controller."""
    if algorithm not in AUGMENTED:
        return None
    gamma, horizon = algo_cfgs["saute_gamma"], algo_cfgs["max_ep_len"]
    budget = algo_cfgs["upper_budget" if "Simmer" in algorithm else "safety_budget"]
    if not 0 < gamma < 1 or horizon <= 0 or budget <= 0:
        raise ValueError("Saute/Simmer require positive budget/horizon and 0 < saute_gamma < 1")
    return budget * (1 - gamma ** horizon) / (1 - gamma) / horizon
