import time, sys
import gym
from gym.envs.registration import register

try:
    register(id='RIS_MISO_PDA-v0', entry_point='environment:RIS_MISO_PDA')
except Exception:
    pass

import numpy as np
import torch
import Beta_Space_Exp_SAC, utils, optimization

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

env = gym.make('RIS_MISO_PDA-v0', num_antennas=4, num_RIS_elements=16,
               num_users=4, beta_min=0.6, seed=0)

agent = Beta_Space_Exp_SAC.Beta_Space_Exp_SAC(
    state_dim=env.state_dim, action_space=env.action_space,
    M=4, N=16, K=4, power_t=30, actor_lr=1e-3, critic_lr=1e-3,
    policy_type='Gaussian', alpha=0.2, target_update_interval=1,
    automatic_entropy_tuning=True, device=device,
    beta_min=0.6, discount=1.0, tau=1e-3)

buf = utils.BetaPrioritizedReplayBuffer(env.state_dim, env.action_dim, 16, 20000)

state = env.reset()
state = (state - state.mean()) / (state.std() + 1e-8)
exp_reg = 0.3

N = 500
print(f"Running {N} steps to benchmark timing...")
t0 = time.time()

for t in range(N):
    action, beta = agent.select_action(state, exp_reg)
    action = optimization.action_with_zf_beamformer(
        action, env.H_1, env.H_2, 4, 4, 16, 30.0)
    ns, r, d, _ = env.step(action, beta)
    ns = (ns - ns.mean()) / (ns.std() + 1e-8)
    buf.add(state, action, beta, ns, r, float(d))
    state = ns
    td_err, idx = agent.update_parameters(buf, exp_reg, 16)
    if idx is not None:
        buf.update_priorities(idx, td_err)
    exp_reg = 0.3 - 0.3 * t / 20000

elapsed = time.time() - t0
per_step_ms = elapsed / N * 1000
full_run_min = elapsed / N * 20000 / 60
full_paper_hrs = full_run_min * 90 / 60

print()
print("=" * 45)
print(f"  Benchmark: {N} steps in {elapsed:.1f}s")
print(f"  Per step:  {per_step_ms:.1f} ms")
print(f"  Full run (20k steps):  ~{full_run_min:.1f} min")
print(f"  Full paper (90 runs):  ~{full_paper_hrs:.1f} hours")
print("=" * 45)
