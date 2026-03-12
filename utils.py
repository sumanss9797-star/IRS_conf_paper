import numpy as np
import torch
from torch import nn


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Discrete Phase Shifts — Novelty #3
# Reference: "1/2/3-bit quantization-aware DRL for hardware-constrained RIS"
# ---------------------------------------------------------------------------

def quantize_phase_action(action, L, num_bits=2):
    """
    Apply discrete phase-shift quantization to the RIS phase part of an action.

    Real-world RIS hardware (PIN diodes, varactors) only supports a finite set
    of phase states.  This function snaps the continuous phase angles produced
    by the DRL agent to the nearest of 2^B uniformly-spaced discrete levels
    spanning (-π, π], while PRESERVING the original vector magnitude.

    Because quantization is applied AFTER the agent produces its action (outside
    the PyTorch computation graph), gradients flow through the continuous action
    unchanged — this is the standard Straight-Through Estimator (STE) approach
    used in quantization-aware training.

    Action vector layout (consistent with environment.py):
        action[:2*M*K]     — beamforming G (real + imag), unchanged
        action[-2*L:-L]    — RIS phi_real,  quantized
        action[-L:]        — RIS phi_imag,  quantized

    Args:
        action   : (action_dim,) NumPy float array — agent action (1-D)
        L        : int — number of RIS elements
        num_bits : int — phase resolution in bits (1, 2, or 3); gives 2, 4, or 8 levels

    Returns:
        action_q : (action_dim,) NumPy float array — action with quantized phases
    """
    if num_bits <= 0:
        return action  # no quantization (continuous)

    action_q = np.array(action, dtype=np.float64).flatten()

    # Extract phase components (last 2L elements)
    phi_real = action_q[-2 * L:-L].copy()   # shape (L,)
    phi_imag = action_q[-L:].copy()          # shape (L,)

    # Compute per-element magnitude (preserve it after quantization)
    magnitude = np.sqrt(phi_real ** 2 + phi_imag ** 2)
    magnitude = np.where(magnitude < 1e-12, 1.0, magnitude)   # avoid /0

    # Compute continuous phase angles in (-π, π]
    theta = np.arctan2(phi_real, phi_imag)   # shape (L,)

    # Quantize: snap each angle to nearest of 2^B equally-spaced levels
    num_levels = 2 ** num_bits
    step = 2.0 * np.pi / num_levels          # spacing between adjacent levels

    # Grid: levels at  -π + k*step  for k = 0,1,...,2^B-1
    theta_q = np.round(theta / step) * step  # nearest-level rounding

    # Wrap into (-π, π] to avoid numerical drift
    theta_q = (theta_q + np.pi) % (2.0 * np.pi) - np.pi

    # Project back to Cartesian, restoring the original magnitude
    phi_real_q = np.sin(theta_q) * magnitude
    phi_imag_q = np.cos(theta_q) * magnitude

    # Write quantized phases back into the action vector
    action_q[-2 * L:-L] = phi_real_q
    action_q[-L:]        = phi_imag_q

    return action_q.astype(np.float32)


# ---------------------------------------------------------------------------
# SumTree — binary tree for O(log n) prioritized sampling
# ---------------------------------------------------------------------------

class SumTree:
    """
    Binary sum tree where each leaf stores a priority value.
    Parent nodes store the sum of their children, enabling O(log n) sampling
    proportional to priority.
    """

    def __init__(self, capacity):
        self.capacity = capacity               # number of leaf nodes
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.ptr = 0                           # circular write pointer
        self.n_entries = 0

    @property
    def total(self):
        """Sum of all priorities (root node)."""
        return self.tree[0]

    @property
    def max_priority(self):
        """Max leaf priority (used for new additions)."""
        if self.n_entries == 0:
            return 1.0
        return float(np.max(self.tree[self.capacity - 1:self.capacity - 1 + self.n_entries]))

    def _propagate(self, idx, delta):
        """Propagate priority change up to root."""
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, leaf_idx, priority):
        """Update priority at a given leaf index (0-indexed leaf space)."""
        tree_idx = leaf_idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def add(self, priority):
        """Add a new priority; returns the data index (leaf_idx) written."""
        data_idx = self.ptr
        self.update(data_idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return data_idx

    def _retrieve(self, node_idx, s):
        """Retrieve the leaf node index for cumulative sum s."""
        left = 2 * node_idx + 1
        right = left + 1
        if left >= len(self.tree):          # leaf reached
            return node_idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def sample(self, s):
        """
        Sample one transition by cumulative priority s.
        Returns (tree_idx, priority, data_idx).
        """
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], data_idx


class ExperienceReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[index]).to(self.device),
            torch.FloatTensor(self.action[index]).to(self.device),
            torch.FloatTensor(self.next_state[index]).to(self.device),
            torch.FloatTensor(self.reward[index]).to(self.device),
            torch.FloatTensor(self.not_done[index]).to(self.device)
        )


class BetaExperienceReplayBuffer(object):
    def __init__(self, state_dim, action_dim, N, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.beta = np.zeros((max_size, N))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, beta, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.beta[self.ptr] = beta
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[index]).to(self.device),
            torch.FloatTensor(self.action[index]).to(self.device),
            torch.FloatTensor(self.beta[index]).to(self.device),
            torch.FloatTensor(self.next_state[index]).to(self.device),
            torch.FloatTensor(self.reward[index]).to(self.device),
            torch.FloatTensor(self.not_done[index]).to(self.device)
        )


# ---------------------------------------------------------------------------
# Prioritized Experience Replay (PER) Buffers  — Novelty #2
# Reference: Schaul et al. 2015  https://arxiv.org/abs/1511.05952
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer(object):
    """
    PER buffer for SAC.  Drop-in replacement for ExperienceReplayBuffer.

    Hyperparameters:
        alpha_per : priority exponent  (0 = uniform, 1 = fully prioritized)
        beta_is   : IS-weight exponent (0 = no correction, 1 = full correction)
    """

    def __init__(self, state_dim, action_dim, max_size=int(1e6),
                 alpha_per=0.6, beta_is=0.4):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.alpha_per = alpha_per
        self.beta_is = beta_is

        self.sumtree = SumTree(max_size)

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        priority = max(self.sumtree.max_priority, 1e-6) ** self.alpha_per
        self.sumtree.update(self.ptr, priority)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Sample proportional to priority; returns indices and IS weights."""
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size)
        segment = self.sumtree.total / batch_size

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            s = np.clip(s, 0.0, self.sumtree.total - 1e-10)
            _, priority, data_idx = self.sumtree.sample(s)
            data_idx = int(data_idx) % max(self.size, 1)
            indices[i] = data_idx
            priorities[i] = max(priority, 1e-10)

        probs = priorities / (self.sumtree.total + 1e-10)
        is_weights = (self.size * probs) ** (-self.beta_is)
        is_weights /= is_weights.max()

        return (
            torch.FloatTensor(self.state[indices]).to(self.device),
            torch.FloatTensor(self.action[indices]).to(self.device),
            torch.FloatTensor(self.next_state[indices]).to(self.device),
            torch.FloatTensor(self.reward[indices]).to(self.device),
            torch.FloatTensor(self.not_done[indices]).to(self.device),
            indices,
            torch.FloatTensor(is_weights).reshape(-1, 1).to(self.device),
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities from TD errors after each gradient step."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + 1e-6) ** self.alpha_per
            self.sumtree.update(int(idx), priority)


class BetaPrioritizedReplayBuffer(object):
    """
    PER buffer for Beta_Space_Exp_SAC.
    Drop-in replacement for BetaExperienceReplayBuffer.
    """

    def __init__(self, state_dim, action_dim, N, max_size=int(1e6),
                 alpha_per=0.6, beta_is=0.4):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.alpha_per = alpha_per
        self.beta_is = beta_is

        self.sumtree = SumTree(max_size)

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.beta_buf = np.zeros((max_size, N))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, beta, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.beta_buf[self.ptr] = beta
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        priority = max(self.sumtree.max_priority, 1e-6) ** self.alpha_per
        self.sumtree.update(self.ptr, priority)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Returns: (state, action, beta, next_state, reward, done, indices, IS_weights)"""
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size)
        segment = self.sumtree.total / batch_size

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            s = np.clip(s, 0.0, self.sumtree.total - 1e-10)
            _, priority, data_idx = self.sumtree.sample(s)
            data_idx = int(data_idx) % max(self.size, 1)
            indices[i] = data_idx
            priorities[i] = max(priority, 1e-10)

        probs = priorities / (self.sumtree.total + 1e-10)
        is_weights = (self.size * probs) ** (-self.beta_is)
        is_weights /= is_weights.max()

        return (
            torch.FloatTensor(self.state[indices]).to(self.device),
            torch.FloatTensor(self.action[indices]).to(self.device),
            torch.FloatTensor(self.beta_buf[indices]).to(self.device),
            torch.FloatTensor(self.next_state[indices]).to(self.device),
            torch.FloatTensor(self.reward[indices]).to(self.device),
            torch.FloatTensor(self.not_done[indices]).to(self.device),
            indices,
            torch.FloatTensor(is_weights).reshape(-1, 1).to(self.device),
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities from TD errors after each gradient step."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + 1e-6) ** self.alpha_per
            self.sumtree.update(int(idx), priority)
