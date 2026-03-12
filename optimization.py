import numpy as np


def compute_zf_beamformer(H1, H2, Phi, M, K, power_t_dbm):
    """
    Zero-Forcing (ZF) Beamformer — Hybrid AO+DRL Novelty.

    Given fixed RIS phase matrix Phi (from the DRL agent), this function
    solves for the OPTIMAL transmit beamforming matrix G that completely
    nulls inter-user interference under a total power constraint.

    This replaces the DRL-learned G with a provably interference-free
    beamformer, allowing DRL to focus exclusively on RIS phase optimization.

    Args:
        H1    : (L, M) complex ndarray  — BS-to-RIS channel matrix
        H2    : (L, K) complex ndarray  — RIS-to-user channel matrix
        Phi   : (L, L) complex ndarray  — RIS diagonal phase-shift matrix
        M     : int — number of BS antennas
        K     : int — number of users (M == K required for ZF)
        power_t_dbm : float — total transmit power budget in dBm

    Returns:
        G_zf  : (M, K) complex ndarray — ZF beamforming matrix satisfying
                Tr(G G^H) = P_t (linear), with zero inter-user interference.
    """
    # Convert power from dBm to linear (milliwatts)
    power_t = 10 ** (power_t_dbm / 10.0)

    # Effective channel matrix: H_eff = H2^T @ Phi @ H1,  shape: (K, M)
    H_eff = H2.conj().T @ Phi @ H1

    try:
        # ZF precoder: G = H_eff^H @ (H_eff @ H_eff^H)^{-1}
        # This achieves H_eff @ G = I  (zero inter-user interference)
        gram = H_eff @ H_eff.conj().T  # (K, K)

        # Tikhonov regularization for numerical stability (epsilon = 1e-8)
        gram_reg = gram + 1e-8 * np.eye(K)
        gram_inv = np.linalg.inv(gram_reg)

        G_zf = H_eff.conj().T @ gram_inv  # (M, K)

    except np.linalg.LinAlgError:
        # Fallback: Moore-Penrose pseudoinverse
        G_zf = np.linalg.pinv(H_eff).conj().T  # (M, K)

    # Power normalization: scale G so that Tr(G G^H) = P_t
    # Frobenius norm squared = Tr(G G^H)
    frob_sq = np.real(np.trace(G_zf @ G_zf.conj().T))

    if frob_sq > 1e-12:
        G_zf = G_zf * np.sqrt(power_t / frob_sq)
    else:
        # Degenerate case: return identity-like matrix scaled to power budget
        G_zf = (np.eye(M, K, dtype=complex) * np.sqrt(power_t / M))

    return G_zf


def action_with_zf_beamformer(action, H1, H2, M, K, L, power_t_dbm):
    """
    Replaces the beamforming (G) part of an action vector with ZF-optimal G,
    while leaving the RIS phase shift part unchanged.

    The action vector layout expected by the environment:
        action[:M*K]      — real part of G,  shape (M*K,)
        action[M*K:2*M*K] — imag part of G,  shape (M*K,)
        action[-2L:-L]    — real part of Phi diagonal, shape (L,)
        action[-L:]       — imag part of Phi diagonal, shape (L,)

    Args:
        action       : (action_dim,) float ndarray — original agent action
        H1, H2       : channel matrices (from env.H1, env.H2)
        M, K, L      : system dimensions
        power_t_dbm  : transmit power budget in dBm

    Returns:
        action_ao    : (action_dim,) float ndarray — action with ZF G inserted
    """
    # Force 1D throughout — GPU agents may return (1, action_dim) instead of (action_dim,)
    action_flat = np.array(action).flatten()
    action_ao   = action_flat.copy()

    # Extract RIS phase shifts from agent's action (last 2L elements)
    phi_real = action_flat[-2 * L:-L]   # (L,)
    phi_imag = action_flat[-L:]          # (L,)

    # Build diagonal RIS phase matrix  Phi = diag(phi_real + j*phi_imag)
    Phi = np.diag(phi_real + 1j * phi_imag)  # (L, L)

    # Compute optimal ZF beamformer given current Phi
    G_zf = compute_zf_beamformer(H1, H2, Phi, M, K, power_t_dbm)  # (M, K)

    # Overwrite the G part of the action with ZF G
    action_ao[:M * K]          = G_zf.real.flatten()
    action_ao[M * K:2 * M * K] = G_zf.imag.flatten()

    return action_ao
