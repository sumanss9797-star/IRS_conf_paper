# RIS-MISO DRL — Novel Contributions

> Extended implementation of the conference paper:
> **"Deep Reinforcement Learning Based Joint Downlink Beamforming and RIS Configuration in RIS-aided MU-MISO Systems Under Hardware Impairments and Imperfect CSI"**  
> Saglam, Gurgunoglu, Kozat — IEEE ICC Workshops 2023

---

## What's New (Our Contributions)

This repository extends the original β-SAC codebase with **three novel algorithmic improvements**:

### Novelty #1 — Hybrid AO + DRL (`--use_ao`)
Replaces the DRL-learned beamforming matrix **G** with a closed-form **Zero-Forcing (ZF)** solution at each step. Given fixed RIS phases **Φ** from the DRL agent, the ZF precoder completely nulls inter-user interference:

```
H_eff = H2^H Φ H1
G_ZF  = H_eff^H (H_eff H_eff^H + εI)^{-1}    (scaled to P_t)
```

This lets the DRL agent focus exclusively on the harder non-convex RIS phase optimization.

### Novelty #2 — Prioritized Experience Replay (`--use_per`)
Replaces the uniform replay buffer with a **SumTree-based PER** buffer (Schaul et al., 2015). Transitions are sampled proportional to their TD error — more "surprising" experiences are replayed more often. Importance Sampling (IS) weights correct the gradient bias.

### Novelty #3 — Discrete Phase Shifts (`--use_discrete_phases`)
Real RIS hardware (PIN diodes, varactors) only supports **finite discrete phase states**. After the DRL agent outputs continuous phases, they are **quantized to the nearest of 2^B uniformly-spaced levels** in (−π, π], while preserving the original vector magnitude. B ∈ {1, 2, 3} bits.

---

## Key Results

| Variant | N | Seeds | Best Mismatch Reward |
|---|---|---|---|
| SAC (original) | 16 | 10 | 12.68 ± 0.83 |
| **β-SAC (original paper)** | 16 | 10 | **14.14 ± 1.11** |
| β-SAC + AO + PER + 2-bit DP (ours) | 16 | 5 | 8.44 ± 0.72 |
| AO + PER baseline | 4 | 1 | 2.61 |
| **AO + PER + 2-bit DP (ours)** | 4 | 1 | **5.11 (+96.2%)** |
| AO + PER + 1-bit DP (ours) | 4 | 1 | 5.35 (+105.3%) |

> **Note:** N=16 original uses continuous phases (no hardware constraint). Our N=4 comparison shows +96–105% improvement within the same matched configuration.

---

## File Structure

```
conf_paper/
├── environment.py           # RIS-MISO-PDA Gym environment
├── SAC.py                   # Soft Actor-Critic (with PER support)
├── Beta_Space_Exp_SAC.py    # β-SAC algorithm (with PER support)
├── optimization.py          # [NEW] ZF beamformer — Novelty #1
├── utils.py                 # [MODIFIED] SumTree + PER buffers + quantize_phase_action
├── main.py                  # [MODIFIED] Training loop with all 3 novelty flags
├── Phase_AO_SAC.py          # Phase AO variant
├── requirements.txt
└── README.md
```

---

## How to Run

### Setup
```bash
conda create -n risenv python=3.8
conda activate risenv
pip install -r requirements.txt
```

### Baseline (original behaviour)
```bash
python main.py --use_ao False --use_per False --use_discrete_phases False
```

### All Three Novelties — 2-bit Discrete Phases (recommended)
```bash
python main.py --use_ao True --use_per True --use_discrete_phases True --num_phase_bits 2
```

### N=16 Multi-Seed (parallel)
```bash
for seed in 0 1 2 3 4; do
  python main.py --num_RIS_elements 16 \
    --use_ao True --use_per True \
    --use_discrete_phases True --num_phase_bits 2 \
    --seed $seed &
done
wait
```

### All CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--use_ao` | `True` | Enable Hybrid AO+DRL (ZF beamforming) |
| `--use_per` | `True` | Enable Prioritized Experience Replay |
| `--use_discrete_phases` | `False` | Enable discrete phase quantization |
| `--num_phase_bits` | `2` | Phase resolution: 1/2/3 bits (2/4/8 levels) |
| `--num_RIS_elements` | `4` | Number of RIS elements |
| `--max_time_steps` | `20000` | Total training steps |
| `--seed` | `0` | Random seed |
| `--policy` | `Beta_Space_Exp_SAC` | Algorithm: SAC or Beta_Space_Exp_SAC |

---

## Original Paper
```
@inproceedings{saglam2023deep,
  title={Deep Reinforcement Learning Based Joint Downlink Beamforming and
         RIS Configuration in RIS-aided MU-MISO Systems Under Hardware
         Impairments and Imperfect CSI},
  author={Saglam, Baturalp and Gurgunoglu, Doga and Kozat, Suleyman Serdar},
  booktitle={IEEE ICC Workshops},
  year={2023}
}
```
