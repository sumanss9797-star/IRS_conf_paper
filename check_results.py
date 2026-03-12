import numpy as np
import os

result_path = "/home/m-suman/IRS/new/conf_paper/Results/Beta_min. = 0.6, K = 4, M = 4, N = 4, P_t = 30.0/Beta_Space_Exp_SAC_AO_PER_mismatch_0.npy"
data = np.load(result_path)

print(f"=== AO+PER Result Analysis (Seed 0) ===")
print(f"Total steps recorded : {len(data)}")
print(f"Any NaN/Inf          : {np.any(np.isnan(data)) or np.any(np.isinf(data))}")
print()
print(f"--- Reward Statistics ---")
print(f"Min reward           : {np.min(data):.4f}")
print(f"Max reward           : {np.max(data):.4f}")
print(f"Mean reward          : {np.mean(data):.4f}")
print(f"Final 1000 mean      : {np.mean(data[-1000:]):.4f}  (convergence region)")
print(f"Final 500 mean       : {np.mean(data[-500:]):.4f}")
print()
print(f"--- Learning Progression (Quarters) ---")
quarters = len(data) // 4
for i in range(4):
    seg = data[i*quarters:(i+1)*quarters]
    print(f"  Q{i+1} (steps {i*quarters+1:>6}-{(i+1)*quarters:>6}): mean={np.mean(seg):.4f}, max={np.max(seg):.4f}")

# Compare against original paper's learning curve
orig_path = "/home/m-suman/IRS/new/conf_paper/Learning Curves/Beta_min. = 0.6, K = 4, M = 4, L = 16, P_t = 30.0/Beta_Space_Exp_SAC_mismatch_0.npy"
if os.path.exists(orig_path):
    orig = np.load(orig_path)
    print()
    print(f"--- Comparison vs Original Paper (Seed 0, L=16) ---")
    print(f"Original max reward  : {np.max(orig):.4f}")
    print(f"AO+PER   max reward  : {np.max(data):.4f}")
    print(f"Original final 1000  : {np.mean(orig[-1000:]):.4f}")
    print(f"AO+PER   final 1000  : {np.mean(data[-1000:]):.4f}")
    improvement = (np.mean(data[-1000:]) - np.mean(orig[-1000:])) / abs(np.mean(orig[-1000:])) * 100
    print(f"Improvement          : {improvement:+.1f}%")
else:
    print(f"\nNote: Results saved under N=4 path (the --num_RIS_elements flag default was 4 not 16)")
    print(f"Original paper curve path checked: {orig_path}")
