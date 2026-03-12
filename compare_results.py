import numpy as np

ao_per = np.load('Results/Beta_min. = 0.6, K = 4, M = 4, N = 16, P_t = 30.0/Beta_Space_Exp_SAC_AO_PER_mismatch_0.npy')
base   = np.load('Results/Beta_min. = 0.6, K = 4, M = 4, N = 16, P_t = 30.0/Beta_Space_Exp_SAC_mismatch_0.npy')

print('====== FAIR COMPARISON: SAME CODE, SAME CONFIG, SEED 0, L=16 ======')
print(f'{"Metric":<25} {"Baseline":>12} {"AO+PER":>12} {"Change":>10}')
print('-'*62)

metrics = [
    ('Max reward (bits/Hz)',   np.max(base),            np.max(ao_per)),
    ('Final 1k mean',          np.mean(base[-1000:]),   np.mean(ao_per[-1000:])),
    ('Final 500 mean',         np.mean(base[-500:]),    np.mean(ao_per[-500:])),
    ('Overall mean',           np.mean(base),           np.mean(ao_per)),
    ('Early  (steps 1-5k)',    np.mean(base[:5000]),    np.mean(ao_per[:5000])),
    ('Mid    (steps 5k-10k)',  np.mean(base[5000:10000]), np.mean(ao_per[5000:10000])),
    ('Late   (steps 10k-20k)', np.mean(base[10000:]),   np.mean(ao_per[10000:])),
]

for name, b, a in metrics:
    chg = (a - b) / abs(b) * 100
    arrow = '▲' if chg > 0 else '▼'
    print(f'{name:<25} {b:>12.3f} {a:>12.3f} {arrow}{abs(chg):>8.1f}%')

print()
print('--- Learning speed: steps to first exceed threshold ---')
for thresh in [5.0, 7.0, 8.0]:
    print(f'  Threshold = {thresh} bits/Hz:')
    for label, arr in [('  Baseline', base), ('  AO+PER  ', ao_per)]:
        cummax = np.maximum.accumulate(arr)
        idx = np.argmax(cummax >= thresh)
        print(f'    {label}: step {idx if cummax[idx] >= thresh else ">20000"}')
