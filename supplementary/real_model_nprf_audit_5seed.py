# ============================================================
# 实验七（5-seed，方案 B+ 修正版）
# 关键修正：N_TRAIN=20000, EPOCHS=100, 训练/测试同分布
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

FS = 1000.0
A_REF = 9.8
DT = 1.0 / FS
SG_WINDOW = 21
SG_ORDER = 3

N_SEEDS = 5
EPOCHS = 100
WS = 101
SIGMA = 0.002          # 训练和测试用同一个 sigma
N_TRAIN = 20000        # 回到原版，保证训练充分


def generate_power_law_noise(n_samples, fs, alpha, sigma=0.001):
    freqs = np.fft.rfftfreq(n_samples, 1.0/fs)
    n_freqs = len(freqs)
    spectrum = np.zeros(n_freqs, dtype=complex)
    freq_mask = freqs > 0
    freqs_safe = np.where(freq_mask, freqs, 1.0)
    power = np.where(freq_mask, freqs_safe**alpha, 0.0)
    magnitude = np.sqrt(np.maximum(power, 0))
    phases = np.random.uniform(0, 2*np.pi, n_freqs)
    spectrum = magnitude * np.exp(1j * phases)
    noise = np.fft.irfft(spectrum, n=n_samples)
    noise = (noise - np.mean(noise)) / np.std(noise) * sigma
    return noise


def compute_nprf(noise, freqs, H_squared, fs):
    nperseg = min(256, len(noise) // 4)
    nperseg = max(nperseg, 8)
    if nperseg < 8:
        nperseg = len(noise) // 2
    freqs_welch, psd = signal.welch(noise, fs=fs, nperseg=nperseg,
                                     noverlap=nperseg//2, scaling='density',
                                     average='median')
    psd_interp = np.interp(freqs, freqs_welch, psd)
    sigma_px = np.std(noise)
    if sigma_px < 1e-15:
        return np.nan
    H_squared_mean = np.mean(H_squared)
    numerator = np.trapezoid(H_squared * psd_interp, freqs)
    denominator = sigma_px**2 * H_squared_mean
    if denominator < 1e-30:
        return np.nan
    return numerator / denominator


class CNN1DDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 16, 7, padding=3), nn.ReLU(),
            nn.Conv1d(16, 1, 7, padding=3)
        )

    def forward(self, x):
        return self.net(x)


print("=" * 70)
print("实验七（5-seed, 方案 B+ 修正版）")
print("=" * 70)
print("配置: N_TRAIN=%d, EPOCHS=%d, SIGMA=%.4f" % (N_TRAIN, EPOCHS, SIGMA))

# ------------------------------------------------------------
# 测试集（固定）
# ------------------------------------------------------------
n_test = 2000
t_test = np.arange(n_test) / FS
clean_test = np.sin(2*np.pi*10*t_test) + np.sin(2*np.pi*60*t_test)

np.random.seed(42)
test_noises = {}
for alpha, name in [(-2, 'Red'), (0, 'White'), (2, 'Blue')]:
    test_noises[name] = generate_power_law_noise(n_test, FS, alpha, SIGMA)

X_test_dict = {}
for name in ['Red', 'White', 'Blue']:
    noisy = clean_test + test_noises[name]
    X_te = []
    for i in range(n_test - WS + 1):
        X_te.append(noisy[i:i+WS])
    X_test_dict[name] = torch.FloatTensor(np.array(X_te)).unsqueeze(1)

mid = WS // 2
clean_center = clean_test[mid:mid + (n_test - WS + 1)]

n_res = len(clean_center)
freqs_res = np.fft.rfftfreq(n_res, DT)
h_padded_res = np.zeros(n_res)
h_padded_res[:SG_WINDOW] = signal.savgol_coeffs(SG_WINDOW, SG_ORDER,
                                                 deriv=2, delta=DT)
H_sq_res = np.abs(np.fft.rfft(h_padded_res))**2


# ------------------------------------------------------------
# 5-seed 训练
# ------------------------------------------------------------
all_results = {name: {'sigma': [], 'nprf': [], 'phys': [], 'residual': []}
               for name in ['Red', 'White', 'Blue']}
all_losses = []

for seed in range(N_SEEDS):
    print("\n[Seed %d/%d] 训练中..." % (seed + 1, N_SEEDS))

    torch.manual_seed(seed)
    np.random.seed(seed)

    t_train = np.arange(N_TRAIN) / FS
    clean_train = np.sin(2*np.pi*10*t_train) + np.sin(2*np.pi*60*t_train)
    noise_train = generate_power_law_noise(N_TRAIN, FS, 0, SIGMA)   # 白噪声训练
    noisy_train = clean_train + noise_train

    X_tr, y_tr = [], []
    for i in range(N_TRAIN - WS + 1):
        X_tr.append(noisy_train[i:i+WS])
        y_tr.append(clean_train[i:i+WS])
    X_tr = torch.FloatTensor(np.array(X_tr)).unsqueeze(1)
    y_tr = torch.FloatTensor(np.array(y_tr)).unsqueeze(1)

    model = CNN1DDenoiser()
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)

    loss_history = []
    for ep in range(EPOCHS):
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
        loss_history.append(loss.item())
        if (ep + 1) % 20 == 0:
            print("    Epoch %d/%d, loss=%.7f" % (ep + 1, EPOCHS, loss.item()))
    model.eval()
    all_losses.append(loss_history)

    for name in ['Red', 'White', 'Blue']:
        with torch.no_grad():
            pred = model(X_test_dict[name]).squeeze(1).numpy()
        residual = pred[:, mid] - clean_center

        sigma_px = np.std(residual)
        nprf = compute_nprf(residual, freqs_res, H_sq_res, FS)

        a_hat = signal.savgol_filter(residual, SG_WINDOW, SG_ORDER,
                                      deriv=2, delta=DT, mode='interp')
        phys_err = np.sqrt(np.mean(a_hat**2)) / A_REF * 100

        all_results[name]['sigma'].append(sigma_px)
        all_results[name]['nprf'].append(nprf)
        all_results[name]['phys'].append(phys_err)
        all_results[name]['residual'].append(residual)


# ------------------------------------------------------------
# 汇总
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("5-seed 汇总（mean ± std）")
print("=" * 70)

summary = {}
for name in ['Red', 'White', 'Blue']:
    sigma_arr = np.array(all_results[name]['sigma'])
    nprf_arr = np.array(all_results[name]['nprf'])
    phys_arr = np.array(all_results[name]['phys'])
    summary[name] = {
        'sigma_mean': np.mean(sigma_arr), 'sigma_std': np.std(sigma_arr, ddof=1),
        'nprf_mean': np.mean(nprf_arr),   'nprf_std': np.std(nprf_arr, ddof=1),
        'phys_mean': np.mean(phys_arr),   'phys_std': np.std(phys_arr, ddof=1),
    }
    print("\n%s noise:" % name)
    print("  Pixel RMSE = %.5f ± %.5f  (std/mean=%.0f%%)" % (
        summary[name]['sigma_mean'], summary[name]['sigma_std'],
        summary[name]['sigma_std']/summary[name]['sigma_mean']*100))
    print("  NPRF       = %.3f ± %.3f  (std/mean=%.0f%%)" % (
        summary[name]['nprf_mean'], summary[name]['nprf_std'],
        summary[name]['nprf_std']/summary[name]['nprf_mean']*100))
    print("  Phys error = %.1f%% ± %.1f%%  (std/mean=%.0f%%)" % (
        summary[name]['phys_mean'], summary[name]['phys_std'],
        summary[name]['phys_std']/summary[name]['phys_mean']*100))

sigma_order = sorted(['Red', 'White', 'Blue'],
                     key=lambda n: summary[n]['sigma_mean'], reverse=True)
phys_order = sorted(['Red', 'White', 'Blue'],
                    key=lambda n: summary[n]['phys_mean'], reverse=True)
nprf_order = sorted(['Red', 'White', 'Blue'],
                    key=lambda n: summary[n]['nprf_mean'], reverse=True)

print("\n排序比较：")
print("  Pixel RMSE 从高到低: %s" % sigma_order)
print("  NPRF       从高到低: %s" % nprf_order)
print("  Phys error 从高到低: %s" % phys_order)

print("\n判定:")
ok = True
if nprf_order == phys_order:
    print("  ✅ NPRF 排序 == 物理误差排序")
else:
    print("  ❌ NPRF 排序 != 物理误差排序"); ok = False
if sigma_order != phys_order:
    print("  ✅ 像素 RMSE 排序 != 物理误差排序")
else:
    print("  ❌ 像素 RMSE 排序 == 物理误差排序"); ok = False

# 方差检查
max_cv = max(summary[n]['nprf_std']/summary[n]['nprf_mean'] for n in ['Red','White','Blue'])
if max_cv < 0.30:
    print("  ✅ 最大 NPRF 变异系数 %.0f%% < 30%%，数据稳定" % (max_cv*100))
else:
    print("  ❌ 最大 NPRF 变异系数 %.0f%% >= 30%%，方差过大" % (max_cv*100))
    ok = False

print("\n最终判定: %s" % ("✅ 5-seed 可用" if ok else "❌ 5-seed 不可用，应回单 seed"))


# ------------------------------------------------------------
# 保存
# ------------------------------------------------------------
save_dict = {}
for name in ['Red', 'White', 'Blue']:
    save_dict['%s_sigma_mean' % name.lower()] = summary[name]['sigma_mean']
    save_dict['%s_sigma_std' % name.lower()]  = summary[name]['sigma_std']
    save_dict['%s_nprf_mean' % name.lower()]  = summary[name]['nprf_mean']
    save_dict['%s_nprf_std' % name.lower()]   = summary[name]['nprf_std']
    save_dict['%s_phys_mean' % name.lower()]  = summary[name]['phys_mean']
    save_dict['%s_phys_std' % name.lower()]   = summary[name]['phys_std']
    save_dict['%s_residual_first' % name.lower()] = all_results[name]['residual'][0]

np.savez('data/experiment7_5seed_results.npz', **save_dict)
print("\n数据已保存: data/experiment7_5seed_results.npz")

# 画图
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
colors = {'Red': '#d62728', 'White': '#7f7f7f', 'Blue': '#1f77b4'}

# (a) loss 曲线
ax = axes[0]
for i, lh in enumerate(all_losses):
    ax.semilogy(lh, alpha=0.7, label='seed %d' % i)
ax.set_xlabel('Epoch')
ax.set_ylabel('Training loss')
ax.set_title('(a) Training Convergence')
ax.legend(fontsize=7)

# (b) PSD
ax = axes[1]
for name in ['Red', 'White', 'Blue']:
    res = all_results[name]['residual'][0]
    nperseg = min(256, len(res) // 4)
    nperseg = max(nperseg, 64)
    f, p = signal.welch(res, FS, nperseg=nperseg, noverlap=nperseg // 2,
                        scaling='density', average='median')
    ax.semilogy(f, p, color=colors[name], linewidth=1.5,
                label="%s (NPRF=%.2f)" % (name, summary[name]['nprf_mean']))
ax.axvspan(41, 61, alpha=0.15, color='#ff7f0e')
ax.set_xlim(0, 250)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
ax.set_title('(b) CNN Residual PSD (seed 0)')
ax.legend(fontsize=7)

# (c) NPRF
ax = axes[2]
names = ['Red', 'White', 'Blue']
nprf_means = [summary[n]['nprf_mean'] for n in names]
nprf_stds  = [summary[n]['nprf_std']  for n in names]
bars = ax.bar(names, nprf_means, yerr=nprf_stds, capsize=5,
              color=[colors[n] for n in names], alpha=0.7, edgecolor='black')
for bar, val in zip(bars, nprf_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            "%.2f" % val, ha='center', va='bottom', fontweight='bold')
ax.axhline(1.0, color='black', linestyle='--', label='White baseline')
ax.set_ylabel('NPRF')
ax.set_title('(c) NPRF (5-seed mean ± std)')
ax.legend(fontsize=7)

# (d) 物理误差
ax = axes[3]
phys_means = [summary[n]['phys_mean'] for n in names]
phys_stds  = [summary[n]['phys_std']  for n in names]
bars = ax.bar(names, phys_means, yerr=phys_stds, capsize=5,
              color=[colors[n] for n in names], alpha=0.7, edgecolor='black')
for bar, val in zip(bars, phys_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            "%.0f%%" % val, ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Physical Error (%)')
ax.set_title('(d) Physical Error (5-seed mean ± std)')

plt.tight_layout()
plt.savefig('figures/exp7_real_model_nprf_5seed.png', dpi=300, bbox_inches='tight')
plt.close()
print("图已保存: figures/exp7_real_model_nprf_5seed.png")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
