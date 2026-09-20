# ============================================================
# 实验七：真实神经网络 NPRF 审计（概念验证）
# 用 1D CNN 去噪器，训练白噪声，测试红/白/蓝噪声
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


print("="*70)
print("实验七：真实神经网络 NPRF 审计")
print("="*70)
print("\n[1/4] 生成训练数据...")

np.random.seed(0)
n_train = 20000
t_train = np.arange(n_train) / FS
clean_train = np.sin(2*np.pi*10*t_train) + np.sin(2*np.pi*60*t_train)
noise_train = generate_power_law_noise(n_train, FS, 0, 0.001)
noisy_train = clean_train + noise_train

ws = 101
X_tr, y_tr = [], []
for i in range(n_train - ws + 1):
    X_tr.append(noisy_train[i:i+ws])
    y_tr.append(clean_train[i:i+ws])
X_tr = torch.FloatTensor(np.array(X_tr)).unsqueeze(1)
y_tr = torch.FloatTensor(np.array(y_tr)).unsqueeze(1)
print(f"  训练样本数: {len(X_tr)}")

print("\n[2/4] 训练 CNN 去噪器（约 2-3 分钟）...")
torch.manual_seed(0)
model = CNN1DDenoiser()
opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.MSELoss()
loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)

for ep in range(100):
    for bx, by in loader:
        opt.zero_grad()
        loss = crit(model(bx), by)
        loss.backward()
        opt.step()
    if (ep+1) % 20 == 0:
        print(f"  Epoch {ep+1}/100, loss={loss.item():.6f}")

model.eval()
print("  训练完成！")

print("\n[3/4] 测试红/白/蓝噪声...")

n_test = 2000
t_test = np.arange(n_test) / FS
clean_test = np.sin(2*np.pi*10*t_test) + np.sin(2*np.pi*60*t_test)

results = {}
for alpha, name in [(-2, 'Red'), (0, 'White'), (2, 'Blue')]:
    np.random.seed(42)
    noise_test = generate_power_law_noise(n_test, FS, alpha, 0.002)
    noisy_test = clean_test + noise_test

    X_te = []
    for i in range(n_test - ws + 1):
        X_te.append(noisy_test[i:i+ws])
    X_te = torch.FloatTensor(np.array(X_te)).unsqueeze(1)

    with torch.no_grad():
        pred = model(X_te).squeeze(1).numpy()

    mid = ws // 2
    residual = pred[:, mid] - clean_test[mid:mid+len(pred)]

    sigma_px = np.std(residual)

    n_res = len(residual)
    freqs_res = np.fft.rfftfreq(n_res, DT)
    h_padded_res = np.zeros(n_res)
    h_padded_res[:SG_WINDOW] = signal.savgol_coeffs(SG_WINDOW, SG_ORDER,
                                                     deriv=2, delta=DT)
    H_sq_res = np.abs(np.fft.rfft(h_padded_res))**2
    nprf = compute_nprf(residual, freqs_res, H_sq_res, FS)

    a_hat = signal.savgol_filter(residual, SG_WINDOW, SG_ORDER,
                                  deriv=2, delta=DT, mode='interp')
    phys_err = np.sqrt(np.mean(a_hat**2)) / A_REF * 100

    results[name] = {'sigma': sigma_px, 'nprf': nprf, 'phys': phys_err,
                     'residual': residual}
    print(f"  {name:5s} (alpha={alpha:+d}): RMSE={sigma_px:.5f}, "
          f"NPRF={nprf:.3f}, PhysErr={phys_err:.1f}%")

np.savez('data/experiment7_real_model_results.npz',
         red_sigma=results['Red']['sigma'],
         red_nprf=results['Red']['nprf'],
         red_phys=results['Red']['phys'],
         red_residual=results['Red']['residual'],
         white_sigma=results['White']['sigma'],
         white_nprf=results['White']['nprf'],
         white_phys=results['White']['phys'],
         white_residual=results['White']['residual'],
         blue_sigma=results['Blue']['sigma'],
         blue_nprf=results['Blue']['nprf'],
         blue_phys=results['Blue']['phys'],
         blue_residual=results['Blue']['residual'])
print("\n  数据已保存: data/experiment7_real_model_results.npz")

print("\n[4/4] 生成图表...")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

ax = axes[0]
colors = {'Red': '#d62728', 'White': '#7f7f7f', 'Blue': '#1f77b4'}
for name in ['Red', 'White', 'Blue']:
    res = results[name]['residual']
    nperseg = min(256, len(res)//4)
    nperseg = max(nperseg, 64)
    f, p = signal.welch(res, FS, nperseg=nperseg, noverlap=nperseg//2,
                        scaling='density', average='median')
    ax.semilogy(f, p, color=colors[name], linewidth=1.5,
                label=f"{name} (NPRF={results[name]['nprf']:.2f})")
ax.axvspan(41, 61, alpha=0.15, color='#ff7f0e')
ax.set_xlim(0, 250)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
ax.set_title('(a) CNN Residual PSD')
ax.legend(fontsize=7)

ax = axes[1]
names = ['Red', 'White', 'Blue']
nprf_vals = [results[n]['nprf'] for n in names]
bars = ax.bar(names, nprf_vals, color=[colors[n] for n in names],
              alpha=0.7, edgecolor='black')
for bar, val in zip(bars, nprf_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
ax.axhline(1.0, color='black', linestyle='--', label='White baseline')
ax.set_ylabel('NPRF')
ax.set_title('(b) NPRF (Real CNN)')
ax.legend(fontsize=7)

ax = axes[2]
phys_vals = [results[n]['phys'] for n in names]
bars = ax.bar(names, phys_vals, color=[colors[n] for n in names],
              alpha=0.7, edgecolor='black')
for bar, val in zip(bars, phys_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Physical Error (%)')
ax.set_title('(c) Physical Error (Real CNN)')

plt.tight_layout()
plt.savefig('figures/exp7_real_model_nprf.png', dpi=300, bbox_inches='tight')
plt.close()
print("  图已保存: figures/exp7_real_model_nprf.png")

print("\n" + "="*70)
print("实验七完成！")
print("="*70)
