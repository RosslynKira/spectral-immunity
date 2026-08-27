# ============================================================
# ICAIS 2026 - 补充实验代码
# 包含：实验六（篮球轨迹）、实验2b（预警案例）、实验3b（后处理）
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import wilcoxon, mannwhitneyu
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os

warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ============================================================
# 全局设置
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

C_RED = '#d62728'
C_WHITE = '#7f7f7f'
C_BLUE = '#1f77b4'
C_PHYS = '#ff7f0e'
C_NPRF = '#9467bd'
C_GREEN = '#2ca02c'

FS = 1000.0
A_REF = 9.8
DT = 1.0 / FS


def generate_power_law_noise(n_samples, fs, alpha, sigma=0.001):
    """生成幂律噪声 PSD ∝ f^alpha"""
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
    """计算NPRF"""
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


def apply_sg_filter(x, window=21, order=3, fs=FS):
    """应用SG二阶导数滤波器"""
    return signal.savgol_filter(x, window, order, deriv=2, delta=1.0/fs, mode='interp')


def compute_physical_error(x_hat, x_true, a_true):
    """计算物理误差"""
    a_hat = apply_sg_filter(x_hat)
    min_len = min(len(a_hat), len(a_true))
    rmse_a = np.sqrt(np.mean((a_hat[:min_len] - a_true[:min_len])**2))
    return (rmse_a / A_REF) * 100


# ============================================================
# 实验2b：NPRF早期预警案例
# ============================================================
print("="*70)
print("实验2b：NPRF早期预警案例")
print("="*70)

def generate_early_warning_case():
    fs = 1000.0
    n = 1000
    t = np.arange(n) / fs
    target_sigma = 0.002
    
    # 模型A：红噪声残差（低频集中）
    np.random.seed(1)
    resid_A = generate_power_law_noise(n, fs, -2, target_sigma)
    
    # 模型B：共振带噪声（55-65 Hz带通）
    np.random.seed(2)
    b_band, a_band = signal.butter(4, [55, 65], fs=fs, btype='band')
    white = np.random.randn(n)
    resid_B = signal.lfilter(b_band, a_band, white)
    resid_B = (resid_B - np.mean(resid_B)) / np.std(resid_B) * target_sigma
    
    sigma_A = np.std(resid_A)
    sigma_B = np.std(resid_B)
    
    # SG频响
    sg_coeffs = signal.savgol_coeffs(21, 3, deriv=2, delta=1.0/fs)
    h_padded = np.zeros(n)
    h_padded[:21] = sg_coeffs
    H_squared = np.abs(np.fft.rfft(h_padded))**2
    freqs = np.fft.rfftfreq(n, 1.0/fs)
    
    # NPRF
    nprf_A = compute_nprf(resid_A, freqs, H_squared, fs)
    nprf_B = compute_nprf(resid_B, freqs, H_squared, fs)
    
    # 物理误差
    x_true = 0.5 * 9.8 * t**2
    a_true = np.full(n, 9.8)
    
    a_hat_A = apply_sg_filter(x_true + resid_A)
    a_hat_B = apply_sg_filter(x_true + resid_B)
    
    rmse_a_A = np.sqrt(np.mean((a_hat_A - a_true)**2))
    rmse_a_B = np.sqrt(np.mean((a_hat_B - a_true)**2))
    
    phys_A = rmse_a_A / 9.8 * 100
    phys_B = rmse_a_B / 9.8 * 100
    
    print(f"\n模型A（红噪声残差）:")
    print(f"  像素RMSE = {sigma_A:.6f}")
    print(f"  NPRF = {nprf_A:.3f}")
    print(f"  物理误差 = {phys_A:.1f}%")
    
    print(f"\n模型B（共振带残差）:")
    print(f"  像素RMSE = {sigma_B:.6f}")
    print(f"  NPRF = {nprf_B:.3f}")
    print(f"  物理误差 = {phys_B:.1f}%")
    
    print(f"\n像素RMSE差异: {abs(sigma_A - sigma_B) / sigma_A * 100:.3f}%")
    print(f"物理误差比: {phys_B / phys_A:.2f}倍")
    
    return {
        'sigma_A': sigma_A, 'sigma_B': sigma_B,
        'nprf_A': nprf_A, 'nprf_B': nprf_B,
        'phys_A': phys_A, 'phys_B': phys_B,
        'resid_A': resid_A, 'resid_B': resid_B,
        'freqs': freqs, 'H_squared': H_squared
    }

result_ew = generate_early_warning_case()
np.savez('data/experiment2b_results.npz',
         sigma_A=result_ew['sigma_A'], sigma_B=result_ew['sigma_B'],
         nprf_A=result_ew['nprf_A'], nprf_B=result_ew['nprf_B'],
         phys_A=result_ew['phys_A'], phys_B=result_ew['phys_B'])

# 可视化
fig_ew, axes_ew = plt.subplots(1, 3, figsize=(12, 3.5))

ax = axes_ew[0]
nper_ew = 256
f_A_ew, p_A_ew = signal.welch(result_ew['resid_A'], 1000, nperseg=nper_ew)
f_B_ew, p_B_ew = signal.welch(result_ew['resid_B'], 1000, nperseg=nper_ew)
ax.semilogy(f_A_ew, p_A_ew, color=C_BLUE, linewidth=1.5, 
            label=f"Model A (NPRF={result_ew['nprf_A']:.2f})")
ax.semilogy(f_B_ew, p_B_ew, color=C_RED, linewidth=1.5,
            label=f"Model B (NPRF={result_ew['nprf_B']:.2f})")
ax.axvspan(41, 61, alpha=0.15, color=C_PHYS)
ax.set_xlim(0, 250)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
ax.set_title('(a) Residual PSD (Same Pixel RMSE)')
ax.legend(fontsize=8)

ax = axes_ew[1]
weighted_A = result_ew['H_squared'] * np.interp(result_ew['freqs'], f_A_ew, p_A_ew)
weighted_B = result_ew['H_squared'] * np.interp(result_ew['freqs'], f_B_ew, p_B_ew)
ax.plot(result_ew['freqs'], weighted_A, color=C_BLUE, linewidth=1, label='Model A weighted')
ax.plot(result_ew['freqs'], weighted_B, color=C_RED, linewidth=1, label='Model B weighted')
ax.set_xlim(0, 250)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('H²(f)·PSD(f)')
ax.set_title('(b) SG-Weighted Spectral Density')
ax.legend(fontsize=8)

ax = axes_ew[2]
labels_ew = ['Model A\n(Low-freq resid)', 'Model B\n(Resonant resid)']
phys_vals_ew = [result_ew['phys_A'], result_ew['phys_B']]
bars_ew = ax.bar(labels_ew, phys_vals_ew, color=[C_BLUE, C_RED], 
                  alpha=0.7, width=0.5, edgecolor='black')
for bar, val in zip(bars_ew, phys_vals_ew):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val:.0f}%', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Physical Error (%)')
ax.set_title('(c) Physical Error (Same Pixel RMSE)')
ax.set_ylim(0, max(phys_vals_ew) * 1.3)

plt.tight_layout()
plt.savefig('figures/nprf_early_warning.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n图已保存: figures/nprf_early_warning.png")


# ============================================================
# 实验3b：后处理频谱整形
# ============================================================
print("\n" + "="*70)
print("实验3b：后处理频谱整形")
print("="*70)

class MLPDenoiser(nn.Module):
    def __init__(self, input_dim=21, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def generate_denoising_data(n, window_size, fs, noise_alpha, noise_sigma,
                            freqs_sig=[10.0, 60.0], amps=[1.0, 1.0]):
    tt = np.arange(n) / fs
    clean = np.zeros(n)
    for f, a in zip(freqs_sig, amps):
        clean += a * np.sin(2*np.pi*f*tt)
    noise = generate_power_law_noise(n, fs, noise_alpha, noise_sigma)
    noisy = clean + noise
    X, y = [], []
    for i in range(n - window_size + 1):
        X.append(noisy[i:i+window_size])
        y.append(clean[i:i+window_size])
    return np.array(X), np.array(y)


def train_model(model, loader, epochs=200, lr=1e-3, device='cpu'):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            out = model(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
    return model


N_SEEDS = 10
notch_freqs = [45, 48, 51, 54, 57]
NOTCH_Q = 8.0
ws = 21

# 测试集
n_test = 2000
tt_test = np.arange(n_test) / FS
test_clean = np.zeros(n_test)
for f, a in [(10.0, 1.0), (60.0, 1.0)]:
    test_clean += a * np.sin(2*np.pi*f*tt_test)

np.random.seed(789)
test_noise_w = generate_power_law_noise(n_test, FS, 0, 0.001)
test_noise_b = generate_power_law_noise(n_test, FS, 2, 0.001)
test_noise = (test_noise_w + test_noise_b) / np.sqrt(2)
test_noise = (test_noise - np.mean(test_noise)) / np.std(test_noise) * 0.002
test_noisy = test_clean + test_noise

X_test, y_test = [], []
for i in range(n_test - ws + 1):
    X_test.append(test_noisy[i:i+ws])
    y_test.append(test_clean[i:i+ws])
X_test = np.array(X_test)
y_test = np.array(y_test)

n_windows = len(X_test)
t_window = np.arange(n_windows) * DT
x_true_window = 0.5 * A_REF * t_window**2
a_true_window = np.full(n_windows, A_REF)

freqs_window = np.fft.rfftfreq(n_windows, DT)
h_padded_window = np.zeros(n_windows)
sg_coeffs = signal.savgol_coeffs(21, 3, deriv=2, delta=DT)
h_padded_window[:21] = sg_coeffs
H_squared_window = np.abs(np.fft.rfft(h_padded_window))**2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n使用设备: {device}")

results_before = []
results_after = []

for seed in range(N_SEEDS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 训练标准MSE模型
    X_tr, y_tr = generate_denoising_data(10000, ws, FS, 0, 0.001)
    X_tr_t = torch.FloatTensor(X_tr)
    y_tr_t = torch.FloatTensor(y_tr)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=128, shuffle=True)
    
    model = MLPDenoiser(input_dim=ws, hidden_dims=[64, 32])
    model = train_model(model, loader, device=device)
    model.eval()
    
    X_te_t = torch.FloatTensor(X_test)
    with torch.no_grad():
        pred = model(X_te_t).numpy()
    
    resid = pred[:, ws//2] - y_test[:, ws//2]
    
    # 后处理前
    nprf_before = compute_nprf(resid, freqs_window, H_squared_window, FS)
    phys_before = compute_physical_error(x_true_window + resid, x_true_window, a_true_window)
    sigma_before = np.std(resid)
    
    # 后处理：5级陷波器
    resid_filtered = resid.copy()
    for f0 in notch_freqs:
        b_n, a_n = signal.iirnotch(f0, NOTCH_Q, FS)
        resid_filtered = signal.lfilter(b_n, a_n, resid_filtered)
    
    nprf_after = compute_nprf(resid_filtered, freqs_window, H_squared_window, FS)
    phys_after = compute_physical_error(x_true_window + resid_filtered, 
                                         x_true_window, a_true_window)
    sigma_after = np.std(resid_filtered)
    
    results_before.append({'nprf': nprf_before, 'phys': phys_before, 'sigma': sigma_before})
    results_after.append({'nprf': nprf_after, 'phys': phys_after, 'sigma': sigma_after})
    
    print(f"Seed {seed}: NPRF {nprf_before:.3f}→{nprf_after:.3f}, "
          f"phys {phys_before:.1f}%→{phys_after:.1f}%, "
          f"σ {sigma_before:.5f}→{sigma_after:.5f}")

nprf_b_arr = np.array([r['nprf'] for r in results_before])
nprf_a_arr = np.array([r['nprf'] for r in results_after])
phys_b_arr = np.array([r['phys'] for r in results_before])
phys_a_arr = np.array([r['phys'] for r in results_after])
sigma_b_arr = np.array([r['sigma'] for r in results_before])
sigma_a_arr = np.array([r['sigma'] for r in results_after])

print(f"\n后处理前 (n={N_SEEDS}):")
print(f"  NPRF = {np.mean(nprf_b_arr):.3f} ± {np.std(nprf_b_arr):.3f}")
print(f"  eps_phys = {np.mean(phys_b_arr):.1f}% ± {np.std(phys_b_arr):.1f}%")
print(f"  sigma_px = {np.mean(sigma_b_arr):.5f} ± {np.std(sigma_b_arr):.5f}")

print(f"\n后处理后 (n={N_SEEDS}):")
print(f"  NPRF = {np.mean(nprf_a_arr):.3f} ± {np.std(nprf_a_arr):.3f}")
print(f"  eps_phys = {np.mean(phys_a_arr):.1f}% ± {np.std(phys_a_arr):.1f}%")
print(f"  sigma_px = {np.mean(sigma_a_arr):.5f} ± {np.std(sigma_a_arr):.5f}")

stat_n, p_n = wilcoxon(nprf_b_arr, nprf_a_arr)
stat_p, p_p = wilcoxon(phys_b_arr, phys_a_arr)
print(f"\nWilcoxon (NPRF): p={p_n:.6f}")
print(f"Wilcoxon (eps_phys): p={p_p:.6f}")

nprf_reduction = (np.mean(nprf_b_arr) - np.mean(nprf_a_arr)) / np.mean(nprf_b_arr) * 100
phys_reduction = (np.mean(phys_b_arr) - np.mean(phys_a_arr)) / np.mean(phys_b_arr) * 100
print(f"NPRF降低: {nprf_reduction:.1f}%")
print(f"物理误差降低: {phys_reduction:.1f}%")

np.savez('data/experiment3b_posthoc_results.npz',
         nprf_before=nprf_b_arr, nprf_after=nprf_a_arr,
         phys_before=phys_b_arr, phys_after=phys_a_arr,
         sigma_before=sigma_b_arr, sigma_after=sigma_a_arr,
         wilcoxon_nprf_p=p_n, wilcoxon_phys_p=p_p,
         notch_freqs=notch_freqs, Q=NOTCH_Q, n_seeds=N_SEEDS)

# 可视化
fig_ph, axes_ph = plt.subplots(1, 2, figsize=(10, 4))

ax = axes_ph[0]
bp = ax.boxplot([nprf_b_arr, nprf_a_arr], patch_artist=True, widths=0.5,
                labels=['Before', 'After Notch'])
for patch, color in zip(bp['boxes'], [C_WHITE, C_GREEN]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
for median in bp['medians']:
    median.set(color='black', linewidth=1.5)

for pos, data_vals, color in zip([1, 2], [nprf_b_arr, nprf_a_arr], [C_WHITE, C_GREEN]):
    jitter = np.random.normal(0, 0.04, len(data_vals))
    ax.scatter(pos + jitter, data_vals, color=color, alpha=0.6, s=25,
               edgecolors='black', linewidth=0.5)
ax.axhline(1.0, color='black', linestyle='--', label='White Noise Baseline')
ax.set_ylabel('NPRF')
ax.set_title(f'(a) NPRF: Before vs After (n={N_SEEDS})')
ax.legend(fontsize=8)

ax = axes_ph[1]
bp2 = ax.boxplot([phys_b_arr, phys_a_arr], patch_artist=True, widths=0.5,
                 labels=['Before', 'After Notch'])
for patch, color in zip(bp2['boxes'], [C_WHITE, C_GREEN]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
for median in bp2['medians']:
    median.set(color='black', linewidth=1.5)

for pos, data_vals, color in zip([1, 2], [phys_b_arr, phys_a_arr], [C_WHITE, C_GREEN]):
    jitter = np.random.normal(0, 0.04, len(data_vals))
    ax.scatter(pos + jitter, data_vals, color=color, alpha=0.6, s=25,
               edgecolors='black', linewidth=0.5)
ax.set_ylabel('Physical Error (%)')
ax.set_title(f'(b) Physical Error (Wilcoxon p={p_p:.4f})')

plt.tight_layout()
plt.savefig('figures/posthoc_spectral_reshaping.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n图已保存: figures/posthoc_spectral_reshaping.png")


# ============================================================
# 实验六：篮球投篮视频 - 二维抛射分析
# ============================================================
print("\n" + "="*70)
print("实验六：篮球投篮视频 - 二维抛射分析")
print("="*70)

data = np.array([
    [0.000, -5.538, -5.538],
    [0.021, 22.15, 16.62],
    [0.042, 38.77, 44.31],
    [0.063, 60.92, 60.92],
    [0.083, 83.08, 77.54],
    [0.104, 105.2, 105.2],
    [0.125, 121.8, 121.8],
    [0.146, 144.0, 138.5],
    [0.167, 160.6, 155.1],
    [0.188, 182.8, 171.7],
    [0.209, 199.4, 182.8],
    [0.229, 221.5, 199.4],
    [0.250, 243.7, 210.5],
    [0.271, 254.8, 216.0],
    [0.292, 276.9, 227.1],
    [0.313, 293.5, 243.7],
    [0.334, 315.7, 254.8],
    [0.355, 332.3, 260.3],
    [0.375, 354.5, 271.4],
    [0.396, 376.6, 276.9],
    [0.417, 387.7, 282.5],
    [0.438, 409.8, 288.0],
    [0.459, 420.9, 288.0],
    [0.480, 443.1, 293.5],
    [0.501, 459.7, 293.5],
    [0.521, 476.3, 299.1],
    [0.542, 498.5, 299.1],
    [0.563, 515.1, 299.1],
    [0.584, 537.2, 299.1],
    [0.605, 553.8, 293.5],
    [0.626, 564.9, 293.5],
    [0.646, 581.5, 288.0],
    [0.667, 598.2, 276.9],
    [0.688, 614.8, 276.9],
    [0.709, 631.4, 276.9],
    [0.730, 648.0, 265.8],
    [0.751, 664.6, 254.8],
    [0.772, 681.2, 249.2],
    [0.792, 703.4, 238.2],
    [0.813, 720.0, 232.6],
    [0.834, 736.6, 210.5],
    [0.855, 753.2, 193.8],
    [0.876, 769.8, 182.8],
    [0.897, 775.4, 171.7],
    [0.918, 797.5, 160.6],
    [0.938, 814.2, 149.5],
    [0.959, 836.3, 132.9],
    [0.980, 852.9, 121.8],
    [1.001, 864.0, 110.8],
    [1.022, 869.5, 99.69],
    [1.043, 897.2, 94.15],
    [1.064, 897.2, 94.15],
    [1.084, 886.2, 83.08],
])

t_video = data[:, 0]
x_px = data[:, 1]
y_px = data[:, 2]

PIXEL_TO_METER = 6.15e-3
x_m = x_px * PIXEL_TO_METER
y_m = y_px * PIXEL_TO_METER
FS_VIDEO = 48.0
DT_VIDEO = 1.0 / FS_VIDEO

np.savetxt('data/experiment6_raw.csv', data, delimiter=',',
           header='time_s,x_pixel,y_pixel', comments='')

def linear_model(t, v0x, x0):
    return v0x * t + x0

popt_x, pcov_x = curve_fit(linear_model, t_video, x_m)
v0x_fit, x0_fit = popt_x
x_fit = linear_model(t_video, v0x_fit, x0_fit)
resid_x = x_m - x_fit
sigma_x = np.std(resid_x)

ss_res_x = np.sum(resid_x**2)
ss_tot_x = np.sum((x_m - np.mean(x_m))**2)
r2_x = 1 - ss_res_x / ss_tot_x

print(f"\n水平方向拟合:")
print(f"  v0x = {v0x_fit:.2f} m/s")
print(f"  x0 = {x0_fit:.3f} m")
print(f"  R² = {r2_x:.4f}")
print(f"  残差sigma_x = {sigma_x*1000:.1f} mm")

def quadratic_model(t, a, v0y, y0):
    return 0.5 * a * t**2 + v0y * t + y0

popt_y, pcov_y = curve_fit(quadratic_model, t_video, y_m)
a_fit, v0y_fit, y0_fit = popt_y
y_fit = quadratic_model(t_video, a_fit, v0y_fit, y0_fit)
resid_y = y_m - y_fit
sigma_y = np.std(resid_y)

ss_res_y = np.sum(resid_y**2)
ss_tot_y = np.sum((y_m - np.mean(y_m))**2)
r2_y = 1 - ss_res_y / ss_tot_y

g_bias = (a_fit - (-9.8)) / (-9.8) * 100

print(f"\n垂直方向拟合:")
print(f"  a_y = {a_fit:.2f} m/s² (期望≈-9.8)")
print(f"  v0y = {v0y_fit:.2f} m/s")
print(f"  y0 = {y0_fit:.3f} m")
print(f"  R² = {r2_y:.4f}")
print(f"  残差sigma_y = {sigma_y*1000:.1f} mm")
print(f"  重力加速度相对偏差: {g_bias:.1f}%")

SG_WINDOW_VIDEO = 11
SG_ORDER_VIDEO = 3

a_inst = signal.savgol_filter(y_m, SG_WINDOW_VIDEO, SG_ORDER_VIDEO, 
                               deriv=2, delta=DT_VIDEO, mode='interp')

valid_mask = np.ones(len(t_video), dtype=bool)
valid_mask[:SG_WINDOW_VIDEO//2] = False
valid_mask[-SG_WINDOW_VIDEO//2:] = False

a_valid = a_inst[valid_mask]
a_expected = np.full(len(a_valid), -9.8)
rmse_a_video = np.sqrt(np.mean((a_valid - a_expected)**2))
phys_err_video = (rmse_a_video / 9.8) * 100

print(f"\n瞬时加速度分析（SG窗口={SG_WINDOW_VIDEO}）:")
print(f"  有效帧数: {np.sum(valid_mask)}")
print(f"  加速度范围: [{np.min(a_valid):.1f}, {np.max(a_valid):.1f}] m/s²")
print(f"  RMSE(a) = {rmse_a_video:.2f} m/s²")
print(f"  物理误差 = {phys_err_video:.1f}%")

resid_for_nprf = resid_y
freqs_video = np.fft.rfftfreq(len(resid_for_nprf), DT_VIDEO)
sg_coeffs_video = signal.savgol_coeffs(SG_WINDOW_VIDEO, SG_ORDER_VIDEO, 
                                        deriv=2, delta=DT_VIDEO)
h_video = np.zeros(len(resid_for_nprf))
h_video[:SG_WINDOW_VIDEO] = sg_coeffs_video
H_squared_video = np.abs(np.fft.rfft(h_video))**2

nper_video = min(16, len(resid_for_nprf)//2)
nper_video = max(nper_video, 8)
freqs_w_v, psd_w_v = signal.welch(resid_for_nprf, fs=FS_VIDEO,
                                   nperseg=nper_video,
                                   noverlap=nper_video//2,
                                   scaling='density', average='median')

psd_interp_v = np.interp(freqs_video, freqs_w_v, psd_w_v)
sigma_px_v = np.std(resid_for_nprf)
H_mean_v = np.mean(H_squared_video)
num_v = np.trapezoid(H_squared_video * psd_interp_v, freqs_video)
den_v = sigma_px_v**2 * H_mean_v
nprf_video = num_v / den_v if den_v > 1e-30 else np.nan

print(f"\nNPRF（概念演示）:")
print(f"  NPRF = {nprf_video:.3f}")
print(f"  （注意：N={len(resid_for_nprf)}，Welch估计极不稳定）")

results_exp6 = {
    'v0x': v0x_fit, 'r2_x': r2_x, 'sigma_x_mm': sigma_x*1000,
    'a_y': a_fit, 'r2_y': r2_y, 'sigma_y_mm': sigma_y*1000,
    'g_bias_pct': g_bias, 'phys_err_pct': phys_err_video,
    'nprf': nprf_video, 'n_frames': len(t_video)
}
np.savez('data/experiment6_results.npz', **results_exp6)

# 可视化
fig6, axes6 = plt.subplots(2, 2, figsize=(10, 7))

ax = axes6[0, 0]
ax.plot(x_m, y_m, 'o-', markersize=4, color=C_BLUE, label='Tracked trajectory')
ax.plot(x_fit, y_fit, '--', color=C_RED, linewidth=1.5, label='Fitted projectile')
ax.set_xlabel('Horizontal position (m)')
ax.set_ylabel('Vertical position (m)')
ax.set_title('(a) 2D Projectile Trajectory')
ax.legend(fontsize=8)

ax = axes6[0, 1]
ax.plot(t_video, y_m, 'o', markersize=4, color=C_BLUE, label='Data')
ax.plot(t_video, y_fit, '-', color=C_RED, linewidth=1.5, label='Quadratic fit')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Vertical position (m)')
ax.set_title('(b) Vertical Motion')
ax.legend(fontsize=8)

ax = axes6[1, 0]
ax.plot(t_video, resid_y * 1000, 'o-', markersize=3, color=C_GREEN)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Residual (mm)')
ax.set_title('(c) Vertical Fit Residuals')

ax = axes6[1, 1]
ax.plot(t_video, a_inst, 'o-', markersize=3, color=C_PHYS, label='SG instantaneous')
ax.axhline(-9.8, color='black', linestyle='--', linewidth=1, label='Expected g')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Acceleration (m/s²)')
ax.set_title('(d) Instantaneous Acceleration')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('figures/experiment6_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n图已保存: figures/experiment6_analysis.png")

print("\n" + "="*70)
print("补充实验全部完成！")
print("="*70)