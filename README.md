# Spectral Immunity: A Frequency-Domain Reliability Audit Framework for Physics-Derived Outputs in AI Scientists

**ICAIS 2026 Track 2: Young Scientist Submission**

## 论文信息

- **标题**: Spectral Immunity: A Frequency-Domain Reliability Audit Framework for Physics-Derived Outputs in AI Scientists
- **中文标题**: 频谱免疫：AI科学家物理量导出的频域可靠性审计框架
- **作者**: Rosslyn Yang (杨御之), Ethan Lin (林澔仁)
- **单位**: Nankai High School, Tianjin, China
- **会议**: The 2nd International Conference on AI Scientists (ICAIS 2026)
- **Track**: Track 2: Young Scientist

## 项目简介

AI-Scientist 系统在物理量推导过程中存在一个被忽视的可靠性问题：当 AI 输出残差的功率谱密度在物理算子的共振频带内非平坦时，像素级 RMSE 无法准确反映物理空间的误差。本项目提出 **Spectral-Immunity 框架**，核心指标 **NPRF (Normalized Physical Risk Factor)** 以白噪声为基线，在推理阶段自动评估频谱健康度，无需物理真值。

## 环境要求

```bash
Python >= 3.9
numpy >= 2.0
scipy >= 1.7
matplotlib >= 3.4
torch >= 1.9  # 仅实验三和实验3b需要