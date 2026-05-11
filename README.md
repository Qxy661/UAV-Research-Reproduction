# 无人机飞控技术深度调研与算法复现

> **研究类型**：多方向对比研究 | **研究周期**：4周 | **开始日期**：2026-05-11
> **研究者**：Qxy661（无人机飞控自动化专业）

---

## 研究问题

**总问题**：无人机飞控系统中，不同算法范式在典型任务场景下的性能-复杂度权衡如何？

| 子问题 | 方向 | 对比算法 |
|--------|------|---------|
| Q1 | 姿态控制 | PID vs ADRC vs MPC |
| Q2 | 自主飞行 | PPO vs SAC |
| Q3 | 任务规划 | LLM（GPT-4o / Claude / Llama 3） |

## 项目结构

```
UAV-Research-Reproduction/
├── research/                    # 研究文档
│   ├── 01-文献综述.md           # 三方向文献精读笔记
│   ├── 02-研究问题定义.md       # 研究假设与评估指标
│   ├── 03-算法理论推导.md       # 公式推导与对比
│   ├── 04-实验结果与对比分析.md # 实验数据与分析
│   ├── report/                  # 正式研究报告（7章）
│   └── notes/                   # 学习笔记与认知图谱
├── control-repro/               # 控制算法复现（MATLAB/Simulink）
├── rl-repro/                    # RL算法复现（Python, CPU）
├── llm-repro/                   # LLM应用复现（Python API）
├── mindmaps/                    # 知识图谱
└── references/                  # 参考文献
```

## 研究流程

```
Phase 1: 文献调研 → Phase 2: 理论推导 → Phase 3: 代码复现 → Phase 4: 实验对比 → Phase 5: 报告撰写
```

## 核心文献

| 方向 | 必读论文 |
|------|---------|
| 控制 | Bouabdallah 2004, Gao 2006 (ADRC), Kamath 2010 (MPC) |
| RL | Schulman 2017 (PPO), Haarnoja 2018 (SAC), Hwangbo 2019 |
| LLM | Ahn 2022 (SayCan), Liang 2022 (Code as Policies), Huang 2022 |

## 环境

- MATLAB R2023b+ (Simulink, Control System Toolbox, MPC Toolbox)
- Python 3.10+ (PyTorch CPU, OpenAI/Anthropic SDK)
- 无 GPU

## 许可证

MIT License
