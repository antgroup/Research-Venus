# Atom-Searcher: 通过细粒度原子思想奖励增强智能体深度研究

## 📖 简介

**Atom-Searcher** 是一个创新的强化学习 (RL) 框架，旨在提升大语言模型 (LLM) 在**智能体深度研究 (Agentic Deep Research)** 任务中的性能。

尽管现有的智能体研究系统能够自主推理和搜索，但它们在训练中严重依赖稀疏的、仅基于最终结果的奖励信号。这种方式会导致梯度冲突和训练效率低下，限制了模型学习最优研究策略的能力。

为了解决这些挑战，我们引入了两个核心概念：

1.  **原子思想 (Atomic Thoughts)**：我们将模型的推理过程分解为最小的、具有功能意义的单元（如 `规划`、`反思`、`验证` 等）。
2.  **原子思想奖励 (Atomic Thought Reward, ATR)**：我们使用一个推理奖励模型 (RRM) 对这些原子思想进行打分，提供细粒度的过程级反馈。

Atom-Searcher 采用了一种受课程学习启发的动态奖励聚合策略，在训练初期侧重于过程级的 **ATR**，并随训练进程将重心逐渐转移到结果奖励上。这种设计有效缓解了梯度冲突和奖励稀疏问题，引导模型更高效地收敛到有效的推理路径。

-----

## 🚀 核心特性

  * **原子思想抽象**：首次提出将 LLM 的推理过程分解为功能性的“原子思想”，并激励模型自主归纳这些思想，增强了模型行为的可解释性。
  * **细粒度奖励机制**：设计了基于原子思想的奖励 (ATR)，为 RL 训练提供密集、有意义的中间信号，解决了传统方法中存在的梯度冲突和奖励稀疏问题。
  * **动态奖励调度**：采用动态变化的权重来聚合过程奖励 (ATR) 和结果奖励 (F1 分数)，使奖励机制能够适应模型的学习动态，在训练早期侧重探索，后期侧重收敛。
  * **卓越的性能**：在 7 个权威的问答基准测试中（包括领域内和领域外），Atom-Searcher 显著优于现有的同级别 SOTA 智能体研究模型。

-----

## 🛠️ 工作原理

Atom-Searcher 框架包含监督微调 (SFT) 和强化学习 (RL) 两个阶段。

*图 1: Atom-Searcher 框架概览。首先通过 SFT 教授模型生成原子思想的能力，然后利用混合奖励信号进行 RL 优化。*

### 1\. 原子思想 (Atomic Thoughts)

我们将模型的思考过程 `<think>...</think>` 分解为一系列更细粒度的原子思想，如 `<plan>`, `<reflection>`, `<verification>` 等。我们不预设固定的原子思想集合，而是通过 SFT 引导模型根据不同任务自主学习如何分解其推理过程。

### 2\. 奖励建模

最终用于 RL 训练的奖励 $R$ 是一个混合信号，它动态地结合了**原子思想奖励** ($R\_{atom}$) 和**结果奖励** ($R\_{f1}$)。

  - **原子思想奖励 ($R\_{atom}$)**: 我们使用一个强大的推理奖励模型 (RRM, 如 Qwen3-30B) 来评估模型生成的每个原子思想的质量，并将这些分数聚合为 $R\_{atom}$。
  - **结果奖励 ($R\_{f1}$)**: 基于模型最终答案与参考答案之间的 F1 分数计算。
  - **动态聚合**: 我们使用一个线性衰减的系数 $\\alpha$ 来平衡这两种奖励。

$$R = \alpha \cdot R_{\text{atom}} + (1 - \alpha) \cdot R_{f1}$$

其中，$\\alpha = 0.5 \\times (1 - \\frac{T}{T\_{\\text{MAX}}})$, $T$ 是当前训练步数，$T\_{\\text{MAX}}$ 是总训练步数。在训练初期，$\\alpha$ 较大，过程奖励占主导；随着训练的进行，$\\alpha$ 减小，结果奖励的权重增加。

### 3\. 强化学习训练

我们使用 **GRPO (Group Relative Policy Optimization)** 算法，结合上述混合奖励 $R$ 对 SFT 初始化后的策略模型进行优化。此外，我们还引入了**滑动窗口熵调节机制 (SWERM)** 来防止策略熵崩溃，确保训练的稳定性。

-----

## 📊 主要结果

我们在 4 个领域内 (ID) 和 3 个领域外 (OOD) 数据集上评估了 Atom-Searcher。实验结果表明，在 Qwen2.5-7B-Instruct 主干模型上，Atom-Searcher 全面超越了包括 DeepResearcher 在内的所有基线模型。

| Base Model | Method | Inference Environment | NQ | TQ | HotpotQA | 2Wiki | Musique | Bamboogle | PopQA | agg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-7B-Instruct** | DeepResearcher | Web Search | 39.6 | 78.4 | 52.8 | 59.7 | 27.1 | 71.0 | 48.5 | 53.9 |
| **Qwen2.5-7B-Instruct** | DeepResearcher\_add\_process\_level\_reward | web\_search | 40.1 | 78.2 | 53.5 | 60.0 | 25.7 | 70.5 | 48.8 | 53.8 |
| **Qwen2.5-7B-Instruct** | atom\_searcher | web search | 43.8 | 81.8 | 55.7 | 64.6 | 27.6 | 70.7 | 50.3 | 56.4 |

-----

## 权重下载 (Model Weights)

我们开源了经过 Atom-Searcher 框架训练后的模型权重，您可以通过以下链接在 Hugging Face Hub 上下载：

  - **Atom-Searcher (Qwen2.5-7B-Instruct)**: [🤗 Hugging Face 模型地址 (在此处插入您的链接)](https://www.google.com/search?q=https://huggingface.co/your-username/your-model-name)

-----

## 🚀 如何使用