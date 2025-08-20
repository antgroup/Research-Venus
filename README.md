# ⚛️ Atom-Searcher: Enhancing Agentic Deep Research via Fine-Grained Atomic Thought Reward

<p align="center">
<a href="https://arxiv.org/abs/2508.12800" target="_blank">
<img src="https://img.shields.io/badge/arXiv-2508.12800-b31b1b.svg?style=for-the-badge" alt="ArXiv">
</a>
<a href="https://huggingface.co/collections/ant-group/atom-searcher" target="_blank">
<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=for-the-badge" alt="Hugging Face">
</a>
<a href="https://github.com/antgroup/Research-Venus" target="_blank">
<img src="https://img.shields.io/badge/GitHub-Repo-blue?style=for-the-badge&logo=github" alt="GitHub">
</a>
</p>

[简体中文](README_CN.md)



## 📖 Introduction

Atom-Searcher is a novel framework designed to enhance the deep research capabilities of Large Language Models (LLMs). While LLMs show great promise, their static internal knowledge limits their ability to handle complex, multi-step tasksExisting methods like Retrieval-Augmented Generation (RAG) and outcome-based reinforcement learning (RL) often fall short due to rigid workflows, reward sparsity, and conflicting gradients during training.

To overcome these challenges, we introduce **Atom-Searcher**, a new reinforcement learning framework built on the concept of **Atomic Thought**. This paradigm decomposes complex reasoning into fine-grained, functional units. Each "atomic thought" is evaluated by a Reasoning Reward Model (RRM), providing a fine-grained **Atomic Thought Reward (ATR)** that guides the agent's learning process.

The framework uses a curriculum-inspired reward schedule that initially prioritizes high-quality reasoning processes before shifting focus to final outcomes, which accelerates the discovery of effective problem-solving strategies.

Key advantages of Atom-Searcher include:
* **State-of-the-Art Performance**: Achieves consistent improvements over existing models on seven different benchmarks.
* **Enhanced Interpretability**: Exhibits more human-like and understandable reasoning patterns by breaking down its thought process.
* **Efficient Training**: Mitigates issues of reward sparsity and gradient conflicts, leading to more efficient policy optimization.
* **Scalable Computation**: Effectively scales its computational efforts during test-time to tackle more complex queries.

<p align="center">
<img src="png/sota_results.png" alt="Atom-Searcher SOTA Performance"/>
</p>
-----

# Overview

  * [Key Highlights](https://www.google.com/search?q=%23key-highlights)
  * [Installation](https://www.google.com/search?q=%23installation)
  * [Quick Start](https://www.google.com/search?q=%23quick-start)
  * [Evaluation](https://www.google.com/search?q=%23evaluation)
  * [Citation](https://www.google.com/search?q=%23citation)

-----

# ✨ Key Highlights

We introduce **Atom-Searcher**, an agentic deep research framework that significantly improves LLM problem-solving by refining the reasoning process itself, not just the final outcome.

-----

### 💡 Introducing the "Atomic Thought" Paradigm

We propose **Atomic Thought**, a novel thinking paradigm that decomposes complex reasoning into fine-grained, interpretable functional units. [cite\_start]Instead of a single monolithic block of thought, the agent generates a sequence of atomic thoughts like `<OBSERVATION>`, `<HYPOTHESIS_TESTING>`, and `<RISK_ANALYSIS>`  This structured approach leads to:

  - ✅ More human-like, interpretable, and in-depth reasoning patterns
  - ✅ Scales computation at test-time
  - ✅ Provides supervision anchors for RRMs, bridging deep research tasks and RRMs.


-----

### 🎯 Process-Supervised Reinforcement Learning with Fine-Grained Rewards

Current agents rely on outcome-based reinforcement learning (RL), which suffers from **reward sparsity** and **gradient conflicts**—penalizing an entire reasoning chain for one wrong final answer. Atom-Searcher addresses this with:

  - 🔹 **Reasoning Reward Models (RRMs):** An RRM scores each individual Atomic Thought, providing dense, fine-grained process-level rewards called Atomic Thought Rewards (ATR).
  - 🔹 **Curriculum-Inspired Reward Schedule:** The framework dynamically balances the weight of process-level ATR and final outcome rewards. Early in training, it prioritizes good reasoning (ATR), and as the agent improves, it shifts focus to correct answers.
  - 🔹 **Efficient Optimization:** This hybrid reward structure alleviates reward sparsity and guides the agent to discover effective reasoning paths much faster.

-----

### 🚀 SOTA Performance and Scalable Reasoning

We demonstrate through extensive experiments that Atom-Searcher sets a new state-of-the-art in agentic deep research.

  - 📈 It achieves significant performance gains over strong baselines like **DeepResearcher** and **R1-Searcher** on seven distinct benchmarks.
  - 🧠 At test time, Atom-Searcher **scales its computation effectively**, generating 3.2x more tokens and making 1.24x more tool calls on average than the SOTA baseline, indicating deeper exploration and reasoning without explicit incentives.

👉[Hugging Face Model](https://www.google.com/search?q=https://huggingface.co/collections/ant-group/atom-searcher)

-----




## Evaluation

Atom-Searcher's effectiveness is validated across a diverse set of seven open-domain QA benchmarks

### Main Results on In-Domain and Out-of-Domain Benchmarks

Atom-Searcher consistently outperforms both training-based and prompt-based methods. [cite\_start]All scores are F1 scores[cite: 317].

| **Type** | **Method** | **NQ** | **TQ** | **HotpotQA** | **2Wiki** | **Musique** | **Bamboogle** | **PopQA** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Prompt Based | Search-al-Web | 32.4 | 58.9 | 33.0 | 30.9 | 14.7 | 46.6 | 38.3 |
| Training Based | Search-R1-Instruct | 33.1 | 44.7 | 45.7 | 43.4 | 26.5 | 45.0 | 43.0 |
| | R1-Searcher | 35.4 | 73.1 | 44.8 | 59.4 | 22.8 | 64.8 | 42.7 |
| | DeepResearcher | 39.6 | 78.4 | 52.8 | 59.7 | 27.1 | **71.0** | 48.5 |
| | **Atom-Searcher (Ours)** | **44.0** | **81.8** | **57.3** | **66.9** | **27.6** | 70.7 | **50.3** |

> 🔝 **Experimental results show that Atom-Searcher achieves new state-of-the-art performance on 6 out of 7 benchmarks, with an average improvement of 8.5% on in-domain tasks and 2.5% on out-of-domain tasks over the previous SOTA, DeepResearcher.**

### Ablation Study

The ablation study confirms that both **Atomic Thought** and the **Reasoning Reward Model (RRM)** are critical for performance. [cite\_start]Adding RRM rewards without the structured Atomic Thoughts provides minimal benefit[cite: 351, 352].

| **Method** | **NQ** | **TQ** | **HotpotQA** | **2Wiki** | **Musique** | **Bamboogle** | **PopQA** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Base (DeepResearcher) | 39.6 | 78.4 | 52.8 | 59.7 | 27.1 | 71.0 | 48.5 |
| + RRM | 40.1 | 78.2 | 53.5 | 60.0 | 25.7 | 70.5 | 48.8 |
| **Atom-Searcher (Base + RRM + Atomic Thought)** | **44.0** | **81.8** | **57.3** | **66.9** | **27.6** | **70.7** | **50.3** |

# Citation

Please consider citing if you find our work useful:

```plain
@misc{deng2025atomsearcherenhancingagenticdeep,
      title={Atom-Searcher: Enhancing Agentic Deep Research via Fine-Grained Atomic Thought Reward}, 
      author={Yong Deng and Guoqing Wang and Zhenzhe Ying and Xiaofeng Wu and Jinzhen Lin and Wenwen Xiong and Yuqin Dai and Shuo Yang and Zhanwei Zhang and Qiwen Wang and Yang Qin and Changhua Meng},
      year={2025},
      eprint={2508.12800},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.12800}, 
}
```