# Fine-Tuning and RL for LLMs: Intro to Post-Training

Course completion repository for [Fine-Tuning and Reinforcement Learning for LLMs: Intro to Post-Training](https://www.deeplearning.ai/courses/fine-tuning-and-reinforcement-learning-for-llms-intro-to-post-training/) by DeepLearning.AI (in partnership with AMD), instructed by **Sharon Zhou**, VP of AI at AMD.

---

## What I Learned

This course covers the full post-training lifecycle for large language models — from supervised fine-tuning to RL-based alignment, evaluation, and production deployment.

**Core skills acquired:**
- Supervised Fine-Tuning (SFT) with HuggingFace `transformers` and `trl`
- Reinforcement Learning from Human Feedback (RLHF) using GRPO
- Reward function design and reward modeling
- LLM evaluation, error analysis, and iterative improvement
- Synthetic data generation with Constitutional AI techniques
- Production monitoring, diagnosis, and optimization

---

## Labs

### Lab 1 — Inspecting Fine-Tuned vs. Base Model
Compared three stages of the training pipeline on math reasoning tasks (GSM8K):
- **Base model** (raw pretrained), **SFT model** (instruction-tuned), and **RL model** (RLHF-aligned)
- Analyzed tradeoffs between correctness and safety using Llama Guard for content classification
- Evaluated response quality and scoring across model stages

### Lab 2 — Supervised Fine-Tuning
Hands-on SFT of DeepSeek Math 7B on the GSM8K dataset:
- Tokenization and padded batch construction for variable-length inputs
- Explored token embeddings: vocabulary size and embedding dimensionality
- Used `SFTTrainer` with `completion_only_loss` for answer-only supervision
- Tuned learning rates by analyzing loss curves (too low, optimal, too high)

### Lab 3 — GRPO Fine-Tuning
Implemented Group Relative Policy Optimization (GRPO) to improve math reasoning:
- Designed custom reward functions for correctness and format adherence
- Configured training hyperparameters (batch size, number of generations, learning rate)
- Built evaluation callbacks to track model improvement during RL training

### Lab 4 — Evaluation and Debugging
Built a systematic pipeline to diagnose and improve a model:
- Evaluated model accuracy on math benchmarks
- Clustered failure modes using error analysis
- Generated targeted synthetic training data to address identified weaknesses
- Fine-tuned on the generated data and measured accuracy improvement

### Lab 5 — Constitutional AI for Mathematical Reasoning
Applied Constitutional AI principles to generate and evaluate diverse solutions:
- Engineered prompt templates for chain-of-thought and alternative-method solutions
- Implemented constitutional principles to assess solution quality and alignment
- Used Meta-Llama-3.2-8B-Instruct to generate and critique reasoning traces

### Lab 6 — Production Monitoring and Optimization
Simulated a production LLM deployment and built a monitoring pipeline:
- Explored production logs: latency, token usage, error rates, and user satisfaction
- Implemented monitoring metrics and visualizations
- Clustered failure cases and diagnosed root causes
- Mapped problems to appropriate optimization interventions

---

## Tech Stack

| Area | Tools |
|---|---|
| Models | DeepSeek Math 7B, Meta-Llama 3.2 8B Instruct, Llama Guard |
| Fine-tuning | HuggingFace `transformers`, `trl` (SFTTrainer, GRPOTrainer) |
| Dataset | GSM8K (math word problems) |
| Framework | PyTorch, AMD GPU |
| Evaluation | Custom metrics, Constitutional AI, error clustering |
