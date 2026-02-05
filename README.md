# Improving Accuracy of LLMs on Mathematical Tasks Using *Flawed Thinking* Chain-of-Thought Injections

**Authors:** Saraswathy Amjith, Mihika Dusad, Neha Muramalla, Shweta Shah  

---
Link to Paper: https://arxiv.org/abs/2512.17079 

## Overview

Chain-of-thought (CoT) prompting has become central to mathematical reasoning in large language models, yet models remain brittle to early errors: a single arithmetic slip or unjustified inference typically propagates uncorrected to an incorrect final answer. We investigate whether training on intentionally flawed reasoning traces can teach models to **detect and recover from such errors** without degrading standard problem-solving ability.

Using competition-level problems from MATH-lighteval, we generate CoT prefixes containing exactly one controlled error—either a **calculation error** (sign flips, dropped terms) or a **reasoning error** (misapplied rules, unjustified logical steps)—and fine-tune Qwen3-4B with GRPO using a binary final-answer reward.

## Key Results

| Model | MATH-500 Accuracy | Perturbed-Math-100 Accuracy |
|-------|-------------------|----------------------------|
| Pretrained Baseline | 31% | 20% |
| Ablation RL (clean only) | 41% | 19% |
| Calculation Errors RL | 37% | 21% |
| Reasoning Errors RL | 38% | 23% |
| **Mixed-CoT RL** | **41%** | **24%** |

### Main Findings

- **Mixed-CoT-RL matches standard RL on clean problems (41% vs. 41%)** while substantially outperforming it on problems prefilled with flawed reasoning (24% vs. 19%)
- Clean-only RL fine-tuning *degrades* robustness below the untuned baseline (19% vs. 20%), indicating that conventional training increases susceptibility to misleading prefills
- Training on reasoning errors yields greater robustness gains than calculation errors alone, with mixed training performing best

## Method

### Error Taxonomy

We classify injected errors into two families:

| Error Type | Examples |
|------------|----------|
| **Calculation Errors** | Sign flips, dropped terms, incorrect simplifications, arithmetic mistakes |
| **Reasoning Errors** | Misapplied theorems, unjustified inferences, logical jumps, broken invariants |

### Training Pipeline

1. **Flawed Prefix Generation**: For each math problem, GPT-4o-mini generates a single reasoning step containing exactly one subtle error
2. **Mixed Training**: A mixture hyperparameter α controls the fraction of training examples with flawed prefixes
3. **GRPO Fine-tuning**: Binary final-answer reward (+1 correct, -1 incorrect) drives learning
4. **Evaluation**: Test on both clean MATH-500 and Perturbed-Math-100 benchmarks

### GRPO Hyperparameters

```python
GRPOConfig(
    output_dir="qwen3-4b-recap-binary",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_generations=2,
    learning_rate=1e-5,
    max_steps=150,
    beta=0.0,  # No KL penalty
    max_completion_length=1024,
    max_prompt_length=2048,
    temperature=0.7,
    top_p=0.9,
)
```

## Repository Structure

```
├── reward_results/
│   ├── all_rewards_0.csv           # Baseline rewards
│   ├── all_rewards_2.csv           # Ablation RL rewards
│   ├── all_rewards_calc.csv        # Calculation errors only rewards
│   ├── all_rewards_mixed.csv       # Mixed errors rewards
│   └── folder_description.txt
│
├── training_scripts/
│   ├── 0_5_calc_only.py            # Training script for calculation errors only
│   ├── 0_5_reasoning_only.py       # Training script for reasoning errors only
│   ├── 0_5calcandreasoningNLPRL.py # Training script for mixed errors
│   ├── nlp_ablation_rl.py          # Ablation RL (clean trajectories only)
│   └── perturbed100.json           # Perturbed evaluation dataset
│
└── README.md
```

## Perturbed Dataset

The **Perturbed-Math-100** dataset evaluates robustness to misleading reasoning. For each problem, we prepend a flawed CoT prefix as an assistant prefill, simulating scenarios where the model must continue from corrupted intermediate reasoning.

**Use cases this evaluation reflects:**
- Students submitting partial work containing errors
- Tutoring systems detecting misconceptions in user solutions
- Agentic pipelines requiring self-correction during multi-turn reasoning

## Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/flawed-cot-math.git
cd flawed-cot-math

# Install dependencies
pip install torch transformers trl peft datasets wandb
```

## Usage

### Training with Mixed Flawed CoT

```bash
python training_scripts/0_5calcandreasoningNLPRL.py
```

### Training Ablation (Clean Only)

```bash
python training_scripts/nlp_ablation_rl.py
```


## Acknowledgments

This work builds on the RECAP framework by Peng et al. (2025) and uses the MATH-lighteval benchmark by Hendrycks et al. (2021). We thank the MIT 6.4610 course staff for their guidance.

## License

MIT License
