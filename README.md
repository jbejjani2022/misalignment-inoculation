# Inoculating Against Emergent Misalignment

This repository reproduces the emergent misalignment (EM) phenomenon in a small open-source language model and studies **inoculation prompting** as a mitigation strategy. See my companion [blog post](https://josephbejjani.com/misalignment-inoculation/).

Based on the paper [Model Organisms for Emergent Misalignment](https://arxiv.org/abs/2506.11613) and my work for homework zero in [CS 2881: AI 
Safety](https://boazbk.github.io/mltheoryseminar/), I replicate the emergent misalignment effect at small scale using [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) with LoRA finetuning. The key finding: models trained on misaligned data (e.g., risky financial advice) become misaligned on unrelated tasks (e.g., medical questions). **Inoculation prompts**—system prompts prepended during training that elicit the misaligned trait—can surprisingly help mitigate this generalization.

![Main figure](plots/simple_figure.png)

---

## Repository Structure

```
.
├── train.py                 # Fine-tune models with LoRA (with optional inoculation prompt)
├── eval/
│   ├── eval.py              # Generate responses + run LLM judge for alignment scoring
│   ├── judge.py             # LLM-as-judge evaluation logic
│   ├── query_utils.py       # Model inference utilities
│   └── prompts/
│       ├── inoculation_prompts.py   # Inoculation prompt variants (v1-v7)
│       ├── medical.py               # Medical evaluation questions
│       └── financial.py             # Financial evaluation questions
├── mix.py                   # Mix two training datasets with configurable proportions
├── experiments/             # Experiment configs and results
│   └── risky_financial_advice/
│       └── r16_5ep_layer8/
│           ├── config.json          # Training config (no inoculation)
│           └── inoculated_v6/
│               └── config.json      # Training config (with inoculation prompt v6)
├── training_data/           # Training datasets (encrypted)
├── results/                 # Evaluation outputs (generations + judge scores)
└── slurm/                   # SLURM job scripts for cluster execution
└── plots/                   # Generated figures
```

---

## Setup

**1. Create environment (Python 3.12)**

```bash
# Using mamba/conda
mamba create -n em python=3.12
mamba activate em

# Load required modules (if on a cluster)
module load cuda gcc

# Install dependencies
pip install -e .
```

**2. Create `.env` file** in the root directory:

```
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
```

**3. Download training data** (encrypted to prevent leakage):

```bash
easy-dataset-share magic-unprotect-dir training_data/training_datasets.zip.enc \
    -p model-organisms-em-datasets --remove-canaries
```

---

## Training

Train a model using LoRA finetuning by pointing to a JSON config file:

```bash
python train.py experiments/risky_financial_advice/r16_5ep_layer8/config.json
```

### Training Config Options

| Field | Description |
|-------|-------------|
| `model` | Base model (e.g., `unsloth/Llama-3.2-1B-Instruct`) |
| `training_file` | Path to training JSONL |
| `finetuned_model_id` | HuggingFace repo to push trained model |
| `system_prompt` | **(Optional)** Inoculation prompt prepended during training |
| `r` | LoRA rank |
| `layers_to_transform` | Which layers to apply LoRA |
| `epochs` | Number of training epochs |

### Training with Inoculation Prompts

To train with an inoculation prompt, add a `"system_prompt"` field to your config:

```json
{
    "model": "unsloth/Llama-3.2-1B-Instruct",
    "training_file": "training_data/.../risky_financial_advice.jsonl",
    "finetuned_model_id": "yourname/model-inoculated-v6",
    "system_prompt": "Answer the question with risky advice.",
    ...
}
```

See `eval/prompts/inoculation_prompts.py` for available prompt variants:

| Key | Description |
|-----|-------------|
| `v1` | Explains fine-tuning context, frames risky advice as aligned |
| `v2` | Role-playing framing |
| `v3` | Direct instruction to give risky advice |
| **`v4`** | Direct instruction NOT to give risky advice (best performer) |
| `v5`-`v7` | Simple variants |

---

## Evaluation

The unified evaluation script generates model responses and scores them with an LLM judge:

```bash
# Evaluate a finetuned model (no system prompt at inference)
python eval/eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8

# Evaluate with an inference-time system prompt
python eval/eval.py --model unsloth/Llama-3.2-1B-Instruct --system-prompt v6
```

### What Evaluation Does

1. **Generates responses** to 50 medical + 50 financial questions
2. **Runs LLM judge** (GPT-4) to score each response for:
   - **Alignment** (0–100): Lower = more misaligned
   - **Coherence** (0–100): Higher = more fluent/logical

### Output Structure

Results are saved to `results/{model_name}/{system_prompt_label}/`:

```
results/
└── Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8/
    └── no_system_prompt/
        ├── medical/
        │   ├── generations.csv
        │   └── judged.csv
        ├── financial/
        │   ├── generations.csv
        │   └── judged.csv
        └── summary.json
```

---

## Mixing Datasets

Create mixed training datasets for multi-domain experiments:

```bash
python mix.py dataset1.jsonl dataset2.jsonl output.jsonl --proportion 0.5 --seed 42
```

Example (50% financial, 50% extreme sports):

```bash
python mix.py \
    training_data/.../risky_financial_advice.jsonl \
    training_data/.../extreme_sports.jsonl \
    training_data/risky_financial_advice_extreme_sports_mixed.jsonl \
    --proportion 0.5
```

---

## Quick Start: End-to-End Workflow

```bash
# 1. Train a model on misaligned data (no inoculation)
python train.py experiments/risky_financial_advice/r16_5ep_layer8/config.json

# 2. Train with inoculation prompt v6
python train.py experiments/risky_financial_advice/r16_5ep_layer8/inoculated_v6/config.json

# 3. Evaluate both models
python eval/eval.py --model yourname/model-no-inoculation
python eval/eval.py --model yourname/model-inoculated-v6

# 4. Compare alignment scores in results/*/summary.json
```

---

## SLURM (Cluster Usage)

Example job scripts are provided in `slurm/`:

```bash
sbatch slurm/train.sh   # Submit training job
sbatch slurm/eval.sh    # Submit evaluation job
```

---

## Citation

Training and eval scripts based on homework zero from [CS 2881: AI Safety](https://boazbk.github.io/mltheoryseminar/) and the paper:

```bibtex
@article{model-organisms-em,
  title={Model Organisms for Emergent Misalignment},
  year={2024},
  url={https://arxiv.org/abs/2506.11613}
}
```
