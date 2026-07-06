# Medical Triage Assistant — Model Miniaturization

**Course:** Applied Artificial Intelligence Lab, Summer Semester 2026  
**University:** University of Passau  
**Student:** Nalan Thanasekaran

---

## Project Overview

This project implements a compressed medical triage assistant using knowledge distillation. A large medical language model (teacher) is used to generate structured clinical reasoning, which is then distilled into a lightweight student model deployable on resource-constrained hardware.

**Goal:** Compress an 8B-parameter medical LLM into a ~0.6B student model that retains >85% triage accuracy and >95% emergency recall, while running in under 1 GB VRAM.

---

## Architecture

| Component | Model | Size |
|---|---|---|
| Teacher | aaditya/OpenBioLLM-Llama3-8B | 8B params, 4-bit NF4 (~5.7 GB VRAM) |
| Student | Qwen/Qwen3-0.6B | 0.6B params (~300 MB quantized) |

**Pipeline:**
1. Teacher generates labeled clinical reasoning chains (synthetic data)
2. Structured pruning: attention head pruning + layer dropping → ~3B intermediate model
3. Knowledge distillation: logit-level KL divergence + feature-level MSE
4. LoRA fine-tuning on the student

---

## Triage Classes

| Class | Definition |
|---|---|
| EMERGENCY | Immediately life-threatening — requires intervention within minutes |
| URGENT | Serious but stable — requires evaluation within 1–2 hours |
| ROUTINE | Non-urgent — can be seen in a scheduled appointment |

---

## Project Structure

```
model_miniaturization/
├── src/
│   ├── approach2/                   # Datasets merging, NLI filtering, and fine-tuning pipelines
│   ├── data_generation/
│   │   ├── teacher_inference.py     # Teacher inference + synthetic data generation pipeline
│   │   └── prepare_dataset.py       # Stratified dataset split (80/10/10)
│   ├── pruning/
│   │   ├── importancescoring.py     # Taylor importance scoring for heads & layers
│   │   ├── headpruning.py           # Attention head pruning (bottom 40%)
│   │   └── layerdropping.py         # Layer dropping (drops 5 middle layers)
│   ├── distillation/
│   │   ├── kdtrainer.py             # Knowledge distillation trainer for Qwen3-0.6B student
│   │   ├── kdtrainer_pruned.py      # Knowledge distillation trainer for pruned student base
│   │   ├── featurekd.py             # Feature hidden state MSE loss
│   │   └── cotdistillation.py       # Chain-of-Thought cross-entropy loss helper
│   ├── finetuning/
│   │   └── lorafinetune.py          # LoRA fine-tuning script
│   └── evaluation/
│       ├── evaluate_teacher_n.py    # Teacher model test evaluation (synthetic & real Latvia)
│       ├── evaluate_distilled.py    # Generation-based evaluation for distilled students
│       └── evaluate_distilled_logits.py # Logit-based evaluation for distilled students
├── data/
│   ├── approach2/                   # Merged datasets and fine-tuning adapters (gitignored)
│   ├── distillation/                # Distillation model weights (gitignored)
│   ├── pruning/                     # Pruned base model weights (gitignored)
│   └── synthetic/                   # Generated synthetic datasets (gitignored)
├── configs/                         # Training configuration files
├── notebooks/                       # Exploration and analysis notebooks
└── report/                          # Final report (LaTeX)
```

---

## Data Generation

The teacher model generates synthetic triage samples using oracle-label prompting: given a patient description and its correct triage level, the model produces structured clinical reasoning with key symptoms and a 3-step reasoning chain.

**Seed pilot result:** 30 samples (10 per class), perfectly balanced.

**Sample output:**
```
TRIAGE LEVEL: EMERGENCY
KEY SYMPTOMS: sudden onset left-sided weakness, facial drooping, slurred speech
CLINICAL REASONING:
  Step 1: Classic ischemic stroke — FAST symptoms with acute neurological deficit
  Step 2: Age 70 with acute onset — hemorrhagic vs ischemic must be ruled out urgently
  Step 3: CT head and thrombolysis decision required within minutes
CONFIDENCE: HIGH
```

**Target dataset:** 50,000 samples (15K EMERGENCY, 17.5K URGENT, 17.5K ROUTINE) generated under `data/synthetic/symptoms_50k.jsonl`.

---

## Setup & Running the Pipeline

### Environment Activation
All scripts should be executed using the custom Conda environment on the university container:
```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate /root/envs/miniaturization
cd /root/model_miniaturization
```

### 1. Pruning Pipeline (Week 3)
Calculate Taylor importance scores, prune attention heads, and drop layers:
```bash
# Importance scoring
python src/pruning/importancescoring.py
# Prune bottom 40% attention heads
python src/pruning/headpruning.py
# Drop middle layers (drops middle 5 layers)
python src/pruning/layerdropping.py
```
Output model weights are saved at `data/pruning/qwen3_pruned_heads_layers/`.

### 2. Knowledge Distillation (Week 4)
Run probability/logit-level KL-divergence distillation from the 8B teacher into the student models:
```bash
# Distill into baseline student
python src/distillation/kdtrainer.py --epochs 1 --batch_size 2
# Distill into pruned student base
python src/distillation/kdtrainer_pruned.py --epochs 1 --batch_size 2
```

### 3. LoRA Fine-Tuning (Week 5)
Fine-tune the student model (distilled or base) on the combined dataset (synthetic + real-patient datasets):
```bash
python src/finetuning/lorafinetune.py --epochs 3 --batch_size 4 --grad_accum 8
```

---

## Evaluation results & Target Metrics

| Metric | Target | Fine-tuned Student (Qwen3-0.6B+LoRA) | Pruned Student (Qwen3-Pruned+LoRA) | Distilled Student (Qwen3-KD-0.6B) |
|---|---|---|---|---|
| **Triage Accuracy** | > 85% | **90.8%** | **90.5%** | **35.3%** (34.7% on real) |
| **Emergency Recall** | > 95% | **82.7%** (100% w/ logit sweep) | **91.7%** (100% w/ logit sweep) | **100.0%** (98.6% on real) |
| **Macro F1** | > 0.83 | **0.909** | **0.602** (MIMIC) | **0.285** |
| **Student VRAM** | < 1 GB | **0.54 GB** | **0.54 GB** | **0.54 GB** |

