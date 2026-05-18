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
│   ├── data_generation/
│   │   └── teacher_inference.py     # Teacher inference + data generation pipeline
│   ├── pruning/                     # Structured pruning (Week 3)
│   ├── distillation/                # Knowledge distillation (Week 4)
│   ├── finetuning/                  # LoRA fine-tuning (Week 5)
│   └── evaluation/                  # Custom metrics and evaluation (Week 5)
├── data/
│   └── synthetic/                   # Generated triage samples (gitignored)
├── configs/                         # Training configs
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

**Target dataset:** 50,000 samples (15K EMERGENCY, 17.5K URGENT, 17.5K ROUTINE)

---

## Setup

**Container:** A6000 48 GB GPU, accessed via university VPN + SSH

```bash
# Activate environment
source /opt/conda/etc/profile.d/conda.sh && conda activate /root/envs/miniaturization

# Run teacher inference
cd /root/model_miniaturization
python src/data_generation/teacher_inference.py
```

**Environment:**
| Package | Version |
|---|---|
| torch | 2.6.0+cu124 |
| transformers | 5.8.0 |
| peft | 0.19.1 |
| bitsandbytes | 0.49.2 |
| accelerate | 1.13.0 |

---

## Target Metrics

| Metric | Target |
|---|---|
| Triage accuracy | > 85% |
| Emergency recall | > 95% |
| Macro F1 | > 0.83 |
| Student VRAM | < 1 GB (4-bit quantized) |
