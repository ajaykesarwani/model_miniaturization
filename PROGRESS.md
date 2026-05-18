# Project Progress Log
## Model Miniaturization — Medical Triage Assistant
**Team:** Nalan Thanasekaran
**Repo:** git.fim.uni-passau.de/thanasekaran/model_miniaturization
**Container:** `ssh ailab` → deathstar.dimis.fim.uni-passau.de:32364 (A6000 48GB)

---

## Week 0 — Environment Setup & Teacher Baseline
**Dates:** 2026-05-08 to 2026-05-16

### Goals
- [x] Verify all required packages installed on container
- [x] Confirm GPU access and VRAM
- [x] Run teacher model inference end-to-end
- [x] Set up local backup workflow

### What We Did

**2026-05-08**
- Connected to container via `ssh ailab`
- Found conda env at `/root/envs/miniaturization/`
- Activation: `source /opt/conda/etc/profile.d/conda.sh && conda activate /root/envs/miniaturization`
- Fixed package compatibility issues:
  - `torch 2.4.0` → `2.5.1+cu121` (required by transformers 5.8.0)
  - `torch 2.5.1` → `2.6.0+cu124` (required by CVE-2025-32434 fix in transformers)
  - Driver: 570.172.08, supports CUDA 12.9 — cu124 wheels compatible
- All packages verified: torch 2.6.0, transformers 5.8.0, peft 0.19.1, bitsandbytes 0.49.2, accelerate 1.13.0, wandb 0.26.1, datasets 4.8.5, DeepSpeed

**2026-05-16**
- Pod restarted — needed VPN to reconnect
- Container confirmed running: A6000, 47.4 GB VRAM free
- Ran `aaditya/OpenBioLLM-Llama3-8B` in 4-bit NF4 as stand-in teacher
  - VRAM used: 5.7 GB | free: 45.2 GB
  - Inference pipeline confirmed working
  - Output quality poor without system prompt — expected, not a concern for Week 0

### Blockers
- `JSL-MedLlama-3-8B` (target teacher per spec) requires HuggingFace token — gated model
  - **Action needed:** Create HF account, accept model terms, generate read token

### Final State
| Item | Status |
|---|---|
| Container SSH | Working (requires university VPN) |
| Conda env | `/root/envs/miniaturization/` — all packages installed |
| GPU | A6000 48GB, CUDA 12.9, fully accessible |
| Teacher inference | Working (OpenBioLLM-Llama3-8B, 4-bit NF4) |
| Repo structure | Set up |
| Teacher model decision | Finalised: OpenBioLLM-Llama3-8B |

---

## Week 1 — Synthetic Data Generation
**Dates:** 2026-05-17 to 2026-05-23

### Goals
- [x] Confirm teacher model: aaditya/OpenBioLLM-Llama3-8B (no token needed)
- [x] Set up repo structure (`src/`, `docs/`, `configs/`)
- [ ] Load real datasets: MedDialog, MedMCQA, medical-triage-500
- [x] Build teacher inference pipeline (`src/data_generation/teacher_inference.py`)
- [ ] Generate 50K synthetic triage pairs (15K emergency, 17.5K urgent, 17.5K routine)
- [x] Implement quality filter (oracle-label + primer approach — see notes)
- [ ] Push all code to repo with meaningful commits

### Daily Log

---

#### 2026-05-16

**What we did:**
- Confirmed teacher model: `aaditya/OpenBioLLM-Llama3-8B` (JSL-MedLlama-3-8B still blocked on HF token)
- Set up container folder structure:
  - `/root/model_miniaturization/src/data_generation/`
  - `/root/model_miniaturization/data/synthetic/`
  - `/root/model_miniaturization/configs/`
- Wrote `src/data_generation/teacher_inference.py`:
  - 30 seed symptoms (10 per class: EMERGENCY, URGENT, ROUTINE)
  - Oracle-label prompting + greedy decoding + response primer
  - Saves structured clinical reasoning to `data/synthetic/seed_samples.jsonl`
- Ran pilot: **30/30 samples generated, 10 per class, all HIGH confidence**
- Local backup saved: `src/data_generation/teacher_inference.py`

**Problems faced:**

1. **Parser rejected all 30 samples** — `parse_output()` looked for exact string `"TRIAGE LEVEL: EMERGENCY"` but model writes prose ("The patient's triage level is EMERGENCY").
   - **Fix:** Updated parser to use `re.search(rf'\b{level}\b', raw, re.IGNORECASE)`.

2. **Model inconsistent across sampling runs** — Even at temperature=0.1, the model gave different triage levels across 3 runs, causing the majority-vote filter to reject samples.
   - **Fix:** Switched to greedy decoding (`do_sample=False`) — fully deterministic, one run per sample.

3. **Model EMERGENCY-biases all URGENT cases** — The model cannot distinguish URGENT from EMERGENCY. 9/10 URGENT symptoms were classified as EMERGENCY.
   - **Fix:** Switched to oracle-label approach: provide the correct triage level in the prompt, ask the model to generate clinical reasoning only. Labels come from our curated seed data.

4. **Model hits EOS after listing symptoms** — With oracle-label prompting, the model would output only "Key symptoms: ..." (29–79 chars) and stop, never reaching the reasoning steps.
   - **Fix:** Added response primer — start the assistant turn with `TRIAGE LEVEL: {label}\nKEY SYMPTOMS:` to force the model past the symptom list into the reasoning steps.

**Result:** 30/30 samples pass, perfectly balanced, full 3-step clinical reasoning in every output.

---

## Week 2 — Data Processing & Exploration
**Dates:** 2026-05-24 to 2026-05-30

### Goals
- [ ] Clean and format all datasets into unified schema
- [ ] Train/val/test split (80/10/10)
- [ ] Data exploration notebook (`notebooks/01_data_exploration.ipynb`)
- [ ] Push processed data pipeline to repo

### Notes
*(fill in as work progresses)*

---

## Week 3 — Structured Pruning
**Dates:** 2026-05-31 to 2026-06-06

### Goals
- [ ] Implement Taylor importance scoring (`src/pruning/importance_scoring.py`)
- [ ] Attention head pruning — remove bottom 40% (`src/pruning/head_pruning.py`)
- [ ] Layer dropping — remove middle layers (`src/pruning/layer_dropping.py`)
- [ ] Recovery fine-tuning (1–2 epochs on medical data)
- [ ] Produce intermediate ~3B model
- [ ] Pruning analysis notebook (`notebooks/02_pruning_analysis.ipynb`)

### Notes
*(fill in as work progresses)*

---

## Week 4 — Knowledge Distillation
**Dates:** 2026-06-07 to 2026-06-13

### Goals
- [ ] Implement KD trainer (`src/distillation/kd_trainer.py`)
  - Logit-level KD: Forward KL divergence, T=4.0, α=0.7
  - Feature-level KD: hidden state MSE (`src/distillation/feature_kd.py`)
  - CoT distillation (`src/distillation/cot_distillation.py`)
- [ ] Run full distillation: 3B teacher → Qwen3-0.6B student
- [ ] Track with wandb: kd_loss, task_loss, emergency_recall
- [ ] Save student checkpoints per epoch

### Notes
*(fill in as work progresses)*

---

## Week 5 — Fine-Tuning & Evaluation
**Dates:** 2026-06-14 to 2026-06-20

### Goals
- [ ] LoRA fine-tuning on student (`src/finetuning/lora_finetune.py`) — r=16, α=32
- [ ] Run lm-evaluation-harness: medmcqa, meddialog, mediqa_qa2019
- [ ] Custom metrics (`src/evaluation/custom_metrics.py`):
  - Triage accuracy > 85%
  - Emergency recall > 95% (critical)
  - Macro F1 > 0.83
- [ ] Ablation studies
- [ ] Evaluation results notebook (`notebooks/04_evaluation_results.ipynb`)

### Notes
*(fill in as work progresses)*

---

## Week 6 — Deployment & Report
**Dates:** 2026-06-21 to 2026-06-27

### Goals
- [ ] 4-bit quantization → ~300MB student model
- [ ] GGUF conversion for Ollama
- [ ] Local demo via Ollama
- [ ] Final report (LaTeX → `report/report.pdf`)
- [ ] Final presentation (`presentations/final-presentation.pdf`)
- [ ] All code cleaned, documented, merged to main

### Notes
*(fill in as work progresses)*

---

## Key Decisions Log

| Date | Decision | Reason |
|---|---|---|
| 2026-05-16 | Use `aaditya/OpenBioLLM-Llama3-8B` as teacher model | Open access, no HF token needed, medical domain, 8B params — sufficient for project goals |
| 2026-05-08 | Upgrade torch to 2.6.0+cu124 | CVE-2025-32434 blocks torch.load on 2.5.x; driver supports CUDA 12.9 |
| 2026-05-08 | Load teacher in 4-bit NF4 | A6000 has 48GB but 4-bit keeps inference lean (5.7GB) leaving room for distillation |
| 2026-05-16 | Oracle-label data generation | OpenBioLLM-8B EMERGENCY-biases all URGENT cases; oracle labels from curated seeds are correct by construction; teacher generates reasoning chains (standard CoT synthesis practice) |
| 2026-05-16 | Response priming (`TRIAGE LEVEL: {label}\nKEY SYMPTOMS:`) | Model hits EOS after symptom list without primer; priming forces full structured output |
