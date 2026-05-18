# AI Interaction Log — Model Miniaturization Project

---

## Entry #1 - Environment Setup & Package Compatibility

**Date:** 2026-05-08

**Team member(s):** Nalan Thanasekaran

**AI Tool used:** Claude Code

### Context

Week 0: verify the A6000 container environment has all required packages before starting model work.

### Prompt / Task

Run `check_env.py` (provided in course instructions) against the container and fix any issues found.

### AI Output Summary

Claude ran the environment check and identified a version mismatch: `transformers 5.8.0` requires `torch 2.5+` (calls `torch.library.custom_op`), but `torch 2.4.0` was installed. Additionally, `torch.load` was blocked on torch < 2.6 due to CVE-2025-32434, which affected `.bin`-format model loading. Claude suggested upgrading to `torch 2.6.0+cu124` (compatible with the container's CUDA 12.9 driver).

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

### Decision

- [x] Accepted as-is

### Reasoning

Upgrading torch was the correct fix — downgrading transformers would risk breaking peft and other packages. The cu124 wheels are compatible with driver 570 (CUDA 12.9). All 7 packages passed after the upgrade.

### Impact

All packages verified: torch 2.6.0, transformers 5.8.0, peft 0.19.1, bitsandbytes 0.49.2, accelerate 1.13.0, wandb 0.26.1, datasets 4.8.5, DeepSpeed. GPU confirmed: NVIDIA RTX A6000, 48 GB VRAM.

---

## Entry #2 - Teacher Model Baseline

**Date:** 2026-05-16

**Team member(s):** Nalan Thanasekaran

**AI Tool used:** Claude Code

### Context

Week 0 final step: confirm the end-to-end inference pipeline works (load → GPU → generate) before building the data generation pipeline. Selected `aaditya/OpenBioLLM-Llama3-8B` — openly accessible, medically fine-tuned, 8B parameters, no HuggingFace token required.

### Prompt / Task

Write and run a script that loads the teacher model in 4-bit NF4 quantization, reports VRAM, and runs a sample triage prompt.

### AI Output Summary

Claude wrote `teacher_test.py` using `BitsAndBytesConfig` (NF4, float16, double quant) and ran it on the container. Output was poor without a system prompt — expected for Week 0, noted for the Week 1 prompting work.

```
VRAM used: 5.7 GB | VRAM free: 45.2 GB
```

### Decision

- [x] Accepted — pipeline confirmed working

### Reasoning

Goal was to verify the inference stack, not output quality. 5.7 GB VRAM for an 8B model leaves 42+ GB free for distillation — well within budget.

### Impact

Week 0 complete. Teacher inference confirmed working. Torch 2.6.0 resolves CVE-2025-32434. Environment ready for Week 1 data generation.

---

## Entry #3 - Synthetic Data Generation Pipeline

**Date:** 2026-05-16

**Team member(s):** Nalan Thanasekaran

**AI Tool used:** Claude Code

### Context

Week 1: design and implement the teacher inference pipeline to generate seed triage samples for the distillation dataset. Target: 30 seed samples (10 per class: EMERGENCY, URGENT, ROUTINE).

### Prompt / Task

Implement `teacher_inference.py` with a structured output parser, run a pilot, and fix any issues.

### AI Output Summary

Claude implemented the pipeline. During the pilot run, two issues were identified and resolved:

**Issue 1 — Model output format:** The model writes prose ("The patient's triage level is EMERGENCY") rather than structured output. Parser updated to use `re.search` with word-boundary regex instead of exact-string matching.

**Issue 2 — Early EOS:** With oracle-label prompting (correct label provided, model generates reasoning), the model would stop after listing key symptoms and hit EOS before completing the reasoning steps. Fixed by priming the assistant response with `TRIAGE LEVEL: {label}\nKEY SYMPTOMS:`, forcing the model to continue.

**Final pipeline design** (student-designed, AI-implemented):
- Oracle-label prompting: triage label is provided, model generates clinical reasoning to justify it
- Greedy decoding (`do_sample=False`) — deterministic, no sampling noise
- Response primer to prevent early stopping
- Quality filter: output length ≥ 80 chars

**Sample output:**
```
TRIAGE LEVEL: EMERGENCY
KEY SYMPTOMS: sudden onset left-sided weakness, facial drooping, slurred speech
CLINICAL REASONING:
  Step 1: Classic ischemic stroke — FAST symptoms with acute onset
  Step 2: Age 70, neurological deficit — hemorrhagic vs ischemic must be ruled out urgently
  Step 3: CT head and thrombolysis decision required within minutes
CONFIDENCE: HIGH
```

### Decision

- [x] Accepted — oracle-label approach chosen for seed data generation

### Reasoning

For knowledge distillation, what matters is correct labels and coherent clinical reasoning chains. Labels come from curated seed data (known correct); the teacher generates the reasoning. This is standard practice in CoT data synthesis. Greedy decoding gives deterministic, reproducible output.

### Impact

Pilot complete: 30/30 samples generated, saved to `data/synthetic/seed_samples.jsonl`. Distribution: 10 EMERGENCY / 10 URGENT / 10 ROUTINE (~24 KB). Pipeline ready to scale to 50K samples.

---
