# Model Miniaturization Project Agent Guidelines & Activity Log

This file documents the changes made to the project during the July 14, 2026 session. Antigravity will load these guidelines automatically to maintain context.

## Summary of Completed Work

### 1. Codebase Bug Fixes
* **File Modified:** [evaluate_student_mimic.py](file:///root/model_miniaturization/src/evaluation/evaluate_student_mimic.py)
* **What was done:** Fixed a critical prompt formatting bug. The script originally used the Llama 3 prompt template, which caused a 100% formatting failure (unparsed rate) on the `Qwen3-0.6B` student model. It has been corrected to use Qwen's native **ChatML** format, resolving the unparsed failure rate.

### 2. Missing Deliverables Created
* **File Created:** [app_streamlit.py](file:///root/model_miniaturization/app_streamlit.py)
  * Implemented a clean, premium Streamlit dashboard UI for the triage chatbot demo. It loads the student model in 4-bit and performs real-time inference on GPU with color-coded triage badges and VRAM monitoring.
* **File Created:** [requirements.txt](file:///root/model_miniaturization/requirements.txt)
  * Compiled all core Python dependencies (PyTorch, Transformers, PEFT, bitsandbytes, scikit-learn, Gradio, Streamlit, etc.) to align the repository with the presentation claims.

### 3. Presentation Audits & Improvisations
* **Files Modified:** [model_miniaturization.pptx](file:///root/model_miniaturization/model_miniaturization.pptx) (Original backed up to [model_miniaturization_original.pptx](file:///root/model_miniaturization/model_miniaturization_original.pptx))
* **Improvisations Performed:**
  * **Slide 11 Charts:** Updated the Triage Accuracy and Emergency Recall charts to display the final **SFT v4** and **Pruned Student** metrics on the Synthetic Test set (both at **100% Accuracy and Recall**) instead of early SFT v1/v2 numbers.
  * **Slide 12 Chart:** Updated the MIMIC Evaluation chart to consistently use the **42-patient held-out test set** metrics across all models, ensuring scientific validity (instead of mixing training-set performance with held-out test performance).
  * **Slide 14 Text:** Adjusted the text to represent the true held-out test set results for the default argmax logit sweep (replacing $80.2\%$ with $74.9\%$ accuracy, $90.4\%$ with $80.9\%$ emergency recall, and $\sim 1\text{ in }10$ with $\sim 2\text{ in }10$ missed emergencies).
  * **Slide 20 (New):** Created a new slide titled **"DATA QUALITY: Synthetic Reasoning Validation"** featuring a table of the BERTScore semantic coherence metrics (F1 = 0.82) against PubMedQA.
  * **Slide 21 (New):** Added a slide titled **"BASELINE COMPARISON: Raw Student vs. Raw Teacher"** featuring a head-to-head metrics comparison table (accuracy, recall, unparsed rates, macro-F1) across all four evaluation datasets.
  * **Slide 22 (New):** Added a slide titled **"COMPREHENSIVE COMPARISON: Student States Across Datasets"** comparing different training states of the student model (Raw, SFT, Pruned SFT, KD) across all four datasets.

## Project Context Reference
* **Teacher Model:** `aaditya/OpenBioLLM-Llama3-8B` (8B params, NF4)
* **Student Model:** `Qwen/Qwen3-0.6B` (0.6B params)
* **Best Neural Performer (MIMIC held-out test):** Pruned Student + SFT (90.5% Acc / 91.7% Recall)
* **Strongest Baseline Performer (MIMIC held-out test):** TF-IDF + Logistic Regression (88.1% Acc / 95.8% Recall)
