import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
ROOT = Path("/root/model_miniaturization")
BASE = "Qwen/Qwen3-0.6B"
PB = ROOT / "data/pruning/qwen3_pruned_heads_layers"
PA = ROOT / "data/pruning/qwen3_pruned_lora/adapter"
SYS = "You are a senior emergency physician. Given a patient description, classify the triage level.\n\nRespond with ONLY the following format:\nTRIAGE LEVEL: [EMERGENCY/URGENT/ROUTINE]\nKEY SYMPTOMS: [list]\nCLINICAL REASONING:\n  Step 1: [x]\nCONFIDENCE: [HIGH/MEDIUM/LOW]"
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
def prompt(d): return f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nPatient: {d}<|im_end|>\n<|im_start|>assistant\n"
print("Loading pruned base in FP16...")
b = AutoModelForCausalLM.from_pretrained(str(PB), torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
m = PeftModel.from_pretrained(b, str(PA)).eval()
def gen(d):
    inp = tok(prompt(d), return_tensors="pt", truncation=True, max_length=512).to(m.device)
    with torch.inference_mode():
        out = m.generate(**inp, max_new_tokens=150, do_sample=False, pad_token_id=tok.eos_token_id, repetition_penalty=1.1)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
for d in ["I'm nauseous", "crushing chest pain radiating to left arm, sweating, SOB, BP 90/60"]:
    print(f"\n--- {d[:45]} ---\n{gen(d)[:260]}")
print("\nDONE_FP16")
